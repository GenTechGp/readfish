"""Messages from the vendored read until client must reach readfish's log handlers."""

import logging
from threading import Event

import grpc
import pytest

import readfish.read_until.base as base


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def readfish_handler():
    logger = logging.getLogger("readfish")
    handler = ListHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def test_client_messages_reach_readfish_logger(monkeypatch, readfish_handler):
    def no_connection(**kwargs):
        raise ConnectionError("no MinKNOW here")

    monkeypatch.setattr(base, "Connection", no_connection)
    with pytest.raises(ConnectionError):
        base.ReadUntilClient(mk_host="localhost", mk_port=1)

    records = [r for r in readfish_handler.records if r.name == "readfish.read_until"]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "Failed to connect to read until" in records[0].getMessage()


class FakeStream:
    """Stands in for the gRPC stream returned by get_live_reads."""

    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class FakeConnection:
    def __init__(self, stream):
        self.data = self
        self.stream = stream

    def get_live_reads(self, requests):
        return self.stream


class Cancelled(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.CANCELLED


@pytest.fixture
def run_stream(readfish_handler):
    """Run the stream thread body with ``process_reads`` standing in for the
    stream; return the client, the stream and the (level, message) log lines."""

    def run(process_reads):
        stream = FakeStream()
        client = base.ReadUntilClient.__new__(base.ReadUntilClient)
        client.logger = logging.getLogger("readfish.read_until")
        client.connection = FakeConnection(stream)
        client.running = Event()
        client._runner = lambda **kwargs: iter(())
        client._process_reads = lambda reads: process_reads(client)
        client._run()
        lines = [(r.levelno, r.getMessage()) for r in readfish_handler.records]
        return client, stream, lines

    return run


def test_stream_closed_by_minknow(run_stream):
    client, stream, lines = run_stream(lambda c: None)
    assert [lvl for lvl, _ in lines] == [logging.WARNING, logging.ERROR]
    assert "closed by MinKNOW" in lines[0][1]
    assert "while readfish is still running" in lines[1][1]
    assert stream.cancelled
    # logging only: the running flag is left as it was
    assert client.is_running


def test_stream_processing_fails(run_stream):
    def fail(client):
        raise RuntimeError("boom")

    client, stream, lines = run_stream(fail)
    assert [lvl for lvl, _ in lines] == [logging.ERROR, logging.ERROR]
    assert "Failed Processing reads" in lines[0][1]
    assert "while readfish is still running" in lines[1][1]
    assert client.is_running


def test_stream_cancelled_by_minknow(run_stream):
    def cancel(client):
        raise Cancelled()

    client, stream, lines = run_stream(cancel)
    assert [lvl for lvl, _ in lines] == [logging.INFO, logging.ERROR]
    assert "finished due to StatusCode.CANCELLED" in lines[0][1]
    assert "while readfish is still running" in lines[1][1]


def test_stream_ends_after_reset(run_stream):
    def reset(client):
        client.running.clear()

    client, stream, lines = run_stream(reset)
    assert [lvl for lvl, _ in lines] == [logging.INFO]
    assert lines[0][1].startswith("Read stream thread exited after")
