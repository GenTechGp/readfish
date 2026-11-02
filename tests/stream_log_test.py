"""The per-message read stream log (targets --stream-log), off by default."""

import argparse
import logging
from threading import Event
from types import SimpleNamespace

import pytest

import readfish.read_until.base as base
from readfish.entry_points import targets

Response = base.data_pb2.GetLiveReadsResponse.ActionResponse


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def stream_rows():
    """A stream logger like the one targets sets up, collecting rows."""
    logger = logging.getLogger("test_read_stream_stats")
    handler = ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield logger, handler
    logger.removeHandler(handler)


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


def parse_targets_args(*extra):
    parser = argparse.ArgumentParser()
    for flags, opts in targets._cli:
        parser.add_argument(flags, **opts)
    required = ["--device", "X1", "--experiment-name", "e", "--toml", "t.toml"]
    return parser.parse_args(required + list(extra))


def test_stream_log_flag_off_by_default():
    assert parse_targets_args().stream_log is False
    assert parse_targets_args("--stream-log").stream_log is True
    assert base.ReadUntilClient.stream_logger is None


def chunk(start, length, classes=(1,)):
    return SimpleNamespace(
        id=f"r{start}",
        chunk_start_sample=start,
        chunk_length=length,
        raw_data=b"",
        chunk_classifications=list(classes),
    )


def message(seconds, samples, channels=None, responses=()):
    return SimpleNamespace(
        seconds_since_start=seconds,
        samples_since_start=samples,
        channels=channels or {},
        action_responses=[
            SimpleNamespace(action_id="a", response=r) for r in responses
        ],
    )


def make_client(monkeypatch, progresses, stream_logger=None):
    """A client whose get_progress() returns ``progresses`` in turn, with
    time.time() fixed at 1000 and each get_progress() taking 2.5 ms."""
    monkeypatch.setattr(base.time, "time", lambda: 1000.0)
    perf = iter(x * 0.0025 for x in range(1000))
    monkeypatch.setattr(base.time, "perf_counter", lambda: next(perf))

    client = base.ReadUntilClient.__new__(base.ReadUntilClient)
    client.logger = logging.getLogger("readfish.read_until")
    client.running = Event()
    client.running.set()
    progresses = iter(progresses)
    client.connection = SimpleNamespace(
        acquisition=SimpleNamespace(
            get_progress=lambda: SimpleNamespace(raw_per_channel=next(progresses))
        )
    )
    client.data_queue = base.ReadCache(size=8)
    client.one_chunk = False
    client.filter_strands = True
    client.strand_classes = {1}
    client.sent_actions = {"a": "unblock"}
    if stream_logger is not None:
        client.stream_logger = stream_logger
    return client


def test_stream_log_rows(monkeypatch, stream_rows):
    logger, handler = stream_rows
    client = make_client(
        monkeypatch,
        [
            SimpleNamespace(acquired=10_000, processed=9_500),
            SimpleNamespace(acquired=11_500, processed=11_000),
        ],
        stream_logger=logger,
    )
    client._process_reads(
        [
            message(
                2.0,
                10_000,
                channels={
                    1: chunk(5_000, 4_000),
                    2: chunk(8_000, 1_000, classes=(2,)),
                    3: chunk(9_000, 500),
                },
                responses=[
                    Response.SUCCESS,
                    Response.SUCCESS,
                    Response.FAILED_READ_FINISHED,
                    7,
                ],
            ),
            message(2.3, 11_500),
        ]
    )
    rows = [r.getMessage().split("\t") for r in handler.records]
    assert all(len(row) == len(base.STREAM_LOG_FIELDS) for row in rows)
    assert [dict(zip(base.STREAM_LOG_FIELDS, row)) for row in rows] == [
        {
            "recv_time": "1000.000000",
            "mk_seconds_since_start": "2.0",
            "mk_samples_since_start": "10000",
            "progress_rpc_ms": "2.500",
            "acquired": "10000",
            "processed": "9500",
            "n_chunks": "3",
            "n_dropped": "1",
            # behind start: 5000, 2000, 1000; behind end: 1000, 1000, 500
            "behind_start_min": "1000",
            "behind_start_median": "2000",
            "behind_start_max": "5000",
            "behind_end_min": "500",
            "behind_end_median": "1000",
            "behind_end_max": "1000",
            "n_responses": "4",
            "success": "2",
            "failed_read_finished": "1",
            "failed_read_too_long": "0",
            "other": "1",
            "queue_length": "2",
        },
        {
            "recv_time": "1000.000000",
            "mk_seconds_since_start": "2.3",
            "mk_samples_since_start": "11500",
            "progress_rpc_ms": "2.500",
            "acquired": "11500",
            "processed": "11000",
            "n_chunks": "0",
            "n_dropped": "0",
            "behind_start_min": "NA",
            "behind_start_median": "NA",
            "behind_start_max": "NA",
            "behind_end_min": "NA",
            "behind_end_median": "NA",
            "behind_end_max": "NA",
            "n_responses": "0",
            "success": "0",
            "failed_read_finished": "0",
            "failed_read_too_long": "0",
            "other": "0",
            "queue_length": "2",
        },
    ]
    # Behaviour unchanged: the two strand chunks were queued, the pore one dropped.
    assert client.queue_length == 2


def test_stream_log_off_writes_nothing(monkeypatch):
    client = make_client(monkeypatch, [SimpleNamespace(acquired=10, processed=10)])

    def fail(*args):
        raise AssertionError("stream log row written while off")

    monkeypatch.setattr(client, "_log_stream_message", fail)
    client._process_reads([message(1.0, 10, channels={1: chunk(0, 10)})])
    assert client.queue_length == 1


def test_stream_log_failure_warned_once_and_stream_continues(
    monkeypatch, stream_rows, readfish_handler
):
    logger, handler = stream_rows
    client = make_client(
        monkeypatch,
        [SimpleNamespace(acquired=100, processed=100)] * 3,
        stream_logger=logger,
    )
    no_length = SimpleNamespace(
        id="bad", chunk_start_sample=0, raw_data=b"", chunk_classifications=[1]
    )
    client._process_reads(
        [
            message(1.0, 100, channels={1: no_length}),
            message(1.1, 100, channels={2: no_length}),
            message(1.2, 100, channels={3: chunk(0, 50)}),
        ]
    )
    warnings = [
        r for r in readfish_handler.records if r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "Could not write a read stream log row" in warnings[0].getMessage()
    # The bad rows are skipped, the good row is written, and all chunks
    #    still reached the queue.
    assert len(handler.records) == 1
    assert handler.records[0].getMessage().split("\t")[6] == "1"
    assert client.queue_length == 3
