"""Messages from the vendored read until client must reach readfish's log handlers."""

import logging

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
