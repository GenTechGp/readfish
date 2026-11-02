"""The read stream's interval update period is configurable, default 15 minutes."""

import argparse
import logging
from threading import Event
from types import SimpleNamespace

import pytest

import readfish.read_until.base as base
from readfish.entry_points import targets


def parse_targets_args(*extra):
    parser = argparse.ArgumentParser()
    for flags, opts in targets._cli:
        parser.add_argument(flags, **opts)
    required = ["--device", "X1", "--experiment-name", "e", "--toml", "t.toml"]
    return parser.parse_args(required + list(extra))


def test_default_interval_is_15_minutes():
    assert base.ReadUntilClient.log_interval_secs == 60 * 15
    assert parse_targets_args().interval_update_seconds == 60 * 15


def test_interval_flag_parsed():
    args = parse_targets_args("--interval-update-seconds", "60")
    assert args.interval_update_seconds == 60.0


@pytest.mark.parametrize("value", ["0", "-5", "nan", "soon"])
def test_interval_flag_rejects_non_positive(value, capsys):
    with pytest.raises(SystemExit):
        parse_targets_args("--interval-update-seconds", value)
    assert "--interval-update-seconds" in capsys.readouterr().err


def test_process_reads_uses_client_interval(monkeypatch, caplog):
    now = [1000.0]
    monkeypatch.setattr(base.time, "time", lambda: now[0])

    client = base.ReadUntilClient.__new__(base.ReadUntilClient)
    client.logger = logging.getLogger("readfish.read_until")
    client.running = Event()
    client.running.set()
    progress = SimpleNamespace(raw_per_channel=SimpleNamespace(acquired=0))
    client.connection = SimpleNamespace(
        acquisition=SimpleNamespace(get_progress=lambda: progress)
    )
    client.data_queue = base.ReadCache(size=2)
    client.filter_strands = True
    client.log_interval_secs = 60

    def messages():
        for _ in range(3):
            now[0] += 61
            yield SimpleNamespace(action_responses=[], channels={})

    with caplog.at_level(logging.INFO, logger="readfish.read_until"):
        client._process_reads(messages())

    updates = [m for m in caplog.messages if m.startswith("Interval update")]
    assert len(updates) == 3
