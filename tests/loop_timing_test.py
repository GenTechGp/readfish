"""Per-loop stage timing appended to the main loop's batch performance line."""

import logging
from collections import Counter
from types import SimpleNamespace

import pytest

from readfish.entry_points import targets
from readfish._statistics import ReadfishStatistics


@pytest.fixture
def clock(monkeypatch):
    """targets.timer as a clock that only moves when ``clock[0]`` is advanced."""
    now = [0.0]
    monkeypatch.setattr(targets, "timer", lambda: now[0])
    return now


def test_timed_passes_items_through_lazily(clock):
    pulled = []

    def source():
        for item in ("a", "b", "c"):
            pulled.append(item)
            clock[0] += 0.5
            yield item

    times = Counter()
    timed = targets._timed(source(), times, "stage")
    assert next(timed) == "a"
    # Lazy: getting one item pulls only one from the source.
    assert pulled == ["a"]
    assert list(timed) == ["b", "c"]
    assert times == {"stage": 1.5, "stage_first": 0.5}


def test_timed_propagates_errors():
    def source():
        yield 1
        raise ValueError("boom")

    timed = targets._timed(source(), Counter(), "stage")
    assert next(timed) == 1
    with pytest.raises(ValueError, match="boom"):
        next(timed)


def test_loop_timing_format():
    times = Counter(
        chunks=0.001,
        basecall_call=0.0,
        basecall=0.4,
        basecall_first=0.2,
        map_call=0.0,
        aligns=0.42,
        loop=0.43,
        actions=0.002,
    )
    assert targets._loop_timing(times, 0.435) == (
        "Loop ms: chunks=1.0, basecall=400.0 (first=200.0), map=20.0, "
        "decide=10.0, actions=2.0, other=2.0"
    )


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_main_loop_logs_stage_times(monkeypatch, clock):
    """One pass of Analysis.run with fake MinKNOW, Dorado and aligner, where
    each stage advances the clock by a known amount."""
    sent = []

    class Client:
        channel_count = 2
        signal_dtype = "i2"
        calibration_values = {}
        mk_run_dir = "."
        connection = None
        wait_for_sequencing_to_start = False

        def __init__(self):
            self.loops = 0

        @property
        def is_sequencing(self):
            self.loops += 1
            return self.loops == 1

        def log_stream_health(self):
            pass

        def get_read_chunks(self, batch_size, last):
            clock[0] += 0.001
            return [(1, "chunk1"), (2, "chunk2")]

        def unblock_read_batch(self, reads, duration):
            clock[0] += 0.002
            sent.append(("unblock", list(reads)))

        def stop_receiving_batch(self, reads):
            sent.append(("stop_receiving", list(reads)))

    def basecall(chunks, dtype, calibration):
        for channel, _ in chunks:
            clock[0] += 0.2
            yield SimpleNamespace(
                channel=channel, read_id=f"r{channel}", seq="ACGT", barcode=None
            )

    def map_reads(calls):
        for result in calls:
            clock[0] += 0.01
            yield result

    def check_override_action(control, action, result, *args):
        clock[0] += 0.003
        unblock_list = args[-1]
        unblock_list.append((result.channel, result.read_id))
        return action, None, False, None

    condition = SimpleNamespace(
        name="c", get_action=lambda decision: targets.Action.unblock
    )
    analysis = targets.Analysis.__new__(targets.Analysis)
    analysis.client = Client()
    analysis.conf = SimpleNamespace(
        write_channels_toml=lambda path: None,
        get_conditions=lambda channel, barcode: (False, condition),
        get_region=lambda channel: None,
    )
    analysis.caller = SimpleNamespace(basecall=basecall, disconnect=lambda: None)
    analysis.mapper = SimpleNamespace(
        describe=lambda regions, barcodes: "",
        initialised=True,
        map_reads=map_reads,
        disconnect=lambda: None,
    )
    analysis.conf.regions = analysis.conf.barcodes = []
    analysis.chunk_tracker = SimpleNamespace(seen=lambda channel, read_id: 1)
    analysis.check_override_action = check_override_action
    analysis.loop_statistics = ReadfishStatistics(None)
    analysis.reload_toml = lambda mtime: mtime
    analysis.readfish_started_during_sequencing = False
    analysis.throttle = 0
    analysis.unblock_duration = 0.1
    handler = ListHandler()
    analysis.logger = logging.getLogger("test_loop_timing")
    analysis.logger.addHandler(handler)
    analysis.logger.setLevel(logging.INFO)
    analysis.logger.propagate = False
    monkeypatch.setattr(targets, "send_message", lambda *args: None)
    monkeypatch.setattr(
        targets, "make_decision", lambda conf, result: targets.Decision.single_off
    )

    try:
        analysis.run()
    finally:
        analysis.logger.removeHandler(handler)

    batch_lines = [
        r.getMessage() for r in handler.records if "R/" in r.getMessage()
    ]
    assert batch_lines == [
        "0002R/0.4290s; Avg: 0002R/0.4290s; Seq:0; Unb:2; Pro:0; "
        "Slow batches (>1.00s): 0/1; "
        "Loop ms: chunks=1.0, basecall=400.0 (first=200.0), map=20.0, "
        "decide=6.0, actions=2.0, other=0.0"
    ]
    # Behaviour unchanged: both reads were unblocked in one batch.
    assert sent == [
        ("unblock", [(1, "r1"), (2, "r2")]),
        ("stop_receiving", []),
    ]
