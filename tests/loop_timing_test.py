"""Per-loop stage timing appended to the main loop's batch performance line."""

import logging
from collections import Counter
from types import SimpleNamespace

import pytest

from readfish.entry_points import targets
from readfish._statistics import ReadfishStatistics, DEBUG_STATS_LOG_FIELDS

# The debug TSV columns before the gtg additions, which analysis scripts read
#    by position
UPSTREAM_DEBUG_FIELDS = (
    "client_iteration",
    "read_in_loop",
    "read_id",
    "channel",
    "seq_len",
    "counter",
    "mode",
    "decision",
    "condition",
    "barcode",
    "previous_action",
    "action_override",
    "timestamp",
)


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


def read_data(read_id, start, samples):
    """A fake ReadData with ``samples`` int16 samples of signal."""
    return SimpleNamespace(
        id=read_id, chunk_start_sample=start, raw_data=b"\0\0" * samples
    )


def run_one_loop(monkeypatch, clock, chunks, last_acquired=10_000):
    """One pass of Analysis.run with fake MinKNOW, Dorado and aligner, where
    each stage advances the clock by a known amount. Returns the batch
    performance lines, the actions sent and the debug TSV rows."""
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
            self.last_acquired = last_acquired

        @property
        def is_sequencing(self):
            self.loops += 1
            return self.loops == 1

        def log_stream_health(self):
            pass

        def get_read_chunks(self, batch_size, last):
            clock[0] += 0.001
            return list(chunks)

        def unblock_read_batch(self, reads, duration):
            clock[0] += 0.002
            sent.append(("unblock", list(reads)))

        def stop_receiving_batch(self, reads):
            sent.append(("stop_receiving", list(reads)))

    def basecall(chunks, dtype, calibration):
        for channel, read in chunks:
            clock[0] += 0.2
            yield SimpleNamespace(
                channel=channel, read_id=read.id, seq="ACGT", barcode=None
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
    debug_handler = ListHandler()
    analysis.loop_statistics.debug_logger.addHandler(debug_handler)
    monkeypatch.setattr(targets, "send_message", lambda *args: None)
    monkeypatch.setattr(
        targets, "make_decision", lambda conf, result: targets.Decision.single_off
    )

    try:
        analysis.run()
    finally:
        analysis.logger.removeHandler(handler)
        analysis.loop_statistics.debug_logger.removeHandler(debug_handler)

    batch_lines = [r.getMessage() for r in handler.records if "R/" in r.getMessage()]
    debug_rows = [r.getMessage().split("\t") for r in debug_handler.records]
    return batch_lines, sent, debug_rows


TWO_READS = [(1, read_data("r1", 5_000, 4_000)), (2, read_data("r2", 9_000, 500))]


def test_main_loop_logs_stage_times(monkeypatch, clock):
    batch_lines, sent, _ = run_one_loop(monkeypatch, clock, TWO_READS)
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


def test_debug_tsv_signal_columns(monkeypatch, clock):
    _, sent, rows = run_one_loop(monkeypatch, clock, TWO_READS, last_acquired=10_000)
    assert all(len(row) == len(DEBUG_STATS_LOG_FIELDS) for row in rows)
    # The upstream columns keep their positions; the new ones are appended.
    assert DEBUG_STATS_LOG_FIELDS[:13] == UPSTREAM_DEBUG_FIELDS
    assert [row[2] for row in rows] == ["r1", "r2"]
    assert [dict(zip(DEBUG_STATS_LOG_FIELDS[13:], row[13:])) for row in rows] == [
        # r1: 4,000 samples from 5,000, so it ends at 9,000; 1,000 behind
        {"signal_start_sample": "5000", "signal_samples": "4000",
         "samples_behind": "1000"},
        # r2: 500 samples from 9,000, so it ends at 9,500; 500 behind
        {"signal_start_sample": "9000", "signal_samples": "500",
         "samples_behind": "500"},
    ]
    assert sent[0] == ("unblock", [(1, "r1"), (2, "r2")])


def test_debug_tsv_samples_behind_unknown_before_first_message(monkeypatch, clock):
    _, _, rows = run_one_loop(monkeypatch, clock, TWO_READS[:1], last_acquired=None)
    assert rows[0][13:] == ["5000", "4000", "None"]


def test_signal_columns_unknown_read():
    assert targets._signal_columns(None, 10_000) == {
        "signal_start_sample": None,
        "signal_samples": None,
        "samples_behind": None,
    }


def test_stream_records_last_acquired():
    """The stream thread keeps the latest acquired for samples_behind."""
    from threading import Event
    import readfish.read_until.base as base

    client = base.ReadUntilClient.__new__(base.ReadUntilClient)
    assert client.last_acquired is None
    client.logger = logging.getLogger("readfish.read_until")
    client.running = Event()
    client.running.set()
    progress = SimpleNamespace(raw_per_channel=SimpleNamespace(acquired=1234))
    client.connection = SimpleNamespace(
        acquisition=SimpleNamespace(get_progress=lambda: progress)
    )
    client._process_reads([SimpleNamespace(action_responses=[], channels={})])
    assert client.last_acquired == 1234
