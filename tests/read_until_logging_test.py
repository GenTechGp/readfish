"""Log messages about the read stream and run state must reach readfish's log handlers."""

import logging
from collections import Counter
from threading import Event
from types import SimpleNamespace

import grpc
import pytest
from minknow_api import acquisition_pb2, protocol_service

import readfish.read_until.base as base
from readfish._read_until_client import RUClient


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


def test_acquisition_state_changes_logged(readfish_handler):
    A = acquisition_pb2.AcquisitionState
    states = [
        acquisition_pb2.AcquisitionRunInfo(state=A.ACQUISITION_RUNNING),
        acquisition_pb2.AcquisitionRunInfo(state=A.ACQUISITION_RUNNING),
        acquisition_pb2.AcquisitionRunInfo(state=A.ACQUISITION_PAUSED),
        acquisition_pb2.AcquisitionRunInfo(state=A.ACQUISITION_RUNNING),
        acquisition_pb2.AcquisitionRunInfo(
            state=A.ACQUISITION_FINISHING,
            stop_reason=acquisition_pb2.AcquisitionStopReason.STOPPED_USER_REQUESTED,
        ),
    ]

    class FakeAcquisition:
        def get_acquisition_info(self):
            return states.pop(0)

    client = RUClient.__new__(RUClient)
    client.log = logging.getLogger("readfish._read_until_client")
    client.connection = SimpleNamespace(acquisition=FakeAcquisition())
    client.current_acquisition_state = None
    seen = []
    while states:
        client.set_acquisition_state()
        seen.append(client.current_acquisition_state)

    assert seen == [
        A.ACQUISITION_RUNNING,
        A.ACQUISITION_RUNNING,
        A.ACQUISITION_PAUSED,
        A.ACQUISITION_RUNNING,
        A.ACQUISITION_FINISHING,
    ]
    assert [r.getMessage() for r in readfish_handler.records] == [
        "Acquisition state changed to ACQUISITION_RUNNING",
        "Acquisition state changed to ACQUISITION_PAUSED",
        "Acquisition state changed to ACQUISITION_RUNNING",
        "Acquisition state changed to ACQUISITION_FINISHING"
        " (stop reason STOPPED_USER_REQUESTED)",
    ]


def test_messages_update_last_message_time():
    client = base.ReadUntilClient.__new__(base.ReadUntilClient)
    client.logger = logging.getLogger("readfish.read_until")
    client.running = Event()
    client.running.set()
    progress = SimpleNamespace(raw_per_channel=SimpleNamespace(acquired=0))
    client.connection = SimpleNamespace(
        acquisition=SimpleNamespace(get_progress=lambda: progress)
    )
    client.last_message_time = None
    client._process_reads([SimpleNamespace(action_responses=[], channels={})])
    assert client.last_message_time is not None


class FakeThread:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive


def test_stream_health_logging(monkeypatch, readfish_handler):
    import readfish._read_until_client as ru

    now = [1000.0]
    monkeypatch.setattr(ru.time, "time", lambda: now[0])
    client = RUClient.__new__(RUClient)
    client.log = logging.getLogger("readfish._read_until_client")
    client._process_thread = FakeThread()
    client.last_message_time = 1000.0
    client._stream_silent_since = None
    client._stream_stopped_logged = False
    client.current_protocol_phase = protocol_service.PHASE_PAUSED
    client.current_acquisition_state = acquisition_pb2.AcquisitionState.ACQUISITION_PAUSED

    def check(at):
        now[0] = at
        before = len(readfish_handler.records)
        client.log_stream_health()
        return [r.getMessage() for r in readfish_handler.records[before:]]

    assert check(1005) == []  # recent data
    assert check(1031) == [
        "No data from MinKNOW for 31 s"
        " (protocol phase PHASE_PAUSED, acquisition state ACQUISITION_PAUSED)."
    ]
    assert check(1100) == []  # still silent: logged once
    client.last_message_time = 1120.0  # a message arrives
    client.current_protocol_phase = protocol_service.PHASE_SEQUENCING
    client.current_acquisition_state = acquisition_pb2.AcquisitionState.ACQUISITION_RUNNING
    assert check(1120.2) == [
        "Data from MinKNOW resumed after 120 s"
        " (protocol phase PHASE_SEQUENCING, acquisition state ACQUISITION_RUNNING)."
    ]
    assert check(1121) == []
    client._process_thread.alive = False
    stopped = check(1122)
    assert len(stopped) == 1 and "read stream thread has stopped" in stopped[0]
    assert check(1200) == []  # logged once


def test_unknown_enum_values_do_not_raise(monkeypatch, readfish_handler):
    """A newer MinKNOW may send states this minknow_api does not know; logging
    them must not break the main loop."""
    import readfish._read_until_client as ru

    info = acquisition_pb2.AcquisitionRunInfo()
    info.state = 99
    info.stop_reason = 98
    client = RUClient.__new__(RUClient)
    client.log = logging.getLogger("readfish._read_until_client")
    client.connection = SimpleNamespace(
        acquisition=SimpleNamespace(get_acquisition_info=lambda: info)
    )
    client.current_acquisition_state = None
    client.set_acquisition_state()
    assert client.current_acquisition_state == 99
    assert readfish_handler.records[-1].getMessage() == (
        "Acquisition state changed to UNKNOWN(99) (stop reason UNKNOWN(98))"
    )

    client.current_protocol_phase = 97
    client._process_thread = FakeThread()
    client._process_thread.alive = False
    client.last_message_time = 0.0
    client._stream_stopped_logged = False
    client.log_stream_health()
    assert "protocol phase UNKNOWN(97), acquisition state UNKNOWN(99)" in (
        readfish_handler.records[-1].getMessage()
    )


def make_client_with_classifications(monkeypatch, classifications, **kwargs):
    """Run ReadUntilClient.__init__ against a fake MinKNOW whose read
    classification map is ``classifications`` (name -> id)."""
    device = SimpleNamespace(
        get_flow_cell_info=lambda: SimpleNamespace(channel_count=2),
        get_calibration=lambda **kw: SimpleNamespace(
            pa_ranges=[1.0, 1.0], offsets=[0.0, 0.0], digitisation=1
        ),
    )
    analysis_configuration = SimpleNamespace(
        get_read_classifications=lambda: SimpleNamespace(
            read_classifications={i: name for name, i in classifications.items()}
        )
    )
    connection = SimpleNamespace(
        device=device, analysis_configuration=analysis_configuration
    )
    monkeypatch.setattr(base, "Connection", lambda **kw: connection)
    numpy_types = SimpleNamespace(calibrated_signal="f4", uncalibrated_signal="i2")
    monkeypatch.setattr(base, "get_numpy_types", lambda conn: numpy_types)
    return base.ReadUntilClient(**kwargs)


def prefilter_records(handler):
    return [
        (r.levelno, r.getMessage())
        for r in handler.records
        if r.name == "readfish.read_until" and "prefilter class" in r.getMessage().lower()
    ]


def test_prefilter_classes_logged(monkeypatch, readfish_handler):
    client = make_client_with_classifications(
        monkeypatch,
        {"strand": 3, "adapter": 1, "pore": 2},
        prefilter_classes={"strand", "adapter"},
    )
    assert client.strand_classes == {3, 1}
    assert prefilter_records(readfish_handler) == [
        (logging.INFO, "Prefilter classes (MinKNOW ids): adapter=1, strand=3.")
    ]


def test_missing_prefilter_classes_warned(monkeypatch, readfish_handler):
    client = make_client_with_classifications(
        monkeypatch,
        {"strand": 3, "pore": 2},
        prefilter_classes={"strand", "adapter", "short_strand"},
    )
    # Behaviour unchanged: missing classes still become placeholders.
    assert client.strand_classes == {
        3,
        "adapter_classification_unavailable",
        "short_strand_classification_unavailable",
    }
    assert prefilter_records(readfish_handler) == [
        (logging.INFO, "Prefilter classes (MinKNOW ids): strand=3."),
        (
            logging.WARNING,
            "MinKNOW has no read classification for prefilter class(es) "
            "adapter, short_strand, so the prefilter cannot match them. "
            "MinKNOW's classifications are: pore, strand.",
        ),
    ]


def test_prefilter_classes_not_logged_without_filtering(
    monkeypatch, readfish_handler
):
    make_client_with_classifications(
        monkeypatch, {"pore": 2}, filter_strands=False, prefilter_classes={"strand"}
    )
    assert prefilter_records(readfish_handler) == []


def test_response_summary_named_per_interval_and_cumulative(
    monkeypatch, readfish_handler
):
    Response = base.data_pb2.GetLiveReadsResponse.ActionResponse.Response
    now = [1000.0]
    monkeypatch.setattr(base.time, "time", lambda: now[0])

    client = base.ReadUntilClient.__new__(base.ReadUntilClient)
    client.logger = logging.getLogger("readfish.read_until")
    client.running = Event()
    client.running.set()
    progress = SimpleNamespace(raw_per_channel=SimpleNamespace(acquired=100))
    client.connection = SimpleNamespace(
        acquisition=SimpleNamespace(get_progress=lambda: progress)
    )
    client.one_chunk = False
    client.filter_strands = False
    client.data_queue = base.ReadCache(size=2)
    client.sent_actions = {"s1": "stop_further_data", "s2": "stop_further_data",
                           "u1": "unblock", "s3": "stop_further_data",
                           "u2": "unblock"}

    def message(*responses):
        read = SimpleNamespace(
            id="r", chunk_start_sample=0, raw_data=b"", chunk_classifications=[]
        )
        return SimpleNamespace(
            action_responses=[
                SimpleNamespace(action_id=a, response=r) for a, r in responses
            ],
            channels={1: read},
        )

    def messages():
        yield message(("s1", Response.SUCCESS), ("s2", Response.FAILED_READ_FINISHED),
                      ("u1", Response.SUCCESS))
        now[0] += 60 * 15 + 1
        yield message()
        yield message(("s3", Response.SUCCESS), ("u2", Response.FAILED_READ_TOO_LONG))
        now[0] += 60 * 15 + 1
        yield message()

    client._process_reads(messages())

    summaries = [
        r.getMessage()
        for r in readfish_handler.records
        if r.getMessage().startswith("Response summary")
    ]
    assert summaries == [
        "Response summary: this interval stop_further_data: SUCCESS=1, "
        "FAILED_READ_FINISHED=1; unblock: SUCCESS=1. Cumulative "
        "stop_further_data: SUCCESS=1, FAILED_READ_FINISHED=1; unblock: SUCCESS=1.",
        "Response summary: this interval stop_further_data: SUCCESS=1; "
        "unblock: FAILED_READ_TOO_LONG=1. Cumulative stop_further_data: "
        "SUCCESS=2, FAILED_READ_FINISHED=1; unblock: SUCCESS=1, "
        "FAILED_READ_TOO_LONG=1.",
    ]


def test_response_summary_never_raises():
    assert base._response_summary({}) == "none"
    assert base._response_summary({"unblock": Counter({99: 2})}) == (
        "unblock: UNKNOWN(99)=2"
    )
    assert base._response_summary(None) == "None"


def test_interval_update_without_reads_does_not_raise(monkeypatch, readfish_handler):
    """A message with no reads after a full interval with none must not end the
    stream thread with ZeroDivisionError."""
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

    def messages():
        now[0] += 60 * 15 + 1
        yield SimpleNamespace(action_responses=[], channels={})

    client._process_reads(messages())

    updates = [
        r.getMessage()
        for r in readfish_handler.records
        if r.getMessage().startswith("Interval update")
    ]
    assert len(updates) == 1
    assert "0 read sections" in updates[0]
    assert "average 0 samples behind" in updates[0]
