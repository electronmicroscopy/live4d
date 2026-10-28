"""Measurement semantics: rates, idle decay, scan boundaries and HTTP availability."""
import json
from pathlib import Path
import sys
import threading
from urllib.request import urlopen

import pytest

from live4d.parallax.streaming.telemetry import Telemetry
from live4d.parallax.streaming.web.server import FrameStore, make_server


def test_rate_excludes_first_batch_and_decays_when_idle():
    telemetry = Telemetry()
    telemetry.begin(3000)
    for i in range(3):
        finished = 10 + i / 120
        telemetry.record(1000, finished - .001, finished)
    state = telemetry.snapshot(now=10.02)
    assert state["frames"] == 3000
    assert state["fps"] == pytest.approx(120000)
    assert state["average_fps"] == pytest.approx(120000)
    assert state["host_batch_ms"]["p95"] == pytest.approx(1)
    assert state["batch_interval_jitter_ms"] == pytest.approx(0, abs=1e-9)
    assert telemetry.snapshot(now=12)["fps"] == 0
    telemetry.finish(3000)
    assert telemetry.snapshot()["status"] == "complete"


def test_integrity_and_next_scan_are_separate():
    telemetry = Telemetry()
    telemetry.begin(3000)
    telemetry.record(1000, 1, 1.001)
    telemetry.record(1000, 2, 2.001)
    telemetry.finish(1500)
    state = telemetry.snapshot(now=2.1)
    assert state["status"] == "incomplete"
    assert state["integrity"] == dict(covered=1500, missing=1500, duplicates=500)
    telemetry.begin(4000)
    state = telemetry.snapshot(now=3)
    assert state["frames"] == 0 and state["integrity"] is None
    assert state["average_fps"] is None
    assert state["previous_scan"]["status"] == "incomplete"
    telemetry.record(500, 4, 4.001)
    assert telemetry.snapshot(now=4.1)["batch_interval_jitter_ms"] is None


def test_repeated_scan_does_not_report_legitimate_repeats_as_duplicates():
    telemetry = Telemetry()
    telemetry.begin(1000)
    telemetry.record(2000, 1, 1.001)
    telemetry.finish(1000, repeated=True)
    state = telemetry.snapshot()
    assert state["status"] == "complete"
    assert state["integrity"]["duplicates"] is None


@pytest.mark.parametrize("enabled", [True, False])
def test_http_metrics_are_available_before_first_image(enabled):
    telemetry = Telemetry() if enabled else None
    server = make_server(FrameStore(), port=0, telemetry=telemetry)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/telemetry", timeout=2) as response:
            state = json.load(response)
        assert state["available"] is enabled
        if enabled:
            assert state["frames"] == 0 and state["fps"] == 0
            json.dumps(state, allow_nan=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_resource_sampler_handles_unsupported_gpu(monkeypatch):
    import subprocess
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", unavailable)
    telemetry = Telemetry("tcp://127.0.0.1:32001")
    telemetry._sample_resources()
    resources = telemetry._sample_resources()
    assert resources["gpus"] == []
    assert resources["process_rss_bytes"] > 0
    assert resources["host_cpu_pct"] is None or 0 <= resources["host_cpu_pct"] <= 100
    json.dumps(resources, allow_nan=False)


def test_sink_reports_real_ids_and_final_integrity():
    import numpy as np
    from types import SimpleNamespace
    from live4d.parallax.ingest.sink import BatchSink
    telemetry = Telemetry()
    acc = SimpleNamespace(scan_shape=(2, 2), process=lambda frames, ids: None)
    publisher = SimpleNamespace(telemetry=telemetry, publish=lambda *args, **kw: None)
    sink = BatchSink(acc, live_publisher=publisher, live_every=1e9)
    # Avoid a preview: this test measures sink counts rather than reconstruction readout.
    sink._throttle._last = float("inf")
    sink.feed(np.zeros((3, 2, 2)), np.array([0, 1, 1]))
    assert sink.frames == telemetry.snapshot()["frames"] == 3
    assert sink.completeness()[0] is False
    assert telemetry.snapshot()["integrity"] == dict(covered=2, missing=2, duplicates=1)


def test_in_process_viewer_reports_completed_file_replay(tmp_path):
    """Real runner wiring: publisher → sink → telemetry → HTTP, with a finite CPU source."""
    import numpy as np
    import signal
    import socket
    import subprocess
    import time
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    source = tmp_path / "frames.npy"
    np.save(source, np.ones((16, 32, 32), dtype=np.uint16))
    from .helpers import offline_args
    flags = ["--scan", "4", "4", "--energy", "300e3", "--semiangle", "4",
             "--angular-sampling", "0.307617", "--scan-sampling", "21",
             "--defocus=-15000", "--rotation=-15", "--center", "16", "16"]
    cmd = [sys.executable, "-m", "live4d.parallax.offline",
           *offline_args(tmp_path, flags, source, batch_size=4, viewer_port=port)]
    with (tmp_path / "runner.log").open("w+") as log:
        proc = subprocess.Popen(cmd, cwd=(Path(__file__).resolve().parents[2] / "python/live4d/parallax"), stdout=log,
                                stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+20
            state = {}
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/telemetry", timeout=.5) as r:
                        state = json.load(r)
                    if state.get("status") == "complete":
                        break
                except OSError:
                    pass
                time.sleep(.1)
            log.seek(0)
            assert state.get("status") == "complete", log.read()
            assert state["frames"] == 16
            assert state["batches"] == 4
            assert state["integrity"] == dict(covered=16, missing=0, duplicates=0)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
