"""tools/detector_setup.py against a small stand-in for the DCU's SIMPLON REST API (no detector)."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import detector_setup


class FakeDcu:
    """Holds the settings; serves GET / PUT on /{module}/api/1.8.0/{task}/{param}."""

    def __init__(self):
        self.config = {
            ("stream", "mode"): "disabled", ("stream", "format"): "cbor",
            ("detector", "compression"): "lz4", ("detector", "binning_mode"): "2x2",
            ("detector", "trigger_mode"): "ints", ("detector", "ntrigger"): 1,
            ("detector", "nimages"): 1000, ("detector", "frame_time"): 1e-4,
            ("detector", "count_time"): 0.99e-4, ("detector", "detector_readout_time"): 1e-7,
            ("detector", "bit_depth_image"): 16, ("filewriter", "mode"): "enabled",
        }
        self.status = {("detector", "state"): "idle", ("stream", "state"): "ready",
                       ("stream", "dropped"): 0}
        self.commands = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, value=None):
                body = b"" if value is None else json.dumps({"value": value}).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/detector/api/version/":
                    return self._reply(200, "1.8.0")
                module, _, _, task, param = self.path.strip("/").split("/")
                table = fake.config if task == "config" else fake.status
                if (module, param) not in table:
                    return self._reply(404)
                self._reply(200, table[(module, param)])

            def do_PUT(self):
                module, _, _, task, param = self.path.strip("/").split("/")
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                if task == "command":
                    fake.commands.append(param)
                    fake.status[("detector", "state")] = {"arm": "ready", "disarm": "idle"}.get(
                        param, "idle")
                    return self._reply(200, {"sequence id": len(fake.commands)}
                                       if param == "arm" else None)
                fake.config[(module, param)] = json.loads(body)["value"]
                self._reply(200, [param])

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def dcu():
    fake = FakeDcu()
    yield fake
    fake.server.shutdown()


def run(dcu, *args):
    return detector_setup.main([args[0], "--ip", "127.0.0.1", "--http-port", str(dcu.port),
                                *args[1:]])


def test_check_reads_only_and_reports_what_differs(dcu, capsys):
    before = dict(dcu.config)
    assert run(dcu, "check", "--scan", "256", "256", "--binned") == 1
    out = capsys.readouterr().out
    assert dcu.config == before and dcu.commands == []
    assert "DIFF  stream/mode: 'disabled'  (needs 'enabled')" in out
    assert "DIFF  detector/compression: 'lz4'  (needs 'bslz4')" in out
    assert "DIFF  detector/ntrigger: 1  (needs 65536)" in out
    assert "OK    detector/binning_mode" in out
    assert "run.py flags for this setup: --scan 256 256 --binned --dtype uint16" in out


def test_apply_sets_what_the_run_needs(dcu, capsys):
    assert run(dcu, "apply", "--scan", "256", "256", "--binned", "--frame-time", "8.334e-6") == 0
    c = dcu.config
    assert c[("stream", "mode")] == "enabled" and c[("detector", "compression")] == "bslz4"
    assert c[("detector", "trigger_mode")] == "exts"
    assert (c[("detector", "ntrigger")], c[("detector", "nimages")]) == (65536, 1)
    assert c[("filewriter", "mode")] == "disabled"
    assert c[("detector", "frame_time")] == 8.334e-6
    assert c[("detector", "count_time")] == pytest.approx(8.334e-6 - 1e-7)
    assert dcu.commands == []                                   # apply never arms
    assert "The DCU is set up for this run." in capsys.readouterr().out


def test_unbinned_run_on_a_binning_detector_is_a_problem(dcu, capsys):
    assert run(dcu, "apply", "--scan", "64", "64") == 1
    assert "PROBLEM: the detector bins 2x2" in capsys.readouterr().out
    assert dcu.config[("detector", "compression")] == "lz4"     # nothing set when a problem remains


def test_arm_and_disarm(dcu, capsys):
    assert run(dcu, "arm") == 0
    assert dcu.status[("detector", "state")] == "ready"
    assert run(dcu, "disarm") == 0
    assert dcu.commands == ["arm", "disarm"]
    assert "Frames dropped by the DCU: 0" in capsys.readouterr().out


def test_scan_arms_waits_for_the_user_and_disarms(dcu, monkeypatch):
    run(dcu, "apply", "--scan", "32", "32", "--binned")
    prompts = []
    monkeypatch.setattr("builtins.input", lambda text: prompts.append(text) or "")
    assert run(dcu, "scan", "--scan", "32", "32", "--binned") == 0
    assert dcu.commands == ["arm", "disarm"]
    assert prompts and "Start the scan now" in prompts[0]


def test_scan_calls_the_sites_start_scan(dcu, monkeypatch):
    run(dcu, "apply", "--scan", "32", "32", "--binned")
    calls = []
    monkeypatch.setattr(detector_setup, "start_scan", lambda ny, nx, ft: calls.append((ny, nx)))
    assert run(dcu, "scan", "--scan", "32", "32", "--binned") == 0
    assert calls == [(32, 32)] and dcu.commands == ["arm", "disarm"]


def test_unreachable_dcu_is_reported(capsys):
    assert detector_setup.main(["check", "--ip", "127.0.0.1", "--http-port", "1",
                                "--scan", "8", "8"]) == 2
    assert "Cannot talk to the DCU's REST API" in capsys.readouterr().out
