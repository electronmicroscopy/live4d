"""Live control of the reconstruction (`run.py --viewer-port`, `apps/control.py`):

  • `ParallaxAccumulator.set_geometry`: "latest" re-renders the frames it holds, exactly what a
    fresh reconstruction with the new geometry gives; the other modes restart their image; the
    bright-field pixels can't change;
  • `LiveControl`: C10 / rotation / C12 (as a, b) into `args` and the running accumulator, the
    accumulate mode switched with a new accumulator handed to the sink, reset, refusals;
  • the viewer's /params (GET, POST) only with a control;
  • and through the runner: tune a replayed recording after the replay, then Ctrl-C saves the
    tuned image and prints its flags.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

REPO = (Path(__file__).resolve().parents[2] / "python/live4d/parallax")

from live4d.parallax.control import LiveControl  # noqa: E402
from live4d.parallax.ingest.sink import BatchSink  # noqa: E402
from live4d.parallax.algorithm import ParallaxAlgorithm  # noqa: E402
from live4d.algorithms.parallax.engine import ParallaxAccumulator, preprocess_geometry  # noqa: E402
from live4d.parallax.streaming.web.server import FrameStore, StorePublisher, make_server  # noqa: E402

_SCAN = (12, 10)
_NPOS = _SCAN[0] * _SCAN[1]
_FLAGS = ["--detector", "32", "32", "--scan", *map(str, _SCAN), "--energy", "300e3",
          "--semiangle", "4", "--angular-sampling", "0.307617", "--scan-sampling", f"{256 / 12}",
          "--defocus=-1.5e4", "--rotation=-15", "--center", "16", "16"]


def _geometry(c10=-1.5e4, rotation=-15.0, c12=0.0, phi12=0.0, semiangle=4.0):
    return preprocess_geometry(
        det_shape=(32, 32), scan_shape=_SCAN, scan_sampling=(256 / 12, 256 / 12), energy=300e3,
        semiangle_cutoff=semiangle, angular_sampling=(0.307617, 0.307617),
        aberration_coefs={"C10": c10, "C12": c12, "phi12": math.radians(phi12)},
        rotation_angle=math.radians(rotation), probe_center=(16, 16))


def _frames(n=_NPOS, seed=0):
    return np.random.default_rng(seed).integers(0, 60000, size=(n, 32, 32), dtype=np.uint16)


def _close(a, b, rel=2e-5):
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=0, atol=rel * max(np.abs(b).max(), 1e-12))


def _feed(acc, frames, ids, batch=40):                   # "latest": a batch ≤ the scan positions
    for s in range(0, len(frames), batch):
        acc.process(frames[s:s + batch], ids[s:s + batch])


def _fresh(geo, frames, ids=None, **kw):
    acc = ParallaxAccumulator(geo, device="cpu", **kw)
    _feed(acc, frames, np.arange(len(frames)) if ids is None else ids)
    return acc


def test_latest_re_renders_the_frames_it_holds():
    frames = _frames(_NPOS + 30, seed=1)                  # a pass and a bit: the store keeps the newest
    acc = ParallaxAccumulator(_geometry(), device="cpu", accumulate="latest", wrap_ids=True)
    _feed(acc, frames, np.arange(len(frames)))
    new = _geometry(c10=-1.2e4, rotation=-12.0, c12=400.0, phi12=30.0)
    note = acc.set_geometry(new)
    assert "re-rendered" in note
    held = np.concatenate([frames[_NPOS:], frames[30:_NPOS]])          # the newest per position
    ids = np.concatenate([np.arange(30), np.arange(30, _NPOS)])
    ref = _fresh(new, held, ids)
    _close(acc.result(), ref.result())
    for name, img in acc.channels().items():                           # χ follows (phase flip)
        _close(img.numpy(), ref.channels()[name].numpy())


@pytest.mark.parametrize("mode", ["all", "reset", "swap"])
def test_the_other_modes_restart(mode):
    acc = _fresh(_geometry(), _frames(_NPOS + 30, seed=2), accumulate=mode, wrap_ids=True)
    assert "restarted" in acc.set_geometry(_geometry(c10=-1.2e4))
    assert float(acc.out.abs().max()) == 0 and acc._n_frames == 0 and acc._front is None
    frames = _frames(_NPOS, seed=3)                       # the next pass: the new geometry alone
    _feed(acc, frames, np.arange(2 * _NPOS, 3 * _NPOS))
    acc.end_series()
    _close(acc.result(), _fresh(_geometry(c10=-1.2e4), frames).result())


def test_the_bright_field_pixels_cannot_change():
    acc = _fresh(_geometry(), _frames(seed=4))
    with pytest.raises(ValueError, match="bright-field pixels"):
        acc.set_geometry(_geometry(semiangle=3.0))


def _args(*extra):
    algo = ParallaxAlgorithm()
    p = argparse.ArgumentParser()
    algo.add_cli_args(p)
    p.add_argument("--accumulate", default="scan")
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(algo.preprocess_argv(_FLAGS + list(extra)))
    return algo, args


class _Publisher:
    def __init__(self):
        self.frames = []

    def publish(self, arr, **kw):
        self.frames.append((np.asarray(arr), kw))
        return len(self.frames)


def test_control_retunes_switches_modes_and_resets():
    algo, args = _args("--accumulate", "latest")
    pub, built = _Publisher(), []
    control = LiveControl(algo, args, publisher=pub, on_new_accumulator=built.append)
    acc = algo.build(args)
    sink = BatchSink(acc, live_publisher=pub, live_every=1e9)
    control.attach(acc, lambda: sink)
    frames = _frames(seed=5)
    for s in range(0, _NPOS, 40):
        sink.feed(frames[s:s + 40], np.arange(s, min(s + 40, _NPOS)))
    n_pub = len(pub.frames)
    state = control.update({"c10": -1.1e4, "rotation": -14.0, "c12a": 300.0, "c12b": 400.0})
    assert args.defocus == -1.1e4 and args.rotation == -14.0
    assert args.c12 == pytest.approx(500.0) and args.phi12 == pytest.approx(
        math.degrees(math.atan2(400, 300)) / 2)
    assert "re-rendered" in state["note"] and len(pub.frames) == n_pub + 1   # published at once
    assert pub.frames[-1][1]["meta"]["C10"] == -1.1e4
    _close(acc.result(), _fresh(_geometry(-1.1e4, -14.0, 500.0, math.degrees(math.atan2(400, 300)) / 2),
                                frames).result())
    values = {p["name"]: p["value"] for p in state["params"]}
    assert values["c12a"] == pytest.approx(300.0) and values["c12b"] == pytest.approx(400.0)
    assert "--defocus=-11000.0" in state["flags"] and "--accumulate latest" in state["flags"]

    state = control.update({"mode": "swap"})              # a new accumulator, in the sink
    assert state["mode"] == "swap" and sink.acc is control.acc is built[-1]
    assert sink.acc.accumulate == "swap" and sink.acc.geometry.aberration_coefs["C10"] == -1.1e4

    state = control.update({"reset": True})
    assert state["mode"] == "latest" and args.defocus == -1.5e4 and args.c12 == pytest.approx(0)
    for bad in ({"nope": 1}, {"mode": "scan"}):
        with pytest.raises(ValueError):
            control.update(bad)


def test_one_scan_has_no_mode_to_switch():
    algo, args = _args()                                  # --accumulate scan
    control = LiveControl(algo, args)
    assert control.state()["modes"] == [] and "--accumulate" not in control.state()["flags"]
    with pytest.raises(ValueError):
        control.update({"mode": "latest"})


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read())


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


@pytest.fixture()
def viewer():
    servers = []

    def start(control=None):
        srv = make_server(FrameStore(), "127.0.0.1", 0, control=control)
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}"
    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def test_params_endpoint(viewer):
    base = viewer()                                       # a standalone viewer: no control
    assert _get(base + "/state")[1]["control"] is False
    for call in (lambda: _get(base + "/params"), lambda: _post(base + "/params", {"c10": 1})):
        with pytest.raises(urllib.error.HTTPError) as e:
            call()
        assert e.value.code == 404
    algo, args = _args("--accumulate", "latest")
    control = LiveControl(algo, args)
    base = viewer(control)
    assert _get(base + "/state")[1]["control"] is True
    status, state = _get(base + "/params")
    assert status == 200 and [p["name"] for p in state["params"]] == ["c10", "rotation", "c12a", "c12b"]
    status, state = _post(base + "/params", {"c10": -1.4e4})
    assert status == 200 and state["params"][0]["value"] == -1.4e4 and args.defocus == -1.4e4
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/params", {"nope": 1})
    assert e.value.code == 400


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_runner_viewer_tunes_a_replay_and_saves_it(tmp_path):
    frames = _frames(seed=6)
    np.save(tmp_path / "scan.npy", frames.reshape(*_SCAN, 32, 32))
    port = _free_port()
    from .helpers import offline_args
    proc = subprocess.Popen(
        [sys.executable, "-m", "live4d.parallax.offline",
         *offline_args(tmp_path, [*_FLAGS, "--accumulate", "latest"],
                       tmp_path / "scan.npy", viewer_port=port)],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):                              # wait for the replay's final snapshot
            try:
                st = _get(base + "/state")[1]
                if st.get("final"):
                    break
            except (urllib.error.URLError, ConnectionError):
                pass
            time.sleep(0.1)
        assert st.get("final") and st["control"]
        seq = st["seq"]
        _post(base + "/params", {"c10": -1.2e4})
        st = _get(base + "/state")[1]
        assert st["seq"] > seq and st["meta"]["C10"] == -1.2e4      # published at once
    finally:
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=60)
    assert "Tuned flags: --defocus=-12000.0" in out, out
    _close(np.load(tmp_path / "out" / "parallax_recon.npy"),
           _fresh(_geometry(c10=-1.2e4), frames).result())


def test_live_http_tuning_is_bounded_and_runs_only_in_callbacks():
    from live4d.parallax.control import QueuedLiveControl
    calls = []
    class Control:
        note = ""
        def state(self):
            return dict(note=self.note)
        def update(self, changes):
            calls.append(changes)
    control = QueuedLiveControl(Control(), capacity=1)
    control.update(dict(c10=-100))
    assert not calls
    with pytest.raises(ValueError, match="queue is full"):
        control.update(dict(c10=-200))
    control.apply_pending()
    assert calls == [dict(c10=-100)]
    control.closed = True
    with pytest.raises(ValueError, match="stopped"):
        control.update(dict(c10=-200))
