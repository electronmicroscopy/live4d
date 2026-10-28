"""Tests for the browser viewer's server (`realtime_stem.streaming.web`): the snapshot store, the
region / spectrum / encoding helpers, and every HTTP endpoint against a real server on a free
port, fed both directly and through the live stream."""

from __future__ import annotations

import io
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_stem.streaming.web.server import (  # noqa: E402
    transfer,
    STATIC,
    FrameStore,
    channel_stats,
    encode,
    make_server,
    power_spectrum,
    subscribe,
    view,
)


def _stack(seed=0, shape=(2, 96, 128)):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32) * 10 + 100


def test_store_history_is_bounded_and_keeps_the_newest():
    store = FrameStore(history_mb=0.2)                # 2 × 96 × 128 × 4 B = 96 kB a frame
    for i in range(5):
        store.put(_stack(i), {"channels": ["a", "b"], "seq": i})
    assert store.seqs() == [3, 4] and store.get().seq == 4
    assert store.get(0) is None and store.get(3).names == ["a", "b"]
    tiny = FrameStore(history_mb=0)
    tiny.put(_stack(), {"channels": ["a", "b"]})
    assert tiny.seqs() == [0]                          # the newest stays whatever the budget


def test_store_wait_and_single_image():
    store = FrameStore()
    assert store.wait(-1, timeout=0.01) is None
    threading.Timer(0.05, lambda: store.put(np.ones((8, 8)), {})).start()
    f = store.wait(-1, timeout=2.0)
    assert f is not None and f.names == ["reconstruction"] and f.stack.shape == (1, 8, 8)


def test_store_stats_cover_only_the_scanned_rows():
    stack = _stack(4)
    stack[:, 40:] = 1e3                                           # rows not scanned yet
    frame = FrameStore().put(stack, {"channels": ["a", "b"], "meta": {"complete_rows": [0, 40]}})
    assert frame.stats[0]["max"] < 1e3 and frame.stack.shape == stack.shape


def test_channel_stats_percentiles():
    img = np.arange(10000, dtype=np.float32).reshape(100, 100)
    s = channel_stats(img)
    assert s["min"] == 0 and s["max"] == 9999
    assert abs(s["pct"]["50" if "50" in s["pct"] else "99"] - 9900) < 20
    assert sum(s["hist"]) > 0.99 * img.size and s["hist_range"][0] < s["hist_range"][1]


def test_view_block_averages_to_the_screen_and_keeps_native_pixels_zoomed_in():
    img = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    data, region, f = view(img, 0, 0, 64, 64, 16, 16)
    assert f == 4 and data.shape == (16, 16) and region == (0, 0, 64, 64)
    np.testing.assert_allclose(data, img.reshape(16, 4, 16, 4).mean(axis=(1, 3)))
    data, region, f = view(img, 10, 20, 8, 8, 100, 100)          # zoomed in: native pixels
    assert f == 1 and region == (10, 20, 8, 8)
    np.testing.assert_array_equal(data, img[10:18, 20:28])
    data, region, f = view(img, 50, 50, 40, 40, 4, 4)             # clipped at the edge
    assert region[2] <= 14 and region[3] <= 14


def test_view_filters_keep_the_shape_and_act():
    rng = np.random.default_rng(1)
    img = rng.standard_normal((128, 128)).astype(np.float32)
    plain, _, _ = view(img, 32, 32, 64, 64, 64, 64)
    low, _, _ = view(img, 32, 32, 64, 64, 64, 64, lp=6.0)          # keeps q < 1/6 of a 1/px
    high, _, _ = view(img, 32, 32, 64, 64, 64, 64, hp=16.0, order=4)
    assert plain.shape == low.shape == high.shape == (64, 64)
    assert low.std() < 0.5 * plain.std() and abs(high.mean()) < 0.1


def test_display_filters_are_butterworth():
    qc = 0.1
    q = np.array([0.5, 0.8, 1.0, 1.25, 2.0]) * qc
    low, high = transfer(q, lp=1 / qc), transfer(q, hp=1 / qc)
    np.testing.assert_allclose([low[2], high[2]], [0.5, 0.5])       # ½ at the cutoff, as quantem
    assert low[1] > 0.99 and low[4] < 1e-3                          # order 12: sharp
    assert high[4] > 0.99 and high[0] < 1e-3                        # mirrored in log q
    np.testing.assert_allclose(low + high, 1.0)                     # complementary
    assert transfer(q, lp=1 / qc, order=2)[4] > 0.05                 # order 2: soft
    assert transfer(np.array([0.0]), hp=10)[0] == 0 and transfer(q)[0] == 1


def test_encode_keeps_precision_at_any_level():
    data = (1e5 + np.linspace(-3, 3, 1000)).astype(np.float32)   # a big offset, small detail
    body, off, scale = encode(data)
    back = np.frombuffer(body, "<f2").astype(np.float64) * scale + off
    np.testing.assert_allclose(back, data, atol=2e-3 * np.ptp(data))


def test_power_spectrum_finds_a_lattice():
    y, x = np.mgrid[:256, :256]
    img = np.cos(2 * np.pi * x / 8.0).astype(np.float32)          # 32 cycles across 256 px
    spec, (cy, cx, n), used, b = power_spectrum(img, 0, 0, 256, 256, n=256)
    assert n == 256 and b == 1 and spec.shape == (256, 256) and used == (0, 256)
    peak = np.unravel_index(np.argmax(spec), spec.shape)
    assert peak[0] == 128 and abs(peak[1] - 128) == 32


def test_power_spectrum_of_a_partly_scanned_image():
    # a lattice on the rows scanned so far, blank below: windowing every row of the square puts
    # the edge of the scan into the spectrum as a streak along q_y (the q_x = 0 column)
    y, x = np.mgrid[:256, :256]
    img = (5.0 + np.cos(2 * np.pi * x / 8.0)).astype(np.float32)
    img[160:] = 0.0
    off_axis = np.abs(np.arange(256) - 128) > 4

    def streak(s):
        return np.median(s[off_axis, 128]) - np.median(s)

    whole, _, _, _ = power_spectrum(img, 0, 0, 256, 256, n=256)
    spec, (cy, cx, n), used, _ = power_spectrum(img, 0, 0, 256, 256, n=256, rows=(20, 150))
    assert (cy, n) == (0, 256) and used == (20, 150)
    assert streak(whole) > 3 and streak(spec) < 0.5
    peak = np.unravel_index(np.argmax(spec), spec.shape)
    assert peak[0] == 128 and abs(peak[1] - 128) == 32
    # a smaller square moves onto the scanned rows; none of them in view: a flat spectrum
    _, (cy, _, _), used, _ = power_spectrum(img, 0, 0, 256, 256, n=64, rows=(20, 150))
    assert cy == 53 and used == (53, 117)
    spec, _, used, _ = power_spectrum(img, 200, 0, 56, 256, n=56, rows=(20, 150))
    assert np.ptp(spec) == 0 and used[1] == used[0]



def test_power_spectrum_shows_the_display_filters_and_bins_large_regions():
    rng = np.random.default_rng(5)
    img = rng.standard_normal((512, 512)).astype(np.float32)      # white: flat spectrum
    q = np.hypot(*np.meshgrid(np.fft.fftshift(np.fft.fftfreq(512)),
                              np.fft.fftshift(np.fft.fftfreq(512)), indexing="ij"))
    plain = power_spectrum(img, 0, 0, 512, 512, n=512)[0]
    low = power_spectrum(img, 0, 0, 512, 512, n=512, lp=4.0)[0]       # q_c = 0.25 1/px, order 12
    high = power_spectrum(img, 0, 0, 512, 512, n=512, hp=4.0)[0]
    outer, inner = q > 0.45, (q > 0.005) & (q < 0.02)
    assert low[outer].mean() < plain[outer].mean() - 5               # log10: 1e5 weaker
    assert high[inner].mean() < plain[inner].mean() - 0.5
    assert abs(high[outer].mean() - plain[outer].mean()) < 0.05      # untouched where it passes
    spec, (_, _, n), _, b = power_spectrum(img, 0, 0, 512, 512, n=2048, out=128)
    assert n == 512 and b == 4 and spec.shape == (128, 128)          # the whole q range, binned


@pytest.fixture()
def server():
    store = FrameStore()
    srv = make_server(store, "127.0.0.1", 0, endpoint="tcp://test")
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield store, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, dict(r.headers), r.read()


def test_endpoints(server):
    store, base = server
    status, _, body = _get(base + "/state")
    assert json.loads(body)["seq"] == -1                          # nothing yet
    stack = _stack(3)
    store.put(stack, {"channels": ["parallax", "phase_flipped"], "seq": 7, "final": False,
                      "meta": {"sampling_A": [0.5, 0.5], "frames": 10}})
    status, _, body = _get(base + "/")
    assert status == 200 and b"<canvas" in body and b"colormaps.js" in body
    status, _, body = _get(base + "/static/colormaps.js")
    assert status == 200 and b"COLORMAPS" in body
    st = json.loads(_get(base + "/state")[2])
    assert st["names"] == ["parallax", "phase_flipped"] and st["shape"] == [96, 128]
    assert st["meta"]["sampling_A"] == [0.5, 0.5] and st["publisher_seq"] == 7
    status, hdr, body = _get(base + "/view?seq=0&ch=phase_flipped&y0=0&x0=0&h=96&w=128&oh=48&ow=64")
    assert hdr["X-Shape"] == "48,64" and hdr["X-Factor"] == "2" and len(body) == 48 * 64 * 2
    back = np.frombuffer(body, "<f2").astype(np.float64).reshape(48, 64) * float(hdr["X-Scale"]) \
        + float(hdr["X-Offset"])
    np.testing.assert_allclose(back, stack[1].reshape(48, 2, 64, 2).mean(axis=(1, 3)), atol=0.05)
    status, hdr, body = _get(base + "/fft?ch=parallax&n=64&hp=2&lp=0.5&order=4")
    assert hdr["X-Shape"] == "64,64" and len(body) == 64 * 64 * 2 and hdr["X-Rows"] == "16,80"
    assert hdr["X-Region"] == "16,32,64,64" and hdr["X-Bin"] == "1"
    store.put(stack, {"channels": ["parallax", "phase_flipped"], "meta": {"complete_rows": [0, 40]}})
    assert _get(base + "/fft?ch=parallax&n=64")[1]["X-Rows"] == "0,40"   # a partly scanned image
    status, hdr, body = _get(base + "/npy?ch=parallax")
    np.testing.assert_array_equal(np.load(io.BytesIO(body)), stack[0])
    for bad in ("/view?ch=nope", "/nope", "/static/../server.py", "/state?seq=99"):
        try:
            code = _get(base + bad)[0]
        except urllib.error.HTTPError as exc:
            code = exc.code
        assert code in (200, 404) if bad == "/state?seq=99" else code == 404, bad


def test_events_stream_new_snapshots(server):
    store, base = server
    store.put(_stack(0), {"channels": ["a", "b"]})
    with urllib.request.urlopen(base + "/events", timeout=5) as r:
        first = json.loads(r.readline().decode().removeprefix("data: "))
        assert first["seq"] == 0
        r.readline()                                              # the blank line ending an event
        threading.Timer(0.05, lambda: store.put(_stack(1), {"channels": ["a", "b"]})).start()
        second = json.loads(r.readline().decode().removeprefix("data: "))
        assert second["seq"] == 1 and second["history"] == [0, 1]


def test_static_page_is_self_contained():
    html = (STATIC / "index.html").read_text()
    assert "http://" not in html and "https://" not in html      # nothing from the internet
    assert "webgl2" in html


def test_fed_from_the_live_stream():
    pytest.importorskip("zmq")
    from realtime_stem.streaming.live_stream import LivePublisher
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    endpoint = f"tcp://127.0.0.1:{port}"
    pub = LivePublisher(endpoint)
    store, stop = FrameStore(), threading.Event()
    subscribe(endpoint, store, stop)
    try:
        deadline = time.time() + 5
        while store.get() is None and time.time() < deadline:     # the SUB joins after a moment
            pub.publish(_stack(2), channels=["parallax", "phase_flipped"], meta={"frames": 3})
            time.sleep(0.05)
        f = store.get()
        assert f is not None and f.names == ["parallax", "phase_flipped"]
        np.testing.assert_array_equal(f.stack, _stack(2))
    finally:
        stop.set()
        pub.close()
