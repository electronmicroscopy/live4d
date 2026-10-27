"""Unit tests for the live-preview transport. Pure Python (numpy + pyzmq) — runs anywhere."""
import sys
import time
from pathlib import Path

import numpy as np


from live4d.parallax.streaming.live_stream import (
    LivePublisher,
    LiveSubscriber,
    Throttle,
    pack,
    unpack,
)


def test_pack_unpack_roundtrip_exact():
    arr = (np.arange(12, dtype=np.float32).reshape(3, 4) * 1.5) - 3.0
    out, header = unpack(pack(arr, seq=7, final=True))
    assert out.dtype == np.float32
    assert out.shape == (3, 4)
    assert np.array_equal(out, arr)
    assert header["seq"] == 7
    assert header["final"] is True
    assert header["shape"] == [3, 4]


def test_throttle_emits_at_most_once_per_interval():
    now = [0.0]
    thr = Throttle(0.2, clock=lambda: now[0])
    assert thr.ready() is True            # first call always emits
    now[0] = 0.1
    assert thr.ready() is False           # 0.1 s < 0.2 s
    now[0] = 0.2
    assert thr.ready() is True            # exactly the interval
    now[0] = 0.35
    assert thr.ready() is False
    now[0] = 0.41
    assert thr.ready() is True


def test_publish_without_subscriber_neither_blocks_nor_raises():
    pub = LivePublisher("tcp://127.0.0.1:5731")
    try:
        t0 = time.perf_counter()
        for _ in range(200):
            pub.publish(np.zeros((16, 16), dtype=np.float32))
        assert time.perf_counter() - t0 < 1.0     # non-blocking: 200 sends are ~instant
    finally:
        pub.close()


def test_pub_sub_delivers_latest_frame():
    endpoint = "tcp://127.0.0.1:5732"
    pub = LivePublisher(endpoint)
    sub = LiveSubscriber(endpoint)
    try:
        time.sleep(0.2)                            # SUB slow-joiner settle
        for i in range(5):
            pub.publish(np.full((4, 4), float(i), dtype=np.float32), final=(i == 4))
            time.sleep(0.02)
        got = None
        for _ in range(30):
            msg = sub.recv_latest(timeout_ms=100)
            if msg is not None:
                got = msg
                if msg[1]["final"]:
                    break
        assert got is not None, "subscriber received nothing"
        arr, header = got
        assert header["final"] is True
        assert np.all(arr == 4.0)                  # CONFLATE → newest frame (i=4)
    finally:
        pub.close()
        sub.close()


def test_pack_channels_and_meta_roundtrip():
    stack = np.random.default_rng(0).random((2, 5, 6)).astype(np.float32)
    meta = {"C10": -2000.0, "coverage": 0.25, "frames": 30}
    out, header = unpack(pack(stack, seq=3, channels=["parallax", "phase_flipped"], meta=meta))
    assert np.array_equal(out, stack)
    assert header["channels"] == ["parallax", "phase_flipped"] and header["meta"] == meta
    _, plain = unpack(pack(stack[0], seq=4))            # 2D frames keep the old header
    assert "channels" not in plain and "meta" not in plain
    import pytest
    with pytest.raises(ValueError):
        pack(stack, seq=5, channels=["only-one"])


def test_final_frame_survives_immediate_close():
    """The runner publishes the final snapshot and closes at once; a connected subscriber must
    still get it (a zero linger used to discard whatever was still queued)."""
    results = []
    for trial in range(5):
        endpoint = f"tcp://127.0.0.1:{5740 + trial}"
        pub = LivePublisher(endpoint)
        sub = LiveSubscriber(endpoint)
        try:
            time.sleep(0.3)                            # SUB slow-joiner settle
            big = np.zeros((1024, 1024), dtype=np.float32)
            pub.publish(big)                           # a live frame still in flight …
            pub.publish(big + trial, final=True)       # … then the final one, and close at once
            pub.close()
            got = None
            deadline = time.time() + 3
            while time.time() < deadline and not (got and got[1]["final"]):
                got = sub.recv_latest(timeout_ms=200) or got
            results.append(bool(got and got[1]["final"] and got[0][0, 0] == trial))
        finally:
            sub.close()
    assert all(results), results
