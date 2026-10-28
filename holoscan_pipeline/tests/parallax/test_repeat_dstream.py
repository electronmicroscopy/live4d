"""`tools/repeat_dstream.py`: a recorded scan looped into passes keeps every payload and the arrival
order, numbers the ids as the detector would (running on, or restarting per series), moves the
positions for a drifting specimen, and writes ids the receiver's C parser reads at any width."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = (Path(__file__).resolve().parents[2] / "python/live4d/parallax")

_SCAN = (12, 10)
_NPOS = _SCAN[0] * _SCAN[1]


def _capture(path, n=_NPOS, seed=0, order=None):
    from live4d.parallax.ingest import codec
    from live4d.parallax.ingest.dstream import write_magic, write_record

    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 1000, size=(n, 16, 16), dtype=np.uint16)
    with open(path, "wb") as fh:
        write_magic(fh)
        write_record(fh, codec.encode_control_message("start"))
        for i in (range(n) if order is None else order):
            write_record(fh, codec.encode_image_message(int(i), frames[i]))
        write_record(fh, codec.encode_control_message("end"))
    return frames


def _read(path):
    from live4d.parallax.ingest.codec import decode_message
    from live4d.parallax.ingest.dstream import iter_records

    msgs = [decode_message(raw) for raw in iter_records(path)]
    return [t for t, _, _ in msgs], [(i, f) for t, i, f in msgs if t == "image"]


def test_repeat_one_series_keeps_the_payloads_and_the_order(tmp_path):
    from live4d.parallax.tools.repeat_dstream import main

    order = np.random.default_rng(1).permutation(_NPOS)   # arrival order across connections
    frames = _capture(tmp_path / "s.dstream", order=order)
    main([str(tmp_path / "s.dstream"), str(tmp_path / "r.dstream"), "--passes", "3"])
    types, images = _read(tmp_path / "r.dstream")
    assert types[0] == "start" and types[-1] == "end" and types.count("start") == 1
    ids = [i for i, _ in images]
    assert ids == [k * _NPOS + int(i) for k in range(3) for i in order]
    for i, f in images:
        np.testing.assert_array_equal(f, frames[i % _NPOS])


def test_repeat_per_pass_series_and_drift(tmp_path):
    from live4d.parallax.tools.repeat_dstream import main

    frames = _capture(tmp_path / "s.dstream")
    main([str(tmp_path / "s.dstream"), str(tmp_path / "p.dstream"), "--passes", "2",
          "--series", "per-pass"])
    types, images = _read(tmp_path / "p.dstream")
    assert types.count("start") == 2 and types.count("end") == 2
    assert [i for i, _ in images] == list(range(_NPOS)) * 2
    # Each pass is its own series, numbered as the DCU numbers its arms.
    from live4d.parallax.ingest.dstream import iter_records
    from live4d.parallax.tools.repeat_dstream import series_id
    ids = [series_id(raw) for raw in iter_records(tmp_path / "p.dstream")]
    first = ids[0]
    assert ids == [first] * (_NPOS + 2) + [first + 1] * (_NPOS + 2)
    main([str(tmp_path / "s.dstream"), str(tmp_path / "d.dstream"), "--passes", "3",
          "--scan", *map(str, _SCAN), "--drift", "1", "2"])
    _, images = _read(tmp_path / "d.dstream")
    ny, nx = _SCAN
    for i, f in images:
        k, p = divmod(i, _NPOS)
        r, c = divmod(p, nx)
        np.testing.assert_array_equal(f, frames[((r - k) % ny) * nx + (c - 2 * k) % nx])


def test_repeated_ids_parse_in_the_c_receiver(tmp_path):
    """Ids past 65535 and 2^32 change the CBOR integer's width; the receiver's parser
    (operators/cpp/common/fast_cbor_parse.c) must read them."""
    from live4d.parallax.ingest import codec
    from live4d.parallax.tools.repeat_dstream import image_id, with_image_id

    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    lib = tmp_path / "libfcp.so"
    subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", str(lib),
                    str(Path(__file__).resolve().parents[2] / "cpp/common/fast_cbor_parse.c")], check=True)
    fcp = ctypes.CDLL(str(lib))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fcp.fast_extract_compressed_bslz4.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(u8p), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64)]
    msg = codec.encode_image_message(7, np.arange(96*96, dtype=np.uint16).reshape(96, 96))
    for new in (7, 300, 70000, 5 * 2**32 + 3):
        patched = with_image_id(msg, new)
        assert image_id(patched) == new and codec.decode_message(patched)[1] == new
        ptr, size, orig, elem, iid = (u8p(), ctypes.c_size_t(), ctypes.c_size_t(),
                                      ctypes.c_size_t(), ctypes.c_uint64())
        width, height = ctypes.c_uint32(), ctypes.c_uint32()
        rc = fcp.fast_extract_compressed_bslz4(patched, len(patched), ctypes.byref(ptr),
                                               ctypes.byref(size), ctypes.byref(orig),
                                               ctypes.byref(elem), ctypes.byref(width), ctypes.byref(height), ctypes.byref(iid))
        assert rc == 0 and iid.value == new and orig.value == 96*96*2


def test_a_per_pass_loop_replays_as_separate_series(tmp_path):
    """`tools/replay_to_zmq.py` sends each series' end in order (not only one at the end of the
    file), so a per-pass loop reaches `run.py --continuous` as one series per pass."""
    import socket
    import threading

    import zmq

    from live4d.parallax.tools.repeat_dstream import main
    from live4d.parallax.tools.replay_to_zmq import replay_dstream
    from live4d.parallax.tools.dstream import envelope_type

    _capture(tmp_path / "s.dstream", n=20)
    main([str(tmp_path / "s.dstream"), str(tmp_path / "p.dstream"), "--passes", "3",
          "--series", "per-pass"])
    with socket.socket() as s:      # a free port; a closed zmq socket frees it only later
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    endpoint = f"tcp://127.0.0.1:{port}"
    ctx = zmq.Context.instance()
    feeder = threading.Thread(target=replay_dstream, args=(str(tmp_path / "p.dstream"), endpoint),
                              kwargs={"linger_s": 0.5}, daemon=True)
    feeder.start()
    pull = ctx.socket(zmq.PULL)
    pull.connect(endpoint)
    types = []
    while len(types) < 3 * 22:
        if pull.poll(5000):
            types.append(envelope_type(pull.recv()))
        else:
            break
    pull.close()
    starts = [i for i, t in enumerate(types) if t == "start"]
    ends = [i for i, t in enumerate(types) if t == "end"]
    assert len(starts) == 3 and len(ends) == 3
    assert all(s < e for s, e in zip(starts, ends)) and all(e < s for e, s in zip(ends, starts[1:]))


def test_refocus_emulates_the_focus_knob(tmp_path):
    """`tools/refocus_dstream.py`: a pass at another microscope defocus measures as that defocus
    (tools/calibrate_parallax.py), with the recording's rotation and flip; a pass at the
    recording's own defocus is the recording."""
    from live4d.parallax.ingest.codec import decode_message
    from live4d.parallax.ingest.dstream import iter_records
    from live4d.parallax.tools.calibrate_parallax import calibrate, file_batches
    from live4d.parallax.tools.refocus_dstream import main
    from live4d.parallax.tools.simulate_4dstem import SimParams, write

    p = SimParams(scan=(48, 48), detector=(64, 64), semiangle=10, angular_sampling=0.5,
                  scan_step=2.0, aberrations={"C10": -800.0}, rotation_deg=20,
                  detector_layout="rows,-cols", dose=20000, n_particles=8, seed=4)
    meta = write(p, tmp_path / "s", formats=("dstream",), device="cpu")
    main([str(tmp_path / "s.dstream"), str(tmp_path / "w.dstream"), "--flags",
          str(tmp_path / "s.flags"), "--defocus-schedule", "-1100", "-800"])
    cal = calibrate(file_batches(tmp_path / "w.dstream"), p.scan, p.energy, p.semiangle,
                    p.angular_sampling, meta["grid"]["scan_step"], rotation_prior=20,
                    loss=False, bootstrap=10, log=lambda *a: None)    # measures pass 0
    g = cal.geometry
    assert g["flip_cols"] and abs(g["rotation_deg"] - 20) < 1.5
    assert g["c10"] == pytest.approx(-1100, rel=0.05)
    n = 48 * 48
    orig = {i: f for t, i, f in map(decode_message, iter_records(tmp_path / "s.dstream")) if t == "image"}
    later = [(i, f) for t, i, f in map(decode_message, iter_records(tmp_path / "w.dstream"))
             if t == "image" and i >= n]
    assert len(later) == n
    for i, f in later:
        np.testing.assert_array_equal(f, orig[i - n])
