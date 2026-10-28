"""Tests for the pure-Python ingest (`live4d.parallax/ingest/python_ingest.py`).

Every source must deliver exactly the frames, in the right scan positions, that feeding the
accumulator directly would — so the reconstruction is compared against `acc.process` on the
decoded frames. ZMQ cases run a PUSH feeder thread on localhost, including the multi-connection
end-of-stream race and two scans back to back on one stream.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("cbor2")
zmq = pytest.importorskip("zmq")

from live4d.parallax.ingest import codec  # noqa: E402
from live4d.parallax.ingest.dstream import iter_records, write_magic, write_record  # noqa: E402
from live4d.parallax.ingest.python_ingest import PythonIngest, ZmqStream  # noqa: E402
from live4d.algorithms.parallax.engine import (  # noqa: E402
    ParallaxAccumulator,
    preprocess_geometry,
)

SAMPLE = (Path(__file__).resolve().parents[2] / "python/live4d/parallax") / "examples" / "data" / "sample.dstream"


def _sample_geometry():
    """The bundled sample capture: 32×32 scan of binned 96×96 frames."""
    return preprocess_geometry(det_shape=(96, 96), scan_shape=(32, 32), scan_sampling=(0.5, 0.5),
                               energy=300e3, semiangle_cutoff=4.0, angular_sampling=(1.0, 1.0),
                               aberration_coefs={"C10": -1.5e4})


def _sample_frames():
    out = {}
    for raw in iter_records(SAMPLE):
        mtype, image_id, frame = codec.decode_message(raw)
        if mtype == "image":
            out[image_id] = frame
    ids = np.array(sorted(out))
    return ids, np.stack([out[i] for i in ids])


def _direct(geo, frames, ids):
    acc = ParallaxAccumulator(geo, device="cpu")
    acc.process(frames, ids)
    return acc


def _assert_same(acc, ref):
    assert acc._n_frames == ref._n_frames
    # float32 sums in a batching-dependent order: compare against the image scale
    scale = np.abs(ref.result()).max()
    np.testing.assert_allclose(acc.result(), ref.result(), rtol=0, atol=1e-5 * scale)
    np.testing.assert_array_equal(acc._visits.numpy(), ref._visits.numpy())


def test_dstream_source_matches_direct():
    geo = _sample_geometry()
    ids, frames = _sample_frames()
    acc = ParallaxAccumulator(geo, device="cpu")
    ing = PythonIngest(acc, source=SAMPLE, batch_size=100)     # 1024 = 10·100 + a partial 24
    ing.run()
    assert ing.sink.frames == 1024                            # the partial final batch is kept
    _assert_same(acc, _direct(geo, frames, ids))


def test_npy_source_matches_direct(tmp_path):
    geo = _sample_geometry()
    ids, frames = _sample_frames()
    path = tmp_path / "frames.npy"
    np.save(path, frames.reshape(32, 32, 96, 96))
    acc = ParallaxAccumulator(geo, device="cpu")
    PythonIngest(acc, source=path, batch_size=128).run()
    _assert_same(acc, _direct(geo, frames, ids))


def test_npy_source_rejects_wrong_detector(tmp_path):
    path = tmp_path / "frames.npy"
    np.save(path, np.zeros((4, 64, 64), np.uint16))
    acc = ParallaxAccumulator(_sample_geometry(), device="cpu")
    with pytest.raises(ValueError, match="--detector"):
        PythonIngest(acc, source=path).run()


def test_source_rate_paces_the_replay(tmp_path):
    geo = _sample_geometry()
    path = tmp_path / "frames.npy"
    np.save(path, np.zeros((256, 96, 96), np.uint16))
    acc = ParallaxAccumulator(geo, device="cpu")
    t0 = time.perf_counter()
    PythonIngest(acc, source=path, source_rate=1000).run()   # 256 frames at 1 kHz ≥ 0.25 s
    assert time.perf_counter() - t0 >= 0.24
    assert acc._n_frames == 256


def _feeder(messages, n_peers_ready: threading.Event, endpoint_box: list):
    """PUSH `messages` from a socket bound on an ephemeral port (published via endpoint_box)."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.bind("tcp://127.0.0.1:*")
    endpoint_box.append(sock.getsockopt_string(zmq.LAST_ENDPOINT))
    n_peers_ready.wait(5)
    time.sleep(0.2)                                   # let every PULL connection attach
    for raw in messages:
        sock.send(raw)
    time.sleep(0.5)
    sock.close(linger=2000)


def _run_stream(messages, scans=1, batch_size=100):
    """Feed `messages` over ZMQ; run `scans` ingests on one stream; return their accumulators."""
    ready, box = threading.Event(), []
    t = threading.Thread(target=_feeder, args=(messages, ready, box), daemon=True)
    t.start()
    while not box:
        time.sleep(0.01)
    stream = ZmqStream(box[0])
    ready.set()
    accs = []
    try:
        for _ in range(scans):
            acc = ParallaxAccumulator(_sample_geometry(), device="cpu")
            PythonIngest(acc, stream=stream, batch_size=batch_size).run()
            accs.append(acc)
    finally:
        stream.close()
        t.join(5)
    return accs


def test_zmq_replay_receives_every_frame():
    """With several connections the feeder's `end` can overtake frames still queued on another
    connection; draining until quiet must still deliver all 1024 (the compiled ingest loses them)."""
    geo = _sample_geometry()
    ids, frames = _sample_frames()
    (acc,) = _run_stream(list(iter_records(SAMPLE)))
    _assert_same(acc, _direct(geo, frames, ids))


def test_zmq_two_scans_back_to_back():
    """A `start` of the next scan arriving while the first drains is handed to the next run."""
    ids, frames = _sample_frames()
    first, second = slice(0, 400), slice(400, 1024)
    msgs = [codec.encode_control_message("start")]
    msgs += [codec.encode_image_message(i, f) for i, f in zip(ids[first], frames[first])]
    msgs += [codec.encode_control_message("end"), codec.encode_control_message("start")]
    msgs += [codec.encode_image_message(i, f) for i, f in zip(ids[second], frames[second])]
    msgs += [codec.encode_control_message("end")]
    acc1, acc2 = _run_stream(msgs, scans=2)
    geo = _sample_geometry()
    _assert_same(acc1, _direct(geo, frames[first], ids[first]))
    _assert_same(acc2, _direct(geo, frames[second], ids[second]))


def test_zmq_rejects_wrong_dtype(tmp_path):
    msgs = [codec.encode_control_message("start"),
            codec.encode_image_message(0, np.zeros((96, 96), np.uint32)),
            codec.encode_control_message("end")]
    with pytest.raises(ValueError, match="dtype"):
        _run_stream(msgs)


def test_written_dstream_roundtrip(tmp_path):
    """encode_image_message + write_record produce a capture the ingest reads back exactly."""
    ids, frames = _sample_frames()
    path = tmp_path / "copy.dstream"
    with open(path, "wb") as fh:
        write_magic(fh)
        write_record(fh, codec.encode_control_message("start"))
        for i, f in zip(ids, frames):
            write_record(fh, codec.encode_image_message(int(i), f))
        write_record(fh, codec.encode_control_message("end"))
    geo = _sample_geometry()
    acc = ParallaxAccumulator(geo, device="cpu")
    PythonIngest(acc, source=path).run()
    _assert_same(acc, _direct(geo, frames, ids))


def test_record_dstream_roundtrip(tmp_path):
    """tools/record_dstream writes back exactly the messages the feeder sent."""
    from live4d.parallax.tools.record_dstream import record

    records = list(iter_records(SAMPLE))
    ready, box = threading.Event(), []
    t = threading.Thread(target=_feeder, args=(records, ready, box), daemon=True)
    t.start()
    while not box:
        time.sleep(0.01)
    out = tmp_path / "rec.dstream"
    ready.set()
    n = record(box[0], out)
    t.join(5)
    assert n == 1024
    got = list(iter_records(out))
    assert got == records        # two connections may interleave the order
