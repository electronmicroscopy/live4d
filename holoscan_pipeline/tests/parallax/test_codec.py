"""Tests for the CPU decoder of DECTRIS Stream2 messages (`live4d.parallax/ingest/codec.py`).

The numpy bslz4 path is pinned against independent implementations where installed — the
`bitshuffle` library (the DCU's own encoder), `dectris-compression` (DECTRIS' C decoder) and the
HDF5 bitshuffle filter via `hdf5plugin` (EIGER/ARINA HDF5 files use the same chunk format) — and
always against itself (encode → decode round trips, incl. partial and raw-tail blocks).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("cbor2")
pytest.importorskip("lz4")

from live4d.parallax.ingest import codec  # noqa: E402
from live4d.parallax.ingest.dstream import iter_records  # noqa: E402

SAMPLE = (Path(__file__).resolve().parents[2] / "python/live4d/parallax") / "examples" / "data" / "sample.dstream"

_CASES = [(dt, shape, bb) for dt in (np.uint8, np.uint16, np.uint32)
          for shape in ((96, 96), (64, 56), (13, 7), (3, 3)) for bb in (8192, 1024)]


def _numpy_decode(blob, shape, dtype):
    return codec._bslz4_decode_numpy(blob, np.dtype(dtype).itemsize).view(dtype).reshape(shape)


@pytest.mark.parametrize("dtype,shape,block_bytes", _CASES)
def test_numpy_roundtrip(dtype, shape, block_bytes, monkeypatch):
    monkeypatch.setattr(codec, "_bitshuffle_compress", None)     # the numpy encoder itself
    rng = np.random.default_rng(0)
    a = rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    blob = codec.bslz4_encode(a, block_bytes)
    assert struct.unpack(">QI", blob[:12]) == (a.nbytes, block_bytes)
    assert np.array_equal(_numpy_decode(blob, shape, dtype), a)


@pytest.mark.parametrize("dtype,shape,block_bytes", _CASES)
def test_matches_bitshuffle_library(dtype, shape, block_bytes):
    bitshuffle = pytest.importorskip("bitshuffle")
    a = np.random.default_rng(1).integers(0, 300, size=shape).astype(dtype)
    payload = bitshuffle.compress_lz4(a, block_bytes // a.dtype.itemsize).tobytes()
    blob = struct.pack(">QI", a.nbytes, block_bytes) + payload
    assert np.array_equal(_numpy_decode(blob, shape, dtype), a)       # theirs → our decoder
    # ours → their decoder (LZ4 output may differ byte-wise between implementations; the
    # bitshuffled blocks and the framing may not)
    ours = np.frombuffer(codec.bslz4_encode(a, block_bytes)[12:], dtype=np.uint8)
    back = bitshuffle.decompress_lz4(ours, shape, np.dtype(dtype), block_bytes // a.dtype.itemsize)
    assert np.array_equal(back, a)


@pytest.mark.parametrize("dtype,shape,block_bytes", _CASES)
def test_matches_dectris_decoder(dtype, shape, block_bytes):
    dectris = pytest.importorskip("dectris.compression")
    a = np.random.default_rng(2).integers(0, 300, size=shape).astype(dtype)
    blob = codec.bslz4_encode(a, block_bytes)
    ref = np.frombuffer(dectris.decompress(blob, "bslz4", elem_size=a.dtype.itemsize), dtype)
    assert np.array_equal(ref.reshape(shape), a)


def test_sample_capture_matches_hdf5_filter():
    """The real-framing sample capture decodes identically through the HDF5 bitshuffle filter."""
    h5py = pytest.importorskip("h5py")
    hdf5plugin = pytest.importorskip("hdf5plugin")
    import cbor2

    n = 0
    with h5py.File("in-memory.h5", "w", driver="core", backing_store=False) as f:
        for raw in iter_records(SAMPLE):
            msg = cbor2.loads(raw)
            if msg["type"] != "image":
                continue
            node = msg["data"]["threshold_1"]
            shape = tuple(node.value[0])
            _, elem_size, blob = node.value[1].value.value
            dtype = np.dtype(f"<u{elem_size}")
            ds = f.create_dataset(f"f{n}", shape=shape, dtype=dtype, chunks=shape,
                                  **hdf5plugin.Bitshuffle(cname="lz4"))
            ds.id.write_direct_chunk((0, 0), blob)
            assert np.array_equal(_numpy_decode(blob, shape, dtype), ds[...])
            n += 1
            if n == 40:
                break
    assert n == 40


def test_decode_message_envelope_and_roundtrip():
    frame = np.random.default_rng(3).integers(0, 4000, size=(96, 96), dtype=np.uint16)
    assert codec.decode_message(codec.encode_control_message("start")) == ("start", None, None)
    assert codec.decode_message(codec.encode_control_message("end")) == ("end", None, None)
    mtype, image_id, got = codec.decode_message(codec.encode_image_message(17, frame))
    assert (mtype, image_id) == ("image", 17)
    assert got.dtype == np.uint16 and np.array_equal(got, frame)
    big = frame.astype(np.uint32) * 70000
    assert np.array_equal(codec.decode_message(codec.encode_image_message(0, big))[2], big)


def test_sample_capture_decodes_with_both_backends(monkeypatch):
    records = list(iter_records(SAMPLE))
    fast = [codec.decode_message(r) for r in records]
    monkeypatch.setattr(codec, "_dectris_decompress", None)
    slow = [codec.decode_message(r) for r in records]
    images = [(i, f) for t, i, f in fast if t == "image"]
    assert len(images) == 1024 and [i for i, _ in images] == list(range(1024))
    assert all(f.shape == (96, 96) and f.dtype == np.uint16 for _, f in images)
    assert all(a[0] == b[0] and a[1] == b[1] and (a[2] is None or np.array_equal(a[2], b[2]))
               for a, b in zip(fast, slow))
