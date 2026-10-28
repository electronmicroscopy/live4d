"""DECTRIS Stream2 message decoding on the CPU: CBOR envelope + bslz4 pixel payload.

The CPU counterpart of the compiled ingest (`ZmqRxOp` parses the CBOR, `DecompressGpuOp` undoes the
bslz4 on the GPU). It lets the pipeline run where there is no NVIDIA GPU — a laptop, CI, an
offline replay — at a few tens of kHz rather than 120 kHz.

Message layout (as in `holoscan_pipeline/cpp/common/fast_cbor_parse.h` and `tools/make_dstream.py`):

    tag 55799 (self-describe) { "type": "image", "image_id": n,
        "data": { "threshold_1": tag 40 [ [H, W], tag 69|70 ( tag 56500 ["bslz4", elem_size, blob] ) ] } }

cbor2 unwraps tag 55799 itself; the others arrive as `CBORTag`s.

bslz4 blob layout (bitshuffle's HDF5 chunk format, which DECTRIS reuses): a 12-byte big-endian header
`(uint64 total bytes, uint32 block bytes)`, then one `[uint32 BE compressed size][LZ4 block]` per
block of `block bytes / elem_size` elements, the last block shortened to a multiple of 8 elements,
then the final `n % 8` elements stored raw. Each decompressed block is bit-transposed: bit `b` of
every element is packed into row `b` of a `(8 * elem_size, n_block / 8)` byte array.

`bslz4_decode` uses the DECTRIS C decoder when `dectris-compression` is installed and a numpy + `lz4`
fallback otherwise; `bslz4_encode` is the numpy inverse, for tests and for writing simulated captures.
"""

from __future__ import annotations

import struct

import numpy as np

__all__ = ["decode_message", "bslz4_decode", "bslz4_encode", "encode_image_message",
           "encode_control_message", "encode_start_message", "encode_end_message"]

_HEADER = struct.Struct(">QI")
_BLOCK_SIZE = struct.Struct(">I")

# CBOR typed-array tags for little-endian unsigned ints (RFC 8746).
_TYPED_ARRAY_DTYPE = {64: np.uint8, 69: np.dtype("<u2"), 70: np.dtype("<u4")}
_DTYPE_TYPED_ARRAY = {np.dtype(np.uint8): 64, np.dtype("<u2"): 69, np.dtype("<u4"): 70}
_SELF_DESCRIBE = 55799
_COMPRESSION_TAG = 56500
_MULTIDIM_TAG = 40

try:  # the DECTRIS C decoder: ~10x faster than the numpy fallback when available
    from dectris.compression import decompress as _dectris_decompress
except ImportError:  # pragma: no cover - depends on the environment
    _dectris_decompress = None

try:  # the bitshuffle C encoder (what the DCU itself uses), for writing captures fast
    from bitshuffle import compress_lz4 as _bitshuffle_compress
except ImportError:  # pragma: no cover - depends on the environment
    _bitshuffle_compress = None


# ── bslz4 ──────────────────────────────────────────────────────────────────────

def _unshuffle_block(shuffled: np.ndarray, n: int, elem_size: int) -> np.ndarray:
    """Inverse bitshuffle of one block of `n` (a multiple of 8) elements, as raw bytes."""
    bits = np.unpackbits(shuffled.reshape(8 * elem_size, n // 8), axis=1, bitorder="little")
    return np.packbits(bits.T.reshape(n, elem_size, 8), axis=-1, bitorder="little").reshape(-1)


def _shuffle_block(raw: np.ndarray, n: int, elem_size: int) -> np.ndarray:
    """Bitshuffle one block of `n` (a multiple of 8) elements given as raw bytes."""
    bits = np.unpackbits(raw.reshape(n, elem_size, 1), axis=-1, bitorder="little")
    return np.packbits(bits.reshape(n, 8 * elem_size).T, axis=1, bitorder="little").reshape(-1)


def _bslz4_decode_numpy(blob: bytes, elem_size: int) -> np.ndarray:
    import lz4.block

    buf = memoryview(blob)
    total_bytes, block_bytes = _HEADER.unpack_from(buf, 0)
    n_total = total_bytes // elem_size
    block_elems = block_bytes // elem_size
    out = np.empty(total_bytes, dtype=np.uint8)

    pos, done = _HEADER.size, 0
    remaining = n_total
    while remaining >= 8:
        n = min(block_elems, remaining - remaining % 8)
        (csize,) = _BLOCK_SIZE.unpack_from(buf, pos)
        pos += _BLOCK_SIZE.size
        raw = lz4.block.decompress(buf[pos:pos + csize], uncompressed_size=n * elem_size)
        pos += csize
        shuffled = np.frombuffer(raw, dtype=np.uint8)
        out[done * elem_size:(done + n) * elem_size] = _unshuffle_block(shuffled, n, elem_size)
        done += n
        remaining -= n
    if remaining:  # the trailing n % 8 elements are stored verbatim
        out[done * elem_size:] = np.frombuffer(buf[pos:pos + remaining * elem_size], np.uint8)
    return out


def bslz4_decode(blob: bytes, shape: tuple[int, ...], dtype) -> np.ndarray:
    """Decompress one bslz4 blob (12-byte header included) into a `shape` array of `dtype`."""
    dtype = np.dtype(dtype)
    if _dectris_decompress is not None:
        raw = np.frombuffer(_dectris_decompress(blob, "bslz4", elem_size=dtype.itemsize), np.uint8)
    else:
        raw = _bslz4_decode_numpy(blob, dtype.itemsize)
    return raw.view(dtype).reshape(shape)


def bslz4_encode(arr: np.ndarray, block_bytes: int = 8192) -> bytes:
    """bslz4-compress an array (header included), byte-compatible with bitshuffle/DECTRIS."""
    import lz4.block

    a = np.ascontiguousarray(arr)
    elem_size = a.dtype.itemsize
    if _bitshuffle_compress is not None:
        payload = _bitshuffle_compress(a, block_bytes // elem_size).tobytes()
        return _HEADER.pack(a.nbytes, block_bytes) + payload
    raw = a.reshape(-1).view(np.uint8)
    n_total = a.size
    block_elems = block_bytes // elem_size

    parts = [_HEADER.pack(a.nbytes, block_bytes)]
    done, remaining = 0, n_total
    while remaining >= 8:
        n = min(block_elems, remaining - remaining % 8)
        shuffled = _shuffle_block(raw[done * elem_size:(done + n) * elem_size], n, elem_size)
        comp = lz4.block.compress(shuffled.tobytes(), store_size=False)
        parts += [_BLOCK_SIZE.pack(len(comp)), comp]
        done += n
        remaining -= n
    if remaining:
        parts.append(raw[done * elem_size:].tobytes())
    return b"".join(parts)


# ── CBOR envelope ──────────────────────────────────────────────────────────────

def _decode_array(node):
    """Decode a tag-40 multi-dim array node into a numpy array."""
    import cbor2

    if not (isinstance(node, cbor2.CBORTag) and node.tag == _MULTIDIM_TAG):
        raise ValueError(f"expected a CBOR tag-40 multi-dimensional array, got {node!r:.80}")
    shape, typed = node.value
    shape = tuple(int(d) for d in shape)
    if not (isinstance(typed, cbor2.CBORTag) and typed.tag in _TYPED_ARRAY_DTYPE):
        raise ValueError(f"unsupported typed-array tag in image data: {typed!r:.80}")
    dtype = np.dtype(_TYPED_ARRAY_DTYPE[typed.tag])
    payload = typed.value

    if isinstance(payload, (bytes, bytearray)):            # uncompressed
        return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    if isinstance(payload, cbor2.CBORTag) and payload.tag == _COMPRESSION_TAG:
        algorithm, elem_size, blob = payload.value
        if algorithm != "bslz4":
            raise ValueError(f"unsupported DECTRIS compression {algorithm!r} (only bslz4)")
        if int(elem_size) != dtype.itemsize:
            raise ValueError(f"compression elem_size {elem_size} != dtype size {dtype.itemsize}")
        return bslz4_decode(blob, shape, dtype)
    raise ValueError(f"unsupported image payload: {payload!r:.80}")


def decode_message(raw: bytes):
    """Decode one Stream2 message.

    Returns `(type, image_id, frame)`: `type` is "start" / "image" / "end" (or whatever the
    message says); `image_id` and `frame` are None for anything but images. The first data
    channel (e.g. "threshold_1") is the one decoded.
    """
    import cbor2

    msg = cbor2.loads(raw)
    if isinstance(msg, cbor2.CBORTag) and msg.tag == _SELF_DESCRIBE:   # older cbor2 versions
        msg = msg.value
    mtype = msg.get("type") if hasattr(msg, "get") else None
    if mtype != "image":
        return mtype, None, None
    channels = msg["data"]
    frame = _decode_array(next(iter(channels.values())))
    return "image", int(msg["image_id"]), frame


def encode_image_message(image_id: int, frame: np.ndarray, block_bytes: int = 8192,
                         series_id: int = 1) -> bytes:
    """A Stream2 image message for `frame`, in the DCU's CBOR framing (inverse of decode)."""
    import cbor2

    frame = np.ascontiguousarray(frame)
    dtype = frame.dtype.newbyteorder("<") if frame.dtype.itemsize > 1 else frame.dtype
    frame = frame.astype(dtype, copy=False)
    compression = cbor2.CBORTag(_COMPRESSION_TAG,
                                ["bslz4", frame.dtype.itemsize, bslz4_encode(frame, block_bytes)])
    typed = cbor2.CBORTag(_DTYPE_TYPED_ARRAY[np.dtype(dtype)], compression)
    multidim = cbor2.CBORTag(_MULTIDIM_TAG, [list(frame.shape), typed])
    msg = {"type": "image", "series_id": series_id, "image_id": int(image_id),
           "data": {"threshold_1": multidim}}
    return cbor2.dumps(cbor2.CBORTag(_SELF_DESCRIBE, msg))


def encode_control_message(mtype: str, series_id: int = 1, **fields) -> bytes:
    """A minimal Stream2 `start` / `end` message."""
    import cbor2

    return cbor2.dumps(cbor2.CBORTag(_SELF_DESCRIBE, {"type": mtype, "series_id": series_id,
                                                      **fields}))


def encode_start_message(shape: tuple[int, int], n_images: int, dtype: str = "uint16",
                         series_id: int = 1) -> bytes:
    """A Stream2 `start` with the fields a DCU always sends and receivers rely on: the frame
    size (`image_size_y`, `image_size_x`), `image_dtype` and `number_of_images` (the Holoscan 5
    receiver takes the geometry from it)."""
    return encode_control_message("start", series_id, series_unique_id=f"simulated-{series_id}",
                                  image_size_x=int(shape[1]), image_size_y=int(shape[0]),
                                  image_dtype=str(dtype), number_of_images=int(n_images))


def encode_end_message(series_id: int = 1) -> bytes:
    """A Stream2 `end`, as a DCU sends it (with the series' unique id)."""
    return encode_control_message("end", series_id, series_unique_id=f"simulated-{series_id}")
