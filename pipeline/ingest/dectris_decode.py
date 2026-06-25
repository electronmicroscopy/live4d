"""
Decoding of DECTRIS stream messages.

A message off the ZMQ socket is a CBOR-encoded blob. Decoding it means parsing the CBOR
*and* decompressing any image payload (tag 56500 -> bitshuffle+LZ4 / lz4). Both happen
inside a single `cbor2.loads(..., tag_hook=...)` call: the tag hook is invoked for each
CBOR tag and turns typed arrays / compressed buffers into NumPy arrays.

This module is vendor-specific but algorithm-agnostic, so any consumer (virtual detectors,
parallax, ...) shares one decoder. The `bslz4` container format and the GIL behavior of
`decompress` are documented in `docs/stream_format.md`.

Public API:
    decode_message(raw_bytes)  -> the decoded message (a dict with a 'type' field); decompresses.
    decode_envelope(raw_bytes) -> CompressedFrame | None -- parse an 'image' message's envelope
        WITHOUT decompressing (the CPU side of the GPU decode path); None for start/end.
"""

from __future__ import annotations

from typing import NamedTuple

import cbor2
import numpy as np
from dectris.compression import decompress


# -- CBOR tag decoders --------------------------------------------------------

def decode_multi_dim_array(tag, column_major):
    dimensions, contents = tag.value
    if isinstance(contents, list):
        array = np.empty((len(contents),), dtype=object)
        array[:] = contents
    elif isinstance(contents, (np.ndarray, np.generic)):
        array = contents
    else:
        raise cbor2.CBORDecodeValueError("expected array or typed array")
    return array.reshape(dimensions, order="F" if column_major else "C")


def decode_typed_array(tag, dtype):
    if not isinstance(tag.value, bytes):
        raise cbor2.CBORDecodeValueError("expected byte string in typed array")
    return np.frombuffer(tag.value, dtype=dtype)


def decode_dectris_compression(tag):
    """Tag 56500: value is (algorithm, elem_size, encoded) -> decompressed bytes.

    `algorithm` is 'bslz4' (bitshuffle+LZ4) or 'lz4'. The decompress call releases the
    GIL, so decoding many messages across threads runs in parallel (see the stream-format
    notes).
    """
    algorithm, elem_size, encoded = tag.value
    return decompress(encoded, algorithm, elem_size=elem_size)


# Tag number -> decoder. Typed-array tags cover the endian/dtype variants of the CBOR spec.
tag_decoders = {
    40: lambda tag: decode_multi_dim_array(tag, column_major=False),
    64: lambda tag: decode_typed_array(tag, dtype="u1"),
    65: lambda tag: decode_typed_array(tag, dtype=">u2"),
    66: lambda tag: decode_typed_array(tag, dtype=">u4"),
    67: lambda tag: decode_typed_array(tag, dtype=">u8"),
    68: lambda tag: decode_typed_array(tag, dtype="u1"),
    69: lambda tag: decode_typed_array(tag, dtype="<u2"),
    70: lambda tag: decode_typed_array(tag, dtype="<u4"),
    71: lambda tag: decode_typed_array(tag, dtype="<u8"),
    72: lambda tag: decode_typed_array(tag, dtype="i1"),
    73: lambda tag: decode_typed_array(tag, dtype=">i2"),
    74: lambda tag: decode_typed_array(tag, dtype=">i4"),
    75: lambda tag: decode_typed_array(tag, dtype=">i8"),
    77: lambda tag: decode_typed_array(tag, dtype="<i2"),
    78: lambda tag: decode_typed_array(tag, dtype="<i4"),
    79: lambda tag: decode_typed_array(tag, dtype="<i8"),
    80: lambda tag: decode_typed_array(tag, dtype=">f2"),
    81: lambda tag: decode_typed_array(tag, dtype=">f4"),
    82: lambda tag: decode_typed_array(tag, dtype=">f8"),
    83: lambda tag: decode_typed_array(tag, dtype=">f16"),
    84: lambda tag: decode_typed_array(tag, dtype="<f2"),
    85: lambda tag: decode_typed_array(tag, dtype="<f4"),
    86: lambda tag: decode_typed_array(tag, dtype="<f8"),
    87: lambda tag: decode_typed_array(tag, dtype="<f16"),
    1040: lambda tag: decode_multi_dim_array(tag, column_major=True),
    56500: lambda tag: decode_dectris_compression(tag),
}


def tag_hook(*args):
    """Dispatch a CBOR tag to its decoder; leave unknown tags untouched.

    cbor2 versions disagree on how they call this hook -- older ones pass `(decoder, tag)`,
    cbor2 6.x passes `(tag, immutable_flag)`. So instead of assuming a position, locate the
    `CBORTag` among the arguments.
    """
    tag = next((a for a in args if isinstance(a, cbor2.CBORTag)), None)
    if tag is None:
        return args[0] if args else None
    tag_decoder = tag_decoders.get(tag.tag)
    return tag_decoder(tag) if tag_decoder else tag


# -- Public entry point -------------------------------------------------------

def decode_message(raw: bytes):
    """Decode one raw ZMQ message (CBOR bytes) into its Python representation.

    Image payloads are returned as NumPy arrays (decompressed). The result is typically a
    dict carrying a 'type' field ('image', plus series start/end) and, for frames, a
    message/image id used downstream for the scan position.
    """
    return cbor2.loads(raw, tag_hook=tag_hook)


def envelope_type(raw: bytes):
    """Return the message 'type' without decompressing the image payload.

    Decoding with no tag_hook leaves the bslz4 blob (tag 56500) as an inert `CBORTag`, so the
    small envelope fields (`type`, `image_size_*`, `number_of_images`) are read for free. Used by
    the producer thread to peek the start/image/end framing before any decode.
    """
    try:
        msg = cbor2.loads(raw)
    except Exception:
        return None
    return msg.get("type") if hasattr(msg, "get") else None


# -- Envelope-only parse (the CPU side of the GPU decode path) -----------------
#
# The GPU decode path (`ingest/gpu_decode.py`) wants the CPU to do only the *cheap* work: parse
# the CBOR envelope for the image id + the still-compressed payload, and ship the compressed bytes
# to the GPU (bslz4 decompress runs there). So we parse the message WITHOUT calling `decompress`.
# The image data is nested: tag 40/1040 (multidim -> shape) -> tag 64-87 (typed array -> dtype) ->
# tag 56500 (compression -> algorithm, elem_size, encoded). We walk that chain without touching a
# pixel; a small dict bubbles up through the hooks, gaining `dtype` then `shape`.

# CBOR typed-array tag -> numpy dtype string (the subset of `tag_decoders` we need to read the
# element dtype off the envelope without materialising the array).
_TYPED_ARRAY_DTYPE = {
    64: "u1", 65: ">u2", 66: ">u4", 67: ">u8", 68: "u1", 69: "<u2", 70: "<u4", 71: "<u8",
    72: "i1", 73: ">i2", 74: ">i4", 75: ">i8", 77: "<i2", 78: "<i4", 79: "<i8",
    80: ">f2", 81: ">f4", 82: ">f8", 83: ">f16", 84: "<f2", 85: "<f4", 86: "<f8", 87: "<f16",
}


class CompressedFrame(NamedTuple):
    """One image's payload, parsed from the CBOR envelope but **not** decompressed.

    Carries everything the GPU decode path needs: the scan id, the compression algorithm + element
    byte width + the raw compressed bytes (feed straight to nvCOMP), and the dtype + (H, W) shape
    the decompressed bytes represent (to reinterpret the GPU output). The CPU never touches a pixel.
    """
    image_id: int
    algorithm: str      # 'bslz4' or 'lz4'
    elem_size: int      # bytes per element (1/2/4)
    encoded: bytes      # the compressed blob
    dtype: str          # numpy dtype string, e.g. '<u4'
    shape: tuple        # (H, W)


def _envelope_tag_hook(*args):
    """Tag hook that captures the image payload's algorithm/dtype/shape WITHOUT decompressing.

    For the image-data tag chain it returns a plain dict that bubbles up, gaining `dtype` (typed
    tag) and `shape` (multidim tag) as the outer tags unwrap; tag 56500 yields the compressed bytes
    verbatim. Any other tag is left untouched (we only read `data`)."""
    tag = next((a for a in args if isinstance(a, cbor2.CBORTag)), None)
    if tag is None:
        return args[0] if args else None
    t = tag.tag
    if t == 56500:                                   # compression: keep the bytes, do NOT decompress
        algorithm, elem_size, encoded = tag.value
        return {"algorithm": algorithm, "elem_size": elem_size, "encoded": encoded}
    if t in _TYPED_ARRAY_DTYPE:                      # typed array: record the element dtype
        inner = tag.value
        if isinstance(inner, dict):
            inner["dtype"] = _TYPED_ARRAY_DTYPE[t]
        return inner
    if t in (40, 1040):                              # multidim: record the frame shape
        dims, contents = tag.value
        if isinstance(contents, dict):
            contents["shape"] = tuple(dims)
            return contents
        return decode_multi_dim_array(tag, column_major=(t == 1040))
    return tag


def decode_envelope(raw: bytes) -> "CompressedFrame | None":
    """Parse one raw 'image' message into a `CompressedFrame` WITHOUT decompressing the pixels.

    Returns None for non-image messages (start/end). The heavy bslz4 decompress is deferred to the
    GPU; this keeps only the small (~microseconds) CPU envelope parse on the host.
    """
    msg = cbor2.loads(raw, tag_hook=_envelope_tag_hook)
    if not hasattr(msg, "get") or msg.get("type") != "image":
        return None
    payload = next(iter(msg["data"].values()))       # unwrap the single channel
    return CompressedFrame(
        image_id=int(msg["image_id"]),
        algorithm=payload["algorithm"],
        elem_size=int(payload["elem_size"]),
        encoded=payload["encoded"],
        dtype=payload["dtype"],
        shape=payload["shape"],
    )
