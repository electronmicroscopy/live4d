"""Generate a synthetic `.dstream` capture for arbitrary detector geometry/dtype.

Frames are bslz4-compressed with the real `bitshuffle` library and wrapped in the same CBOR
structure real DCU messages use (self-describe tag 55799, tag 40 multi-dim array, typed-array
tag 69/70, DECTRIS compression tag 56500, 12-byte bslz4 container header). Every frame is
round-tripped through `dectris.compression.decompress` at generation time, so a format mismatch
fails here, not in the pipeline under test.

Needs the DECTRIS decode dependencies (`bitshuffle`, `cbor2`, `dectris.compression`), which the
image does not carry: run it in a host environment that has them (`pip install -e ".[decode]"`):

    python -m tools.make_dstream 96 96 uint16 4096 --out examples/data/sample.dstream
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → `import tools.*`

from tools.dstream import write_magic, write_record   # noqa: E402

# CBOR typed-array tags for little-endian unsigned ints (RFC 8746)
_TYPED_ARRAY_TAG = {"uint16": 69, "uint32": 70}
_SELF_DESCRIBE = 55799


def make_frame(rng: np.random.Generator, shape: tuple[int, int], dtype: str,
               kind: str) -> np.ndarray:
    """Sparse frames mimic counting data; dense ones stress worst-case compression."""
    if kind == "sparse":
        frame = np.zeros(shape, dtype=dtype)
        n_hot = int(rng.integers(0, frame.size // 50 + 2))
        idx = rng.integers(0, frame.size, size=n_hot)
        frame.flat[idx] = rng.integers(1, 200, size=n_hot).astype(dtype)
        return frame
    info = np.iinfo(dtype)
    return rng.integers(0, int(info.max) + 1, size=shape, dtype=dtype)


def encode_bslz4(frame: np.ndarray, block_bytes: int) -> bytes:
    import bitshuffle
    elem = frame.dtype.itemsize
    block_elems = block_bytes // elem
    payload = bitshuffle.compress_lz4(frame, block_elems).tobytes()
    header = struct.pack(">QI", frame.nbytes, block_bytes)
    return header + payload


def image_message(image_id: int, frame: np.ndarray, block_bytes: int) -> bytes:
    import cbor2
    from dectris.compression import decompress
    dtype = str(frame.dtype)
    blob = encode_bslz4(frame, block_bytes)
    # Generation-time self check: the real decoder must reproduce the frame bytes.
    if decompress(blob, "bslz4", elem_size=frame.dtype.itemsize) != frame.tobytes():
        raise AssertionError("bslz4 roundtrip failed — encoder/decoder mismatch")
    compression = cbor2.CBORTag(56500, ["bslz4", frame.dtype.itemsize, blob])
    typed = cbor2.CBORTag(_TYPED_ARRAY_TAG[dtype], compression)
    multidim = cbor2.CBORTag(40, [list(frame.shape), typed])
    msg = {
        "type": "image",
        "series_id": 1,
        "image_id": image_id,
        "data": {"threshold_1": multidim},
    }
    return cbor2.dumps(cbor2.CBORTag(_SELF_DESCRIBE, msg))


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python -m tools.make_dstream", description=__doc__.splitlines()[0])
    p.add_argument("height", type=int, help="Detector rows.")
    p.add_argument("width", type=int, help="Detector columns.")
    p.add_argument("dtype", choices=tuple(_TYPED_ARRAY_TAG), help="Pixel type.")
    p.add_argument("n_frames", type=int, help="Number of frames (random counts, not physics).")
    p.add_argument("--block-bytes", type=int, default=8192,
                   help="bslz4 block size in bytes (written to the container header).")
    p.add_argument("--seed", type=int, default=1234, help="Random seed.")
    p.add_argument("--out", required=True, help="The .dstream file to write.")
    return p


def main() -> None:
    from realtime_stem.ingest import codec

    p = build_parser()
    args = p.parse_args()

    shape = (args.height, args.width)
    rng = np.random.default_rng(args.seed)
    # START and END as a DCU sends them: the Holoscan 5 receivers take the geometry from START.
    start = codec.encode_start_message(shape, args.n_frames, args.dtype)
    end = codec.encode_end_message()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        write_magic(fh)
        write_record(fh, start)
        for image_id in range(args.n_frames):
            kind = "dense" if image_id % 8 == 7 else "sparse"
            frame = make_frame(rng, shape, args.dtype, kind)
            write_record(fh, image_message(image_id, frame, args.block_bytes))
        write_record(fh, end)

    elems = args.height * args.width
    elem = np.dtype(args.dtype).itemsize
    tail = (elems * elem) % args.block_bytes
    print(f"[synth] {args.n_frames} frames {shape} {args.dtype}, block={args.block_bytes}B "
          f"({'tail ' + str(tail) + 'B' if tail else 'no tail'}) -> {args.out}")


if __name__ == "__main__":
    main()
