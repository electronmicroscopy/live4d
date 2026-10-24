"""Convert a 4D-STEM array (.npy) into a `.dstream` capture with the DCU's message framing.

Lets any recorded dataset go through the full streaming path — `tools/replay_to_zmq.py` into either
ingest (the Holoscan one included) — not just through `run.py --source`. Frames are cast to the
detector's integer type (uint16 by default, as the ARINA streams at full rate) and bslz4-compressed.

    python -m tools.npy_to_dstream data/apoF.npy --out out/apoF.dstream
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → `import realtime_stem`

from realtime_stem.ingest import codec  # noqa: E402
from realtime_stem.ingest.dstream import write_magic, write_record  # noqa: E402


def convert(src: Path, out: Path, dtype: str = "uint16", block_bytes: int = 8192) -> int:
    data = np.load(src, mmap_mode="r")
    frames = data.reshape(-1, *data.shape[-2:])
    info = np.iinfo(dtype)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        write_magic(fh)
        write_record(fh, codec.encode_control_message("start"))
        for i in range(len(frames)):
            f = np.asarray(frames[i])
            if f.dtype.kind == "f":
                f = np.rint(f)
            f = np.clip(f, info.min, info.max).astype(dtype)
            write_record(fh, codec.encode_image_message(i, f, block_bytes))
        write_record(fh, codec.encode_control_message("end"))
    return len(frames)


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python -m tools.npy_to_dstream", description=__doc__.splitlines()[0])
    p.add_argument("npy", type=Path, help="(Ny, Nx, H, W) or (N, H, W) frames")
    p.add_argument("--out", type=Path, required=True, help="The .dstream file to write.")
    p.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16",
                   help="Pixel type of the written frames (rounded and clipped).")
    p.add_argument("--block-bytes", type=int, default=8192,
                   help="bslz4 block size in bytes (written to each frame's header).")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    n = convert(args.npy, args.out, args.dtype, args.block_bytes)
    print(f"[npy_to_dstream] {n} frames → {args.out}")


if __name__ == "__main__":
    main()
