"""Loop a recorded scan into a continuous acquisition: N passes over the same area in one
`.dstream`, as the detector sends them when it is armed for N × positions frames. Use it to try
the continuous modes (`run.py --accumulate latest | all | reset | swap`) and the live view on real
data.

    python -m tools.repeat_dstream scan.dstream scan_x8.dstream --passes 8
    python -m tools.repeat_dstream scan.dstream drift.dstream --passes 6 --scan 256 256 --drift 0 2
    python -m tools.repeat_dstream scan.dstream per_pass.dstream --passes 4 --series per-pass

Every message is copied byte for byte except the image's `image_id`, which is re-encoded in place
(a minimal CBOR unsigned integer, as the DCU writes it); the bslz4 payloads are untouched, so the
file replays through `run.py --source` and through `tools/replay_to_zmq.py` into either ingest.

- `--series one` (default): one start, the passes with ids running on (pass k adds
  k × positions), one end. A single arm for many passes; plays with `--source`.
- `--series per-pass`: start / images / end per pass with ids restarting at 0, as when every
  pass is armed separately; replay it over ZeroMQ to `run.py --continuous`.
- `--drift DY DX`: the specimen moves by (DY, DX) scan pixels per pass (integers, wrapping round
  the field). Pass k places the frame recorded at position q at q + k·(DY, DX) and sends pass k in
  raster order; without it each pass keeps the recorded arrival order.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → `import realtime_stem`

from realtime_stem.ingest.dstream import iter_records, write_magic, write_record  # noqa: E402

_KEY = b"\x68image_id"                      # CBOR text string "image_id"
_TYPE_IMAGE = b"\x65image"                  # "image" (the value of "type")


def _uint_len(head: int) -> int:
    """Bytes taken by a CBOR unsigned integer (major type 0) whose first byte is `head`."""
    if head >> 5 != 0:
        raise ValueError(f"image_id is not an unsigned integer (CBOR head 0x{head:02x})")
    ai = head & 0x1F
    if ai < 24:
        return 1
    return {24: 2, 25: 3, 26: 5, 27: 9}[ai]


def _encode_uint(v: int) -> bytes:
    if v < 24:
        return bytes([v])
    for ai, width in ((24, 1), (25, 2), (26, 4), (27, 8)):
        if v < 1 << (8 * width):
            return bytes([ai]) + v.to_bytes(width, "big")
    raise ValueError(f"image_id {v} too large")


def image_id(raw: bytes) -> int | None:
    """The `image_id` of an image message, or None for any other message."""
    j = raw.find(_KEY)
    if j < 0 or _TYPE_IMAGE not in raw[:j]:
        return None
    n = _uint_len(raw[j + len(_KEY)])
    body = raw[j + len(_KEY):j + len(_KEY) + n]
    return body[0] & 0x1F if n == 1 else int.from_bytes(body[1:], "big")


def with_image_id(raw: bytes, new_id: int) -> bytes:
    """The message with its `image_id` replaced, every other byte unchanged."""
    j = raw.find(_KEY) + len(_KEY)
    n = _uint_len(raw[j])
    return raw[:j] + _encode_uint(int(new_id)) + raw[j + n:]


def repeat(src: Path, dst: Path, passes: int, series: str = "one",
           scan: tuple[int, int] | None = None, drift: tuple[int, int] = (0, 0)) -> dict:
    """Write `passes` passes of the scan in `src` to `dst`; returns a summary."""
    start = end = None
    images: dict[int, bytes] = {}
    order: list[int] = []
    for raw in iter_records(src):
        iid = image_id(raw)
        if iid is not None:
            if iid in images:
                raise ValueError(f"{src}: image_id {iid} twice; give a single scan")
            images[iid] = raw
            order.append(iid)
        elif start is None:
            start = raw
        else:
            end = raw
    if start is None or not images:
        raise ValueError(f"{src}: needs a start message and image messages")
    positions = scan[0] * scan[1] if scan else len(images)
    if max(images) >= positions:
        raise ValueError(f"{src}: image_id {max(images)} is past the {positions} positions "
                         f"(give --scan)")
    if len(images) < positions:
        print(f"[repeat] note: {positions - len(images)} of {positions} positions have no frame",
              flush=True)
    dy, dx = drift
    if (dy or dx) and scan is None:
        raise ValueError("--drift needs --scan NY NX")

    def pass_messages(k: int):
        offset = k * positions if series == "one" else 0
        if not (dy or dx):
            for iid in order:
                yield with_image_id(images[iid], iid + offset)
            return
        ny, nx = scan
        for p in range(positions):                       # raster order
            r, c = divmod(p, nx)
            q = ((r - k * dy) % ny) * nx + (c - k * dx) % nx
            if q in images:
                yield with_image_id(images[q], p + offset)

    t0 = time.perf_counter()
    n_out = 0
    with open(dst, "wb") as fh:
        write_magic(fh)
        for k in range(passes):
            if series == "per-pass" or k == 0:
                write_record(fh, start)
            for msg in pass_messages(k):
                write_record(fh, msg)
                n_out += 1
            if (series == "per-pass" or k == passes - 1) and end is not None:
                write_record(fh, end)
    return {"passes": passes, "positions": positions, "images": n_out, "series": series,
            "drift": drift, "bytes": dst.stat().st_size, "seconds": time.perf_counter() - t0}


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(prog="python -m tools.repeat_dstream",
                                description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("src", type=Path, help="A recorded scan (.dstream), one frame per position.")
    p.add_argument("dst", type=Path, help="The looped .dstream to write.")
    p.add_argument("--passes", type=int, required=True, metavar="N", help="Passes to write.")
    p.add_argument("--series", choices=("one", "per-pass"), default="one",
                   help="one: a single series, ids running on (plays with run.py --source). "
                        "per-pass: a series per pass, ids restarting (replay over ZeroMQ to "
                        "run.py --continuous).")
    p.add_argument("--scan", type=int, nargs=2, default=None, metavar=("NY", "NX"),
                   help="The scan grid (default: as many positions as the recording has frames; "
                        "needed for --drift).")
    p.add_argument("--drift", type=int, nargs=2, default=(0, 0), metavar=("DY", "DX"),
                   help="Specimen drift per pass in scan pixels (integers, wrapping round the "
                        "field); pass k is then sent in raster order.")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    info = repeat(args.src, args.dst, args.passes, args.series,
                  tuple(args.scan) if args.scan else None, tuple(args.drift))
    print(f"[repeat] {info['passes']} passes × {info['positions']} positions → {args.dst} "
          f"({info['images']} images, {info['bytes'] / 1e9:.2f} GB, {info['series']} series"
          + (f", drift {info['drift']} px/pass" if any(info["drift"]) else "")
          + f") in {info['seconds']:.1f}s", flush=True)
    return info


if __name__ == "__main__":
    main()
