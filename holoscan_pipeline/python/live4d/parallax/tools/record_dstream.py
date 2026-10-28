"""Record a live DECTRIS stream (or the replay feeder) verbatim into a `.dstream` capture.

For commissioning: record one acquisition from the DCU, then work out orientation, center and
defocus offline (`run.py --source`, `tools/simulate_4dstem`-style checks) and replay it through
either ingest as often as needed. Records `start`, the images and `end` exactly as received.

    python -m live4d.parallax.tools.record_dstream --endpoint tcp://<DCU_IP>:31001 \\
        --out captures/session1.dstream [--max-frames 65536]

The DCU's PUSH socket load-balances across every connected PULL peer: while recording, nothing
else may listen to the stream (stop run.py first).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


from live4d.parallax.ingest.dstream import write_magic, write_record  # noqa: E402
from live4d.parallax.ingest.python_ingest import ZmqStream  # noqa: E402


def _type(raw: bytes):
    import cbor2

    msg = cbor2.loads(raw)
    return msg.get("type") if hasattr(msg, "get") else None


def record(endpoint: str, out: Path, max_frames: int | None = None,
           idle_timeout_s: float | None = None) -> int:
    """Record one ordered series through END, with bounded reception."""
    stream = ZmqStream(endpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_images, last = 0, time.perf_counter()
    try:
        with out.open("wb") as fh:
            write_magic(fh)
            while True:
                messages = stream.receive(200)
                now = time.perf_counter()
                if not messages:
                    if idle_timeout_s is not None and now - last >= idle_timeout_s:
                        return n_images
                    continue
                last = now
                for raw in messages:
                    write_record(fh, raw)
                    kind = _type(raw)
                    n_images += kind == "image"
                    if kind == "end" or (max_frames is not None and n_images >= max_frames):
                        return n_images
    finally:
        stream.close()


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python -m live4d.parallax.tools.record_dstream", description=__doc__.splitlines()[0])
    p.add_argument("--endpoint", required=True,
                   help="ZMQ endpoint to PULL from (DCU: tcp://<dcu-ip>:31001).")
    p.add_argument("--out", type=Path, required=True, help="The .dstream file to write.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many images (default: at the series' end).")
    p.add_argument("--idle-timeout", type=float, default=None, metavar="S",
                   help="Stop after this many seconds without a message (default: wait for `end`).")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    n = record(args.endpoint, args.out, args.max_frames,
               idle_timeout_s=args.idle_timeout)
    print(f"[record] {n} images → {args.out}")


if __name__ == "__main__":
    main()
