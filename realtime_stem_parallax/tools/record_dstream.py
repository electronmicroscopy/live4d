"""Record a live DECTRIS stream (or the replay feeder) verbatim into a `.dstream` capture.

For commissioning: record one acquisition from the DCU, then work out orientation, center and
defocus offline (`run.py --source`, `tools/simulate_4dstem`-style checks) and replay it through
either ingest as often as needed. Records `start`, the images and `end` exactly as received.

    python -m tools.record_dstream --endpoint tcp://<DCU_IP>:31001 --n-connections 3 \\
        --out captures/session1.dstream [--max-frames 65536]

The DCU's PUSH socket load-balances across every connected PULL peer: while recording, nothing
else may listen to the stream (stop run.py first).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → `import realtime_stem`

from realtime_stem.ingest.dstream import write_magic, write_record  # noqa: E402
from realtime_stem.ingest.python_ingest import ZmqStream  # noqa: E402


def _type(raw: bytes):
    import cbor2

    msg = cbor2.loads(raw)
    return msg.get("type") if hasattr(msg, "get") else None


def record(endpoint: str, out: Path, n_connections: int = 1, max_frames: int | None = None,
           end_drain_s: float = 0.5, idle_timeout_s: float | None = None) -> int:
    """Write messages until the series `end` (plus a drain of the other connections),
    `max_frames` images, or `idle_timeout_s` without a message. Returns the images written."""
    stream = ZmqStream(endpoint, n_connections)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_images, ended, last = 0, False, time.perf_counter()
    try:
        with open(out, "wb") as fh:
            write_magic(fh)
            while True:
                messages = stream.receive(200)
                now = time.perf_counter()
                if not messages:
                    if ended and now - last >= end_drain_s:
                        break
                    if idle_timeout_s is not None and now - last >= idle_timeout_s:
                        break
                    continue
                last = now
                for raw in messages:
                    write_record(fh, raw)
                    kind = _type(raw)
                    n_images += kind == "image"
                    ended |= kind == "end"
                if max_frames is not None and n_images >= max_frames:
                    break
    finally:
        stream.close()
    return n_images


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python -m tools.record_dstream", description=__doc__.splitlines()[0])
    p.add_argument("--endpoint", required=True,
                   help="ZMQ endpoint to PULL from (DCU: tcp://<dcu-ip>:31001).")
    p.add_argument("--out", type=Path, required=True, help="The .dstream file to write.")
    p.add_argument("--n-connections", type=int, default=1,
                   help="TCP connections to the stream (the DCU at full rate needs 3).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many images (default: at the series' end).")
    p.add_argument("--idle-timeout", type=float, default=None, metavar="S",
                   help="Stop after this many seconds without a message (default: wait for `end`).")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    n = record(args.endpoint, args.out, args.n_connections, args.max_frames,
               idle_timeout_s=args.idle_timeout)
    print(f"[record] {n} images → {args.out}")


if __name__ == "__main__":
    main()
