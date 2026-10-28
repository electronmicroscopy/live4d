"""Replay a captured `.dstream` over a ZMQ PUSH socket, in capture order.

Feeds the ingest graph the *identical* raw ARINA messages the capture recorded — so you can run
and demonstrate the full pipeline with no detector attached. Sends `start`, then up to
`max_frames` image records, then `end`. Standalone (host-side; only needs pyzmq + cbor2).

    python -m tools.replay_to_zmq examples/data/sample.dstream --endpoint tcp://127.0.0.1:5555
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → `import tools.*`

from tools.dstream import envelope_type, iter_records   # noqa: E402


def replay_dstream(path: str, endpoint: str, max_frames: int | None = None,
                   linger_s: float = 2.0, fast: bool = False, rate: float = 0.0) -> int:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.bind(endpoint)
    time.sleep(0.2)  # let a PULL peer connect before the first send

    if fast:
        # Max-rate feed for pipeline throughput measurement: no CBOR peek per record
        # (cbor2.loads caps a Python loop near ~30k msg/s — far below a 120 kHz consumer).
        # Assumes a well-formed capture: 'start' first, 'end' last. --max-frames unsupported.
        if max_frames is not None:
            raise ValueError("--fast replays the whole file (no per-record parse).")
        sent = 0
        t0 = time.perf_counter()
        for raw in iter_records(path):
            sock.send(raw)  # blocks on HWM when the consumer lags = natural backpressure
            sent += 1
            if sent % 100_000 == 0:
                rate = sent / (time.perf_counter() - t0)
                print(f"[replay] {sent:,} msgs sent  {rate:,.0f}/s", flush=True)
        rate = sent / (time.perf_counter() - t0)
        print(f"[replay] done: {sent:,} msgs at {rate:,.0f}/s (send side)", flush=True)
        time.sleep(linger_s)
        sock.close(linger=int(linger_s * 1000))
        return max(sent - 2, 0)

    sent_images = 0
    end_raw = None
    ends_sent = 0
    t0 = time.perf_counter()
    for raw in iter_records(path):
        t = envelope_type(raw)
        if t == "start":
            sock.send(raw)
        elif t == "image":
            if max_frames is not None and sent_images >= max_frames:
                continue
            sock.send(raw)
            sent_images += 1
            if rate > 0 and sent_images % 64 == 0:   # pace like a scan, in small bursts
                ahead = sent_images / rate - (time.perf_counter() - t0)
                if ahead > 0:
                    time.sleep(ahead)
        elif t == "end":
            if max_frames is None:
                sock.send(raw)        # in order: a capture of several series replays as several
                ends_sent += 1
            else:
                end_raw = raw         # send after the loop so a capped run stays well-formed
    if ends_sent == 0:
        sock.send(end_raw if end_raw is not None else b"")
    time.sleep(linger_s)  # let the consumer drain before we close
    sock.close(linger=int(linger_s * 1000))
    return sent_images


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python -m tools.replay_to_zmq", description="Replay a .dstream over ZMQ PUSH.")
    p.add_argument("dstream", help="The .dstream capture to replay.")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5555",
                   help="ZMQ PUSH endpoint to bind (run.py --endpoint connects to it).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Send only the first N images (then the end message).")
    p.add_argument("--rate", type=float, default=0.0, metavar="HZ",
                   help="Send at most this many frames/s (0 = as fast as the consumer takes them), "
                        "so a live view fills in like a scan.")
    p.add_argument("--fast", action="store_true",
                   help="No per-record CBOR peek: a max-rate feed for throughput measurement "
                        "(whole file, assumes start first and end last).")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if args.fast and args.rate:
        p.error("--fast replays unpaced; drop --rate")
    n = replay_dstream(args.dstream, args.endpoint, args.max_frames, fast=args.fast, rate=args.rate)
    print(f"[replay] sent {n} image frames from {args.dstream}")


if __name__ == "__main__":
    main()
