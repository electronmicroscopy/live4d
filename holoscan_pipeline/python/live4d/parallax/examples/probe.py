"""Headless probe: subscribe to a live endpoint, report frames, optionally save the last one.

No GUI — useful for scripts and CI: confirm frames arrive mid-run and grab the final frame.

    python examples/probe.py tcp://127.0.0.1:5601 --seconds 15 --save /tmp/final.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


from live4d.parallax.streaming.live_stream import LiveSubscriber   # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python examples/probe.py", description=__doc__)
    p.add_argument("endpoint", help="tcp://host:port of the run.py --live-endpoint PUB.")
    p.add_argument("--seconds", type=float, default=15.0, help="How long to listen.")
    p.add_argument("--save", default=None, help="Path to save the last frame received (.npy).")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    sub = LiveSubscriber(args.endpoint)
    n = 0
    last = None
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        msg = sub.recv_latest(timeout_ms=200)
        if msg is None:
            continue
        arr, header = msg
        n += 1
        last = arr
        print(f"[probe] frame {n}: seq={header['seq']} shape={tuple(arr.shape)} "
              f"channels={header.get('channels')} final={header['final']} "
              f"meta={header.get('meta')}", flush=True)
        if header["final"]:
            print("[probe] saw final=True", flush=True)
    print(f"[probe] received {n} frames total", flush=True)
    if args.save is not None and last is not None:
        np.save(args.save, last)
        print(f"[probe] saved last frame to {args.save}", flush=True)
    sub.close()


if __name__ == "__main__":
    main()
