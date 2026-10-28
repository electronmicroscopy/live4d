"""Time the parallax hot path and the live readout on the available torch devices.

    python scripts/bench_accumulator.py                 # cpu + mps/cuda if present
    python scripts/bench_accumulator.py --scan 1024 1024 --batch 2048 --device cuda

Reports, per device and bright-field size M:
  * `process()` on device-resident uint16 batches (the Holoscan hand-off), in kHz of frames, for
    `--accumulate all` (one scan, or a sum) and `latest` (continuous scanning: each position's
    newest frame replaces its previous one — an extra gather from the per-position store);
  * the share of it spent on the readout side accumulators (the per-BF running sum and the visited
    mask, measured in isolation on the same shapes) — the hot-path cost of the live channels;
  * one `snapshot()` (DC correction + phase flip FFTs + the device→host copy), in ms.

Relative numbers are what transfer between machines; the absolute rate on the GPU machine must be
measured there (and end to end, ingest included).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


from live4d.algorithms.parallax.engine import ParallaxAccumulator, preprocess_geometry  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _timeit(fn, device, repeats):
    fn()                                            # warm-up (kernel compilation, caches)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / repeats


def bench(device: str, scan, batch: int, semiangle: float, repeats: int,
          accumulate: str = "all") -> dict:
    dev = torch.device(device)
    geo = preprocess_geometry((96, 96), tuple(scan), (2.0, 2.0), 300e3, semiangle,
                              angular_sampling=(0.5, 0.5), aberration_coefs={"C10": -2000.0},
                              rotation_angle=np.deg2rad(-15))
    acc = ParallaxAccumulator(geo, device=device, accumulate=accumulate,
                              wrap_ids=accumulate != "all")
    rng = np.random.default_rng(0)
    frames = torch.from_numpy(rng.integers(0, 20, size=(batch, 96, 96), dtype=np.uint16))
    frames = frames.to(dev) if dev.type == "cuda" else frames.to(torch.int32).to(dev)
    n_pos = scan[0] * scan[1]
    # consecutive raster batches; for "latest" every position has already been seen once, so
    # each batch replaces stored frames (the steady state of continuous scanning)
    ids = [torch.arange(i, i + batch, device=dev) % n_pos for i in range(0, 8 * batch, batch)]
    if accumulate == "latest":
        for s in range(0, n_pos, batch):
            acc.process(frames[: min(batch, n_pos - s)], torch.arange(s, min(s + batch, n_pos),
                                                                     device=dev))
    k = [0]

    def step():
        if accumulate in ("reset", "swap"):           # ids run on, so passes complete in the loop
            acc.process(frames, torch.arange(k[0] * batch, (k[0] + 1) * batch, device=dev))
        else:
            acc.process(frames, ids[k[0] % len(ids)])
        k[0] += 1

    t_process = _timeit(step, dev, repeats)

    i_bf = torch.randn(batch, geo.n_bf, device=dev)
    visited = torch.zeros(n_pos, device=dev)
    bf_sum = torch.zeros(geo.n_bf, device=dev)

    def extras():
        bf_sum.add_(i_bf.sum(dim=0))
        visited[ids[0]] = 1.0

    t_extras = _timeit(extras, dev, repeats)
    t_snapshot = _timeit(acc.snapshot, dev, max(3, repeats // 10))
    return {"device": device, "mode": accumulate, "M": geo.n_bf, "kHz": batch / t_process / 1e3,
            "batch_ms": 1e3 * t_process, "extras_pct": 100 * t_extras / t_process,
            "snapshot_ms": 1e3 * t_snapshot}


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python scripts/bench_accumulator.py", description=__doc__.splitlines()[0])
    p.add_argument("--device", nargs="+", default=None,
                   help="Devices to time (default: cpu plus cuda / mps where available).")
    p.add_argument("--scan", type=int, nargs=2, default=(1024, 1024), metavar=("NY", "NX"),
                   help="Scan grid (sets the output buffer and the latest-mode store).")
    p.add_argument("--batch", type=int, default=2048, help="Frames per process() call.")
    p.add_argument("--semiangle", type=float, nargs="+", default=[2.0, 10.0],
                   help="mrad at 0.5 mrad/px: 2 → M≈50 (the binned default), 10 → M≈1250")
    p.add_argument("--repeats", type=int, default=50, help="Timed batches per measurement.")
    p.add_argument("--accumulate", nargs="+", default=["all", "latest"],
                   choices=("all", "latest", "reset", "swap"),
                   help="Accumulation modes to time (run.py --accumulate).")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    devices = args.device or (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
                              + (["mps"] if torch.backends.mps.is_available() else []))
    print(f"scan {args.scan[0]}x{args.scan[1]}, batch {args.batch}, 96x96 uint16 frames")
    print(f"{'device':>6} {'mode':>6} {'M':>5} {'process kHz':>12} {'ms/batch':>9} "
          f"{'side acc.':>10} {'snapshot ms':>12}")
    for dev in devices:
        for sa in args.semiangle:
            for mode in args.accumulate:
                r = bench(dev, args.scan, args.batch, sa, args.repeats, mode)
                print(f"{r['device']:>6} {r['mode']:>6} {r['M']:>5} {r['kHz']:>12,.0f} "
                      f"{r['batch_ms']:>9.2f} {r['extras_pct']:>9.1f}% {r['snapshot_ms']:>12.1f}")


if __name__ == "__main__":
    main()
