"""Time live OBF (live4d.algorithms.obf): the setup, the per-frame deposit and the readout.

    python -m live4d.parallax.scripts.bench_obf          # binned and unbinned, here
    python -m live4d.parallax.scripts.bench_obf --preset binned --batch 512 1024 2048 --device cuda

The presets are the 2026-10-20 Gd2O3 recordings' geometry (200 kV, 30 mrad, a 4.8° rotation,
C12 17 Å): binned 96² frames at 1.066 mrad/px (about 2,500 bright-field pixels) and unbinned 192²
at 0.533 mrad/px (about 9,900). Frames are random; only the timing means anything. Reports, per
device and preset:
  * the setup (the probe, the canvas layout and the Gram), built once per geometry, in s;
  * the sustained deposit rate for batches of `--batch` frames with the setup cached, in kHz, and
    against 1.2 times the detector rate (144 kHz binned, 24 kHz unbinned), the hot-path bar;
  * where one batch's time goes: the detector-side correlations (FFTs and the map onto the patch
    lattice), the patch synthesis (the sub-pixel ramps and the patch's inverse FFT) and the deposit
    (its canvas indices and `index_add_`);
  * one readout (the canvas FFT, the solve and the read onto the output grid), in ms, and how long
    a sweep of the whole scan takes at the measured rate.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from live4d.algorithms.obf.core import ObfSetup, normalize
from live4d.algorithms.parallax.engine import preprocess_geometry

PRESETS = {  # detector, angular sampling (mrad/px), detector rate (kHz)
    "binned": ((96, 96), 1.066, 120.0),
    "unbinned": ((192, 192), 0.533, 20.0),
}


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


def geometry(preset: str, scan, upsampling: int):
    det, angular, _ = PRESETS[preset]
    return preprocess_geometry(
        det, tuple(scan), (1.9766, 1.9766), 200e3, 30.0, angular_sampling=(angular, angular),
        aberration_coefs={"C10": -281.0, "C12": 17.0, "phi12": math.radians(-7.1)},
        rotation_angle=math.radians(4.78), upsampling_factor=upsampling,
        probe_center=(det[0] / 2 - 0.14, det[1] / 2 + 0.41), flip_cols=True)


def _breakdown(setup: ObfSetup, values, pos, device, repeats) -> dict:
    """Where one sub-batch's time goes (the steps of `ObfSetup.deposit`, timed apart)."""
    v, p = values[:setup._batch], pos[:setup._batch]
    q_rows, q_cols = setup._q
    base, fraction = setup._place if setup.placement == "patch" else (None, None)
    buffer = setup.buffer()
    contracted = setup._contract(v)
    t_contract = _timeit(lambda: setup._contract(v), device, repeats)
    if setup.placement != "patch":
        return {"contract": t_contract, "n": v.shape[0]}

    def synth():
        patch = contracted * setup._ramps(q_rows, fraction[p][:, 0])[:, :, None]
        return torch.fft.ifft2(patch * setup._ramps(q_cols, fraction[p][:, 1])[:, None, :])

    patch = synth()
    t_synth = _timeit(synth, device, repeats)

    def place():
        i0, i1 = setup.inner_shape
        rows = (base[p][:, 0:1] + setup._offsets[0][None, :]) % i0
        cols = (base[p][:, 1:2] + setup._offsets[1][None, :]) % i1
        flat = rows[:, :, None] * i1 + cols[:, None, :]
        buffer.index_add_(0, flat.reshape(-1), torch.view_as_real(patch).reshape(-1, 2))

    t_place = _timeit(place, device, repeats)
    return {"contract": t_contract, "synth": t_synth, "place": t_place, "n": v.shape[0]}


def bench(device: str, preset: str, scan, upsampling: int, batches, repeats: int,
          fft_sizes: bool, full_band: bool = False) -> None:
    dev = torch.device(device)
    geo = geometry(preset, scan, upsampling)
    _sync(dev)
    t0 = time.perf_counter()
    setup = ObfSetup.build(geo, device=dev, fft_sizes=fft_sizes,
                           **({"margin": None} if full_band else {}))
    _sync(dev)
    t_setup = time.perf_counter() - t0
    n_pos = scan[0] * scan[1]
    rate_bar = 1.2 * PRESETS[preset][2]
    print(f"{device} {preset}: M = {geo.n_bf}, {setup.placement}, canvas {setup.inner_shape} "
          f"({setup.multiples[0]}x{setup.multiples[1]} fields of view of {setup.per_fov[0]} px), "
          f"patch {setup.grid_shape}, sub-batch {setup._batch}; setup {t_setup:.2f} s, "
          f"{setup.nbytes() / 1e6:,.0f} MB + {setup.buffer().numel() * 4 / 1e6:,.0f} MB per "
          f"numerator", flush=True)
    rng = np.random.default_rng(0)
    mean = torch.full((geo.n_bf,), 1000.0, device=dev)
    for batch in batches:
        raw = torch.as_tensor(rng.poisson(1000.0, size=(batch, geo.n_bf)).astype(np.float32))
        values = normalize(raw.to(dev), mean)
        ids = [torch.arange(i, i + batch, device=dev) % n_pos for i in range(0, 8 * batch, batch)]
        buffer = setup.buffer()
        k = [0]

        def step():
            setup.deposit(buffer, values, ids[k[0] % len(ids)])
            k[0] += 1

        t = _timeit(step, dev, repeats)
        khz = batch / t / 1e3
        verdict = "above" if khz >= rate_bar else "below"
        print(f"  batch {batch:>5}: {khz:8.2f} kHz ({1e6 * t / batch:6.1f} us/frame), {verdict} the "
              f"hot-path bar of {rate_bar:g} kHz; a sweep of the {scan[0]}x{scan[1]} scan "
              f"{n_pos / khz / 1e3:.1f} s", flush=True)
    parts = _breakdown(setup, values, ids[0], dev, repeats)
    total = sum(v for key, v in parts.items() if key != "n")
    print("  one sub-batch of " + str(parts["n"]) + ": " + ", ".join(
        f"{key} {100 * v / total:.0f}%" for key, v in parts.items() if key != "n")
        + f" ({1e6 * total / parts['n']:.1f} us/frame)", flush=True)
    t_image = _timeit(lambda: setup.image(buffer), dev, max(3, repeats // 4))
    print(f"  readout: {1e3 * t_image:.1f} ms", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/parallax/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python -m live4d.parallax.scripts.bench_obf", description=__doc__.splitlines()[0])
    p.add_argument("--device", nargs="+", default=None,
                   help="Devices to time (default: cuda or mps where available, else cpu).")
    p.add_argument("--preset", nargs="+", choices=list(PRESETS), default=list(PRESETS),
                   help="Detector modes of the Gd2O3 recordings to time.")
    p.add_argument("--scan", type=int, nargs=2, default=(256, 256), metavar=("NY", "NX"),
                   help="Scan grid (sets the canvas, and so the Gram and the readout).")
    p.add_argument("--upsampling", type=int, default=2, help="Output-grid upsampling.")
    p.add_argument("--batch", type=int, nargs="+", default=[512, 1024, 2048],
                   help="Frames per deposit call.")
    p.add_argument("--repeats", type=int, default=10, help="Timed batches per measurement.")
    p.add_argument("--fft-sizes", action="store_true",
                   help="Round the canvas's pixels per field of view up to 2^a 3^b 5^c sizes.")
    p.add_argument("--full-band", action="store_true",
                   help="Patches holding OBF's whole band (quantem's patch_margin=None) instead "
                        "of the output grid's band (the default, 2-4x faster per frame).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    devices = args.device or (["cuda"] if torch.cuda.is_available() else
                              ["mps"] if torch.backends.mps.is_available() else ["cpu"])
    for dev in devices:
        for preset in args.preset:
            bench(dev, preset, args.scan, args.upsampling, args.batch, args.repeats,
                  args.fft_sizes, args.full_band)


if __name__ == "__main__":
    main()
