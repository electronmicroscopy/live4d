"""ParallaxAlgorithm — wire parallax / tcBF into the generic runner.

This is the whole "how to add a method" story in one file: declare the CLI flags that shape the
geometry, build the accumulator from them, and save the result. The runner (`apps/run.py`) and the
ingest graph never mention parallax — they see only the `Algorithm` / `Accumulator` contracts.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from realtime_stem.processing.base import Accumulator, Algorithm
from realtime_stem.processing.parallax.config import (
    add_readout_args,
    add_tuning_args,
    config_from_args,
    join_negative_floats,
    missing_acquisition_args,
    readout_from_args,
)
from realtime_stem.processing.parallax.engine import (
    ParallaxAccumulator,
    preprocess_geometry,
)


class ParallaxAlgorithm(Algorithm):
    name = "parallax"

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        # Geometry presets first, then the shared physics/tuning knobs.
        g = parser.add_argument_group("parallax geometry and target aberrations")
        g.add_argument(
            "--detector", type=int, nargs=2, default=None, metavar=("HY", "HX"),
            help="Detector geometry (default: config, 192x192; --binned preset: 96x96). Changing "
                 "the detector changes the BF disk — set --angular-sampling to match the mode.")
        g.add_argument(
            "--binned", action="store_true",
            help="ARINA 2×2-binned preset: 96×96 detector (give --angular-sampling of the "
                 "binned mode).")
        add_tuning_args(g)        # --defocus/--rotation/--upsampling/physics/--scan/--center
        f = parser.add_argument_group("parallax frame processing (per frame, as frames arrive)")
        f.add_argument(
            "--subtract-frame-mean", action="store_true",
            help="Subtract each frame's mean over its bright-field pixels before it is added in "
                 "(as the reference implementation does). This removes the amplitude contrast "
                 "(mass-thickness, diffraction) over regions larger than about the probe, and the "
                 "frame-to-frame changes in beam intensity. Default: off, the mean is kept.")
        add_readout_args(parser)  # --no-phase-flip/--no-dc-correct/--normalize/filters

    def preprocess_argv(self, argv: list[str]) -> list[str]:
        # `--defocus -1.5e4` etc.: scientific-notation negatives read as flags otherwise.
        return join_negative_floats(argv)

    def check_args(self, args: argparse.Namespace) -> list[str]:
        return missing_acquisition_args(args)

    @staticmethod
    def _geometry(args: argparse.Namespace):
        cfg = config_from_args(args)   # incl. --detector / --binned / --detector-layout
        return cfg, preprocess_geometry(
            det_shape=cfg.detector_shape, scan_shape=cfg.scan_shape,
            scan_sampling=cfg.scan_sampling, energy=cfg.energy,
            semiangle_cutoff=cfg.semiangle_cutoff, angular_sampling=cfg.angular_sampling,
            aberration_coefs=cfg.aberration_coefs, rotation_angle=cfg.rotation_angle,
            upsampling_factor=cfg.upsampling_factor, probe_center=cfg.probe_center,
            flip_cols=cfg.flip_cols,
        )

    def build(self, args: argparse.Namespace) -> Accumulator:
        cfg, geo = self._geometry(args)

        rot_deg = 0.0 if cfg.rotation_angle is None else float(np.rad2deg(cfg.rotation_angle))
        center_str = ("frame center" if cfg.probe_center is None
                      else f"({cfg.probe_center[0]:.4g}, {cfg.probe_center[1]:.4g})")
        print(f"[parallax] BF pixels = {geo.n_bf}, "
              f"C10 = {cfg.aberration_coefs.get('C10', 0.0):.4g} Å, rotation = {rot_deg:.4g}°, "
              f"max |shift| = {int(np.abs(geo.shifts).max())} px, upsampling = {cfg.upsampling_factor}, "
              f"angular sampling = {cfg.angular_sampling[0]:.4g} mrad/px, disk center = {center_str}"
              f"{', columns flipped' if cfg.flip_cols else ''}, frame mean "
              f"{'subtracted' if getattr(args, 'subtract_frame_mean', False) else 'kept'}",
              flush=True)
        radius = cfg.semiangle_cutoff / cfg.angular_sampling[0]
        print(f"[parallax] BF disk radius = {radius:.3g} px (semiangle {cfg.semiangle_cutoff:g} mrad"
              f" / {cfg.angular_sampling[0]:g} mrad/px): compare with the disk on a frame",
              flush=True)
        if geo.n_clipped:
            print(f"[parallax] WARNING: {geo.n_clipped} of {geo.n_clipped + geo.n_bf} BF-disk "
                  f"pixels fall outside the {cfg.detector_shape[0]}x{cfg.detector_shape[1]} frame "
                  f"and are left out. Check --center, --semiangle and --angular-sampling.",
                  flush=True)

        mode = getattr(args, "accumulate", "scan")
        acc = ParallaxAccumulator(geo, device=getattr(args, "device", None),
                                  readout=readout_from_args(args),
                                  accumulate="all" if mode == "scan" else mode,
                                  wrap_ids=mode != "scan",
                                  subtract_frame_mean=getattr(args, "subtract_frame_mean", False),
                                  track_passes=bool(getattr(args, "save_passes", False)),
                                  pass_average=getattr(args, "pass_average", 1.0))
        if mode == "latest":
            itemsize = 4 if getattr(args, "dtype", "uint16") == "uint32" else 2
            print(f"[parallax] accumulate=latest: keeping the newest frame per scan position "
                  f"(~{acc.store_bytes(itemsize) / 1e6:,.0f} MB on {acc.device}; the Python "
                  f"ingest stores 32-bit)", flush=True)
        elif mode in ("reset", "swap"):
            avg = getattr(args, "pass_average", 1.0)
            print(f"[parallax] accumulate={mode}: a fresh image per pass, the last completed pass "
                  f"kept" + (f", blended with weight {avg:g}" if mode == "swap" and avg < 1 else "")
                  + f" (~{2 * acc.out.numel() * 4 / 1e6:,.0f} MB on {acc.device})", flush=True)
        return acc

    # ── live tuning: C10, the rotation, and C12 as Cartesian components (like stigmators) ──

    @staticmethod
    def _c12_ab(args) -> tuple[float, float]:
        c12, phi = float(args.c12 or 0.0), math.radians(float(args.phi12 or 0.0))
        return c12 * math.cos(2 * phi), c12 * math.sin(2 * phi)

    def live_params(self, args: argparse.Namespace) -> list[dict]:
        c10 = float(args.defocus)
        a, b = self._c12_ab(args)
        c12_span = max(100.0, 3 * math.hypot(a, b))
        return [
            {"name": "c10", "label": "C10", "unit": "Å", "value": c10, "step": 1,
             "span": max(300.0, 0.5 * abs(c10))},
            {"name": "rotation", "label": "rotation", "unit": "°", "value": float(args.rotation),
             "step": 0.05, "span": 10.0},
            {"name": "c12a", "label": "C12 a", "unit": "Å", "value": round(a, 3), "step": 1,
             "span": c12_span},
            {"name": "c12b", "label": "C12 b", "unit": "Å", "value": round(b, 3), "step": 1,
             "span": c12_span},
        ]

    def retune(self, acc: Accumulator, args: argparse.Namespace, values: dict) -> str:
        """C10 → --defocus, rotation → --rotation, (C12 a, C12 b) = C12 (cos 2φ12, sin 2φ12) →
        --c12 / --phi12; then re-shift the running accumulator (`set_geometry`)."""
        unknown = set(values) - {"c10", "rotation", "c12a", "c12b"}
        if unknown:
            raise ValueError(f"not tunable: {', '.join(sorted(unknown))}")
        if "c10" in values:
            args.defocus = float(values["c10"])
        if "rotation" in values:
            args.rotation = float(values["rotation"])
        if "c12a" in values or "c12b" in values:
            a0, b0 = self._c12_ab(args)
            a, b = float(values.get("c12a", a0)), float(values.get("c12b", b0))
            args.c12 = math.hypot(a, b)
            args.phi12 = math.degrees(math.atan2(b, a)) / 2
        _, geo = self._geometry(args)
        return acc.set_geometry(geo) if acc is not None else "from the first frame"

    def tuned_flags(self, args: argparse.Namespace) -> list[str]:
        flags = [f"--defocus={float(args.defocus):.1f}", f"--rotation={float(args.rotation):.2f}"]
        if args.c12:
            flags += [f"--c12={float(args.c12):.1f}", f"--phi12={float(args.phi12 or 0.0):.1f}"]
        return flags

    # readout channel → saved file (the raw sum keeps its historical name)
    _FILES = {"raw": "parallax_recon.npy", "parallax": "parallax_corrected.npy",
              "phase_flipped": "parallax_phase_flipped.npy", "coverage": "parallax_coverage.npy"}

    def save(self, acc: Accumulator, out_dir: Path, tag: str = "") -> None:
        """Save every readout channel; with a `tag` (continuous mode) also a numbered copy of each,
        so a session keeps all its scans while the plain names always hold the latest one."""
        out_dir.mkdir(parents=True, exist_ok=True)
        images = acc.images() if hasattr(acc, "images") else {"raw": acc.result()}
        names = []
        for channel, image in images.items():
            name = self._FILES.get(channel, f"parallax_{channel}.npy")
            np.save(out_dir / name, image)
            names.append(name)
            if tag:
                tagged = name.replace(".npy", f"_{tag}.npy")
                np.save(out_dir / tagged, image)
                names.append(tagged)
        np.save(out_dir / "parallax_shifts.npy", acc.geometry.shifts)   # parallax-specific extra
        print(f"[parallax] saved {' + '.join(names)} + parallax_shifts.npy to {out_dir}",
              flush=True)

    def save_pass(self, acc: Accumulator, out_dir: Path, k: int, dtype: str = "float32",
                  executor=None) -> None:
        """The readout channels of completed pass `k` → `<out_dir>/passes/parallax_<channel>_
        passNNNN.npy` (`parallax_corrected_…`, `parallax_phase_flipped_…`): a movie of the
        session, one frame per pass. The arrays are copied to the host here; the files are
        written by `executor` when given, off the hot path."""
        images = {k_: v.astype(dtype) for k_, v in acc.images().items() if k_ != "raw"}
        folder = out_dir / "passes"

        def write():
            folder.mkdir(parents=True, exist_ok=True)
            for channel, image in images.items():
                name = self._FILES.get(channel, f"parallax_{channel}.npy")
                np.save(folder / name.replace(".npy", f"_pass{k:04d}.npy"), image)
            print(f"[parallax] pass {k} saved to {folder}", flush=True)

        if executor is None:
            write()
        else:
            executor.submit(write)
