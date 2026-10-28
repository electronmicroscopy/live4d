"""Emulate turning the focus knob during a continuous acquisition, on a recorded scan: pass k of
the output is the recording as it would have been taken at the microscope defocus C10_k.

    python -m tools.refocus_dstream scan.dstream walk.dstream --flags scan.flags \
        --defocus-schedule -500 -450 -400 -350 -300 -316 -316 -316

Under the parallax model a change of defocus ΔC10 only moves each bright-field pixel's virtual
bright-field image, by ΔC10 · α along the pixel's tilt α (rotated into the scan frame, with the
detector's column flip) — the geometry `run.py` reconstructs with. So in pass k every
bright-field pixel's image is moved by round((C10_k − C10) · R(θ) · diag(1, ±1) · α / step) scan
pixels, wrapping round the field as the reconstruction does, where C10, θ, the flip, the disk
centre and the samplings are the recording's own (`--flags`: its `run.py` flags, e.g. from
`tools/calibrate_parallax.py`). Reconstruct with those same flags and the passes sharpen as the
schedule reaches the recording's C10; passes at that C10 are the recording itself.

Approximations: whole-scan-pixel moves (the counts stay integers, as the reconstruction itself
deposits on whole upsampled pixels), the contrast transfer sin χ(q) common to all pixels is left
as recorded, and so are the dark-field pixels. The frames are re-encoded (bslz4, minimal Stream2
messages) and sent in raster order, as one series (ids running on) or one series per pass.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → `import realtime_stem`

from realtime_stem.ingest import codec  # noqa: E402
from realtime_stem.ingest.dstream import iter_records, write_magic, write_record  # noqa: E402
from realtime_stem.processing.parallax.algorithm import ParallaxAlgorithm  # noqa: E402
from realtime_stem.processing.parallax.config import config_from_args, join_negative_floats  # noqa: E402


def geometry_from_flags(flags: list[str]):
    """The recording's configuration (`ParallaxConfig`) from its `run.py` flags."""
    algo = ParallaxAlgorithm()
    p = argparse.ArgumentParser()
    algo.add_cli_args(p)
    return config_from_args(p.parse_args(algo.preprocess_argv(flags)))


def bright_field_moves(cfg, margin_px: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """(pixels, moves): the flat indices of the detector pixels within the bright-field disk
    (+ `margin_px`), and for each the scan-pixel move (dy, dx) per Å of defocus change."""
    h, w = cfg.detector_shape
    if cfg.probe_center is not None:
        c0, c1 = cfg.probe_center
    else:                                   # the engine's default centre (preprocess_geometry)
        c0, c1 = h // 2, (w - 1 - w // 2) if cfg.flip_cols else w // 2
    rows, cols = np.indices((h, w))
    u, v = rows - c0, cols - c1
    radius = cfg.semiangle_cutoff / cfg.angular_sampling[0]
    inside = np.hypot(u, v) <= radius + margin_px
    kappa = np.stack([u[inside] * cfg.angular_sampling[0], v[inside] * cfg.angular_sampling[1]],
                     axis=1) * 1e-3                                        # rad
    t = cfg.rotation_angle or 0.0
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    m = rot @ np.diag([1.0, -1.0 if cfg.flip_cols else 1.0])               # engine shift model
    moves = kappa @ m.T / np.asarray(cfg.scan_sampling)                    # scan px per Å of C10
    return np.flatnonzero(inside), moves


def refocus(src: Path, dst: Path, flags: list[str], schedule: list[float], series: str = "one",
            log=print) -> dict:
    cfg = geometry_from_flags(flags)
    ny, nx = cfg.scan_shape
    n_pos = ny * nx
    c10 = float(cfg.aberration_coefs.get("C10", 0.0))
    start = end = None
    data = None
    for raw in iter_records(src):
        mtype, iid, frame = codec.decode_message(raw)
        if mtype == "image":
            if data is None:
                if tuple(frame.shape) != tuple(cfg.detector_shape):
                    raise ValueError(f"frames are {frame.shape}, the flags say "
                                     f"{cfg.detector_shape}")
                data = np.zeros((n_pos, *frame.shape), frame.dtype)
            if iid < n_pos:
                data[iid] = frame
        elif mtype == "start" and start is None:
            start = raw
        elif mtype == "end":
            end = raw
    if data is None:
        raise ValueError(f"{src}: no image messages")
    flat = data.reshape(n_pos, -1)
    pixels, moves = bright_field_moves(cfg)
    bf = flat[:, pixels].reshape(ny, nx, len(pixels))
    t0 = time.perf_counter()
    with open(dst, "wb") as fh:
        write_magic(fh)
        for k, target in enumerate(schedule):
            shift = np.round((target - c10) * moves).astype(int)          # (P, 2) scan px
            moved = bf
            if np.any(shift):
                moved = np.empty_like(bf)
                keys, inverse = np.unique(shift, axis=0, return_inverse=True)
                for j, (dy, dx) in enumerate(keys):
                    sel = np.flatnonzero(inverse.ravel() == j)
                    # data at C10 + Δ: vBF'(r) = vBF(r + Δs), i.e. rolled by −Δs
                    moved[:, :, sel] = np.roll(bf[:, :, sel], (-dy, -dx), axis=(0, 1))
            moved = moved.reshape(n_pos, -1)
            if series == "per-pass" or k == 0:
                write_record(fh, start if start is not None else codec.encode_control_message("start"))
            offset = k * n_pos if series == "one" else 0
            frame = np.empty_like(flat[0])
            for i in range(n_pos):
                frame[:] = flat[i]
                frame[pixels] = moved[i]
                write_record(fh, codec.encode_image_message(offset + i, frame.reshape(data.shape[1:])))
            if series == "per-pass" or k == len(schedule) - 1:
                write_record(fh, end if end is not None else codec.encode_control_message("end"))
            log(f"[refocus] pass {k}: C10 {target:g} Å (Δ {target - c10:+g} Å, moves up to "
                f"{np.abs(shift).max()} scan px)", flush=True)
    return {"passes": len(schedule), "positions": n_pos, "recording_c10": c10,
            "bytes": dst.stat().st_size, "seconds": time.perf_counter() - t0}


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(prog="python -m tools.refocus_dstream",
                                description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("src", type=Path, help="A recorded scan (.dstream), one frame per position.")
    p.add_argument("dst", type=Path, help="The .dstream to write, one pass per schedule entry.")
    p.add_argument("--flags", type=Path, required=True, metavar="FILE",
                   help="The recording's run.py flags (geometry and its own --defocus), e.g. "
                        "from tools/calibrate_parallax.py.")
    p.add_argument("--defocus-schedule", type=float, nargs="+", required=True, metavar="C10",
                   help="The microscope's C10 (Å) on each pass. Plain numbers (-500, not -5e2).")
    p.add_argument("--series", choices=("one", "per-pass"), default="one",
                   help="one: a single series, ids running on (plays with run.py --source). "
                        "per-pass: a series per pass (replay to run.py --continuous).")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(join_negative_floats(sys.argv[1:] if argv is None else argv))
    info = refocus(args.src, args.dst, args.flags.read_text().split(), args.defocus_schedule,
                   args.series)
    print(f"[refocus] {info['passes']} passes × {info['positions']} positions → {args.dst} "
          f"({info['bytes'] / 1e9:.2f} GB, recording at C10 {info['recording_c10']:g} Å) in "
          f"{info['seconds']:.0f}s", flush=True)
    return info


if __name__ == "__main__":
    main()
