"""Compare the continuous modes (`--accumulate latest | all | reset | swap`) on one looped capture:
what each live view shows part-way through each pass, against the true reconstruction of that
pass and of the one before.

    python -m tools.repeat_dstream scan.dstream loop.dstream --passes 4 --scan 256 256 --drift 3 4
    python scripts/compare_accumulate.py loop.dstream $(cat scan.flags) --upsampling 2 --out cmp.png

Every mode sees the same batches (decoded once). For each pass and fraction the script records
the live image of each mode (`phase_flipped` by default) and its correlation with the pass being
scanned ("now") and with the previous pass ("before"), and prints a table; with --out it saves a
montage: one row per mode, one column per (pass, fraction).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_stem.ingest.codec import decode_message  # noqa: E402
from realtime_stem.ingest.dstream import iter_records  # noqa: E402
from realtime_stem.processing.parallax.algorithm import ParallaxAlgorithm  # noqa: E402
from realtime_stem.processing.parallax.config import config_from_args  # noqa: E402
from realtime_stem.processing.parallax.engine import (  # noqa: E402
    ParallaxAccumulator,
    preprocess_geometry,
)

MODES = {"latest": dict(accumulate="latest"), "all": dict(accumulate="all"),
         "reset": dict(accumulate="reset"), "swap": dict(accumulate="swap"),
         "swap β=0.5": dict(accumulate="swap", pass_average=0.5)}


def batches(path: Path, batch: int):
    ids, frames = [], []
    for raw in iter_records(path):
        t, i, f = decode_message(raw)
        if t != "image":
            continue
        ids.append(i)
        frames.append(f)
        if len(ids) == batch:
            yield np.asarray(ids), np.stack(frames)
            ids, frames = [], []
    if ids:
        yield np.asarray(ids), np.stack(frames)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(prog="python scripts/compare_accumulate.py",
                                description=__doc__.splitlines()[0])
    p.add_argument("source", type=Path, help="A looped capture (tools/repeat_dstream.py, one series).")
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES),
                   help="Modes to compare.")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0],
                   help="Points in each pass at which to take the live images.")
    p.add_argument("--channel", default="phase_flipped", help="Channel to compare and show.")
    p.add_argument("--batch-size", type=int, default=2048, help="Frames per batch, as run.py.")
    p.add_argument("--device", default="auto", help="Torch device (auto: cuda, then mps, then cpu).")
    p.add_argument("--out", type=Path, default=None, help="Montage PNG to write (needs matplotlib).")
    ParallaxAlgorithm().add_cli_args(p)
    return p


def main(argv=None) -> None:
    algo = ParallaxAlgorithm()
    args = build_parser().parse_args(algo.preprocess_argv(sys.argv[1:] if argv is None else argv))
    cfg = config_from_args(args)
    geo = preprocess_geometry(cfg.detector_shape, cfg.scan_shape, cfg.scan_sampling, cfg.energy,
                              cfg.semiangle_cutoff, angular_sampling=cfg.angular_sampling,
                              aberration_coefs=cfg.aberration_coefs,
                              rotation_angle=cfg.rotation_angle,
                              upsampling_factor=cfg.upsampling_factor,
                              probe_center=cfg.probe_center, flip_cols=cfg.flip_cols)
    device = None if args.device == "auto" else args.device
    accs = {m: ParallaxAccumulator(geo, device=device, wrap_ids=True, **MODES[m]) for m in args.modes}
    n_pos = geo.scan_shape[0] * geo.scan_shape[1]
    truth = {}                                          # pass → its own reconstruction
    current = None
    shots, times = [], {m: 0.0 for m in args.modes}
    stops = sorted(args.fractions)

    def snapshot(k, f):
        row = {"pass": k, "fraction": f, "images": {}}
        for m, acc in accs.items():
            row["images"][m] = acc.channels()[args.channel].float().cpu().numpy()
        shots.append(row)

    for ids, frames in batches(args.source, args.batch_size):
        k_first, k_last = int(ids.min()) // n_pos, int(ids.max()) // n_pos
        for k in range(k_first, k_last + 1):
            sel = ids // n_pos == k
            if k not in truth:
                if current is not None:
                    truth[current[0]] = current[1].channels()[args.channel].float().cpu().numpy()
                current = (k, ParallaxAccumulator(geo, device=device))
                done_stop = 0
            current[1].process(frames[sel], ids[sel] % n_pos)
            for m, acc in accs.items():
                t = time.perf_counter()
                acc.process(frames[sel], ids[sel])
                times[m] += time.perf_counter() - t
            cursor = (int(ids[sel].max()) % n_pos + 1) / n_pos
            while done_stop < len(stops) and cursor >= stops[done_stop] - 1e-9:
                snapshot(k, stops[done_stop])
                done_stop += 1
    truth[current[0]] = current[1].channels()[args.channel].float().cpu().numpy()

    print(f"{'pass':>4} {'frac':>5}  " + "  ".join(f"{m:>17}" for m in args.modes))
    print(f"{'':>4} {'':>5}  " + "  ".join(f"{'now / before':>17}" for _ in args.modes))
    for s in shots:
        k = s["pass"]
        cells = []
        for m in args.modes:
            img = s["images"][m]
            now = corr(img, truth[k])
            before = corr(img, truth[k - 1]) if k > 0 else float("nan")
            cells.append(f"{now:+.3f} / {before:+.3f}")
        print(f"{k:>4} {s['fraction']:>5.2f}  " + "  ".join(f"{c:>17}" for c in cells))
    for m, acc in accs.items():
        # "latest": its store; "reset" / "swap": the completed pass, one image
        extra = acc.store_bytes(2) if acc.accumulate == "latest" else \
            acc.out.numel() * 4 if acc.accumulate in ("reset", "swap") else 0
        print(f"[compare] {m}: {times[m]:.1f} s in process(), extra device memory "
              f"{extra / 1e6:,.0f} MB, {acc.passes_completed} passes completed, "
              f"{acc.late_frames} late frames")

    if args.out is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = [s for s in shots if s["pass"] > 0] or shots
        fig, ax = plt.subplots(len(args.modes), len(cols),
                               figsize=(2.6 * len(cols), 2.8 * len(args.modes)), squeeze=False)
        for j, s in enumerate(cols):
            for i, m in enumerate(args.modes):
                img = s["images"][m]
                lo, hi = np.percentile(img, [0.5, 99.5])
                ax[i, j].imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="antialiased",
                                interpolation_stage="data")
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                k = s["pass"]
                ax[i, j].set_title(f"pass {k} at {int(100 * s['fraction'])}%\nr now "
                                   f"{corr(img, truth[k]):+.2f}"
                                   + (f", before {corr(img, truth[k - 1]):+.2f}" if k else ""),
                                   fontsize=7)
                if j == 0:
                    ax[i, j].set_ylabel(m, fontsize=10)
        fig.suptitle(f"{args.source.name}: live {args.channel} per mode", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(args.out, dpi=150)
        print(f"[compare] wrote {args.out}")


if __name__ == "__main__":
    main()
