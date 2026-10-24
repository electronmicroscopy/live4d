"""Plot a saved reconstruction (`run.py` writes `parallax_recon.npy`).

Static counterpart to the live viewer: load the `.npy` and show it (or save a PNG), with the same
robust contrast. Self-contained — numpy + matplotlib only.

    python examples/plot_result.py out/parallax_recon.npy
    python examples/plot_result.py out/parallax_recon.npy --save recon.png --high-pass 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root on path

from examples._display import high_pass, robust_clim   # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python examples/plot_result.py", description=__doc__)
    p.add_argument("recon", help="Path to a reconstruction .npy (e.g. parallax_recon.npy).")
    p.add_argument("--save", default=None, help="Write a PNG here instead of opening a window.")
    p.add_argument("--high-pass", type=float, default=0.0, metavar="SIGMA_PX",
                   help="Gaussian high-pass sigma in pixels (0 = off).")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    import matplotlib.pyplot as plt

    recon = np.load(args.recon)
    disp = high_pass(recon, args.high_pass) if args.high_pass > 0 else recon
    lo, hi = robust_clim(disp)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    im = ax.imshow(disp, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(Path(args.recon).name)
    ax.set_xlabel("scan x")
    ax.set_ylabel("scan y")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
