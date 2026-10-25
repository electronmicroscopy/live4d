"""The standard analysis of a session of recordings: calibrate each recording, reconstruct it, and
make the same figures and summary for every session (sessions/README.md).

    python scripts/session_report.py sessions/2026-10-20_Gd2O3
    python scripts/session_report.py sessions/2026-10-23_Au-grating --steps figures

The session folder holds `session.json`:

    {"title": …, "date": …, "specimen": …,
     "reflections": {"111": 2.355, …},          # d-spacings (Å) ringed on the spectra
     "live_note": "where the live flags come from",
     "recordings": {"<id>": {"file": "data/….dstream",
                             "calibrate": "<the acquisition values for tools/calibrate_parallax.py>",
                             "live": "<the run.py flags used live, if known>"}},
     "variants": {"<name>": "<extra run.py flags>"}}   # optional extra reconstructions

and gets, per recording:

    calibration/<id>.flags / .json / .png    tools/calibrate_parallax.py (C10 refined on the lattice)
    recon/<id>/U<u>/                         run.py at ×1/2/4/8 with calibration/<id>.flags
    recon/<id>/U4_fit/                       ×4 at the shift fit's C10 (what the low frequencies,
                                             e.g. a support film, focus at)
    recon/<id>/U4_live/                      ×4 with the live flags
    recon/<id>/U<u>_<variant>/               each variant at ×1/2/4/8

and for the session `figures/overview.jpg` (each recording live / support focus / lattice focus,
its spectrum and the calibration along C10), `figures/upsampling_<id>.jpg` and `summary.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSAMPLINGS = (1, 2, 4, 8)
STEPS = ("calibrate", "recon", "figures")


def load(folder: Path) -> dict:
    return json.loads((folder / "session.json").read_text())


def calibration(folder: Path, rid: str) -> dict:
    return json.loads((folder / "calibration" / f"{rid}.json").read_text())


def _run(cmd: list[str], log: Path) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return log.read_text()


# ── calibrate ──────────────────────────────────────────────────────────────────

def calibrate(folder: Path, session: dict, python: str) -> None:
    for rid, rec in session["recordings"].items():
        out = folder / "calibration" / rid
        text = _run([python, "-m", "tools.calibrate_parallax", str(folder / rec["file"]),
                     *rec["calibrate"].split(), "--out", str(out)], out.with_suffix(".log"))
        lattice = [ln for ln in text.splitlines() if "focus on the lattice" in ln]
        print(f"[session] {rid}: calibrated; {lattice[0].split('): ', 1)[-1] if lattice else ''}",
              flush=True)


# ── reconstruct ────────────────────────────────────────────────────────────────

def jobs(folder: Path, session: dict) -> list[tuple[str, str, list[str]]]:
    """(recording, output name, run.py flags) of every standard reconstruction."""
    out = []
    for rid, rec in session["recordings"].items():
        flags = (folder / "calibration" / f"{rid}.flags").read_text().split()
        fit = calibration(folder, rid)["geometry"]["c10"]
        for u in UPSAMPLINGS:
            out.append((rid, f"U{u}", flags + ["--upsampling", str(u)]))
        at_fit = [f"--defocus={fit:.0f}" if f.startswith("--defocus=") else f for f in flags]
        out.append((rid, "U4_fit", at_fit + ["--upsampling", "4"]))
        if rec.get("live"):
            out.append((rid, "U4_live", rec["live"].split() + ["--upsampling", "4"]))
        for name, extra in session.get("variants", {}).items():
            for u in UPSAMPLINGS:
                out.append((rid, f"U{u}_{name}", flags + extra.split() + ["--upsampling", str(u)]))
    return out


def reconstruct(folder: Path, session: dict, python: str, device: str) -> None:
    for rid, name, flags in jobs(folder, session):
        out = folder / "recon" / rid / name
        text = _run([python, "-m", "realtime_stem.apps.run", "--ingest", "python", "--device", device,
                     "--source", str(folder / session["recordings"][rid]["file"]), *flags,
                     "--out-dir", str(out)], out.with_suffix(".log"))
        status = [ln for ln in text.splitlines() if "COMPLETE" in ln]
        print(f"[session] {rid} {name}: {status[-1].split('] ', 1)[-1] if status else 'NO STATUS'}",
              flush=True)


def recon_status(folder: Path, rid: str, name: str) -> str:
    log = folder / "recon" / rid / f"{name}.log"
    if not log.exists():
        return "missing"
    lines = [ln for ln in log.read_text().splitlines() if "COMPLETE" in ln]
    return "COMPLETE" if lines and "INCOMPLETE" not in lines[-1] else (lines[-1] if lines else "?")


# ── figures ────────────────────────────────────────────────────────────────────

def _image(folder, rid, name, channel="phase_flipped"):
    return np.load(folder / "recon" / rid / name / f"parallax_{channel}.npy").astype(np.float64)


def _bar_nm(fov_nm: float) -> float:
    return next((b for b in (0.5, 1, 2, 5, 10, 20, 50) if b >= fov_nm / 8), 100)


def _show(ax, img, title, step_A):
    lo, hi = np.percentile(img, [0.5, 99.5])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="antialiased")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    h, w = img.shape
    bar = _bar_nm(w * step_A / 10)
    n = bar * 10 / step_A
    ax.plot([0.04 * w, 0.04 * w + n], [0.95 * h] * 2, color="w", lw=3)
    ax.text(0.04 * w + n / 2, 0.92 * h, f"{bar:g} nm", color="w", ha="center", fontsize=9)


def _spectrum(ax, img, step_A, reflections, qmax, title):
    from matplotlib.patches import Circle

    n = img.shape[0]
    w = np.hanning(n)
    p = np.fft.fftshift(np.abs(np.fft.fft2((img - img.mean()) * np.outer(w, w))) ** 2)
    f = np.fft.fftshift(np.fft.fftfreq(n, step_A))
    lp = np.log10(p + p.mean() * 1e-6)
    v0, v1 = np.percentile(lp, [40, 99.95])
    ax.imshow(lp, cmap="inferno", vmin=v0, vmax=v1, extent=[f[0], f[-1], f[-1], f[0]],
              interpolation="antialiased")
    colors = ("c", "lime", "deepskyblue", "w", "orange", "violet")
    for (hkl, d), c in zip(reflections.items(), colors):
        ax.add_patch(Circle((0, 0), 1 / d, fill=False, color=c, lw=0.6, ls="--"))
        ax.text(0.72 / d, -0.72 / d, hkl, color=c, fontsize=8, clip_on=True)
    ax.set_xlim(-qmax, qmax), ax.set_ylim(qmax, -qmax)
    ax.set_xlabel("1/Å", fontsize=8), ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)


def _c10_curves(ax, cal):
    g, lo, rf = cal["geometry"], cal.get("loss") or {}, cal.get("refine") or {}
    if lo:
        ax.plot(lo["c10"], lo["loss_c10"], ".-", color="tab:blue", ms=3)
        ax.set_ylabel("variance loss (low frequencies)", color="tab:blue", fontsize=8)
    ax.axvline(g["c10"], color="tab:blue", ls="--", lw=0.8)
    if rf:
        a = ax.twinx()
        a.plot(rf["c10"], rf["prominence"], ".-", color="tab:orange", ms=3)
        a.set_ylabel("lattice prominence", color="tab:orange", fontsize=8)
        a.axvline(rf["best_c10"], color="tab:orange", ls="--", lw=0.8)
        a.tick_params(labelsize=7)
    ax.set_xlabel("C10 (Å)", fontsize=8), ax.tick_params(labelsize=7)
    focus = f", lattice {rf['best_c10']:.0f} Å" if rf.get("used") else ""
    ax.set_title(f"along C10: support {g['c10']:.0f} Å{focus}", fontsize=10)


def _qmax(reflections, step_A):
    q = 1.15 / min(reflections.values()) if reflections else 0.8
    return min(q, 0.5 / step_A)


def _lattice_crop(img, step_A, band, size):
    """(y0, x0) of the size × size window with the most power in `band` (1/Å)."""
    n0, n1 = img.shape
    q = np.hypot(np.fft.fftfreq(n0, step_A)[:, None], np.fft.fftfreq(n1, step_A)[None, :])
    hp = np.fft.ifft2(np.fft.fft2(img - img.mean()) * ((q > band[0]) & (q < band[1]))).real
    k = np.zeros((n0, n1))
    k[:size, :size] = 1.0
    local = np.fft.ifft2(np.fft.fft2(hp ** 2) * np.conj(np.fft.fft2(k))).real   # window sums
    local[n0 - size + 1:, :] = -np.inf                                        # windows that wrap
    local[:, n1 - size + 1:] = -np.inf
    y0, x0 = np.unravel_index(np.argmax(local), local.shape)
    return int(y0), int(x0)


def figures(folder: Path, session: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = list(session["recordings"])
    refl = session.get("reflections", {})
    fig_dir = folder / "figures"
    fig_dir.mkdir(exist_ok=True)

    has_live = any(session["recordings"][r].get("live") for r in recs)
    ncol = 5 if has_live else 4
    fig, axs = plt.subplots(len(recs), ncol, figsize=(5.4 * ncol, 5.6 * len(recs)), dpi=110,
                            squeeze=False)
    for row, rid in enumerate(recs):
        cal = calibration(folder, rid)
        s = cal["scan_sampling"] / 4
        c = 0
        if has_live:
            if session["recordings"][rid].get("live"):
                live_c10 = next(f.split("=")[1] for f in session["recordings"][rid]["live"].split()
                                if f.startswith("--defocus="))
                _show(axs[row, c], _image(folder, rid, "U4_live"),
                      f"{rid}: live flags ×4 (C10 {live_c10} Å)", s)
            else:
                axs[row, c].axis("off")
            c += 1
        rf = cal.get("refine") or {}
        _show(axs[row, c], _image(folder, rid, "U4_fit"),
              f"{rid}: shift-fit C10 {cal['geometry']['c10']:.0f} Å (support) ×4", s)
        lattice = f"{rf['best_c10']:.0f} Å (lattice)" if rf.get("used") else "= shift fit"
        _show(axs[row, c + 1], _image(folder, rid, "U4"), f"calibrated C10 {lattice} ×4", s)
        _spectrum(axs[row, c + 2], _image(folder, rid, "U4"), s, refl, _qmax(refl, s),
                  "its power spectrum" + (" (rings: " + ", ".join(refl) + ")" if refl else ""))
        _c10_curves(axs[row, c + 3], cal)
    fig.suptitle(f"{session['title']} ({session['date']}): phase-flipped reconstructions\n"
                 f"live flags: {session.get('live_note', '—')}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_dir / "overview.jpg", pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)

    for rid in recs:
        cal = calibration(folder, rid)
        band = (cal.get("refine") or {}).get("band") or (0.5 / cal["scan_sampling"],
                                                         0.95 / cal["scan_sampling"])
        ref = _image(folder, rid, "U4")
        fov_A = ref.shape[1] * cal["scan_sampling"] / 4
        crop_A = min(60.0, fov_A / 3)
        size4 = int(crop_A / (cal["scan_sampling"] / 4))
        y4, x4 = _lattice_crop(ref, cal["scan_sampling"] / 4, band, size4)
        fig, axs = plt.subplots(3, len(UPSAMPLINGS), figsize=(5.4 * len(UPSAMPLINGS), 16.5),
                                dpi=110)
        for col, u in enumerate(UPSAMPLINGS):
            img, s = _image(folder, rid, f"U{u}"), cal["scan_sampling"] / u
            y0, x0, n = y4 * u // 4, x4 * u // 4, max(4, size4 * u // 4)
            _show(axs[0, col], img, f"×{u} ({img.shape[0]}×{img.shape[1]})", s)
            axs[0, col].add_patch(plt.Rectangle((x0, y0), n, n, fill=False, color="y", lw=1))
            _show(axs[1, col], img[y0:y0 + n, x0:x0 + n], f"×{u}: {crop_A / 10:.1f} nm where the "
                  "lattice is strongest", s)
            _spectrum(axs[2, col], img, s, refl, min(_qmax(refl, cal["scan_sampling"] / 4),
                                                     0.5 / s),
                      f"×{u} power spectrum (Nyquist {0.5 / s:.2f} 1/Å)")
        c10 = cal["refine"]["best_c10"] if (cal.get("refine") or {}).get("used") else \
            cal["geometry"]["c10"]
        fig.suptitle(f"{session['title']} ({session['date']}), {rid}: phase-flipped, C10 "
                     f"{c10:.0f} Å, upsampling ×1/×2/×4/×8", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(fig_dir / f"upsampling_{rid}.jpg", pil_kwargs={"quality": 85, "optimize": True})
        plt.close(fig)
    print(f"[session] figures → {fig_dir}", flush=True)


# ── summary ────────────────────────────────────────────────────────────────────

def summary(folder: Path, session: dict) -> dict:
    out = {"title": session["title"], "date": session["date"], "recordings": {}}
    for rid, rec in session["recordings"].items():
        cal = calibration(folder, rid)
        g, d, rf = cal["geometry"], cal["disk"], cal.get("refine") or {}
        names = sorted({n for r, n, _ in jobs(folder, session) if r == rid})
        out["recordings"][rid] = {
            "file": rec["file"], "frames": cal["frames"],
            "acquisition": rec["calibrate"],
            "disk": {"centre_px": [round(v, 2) for v in d["center"]],
                     "radius_px": round(d["radius"], 2),
                     "radius_mrad": round(d["radius"] * cal["angular_sampling"], 2)},
            "flip_cols": g["flip_cols"], "rotation_deg": round(g["rotation_deg"], 2),
            "c12_A": round(g["c12"], 1), "phi12_deg": round(g["phi12_deg"], 1),
            "c10_shift_fit_A": round(g["c10"], 1),
            "c10_A": round(rf["best_c10"] if rf.get("used") else g["c10"], 1),
            "lattice_refinement": ({"used": rf["used"], "gain": round(rf["gain"], 2),
                                    "band_per_A": [round(v, 3) for v in rf["band"]]}
                                   if rf else None),
            "apparent_c30_um": round(cal["higher_orders"]["c30"] / 1e4, 2),
            "flags": cal["flags"],
            "live_flags": rec.get("live"),
            "recon": {n: recon_status(folder, rid, n) for n in names},
        }
    (folder / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    return out


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(prog="python scripts/session_report.py",
                                description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("session", type=Path, help="A session folder holding session.json.")
    p.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS),
                   help="Which steps to run, in this order.")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"),
                   help="Torch device for the reconstructions.")
    p.add_argument("--python", default=sys.executable,
                   help="The interpreter that runs the calibration and run.py.")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    folder = args.session.resolve()
    session = load(folder)
    if "calibrate" in args.steps:
        calibrate(folder, session, args.python)
    if "recon" in args.steps:
        reconstruct(folder, session, args.python, args.device)
    if "figures" in args.steps:
        figures(folder, session)
    out = summary(folder, session)
    for rid, r in out["recordings"].items():
        print(f"[session] {rid}: C10 {r['c10_A']:.0f} Å (shift fit {r['c10_shift_fit_A']:.0f} Å), "
              f"rotation {r['rotation_deg']}°, {'flipped' if r['flip_cols'] else 'unflipped'}, "
              f"C12 {r['c12_A']} Å; recon "
              f"{sum(v == 'COMPLETE' for v in r['recon'].values())}/{len(r['recon'])} COMPLETE",
              flush=True)
    return out


if __name__ == "__main__":
    main()
