"""Simulate a 4D-STEM parallax dataset with known ground truth (torch; cuda / mps / cpu).

Single-slice, pure-phase object: for each probe position the exit wave is the defocused probe times
`exp(iφ)` over a window of the (periodic) object, its far-field intensity is integrated into
detector pixels, rotated into the detector frame and Poisson-sampled at the requested dose. Big
enough to exercise the live pipeline at the scale it runs at (default 256×256 scan of 96×96
frames, 1.2 GB as .npy) and with the answer known: the phase map, the aberrations, the rotation and
the detector layout are all saved alongside.

    python -m live4d.parallax.tools.simulate_4dstem --scan 256 256 --out out/sim256              # .npy + .dstream
    python -m live4d.parallax.offline --source out/sim256.dstream $(cat out/sim256.flags) ...

Repeated scans model live tuning: `--passes N` scans the same area N times in one series (image ids
continue past the scan, as a continuously scanning detector sends them), and
`--defocus-schedule` / `--c12-schedule` give the microscope's *actual* C10 / C12 on each pass —
e.g. an operator focusing towards the target — while `--defocus` / `--c12` stay the algorithm's
target (what the reconstruction flags carry):

    python -m live4d.parallax.tools.simulate_4dstem --scan 128 128 --passes 4 --defocus -2000 \
        --defocus-schedule -2600 -2400 -2200 -2000 --out out/tuning

Physics and conventions (those of the reconstruction, `live4d.algorithms.parallax`):
  * probe `IFFT(A(k) exp(-iχ(k)))`, χ = π/λ α² (C10 + C12 cos 2(φ − φ12)), abTEM's convention
    (C10 < 0 = underfocus, as the reference notebooks use for the apoferritin data);
  * frequency axis 0 ("kx") runs along scan rows, as in `spatial_frequencies`;
  * the detector pixel at offset k_det from the disk center records the intensity at scan-frame
    frequency R(θ)·k_det — the active rotation the geometry applies (`rotation_angle` = θ);
  * the detector layout (`--detector-layout`) re-lays the canonical frame afterwards.

The simulation k-grid is `bin` (odd) times finer than the detector, so that the real-space window
(`bin` × the detector-conjugate field of view) holds the defocused probe; each detector pixel
integrates the `bin × bin` simulation pixels centered on it. Scan positions sit on the window's
pixel grid, so the scan step is rounded to a multiple of the simulation sampling (the effective
step is reported and used in the reconstruction flags).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch


from live4d.algorithms.parallax.engine import _electron_wavelength, _pick_device  # noqa: E402
from live4d.algorithms.parallax.orientation import (  # noqa: E402
    DETECTOR_LAYOUTS,
    layout_center,
    layout_frames,
    layout_shape,
)
from live4d.algorithms.parallax.readout import quadratic_aberration_surface  # noqa: E402


@dataclass
class SimParams:
    scan: tuple[int, int] = (256, 256)
    detector: tuple[int, int] = (96, 96)        # canonical (H, W); the layout may transpose it
    energy: float = 300e3                       # eV
    semiangle: float = 10.0                     # mrad
    angular_sampling: float = 0.5               # mrad / detector px
    scan_step: float = 2.0                      # Å, requested (rounded to the sim grid)
    aberrations: dict = field(default_factory=lambda: {"C10": -2000.0})   # Å, Å, rad
    rotation_deg: float = -15.0
    detector_layout: str = "rows,cols"
    dose: float = 2000.0                        # e / Å²; 0 → noiseless expected counts
    n_particles: int | None = None              # None → ~30% coverage
    seed: int = 0
    passes: int = 1                             # repeated scans of the same area, one series
    defocus_schedule: tuple | None = None       # actual C10 per pass (Å); None → the target
    c12_schedule: tuple | None = None           # actual C12 per pass (Å); None → the target

    def pass_aberrations(self, k: int) -> dict:
        """The microscope's actual aberrations on pass `k` (schedules repeat their last value)."""
        ab = dict(self.aberrations)
        if self.defocus_schedule:
            ab["C10"] = float(self.defocus_schedule[min(k, len(self.defocus_schedule) - 1)])
        if self.c12_schedule:
            ab["C12"] = float(self.c12_schedule[min(k, len(self.c12_schedule) - 1)])
        return ab


@dataclass
class SimGrid:
    """Derived sampling: everything is fixed by the detector sampling and the probe size."""
    wavelength: float
    bin: int                                    # sim k-pixels per detector pixel (odd)
    n_sim: int                                  # sim window, px
    dx: float                                   # sim real-space sampling, Å
    step_px: int                                # scan step in sim px
    scan_step: float                            # effective scan step, Å
    object_shape: tuple[int, int]


def sim_grid(p: SimParams) -> SimGrid:
    lam = _electron_wavelength(p.energy)
    dk_det = p.angular_sampling * 1e-3 / lam                       # 1/Å per detector pixel
    fov_det = 1.0 / dk_det                                         # detector-conjugate window, Å
    c10 = max(abs(ab.get("C10", 0.0)) + abs(ab.get("C12", 0.0))
              for ab in (p.pass_aberrations(k) for k in range(max(1, p.passes))))
    probe_d = 2 * p.semiangle * 1e-3 * c10 + 4 * lam / (p.semiangle * 1e-3)    # + focused size
    b = max(3, math.ceil(2.0 * probe_d / fov_det))
    b += (b + 1) % 2                                               # odd: centered bins
    n_sim = b * max(p.detector)
    dx = 1.0 / (max(p.detector) * dk_det)
    step_px = max(1, round(p.scan_step / dx))
    return SimGrid(wavelength=lam, bin=b, n_sim=n_sim, dx=dx, step_px=step_px,
                   scan_step=step_px * dx,
                   object_shape=(p.scan[0] * step_px, p.scan[1] * step_px))


def make_object(p: SimParams, g: SimGrid) -> np.ndarray:
    """A periodic phase map (rad) on the simulation grid: shells and blobs on a weak background."""
    rng = np.random.default_rng(p.seed)
    ny, nx = g.object_shape
    phase = np.zeros((ny, nx), np.float64)
    area = ny * nx * g.dx**2
    n = p.n_particles if p.n_particles is not None else max(1, round(0.3 * area / (np.pi * 35**2)))
    for _ in range(n):
        radius = rng.uniform(20.0, 55.0)                           # Å
        amp = rng.uniform(0.1, 0.4)                                # rad
        cy, cx = rng.uniform(0, ny), rng.uniform(0, nx)
        half = int(np.ceil(1.6 * radius / g.dx))
        iy = np.arange(int(cy) - half, int(cy) + half + 1)
        ix = np.arange(int(cx) - half, int(cx) + half + 1)
        r = np.hypot((iy[:, None] - cy) * g.dx, (ix[None, :] - cx) * g.dx)
        if rng.random() < 0.6:                                     # protein-cage-like shell
            width = radius / 5
            blob = amp * (np.exp(-0.5 * ((r - radius) / width) ** 2)
                          + 0.35 * np.exp(-0.5 * (r / (0.6 * radius)) ** 2))
        else:
            blob = amp * np.exp(-0.5 * (r / (0.5 * radius)) ** 2)
        np.add.at(phase, ((iy % ny)[:, None], (ix % nx)[None, :]), blob)
    # weak smooth background (a support film)
    noise = rng.standard_normal((ny, nx))
    ky = np.fft.fftfreq(ny, g.dx)[:, None]
    kx = np.fft.fftfreq(nx, g.dx)[None, :]
    smooth = np.fft.ifft2(np.fft.fft2(noise) * np.exp(-0.5 * (np.hypot(ky, kx) / 0.05) ** 2)).real
    phase += 0.03 * smooth / smooth.std()
    return phase


def make_probe(p: SimParams, g: SimGrid, device, aberrations: dict | None = None) -> torch.Tensor:
    """(n_sim, n_sim) complex probe, centered at (n_sim//2, n_sim//2), Σ|ψ|² = 1."""
    k = np.fft.fftfreq(g.n_sim, g.dx)
    kxa, kya = np.meshgrid(k, k, indexing="ij")
    alpha = np.hypot(kxa, kya) * g.wavelength
    phi = np.arctan2(kya, kxa)
    dalpha = g.wavelength / (g.n_sim * g.dx)                       # one sim pixel, rad
    aperture = np.clip((p.semiangle * 1e-3 - alpha) / dalpha + 0.5, 0.0, 1.0)
    chi = quadratic_aberration_surface(alpha, phi, g.wavelength,
                                       p.aberrations if aberrations is None else aberrations)
    probe = np.fft.fftshift(np.fft.ifft2(aperture * np.exp(-1j * chi)))
    probe /= np.sqrt((np.abs(probe) ** 2).sum())
    return torch.as_tensor(probe.astype(np.complex64), device=device)


def _rotation_grid(g: SimGrid, theta: float, device) -> torch.Tensor | None:
    """grid_sample grid sampling the scan-frame intensity at R(θ)·k_det, or None for θ = 0."""
    if theta == 0.0:
        return None
    n = g.n_sim
    d0 = n // 2 + (g.bin - 1) // 2                                 # the DC pixel after centering
    off = torch.arange(n, dtype=torch.float32, device=device) - d0
    k0, k1 = torch.meshgrid(off, off, indexing="ij")
    c, s = math.cos(theta), math.sin(theta)
    src0 = d0 + c * k0 - s * k1
    src1 = d0 + s * k0 + c * k1
    to_norm = lambda i: 2 * i / (n - 1) - 1                        # noqa: E731
    return torch.stack((to_norm(src1), to_norm(src0)), dim=-1)[None]   # (1, n, n, 2): (x=axis1, y=axis0)


class Simulation:
    """The object, probe and detector mapping for one `SimParams`; `batches()` yields frames."""

    def __init__(self, p: SimParams, device=None):
        self.params = p
        self.device = _pick_device(device)
        self.grid = sim_grid(p)
        self.phase = make_object(p, self.grid)                      # (rad) on the sim grid
        self._trans = torch.as_tensor(np.exp(1j * self.phase).astype(np.complex64),
                                      device=self.device)
        self._rot = _rotation_grid(self.grid, math.radians(p.rotation_deg), self.device)

    def batches(self, batch: int = 256):
        """Yield `(image_id, frames)` in acquisition order: uint16 counts (float32 expected counts
        if noiseless), laid out per `detector_layout`. With `passes` > 1 the ids run on past the
        scan (pass k covers k·Ny·Nx …), each pass with its scheduled aberrations."""
        p = self.params
        rng = torch.Generator(device="cpu").manual_seed(p.seed + 1)
        n_pos = p.scan[0] * p.scan[1]
        for k in range(max(1, p.passes)):
            probe = make_probe(p, self.grid, self.device, p.pass_aberrations(k))
            for start, frames in self._one_pass(probe, rng, batch):
                yield k * n_pos + start, frames

    def _one_pass(self, probe, rng, batch):
        p, g = self.params, self.grid
        n = g.n_sim
        h, w = p.detector
        win = torch.arange(n, device=self.device) - n // 2
        electrons = p.dose * g.scan_step**2 if p.dose > 0 else 1.0
        n_pos = p.scan[0] * p.scan[1]
        shift = (g.bin - 1) // 2                                     # DC to the center of its bin
        pad = ((n - g.bin * h) // 2, (n - g.bin * w) // 2)          # non-square detectors
        for start in range(0, n_pos, batch):
            ids = torch.arange(start, min(start + batch, n_pos), device=self.device)
            cy = (ids // p.scan[1]) * g.step_px
            cx = (ids % p.scan[1]) * g.step_px
            rows = (cy[:, None] + win[None, :]) % g.object_shape[0]
            cols = (cx[:, None] + win[None, :]) % g.object_shape[1]
            exit_wave = probe[None] * self._trans[rows[:, :, None], cols[:, None, :]]
            inten = torch.fft.fft2(exit_wave).abs().square() / (n * n)
            inten = torch.roll(torch.fft.fftshift(inten, dim=(-2, -1)), (shift, shift),
                               dims=(-2, -1))
            if self._rot is not None:
                inten = torch.nn.functional.grid_sample(
                    inten[:, None], self._rot.expand(len(ids), -1, -1, -1), mode="bilinear",
                    padding_mode="zeros", align_corners=True)[:, 0]
            inten = inten[:, pad[0]:pad[0] + g.bin * h, pad[1]:pad[1] + g.bin * w]
            det = inten.reshape(len(ids), h, g.bin, w, g.bin).sum(dim=(2, 4)) * electrons
            if p.dose > 0:
                counts = torch.poisson(det.float().cpu(), generator=rng).numpy()
                frames = np.minimum(counts, np.iinfo(np.uint16).max).astype(np.uint16)
            else:
                frames = det.float().cpu().numpy()
            yield start, np.ascontiguousarray(layout_frames(frames, p.detector_layout))

    def scan_phase(self) -> np.ndarray:
        """The ground-truth phase averaged onto the scan grid (one value per probe position)."""
        return scan_grid_phase(self.phase, self.grid)


def reconstruct_flags(p: SimParams, g: SimGrid) -> list[str]:
    """The `live4d.parallax.offline` flags that describe this dataset exactly."""
    raw = layout_shape(p.detector, p.detector_layout)
    flags = ["--detector", str(raw[0]), str(raw[1]), "--scan", str(p.scan[0]), str(p.scan[1]),
             "--energy", f"{p.energy:g}", "--semiangle", f"{p.semiangle:g}",
             "--angular-sampling", f"{p.angular_sampling:g}",
             "--scan-sampling", f"{g.scan_step:.6g}",
             f"--defocus={p.aberrations.get('C10', 0.0):g}", f"--rotation={p.rotation_deg:g}"]
    center = layout_center(raw, p.detector_layout)            # the disk centre on the raw frames
    flags += ["--center", f"{center[0]:g}", f"{center[1]:g}"]
    if p.aberrations.get("C12"):
        flags += [f"--c12={p.aberrations['C12']:g}",
                  f"--phi12={math.degrees(p.aberrations.get('phi12', 0.0)):g}"]
    if p.detector_layout != "rows,cols":
        flags += ["--detector-layout", p.detector_layout]
    if p.passes > 1:
        flags += ["--accumulate", "latest"]           # ids run past the scan: repeated passes
    return flags


def scan_grid_phase(phase: np.ndarray, g: SimGrid) -> np.ndarray:
    """The ground-truth phase averaged onto the scan grid (one value per probe position)."""
    ny, nx = phase.shape
    s = g.step_px
    # center each block on its probe position: roll by half a step first
    rolled = np.roll(phase, (s // 2, s // 2), axis=(0, 1))
    return rolled.reshape(ny // s, s, nx // s, s).mean(axis=(1, 3))


def write(p: SimParams, out: Path, formats=("npy", "dstream"), device=None,
          batch: int = 256) -> dict:
    """Simulate and write `<out>.npy` / `<out>.dstream` + `<out>.json`, `<out>_phase.npy`,
    `<out>.flags`. Returns the metadata dict."""
    from live4d.parallax.ingest import codec
    from live4d.parallax.ingest.dstream import write_magic, write_record

    out.parent.mkdir(parents=True, exist_ok=True)
    raw_shape = layout_shape(p.detector, p.detector_layout)
    dtype = np.float32 if p.dose <= 0 else np.uint16
    mm = None
    if "npy" in formats:
        lead = (p.passes, *p.scan) if p.passes > 1 else tuple(p.scan)
        mm = np.lib.format.open_memmap(out.with_suffix(".npy"), mode="w+", dtype=dtype,
                                       shape=(*lead, *raw_shape))
        flat = mm.reshape(-1, *raw_shape)
    fh = None
    if "dstream" in formats:
        if dtype != np.uint16:
            raise ValueError("a .dstream carries integer counts: give a dose > 0")
        fh = open(out.with_suffix(".dstream"), "wb")
        write_magic(fh)
        write_record(fh, codec.encode_start_message(raw_shape, p.passes * p.scan[0] * p.scan[1]))
    t0 = time.perf_counter()
    sim = Simulation(p, device=device)
    try:
        for start, frames in sim.batches(batch):
            if mm is not None:
                flat[start:start + len(frames)] = frames
            if fh is not None:
                for i, f in enumerate(frames):
                    write_record(fh, codec.encode_image_message(start + i, f))
        if fh is not None:
            write_record(fh, codec.encode_end_message())
    finally:
        if fh is not None:
            fh.close()
        if mm is not None:
            mm.flush()
    g = sim.grid
    np.save(out.parent / f"{out.name}_phase.npy", sim.scan_phase().astype(np.float32))
    flags = reconstruct_flags(p, g)
    meta = {"params": asdict(p), "grid": asdict(g), "reconstruct_flags": flags,
            "seconds": time.perf_counter() - t0}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    out.with_suffix(".flags").write_text(" ".join(flags) + "\n")
    return meta


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python -m live4d.parallax.tools.simulate_4dstem", description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = SimParams()
    p.add_argument("--out", type=Path, required=True, metavar="PREFIX",
                   help="Output prefix, e.g. out/sim256: writes <prefix>.npy / .dstream, "
                        "<prefix>.json (parameters), <prefix>.flags (run.py flags) and "
                        "<prefix>_phase.npy (ground truth on the scan grid).")
    p.add_argument("--format", nargs="+", choices=("npy", "dstream"), default=["npy", "dstream"],
                   help="What to write: frames as .npy (memory-mapped), a .dstream capture "
                        "(Stream2 framing, bslz4), or both.")
    p.add_argument("--scan", type=int, nargs=2, default=d.scan, metavar=("NY", "NX"),
                   help="Scan grid (probe positions).")
    p.add_argument("--detector", type=int, nargs=2, default=d.detector, metavar=("H", "W"),
                   help="Canonical detector shape (the layout may transpose it).")
    p.add_argument("--energy", type=float, default=d.energy, metavar="EV",
                   help="Beam energy (eV).")
    p.add_argument("--semiangle", type=float, default=d.semiangle, metavar="MRAD",
                   help="Probe convergence semiangle (mrad).")
    p.add_argument("--angular-sampling", type=float, default=d.angular_sampling,
                   metavar="MRAD_PX", help="Detector angular sampling (mrad/px).")
    p.add_argument("--scan-step", type=float, default=d.scan_step, metavar="ANG",
                   help="Scan step (Å); rounded to the simulation grid, the effective step is "
                        "reported and written to the flags.")
    p.add_argument("--defocus", type=float, default=d.aberrations["C10"], metavar="C10_ANG",
                   help="Target defocus C10 (Å), abTEM convention (negative = underfocus); also "
                        "the actual one unless --defocus-schedule.")
    p.add_argument("--c12", type=float, default=0.0, metavar="C12_ANG",
                   help="Target astigmatism C12 (Å).")
    p.add_argument("--phi12", type=float, default=0.0, metavar="DEG",
                   help="Astigmatism angle phi12 (degrees).")
    p.add_argument("--rotation", type=float, default=d.rotation_deg, metavar="DEG",
                   help="Scan/detector rotation (degrees).")
    p.add_argument("--detector-layout", choices=list(DETECTOR_LAYOUTS), default=d.detector_layout,
                   metavar="LAYOUT",
                   help="Write the frames transposed/mirrored like this detector layout "
                        "(run.py --detector-layout): " + " | ".join(DETECTOR_LAYOUTS) + ".")
    p.add_argument("--dose", type=float, default=d.dose, metavar="E_PER_A2",
                   help="Electron dose (e/Å²) for Poisson counts; 0 = noiseless expected counts "
                        "(.npy only).")
    p.add_argument("--n-particles", type=int, default=None, metavar="N",
                   help="Particles in the phase object (default: ~30%% area coverage).")
    p.add_argument("--passes", type=int, default=1, metavar="N",
                   help="Scan the area this many times in one series (ids continue past the scan).")
    p.add_argument("--defocus-schedule", type=float, nargs="+", default=None, metavar="C10",
                   help="The microscope's actual C10 (Å) on each pass; --defocus stays the target. "
                        "Plain numbers (e.g. -2600, not -2.6e3).")
    p.add_argument("--c12-schedule", type=float, nargs="+", default=None, metavar="C12",
                   help="The actual C12 (Å) on each pass (at --phi12); --c12 stays the target.")
    p.add_argument("--seed", type=int, default=d.seed,
                   help="Random seed (object and shot noise).")
    p.add_argument("--device", default="auto",
                   help="Torch device for the simulation (auto: cuda, then mps, then cpu).")
    p.add_argument("--batch", type=int, default=256, metavar="N",
                   help="Probe positions per simulation batch (lower it for large probes).")
    return p


def main() -> None:
    p = build_parser()
    from live4d.algorithms.parallax.config import join_negative_floats
    args = p.parse_args(join_negative_floats(sys.argv[1:]))

    aberrations = {"C10": args.defocus}
    if args.c12 or args.c12_schedule:
        aberrations.update(C12=args.c12, phi12=math.radians(args.phi12))
    params = SimParams(scan=tuple(args.scan), detector=tuple(args.detector), energy=args.energy,
                       semiangle=args.semiangle, angular_sampling=args.angular_sampling,
                       scan_step=args.scan_step, aberrations=aberrations,
                       rotation_deg=args.rotation, detector_layout=args.detector_layout,
                       dose=args.dose, n_particles=args.n_particles, seed=args.seed,
                       passes=args.passes,
                       defocus_schedule=tuple(args.defocus_schedule) if args.defocus_schedule else None,
                       c12_schedule=tuple(args.c12_schedule) if args.c12_schedule else None)
    g = sim_grid(params)
    print(f"[sim] {params.scan[0]}x{params.scan[1]} scan of {params.detector} frames; sim window "
          f"{g.n_sim}px at {g.dx:.3f} Å (bin {g.bin}), step {g.scan_step:.4g} Å, "
          f"{params.dose * g.scan_step**2:,.0f} e/frame", flush=True)
    meta = write(params, args.out, formats=tuple(args.format), device=args.device,
                 batch=args.batch)
    print(f"[sim] wrote {args.out}.{{{','.join(args.format)}}} + json/flags/phase in "
          f"{meta['seconds']:.1f}s", flush=True)
    print(f"[sim] reconstruct with: {' '.join(meta['reconstruct_flags'])}", flush=True)


if __name__ == "__main__":
    main()
