"""Measure the parallax geometry of a recorded scan: the bright-field disk's centre, the detector's
handedness (column flip), the scan/detector rotation, the defocus and the astigmatism, written out
as `run.py` flags.

    python -m tools.calibrate_parallax scan.dstream --scan 256 256 --energy 200000 \
        --semiangle 30 --angular-sampling 1.066 --scan-sampling 1.9766 --rotation-prior 3.75 \
        --out out/scan_cal

Under defocus each bright-field pixel sees the specimen along its own tilt, so its virtual
bright-field (vBF) image is one common image moved by a shift that is linear in the tilt:
s = M·κ, with κ the pixel's angular offset from the disk centre on the raw frame. The tool

1. finds the disk centre and radius on the mean diffraction pattern, and lists the flagged
   (dtype-maximum) and hot pixels;
2. builds the vBF images of small groups of bright-field pixels (one pass over the recording);
3. measures each group's shift by cross-correlation against an iteratively aligned reference;
4. fits M (with a free translation; groups whose peak was lost in the noise are left out) and
   decomposes it in the reconstruction's own model,
   M = (C10·I + C12·Q(φ12)) · R(θ) · diag(1, ±1): the sign of det M is the column flip, and the
   polar decomposition gives θ, C10, C12 and φ12 (`processing/parallax/engine.py` conventions);
5. checks the fit with the parallax variance loss (quantem's `residual_loss`) along C10 and θ,
   and against the other handedness;
6. reports, as a diagnostic only, how far the shifts depart from a quadratic aberration surface
   (C30, C21, C23). Parallax corrects quadratic aberrations alone (`higher_orders`), so only C10,
   C12, θ and the flip go into the `run.py` flags;
7. refines C10 on the lattice (`refine_c10`). The shifts are measured at low spatial frequencies,
   where a support film can dominate, and particles sit about one radius off the support. The
   tool reconstructs ×4 images over a range of C10 and keeps the one whose lattice spots beyond
   the scan's Nyquist frequency stand out most. That C10 goes into `--defocus` (without a lattice,
   the fit's); the centre, rotation, flip and C12 stay from the fit.

The shifts cannot tell (C10, C12, θ) from (−C10, −C12, θ + 180°). The twin whose rotation is nearest
`--rotation-prior` is reported (without one, the twin with |θ| ≤ 90°). The fit measures the
aberrations times the angular sampling: a wrong `--angular-sampling` scales C10 and C12 by the same
factor. Scan-step anisotropy, if any, shows up as astigmatism.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → `import realtime_stem`

from realtime_stem.processing.parallax.engine import (  # noqa: E402
    _electron_wavelength,
    preprocess_geometry,
)

Batches = Callable[[int], Iterator[tuple[np.ndarray, np.ndarray]]]


# ── frames from a recording ────────────────────────────────────────────────────

def file_batches(path: Path, batch: int = 4096) -> Batches:
    """A re-iterable source of `(ids, frames)` batches from a `.dstream` or a `.npy` of frames.
    Calling it with `every=k` yields only every k-th frame (and decodes only those)."""
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r")
        return array_batches(data.reshape(-1, *data.shape[-2:]), batch)

    def gen(every: int = 1):
        from realtime_stem.ingest.codec import decode_message
        from realtime_stem.ingest.dstream import iter_records

        ids, frames = [], []
        for i, raw in enumerate(iter_records(path)):
            if every > 1 and i % every:
                continue                      # skipped records are never parsed
            mtype, image_id, frame = decode_message(raw)
            if mtype != "image":
                if mtype == "end":
                    break
                continue
            ids.append(image_id)
            frames.append(frame)
            if len(ids) == batch:
                yield np.asarray(ids), np.stack(frames)
                ids, frames = [], []
        if ids:
            yield np.asarray(ids), np.stack(frames)
    return gen


def array_batches(frames: np.ndarray, batch: int = 4096) -> Batches:
    """`file_batches` for an in-memory (N, H, W) array whose frame i sits at scan position i."""
    def gen(every: int = 1):
        for start in range(0, len(frames), batch * every):
            idx = np.arange(start, min(start + batch * every, len(frames)), every)
            yield idx, np.asarray(frames[idx])
    return gen


# ── 1. the disk on the mean diffraction pattern ────────────────────────────────

@dataclass
class Disk:
    center: tuple[float, float]           # (row, col) px on the raw frame
    radius: float                         # px
    shape: tuple[int, int]
    flagged: list[tuple[int, int]] = field(default_factory=list)   # dtype-maximum in every frame
    hot: list[tuple[int, int]] = field(default_factory=list)       # far above the disk's level
    frames_used: int = 0

    def inside(self, pixel) -> bool:
        return math.hypot(pixel[0] - self.center[0], pixel[1] - self.center[1]) <= self.radius + 1


def mean_pattern(batches: Batches, every: int = 8) -> tuple[np.ndarray, int, float]:
    """(mean frame, frames averaged, the frames' dtype maximum)."""
    total, n, top = None, 0, None
    for _, frames in batches(every):
        f = frames.astype(np.float64)
        total = f.sum(axis=0) if total is None else total + f.sum(axis=0)
        n += len(frames)
        if top is None:
            top = float(np.iinfo(frames.dtype).max) if frames.dtype.kind in "ui" else np.inf
    if total is None:
        raise ValueError("no image frames in the recording")
    return total / n, n, top


def fit_disk(mean: np.ndarray, top: float = np.inf, frames_used: int = 0) -> Disk:
    """Centre and radius of the bright-field disk, from its partial-pixel area and centroid."""
    m = mean.astype(np.float64).copy()
    flagged = np.argwhere(m >= top - 0.5)               # the detector's defective-pixel value
    fill = float(np.median(m))
    m[tuple(flagged.T)] = fill
    hot = np.zeros(m.shape, bool)
    rows, cols = np.indices(m.shape)
    bg = fill
    mask = m > 0.5 * (bg + np.percentile(m, 99))
    for _ in range(3):
        plateau = float(np.median(m[mask]))
        hot |= m > 5 * plateau
        m[hot] = bg
        mask &= ~hot
        w = mask.astype(np.float64)
        cy, cx = (rows * w).sum() / w.sum(), (cols * w).sum() / w.sum()
        r = math.sqrt(w.sum() / math.pi)
        near = np.hypot(rows - cy, cols - cx) <= 1.3 * r + 2
        bg = float(np.median(m[~near])) if (~near).any() else float(np.percentile(m, 1))
        mask = near & (m > 0.5 * (bg + plateau))
    # partial-pixel refinement: each pixel counts by its fraction of the plateau
    frac = np.clip((m - bg) / max(plateau - bg, 1e-12), 0.0, 1.0) * near
    cy, cx = (rows * frac).sum() / frac.sum(), (cols * frac).sum() / frac.sum()
    radius = math.sqrt(frac.sum() / math.pi)
    return Disk(center=(float(cy), float(cx)), radius=float(radius), shape=tuple(m.shape),
                flagged=[tuple(int(v) for v in p) for p in flagged],
                hot=[tuple(int(v) for v in p) for p in np.argwhere(hot)],
                frames_used=frames_used)


def _disk_mask(disk: Disk, shrink: float = 2.0) -> np.ndarray:
    rows, cols = np.indices(disk.shape)
    return np.hypot(rows - disk.center[0], cols - disk.center[1]) <= disk.radius - shrink


# ── 2. virtual bright-field images of pixel groups ─────────────────────────────

@dataclass
class Groups:
    pix: np.ndarray        # flat detector indices, sorted by group
    starts: np.ndarray     # first entry of each group in `pix` (np.add.reduceat)
    offsets: np.ndarray    # (G, 2) mean (row, col) offset of each group from the disk centre, px
    size: int              # group edge, detector px


def bf_groups(disk: Disk, size: int | None = None, target: int = 256,
              margin: float = 1.0, min_px: int = 1) -> Groups:
    """Bright-field pixels at least `margin` px inside the disk edge, minus the flagged and hot
    ones, tiled into size × size groups centred on the disk.

    Near the edge a pixel's vBF image is not a shifted copy of the others: for spatial frequencies
    q beyond its distance to the aperture edge only one of the two interfering beams is inside the
    aperture, and that single-sideband transfer carries an odd phase ∓πλC10q² which moves the
    cross-correlation peak outwards. A margin of q_max / Δk keeps every group clear of it."""
    rows, cols = np.indices(disk.shape)
    u, v = rows - disk.center[0], cols - disk.center[1]
    inside = np.hypot(u, v) <= disk.radius - max(margin, 1.0)
    for p in disk.flagged + disk.hot:
        inside[p] = False
    if not inside.any():
        raise ValueError(f"no bright-field pixels left {margin:.1f} px inside the disk edge "
                         f"(radius {disk.radius:.1f} px): lower the low-pass")
    if size is None:
        size = max(1, round(math.sqrt(inside.sum() / target)), math.ceil(math.sqrt(min_px)))
    gi = np.floor(u / size + 0.5).astype(np.int64)
    gj = np.floor(v / size + 0.5).astype(np.int64)
    key = (gi * 4096 + gj)[inside]
    flat = np.flatnonzero(inside)
    order = np.argsort(key, kind="stable")
    key, flat = key[order], flat[order]
    _, starts, counts = np.unique(key, return_index=True, return_counts=True)
    keep = counts >= max(1, (size * size) // 2)
    pix = np.concatenate([flat[s:s + c] for s, c, k in zip(starts, counts, keep) if k])
    counts = counts[keep]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    off = np.stack([np.add.reduceat(u.ravel()[pix], starts) / counts,
                    np.add.reduceat(v.ravel()[pix], starts) / counts], axis=1)
    return Groups(pix=pix, starts=starts, offsets=off, size=size)


def vbf_stack(batches: Batches, groups: Groups, scan: tuple[int, int]) -> tuple[np.ndarray, int]:
    """(G, Ny, Nx) float32 vBF images and the number of frames placed. Ids past the scan (a
    repeated scan) are left out: the first pass is used."""
    n_pos = scan[0] * scan[1]
    stack = np.zeros((len(groups.starts), n_pos), np.float32)
    placed = 0
    for ids, frames in batches(1):
        keep = ids < n_pos
        if not keep.any():
            continue
        vals = frames.reshape(len(frames), -1)[keep][:, groups.pix].astype(np.float32)
        stack[:, ids[keep]] = np.add.reduceat(vals, groups.starts, axis=1).T
        placed += int(keep.sum())
    return stack.reshape(-1, *scan), placed


# ── 3. shifts by cross-correlation ─────────────────────────────────────────────

def _tukey(n: int, alpha: float = 0.25) -> np.ndarray:
    x = np.linspace(0, 1, n)
    w = np.ones(n)
    edge = x < alpha / 2
    w[edge] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[edge] - alpha / 2)))
    edge = x > 1 - alpha / 2
    w[edge] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[edge] - 1 + alpha / 2)))
    return w


def _freqs(shape, device):
    qy = torch.fft.fftfreq(shape[0], device=device, dtype=torch.float64).view(-1, 1)
    qx = torch.fft.fftfreq(shape[1], device=device, dtype=torch.float64).view(1, -1)
    return qy, qx


def _ramp(d: torch.Tensor, qy, qx) -> torch.Tensor:
    """exp(+2πi q·d) per image: multiplying the spectrum by it moves an image by −d."""
    return (torch.exp(2j * np.pi * qy[None] * d[:, 0, None, None])
            * torch.exp(2j * np.pi * qx[None] * d[:, 1, None, None])).to(torch.complex64)


def normalized(stack: np.ndarray, device="cpu") -> torch.Tensor:
    """Each vBF image as relative contrast, I / ⟨I⟩ − 1."""
    t = torch.as_tensor(stack, dtype=torch.float32, device=device)
    return t / t.mean(dim=(1, 2), keepdim=True).clamp_min(1e-12) - 1


def inliers(resid: np.ndarray, k: float = 4.0) -> np.ndarray:
    """Rows whose residual norm is within k robust standard deviations (1.4826·MAD) of zero:
    a group whose correlation peak was lost in the noise lands far off the model."""
    r = np.hypot(resid[:, 0], resid[:, 1])
    scale = 1.4826 * np.median(r) + 1e-12
    return r <= k * scale


def _affine_prediction(kappa: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The trimmed affine fit d ≈ M·κ + t, evaluated at every κ."""
    a = np.concatenate([kappa, np.ones((len(kappa), 1))], axis=1)
    keep = np.ones(len(a), bool)
    for _ in range(3):
        x, *_ = np.linalg.lstsq(a[keep], d[keep], rcond=None)
        keep = inliers(d - a @ x)
    return a @ x


def measure_shifts(images: torch.Tensor, kappa: np.ndarray | None = None,
                   lowpass: float | None = None, tol: float = 0.01, max_iterations: int = 30,
                   max_shift: int | None = None, refine: int = 3) -> tuple[np.ndarray, int]:
    """(G, 2) displacement (rows, cols; scan px) of each image relative to the common reference,
    image_g(r) ≈ reference(r − d_g), and the iterations used.

    Each iteration moves every image back by its current estimate, correlates it (windowed, and
    low-passed at `lowpass` cycles per scan px) with the mean of the moved images, and adds the
    residual displacement. The window pulls a correlation peak towards zero in proportion to the
    displacement left, so the estimates converge to unbiased ones. With `kappa` (each image's
    tilt) the images are moved by the affine fit of the current estimates rather than by the
    estimates themselves, as quantem's `regularize_shifts` does: the reference is sharp from the
    second iteration on, instead of slowly sharpening from the plain sum, and each residual peak
    is then looked for within `refine` px of the fit. Stops when no estimate changes by more than
    `tol` px."""
    g, ny, nx = images.shape
    dev = images.device
    m = max_shift if max_shift is not None else min(ny, nx) // 4
    m = int(min(m, ny // 2 - 2, nx // 2 - 2))
    win = torch.as_tensor(np.outer(_tukey(ny), _tukey(nx)), dtype=torch.float32, device=dev)
    qy, qx = _freqs((ny, nx), dev)
    lp = 1.0
    if lowpass:
        lp = torch.exp(-0.5 * (qy**2 + qx**2) / (0.5 * lowpass) ** 2).to(torch.float32)
    spec = torch.fft.fft2(images)
    d = np.zeros((g, 2))
    cy, cx = ny // 2, nx // 2
    n = torch.arange(g, device=dev)

    def vertex(lo, c0, hi):
        den = lo - 2 * c0 + hi
        return torch.where(den.abs() > 1e-12, 0.5 * (lo - hi) / den,
                           torch.zeros_like(den)).clamp(-0.5, 0.5)

    for it in range(1, max(1, max_iterations) + 1):
        regularized = kappa is not None and it > 1
        align = _affine_prediction(kappa, d) if regularized else d
        m = min(refine, m) if regularized else m
        moved = torch.fft.ifft2(spec * _ramp(torch.as_tensor(align, device=dev), qy, qx)).real
        a = torch.fft.fft2(moved * win) * lp
        b = torch.fft.fft2(moved.mean(0) * win) * lp
        cc = torch.fft.fftshift(torch.fft.ifft2(a * b.conj()[None]).real, dim=(-2, -1))
        k = cc[:, cy - m:cy + m + 1, cx - m:cx + m + 1].reshape(g, -1).argmax(dim=1)
        iy, ix = k // (2 * m + 1) + cy - m, k % (2 * m + 1) + cx - m
        c0 = cc[n, iy, ix]
        dy = vertex(cc[n, iy - 1, ix], c0, cc[n, iy + 1, ix])
        dx = vertex(cc[n, iy, ix - 1], c0, cc[n, iy, ix + 1])
        resid = torch.stack([(iy - cy).double() + dy.double(),
                             (ix - cx).double() + dx.double()], 1).cpu().numpy()
        new = align + resid
        new -= new.mean(0)                                            # keep the reference centred
        change = np.abs(new - d).max()
        d = new
        if change < tol:
            break
    return d, it


# ── 4. the shift model ─────────────────────────────────────────────────────────

def _rot(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def shift_matrix(c10: float, rotation_deg: float, flip_cols: bool, c12: float = 0.0,
                 phi12_deg: float = 0.0) -> np.ndarray:
    """M (Å per rad of tilt) such that the engine's parallax shift of the bright-field pixel at
    raw-frame offset (u, v) px from the centre is M · (u, v) · angular_sampling (Å). Matches
    `preprocess_geometry`: canonical tilt (u, ±v), rotated by R(θ), times the symmetric
    aberration matrix C10·I + C12·Q(φ12)."""
    p = 2 * math.radians(phi12_deg)
    s = c10 * np.eye(2) + c12 * np.array([[math.cos(p), math.sin(p)], [math.sin(p), -math.cos(p)]])
    return s @ _rot(rotation_deg) @ np.diag([1.0, -1.0 if flip_cols else 1.0])


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def decompose(M: np.ndarray, rotation_prior: float | None = None) -> dict:
    """(C10, C12, φ12, θ, flip) from a fitted M, choosing between the twins (S, θ) and
    (−S, θ + 180°) by `rotation_prior` (degrees) or, without one, |θ| ≤ 90°."""
    flip = bool(np.linalg.det(M) < 0)
    n = M @ np.diag([1.0, -1.0 if flip else 1.0])
    w, sig, vt = np.linalg.svd(n)
    p = w @ np.diag(sig) @ w.T
    u = w @ vt
    theta = math.degrees(math.atan2(u[1, 0], u[0, 0]))
    twins = [(p, _wrap_deg(theta)), (-p, _wrap_deg(theta + 180.0))]
    if rotation_prior is None:
        s, theta = min(twins, key=lambda c: abs(c[1]))
    else:
        s, theta = min(twins, key=lambda c: abs(_wrap_deg(c[1] - rotation_prior)))
    c10 = 0.5 * float(np.trace(s))
    dev = s - c10 * np.eye(2)
    c12 = float(math.hypot(dev[0, 0], dev[0, 1]))
    phi12 = 0.5 * math.degrees(math.atan2(dev[0, 1], dev[0, 0])) if c12 > 0 else 0.0
    return {"c10": c10, "c12": c12, "phi12_deg": phi12, "rotation_deg": theta, "flip_cols": flip}


def fit_shifts(kappa: np.ndarray, shifts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares s = M·κ + t: (M, t, residuals)."""
    a = np.concatenate([kappa, np.ones((len(kappa), 1))], axis=1)
    x, *_ = np.linalg.lstsq(a, shifts, rcond=None)
    return x[:2].T, x[2], shifts - a @ x


def fit_geometry(kappa: np.ndarray, shifts: np.ndarray,
                 rotation_prior: float | None = None) -> dict:
    """The parallax geometry (C10, C12, φ12, θ, flip, and M): a trimmed affine fit of the shifts,
    decomposed. `inliers` marks the groups kept, `resid` the residuals of all of them."""
    keep = np.ones(len(kappa), bool)
    for _ in range(3):
        M, t, _ = fit_shifts(kappa[keep], shifts[keep])
        resid = shifts - kappa @ M.T - t
        keep = inliers(resid)
    return {**decompose(M, rotation_prior), "M": M.tolist(), "inliers": keep, "resid": resid}


def _higher_order_basis(a: np.ndarray) -> np.ndarray:
    """(G, 2, 5) shift fields of unit C30, C21 (x, y) and C23 (cos, sin) at scan-frame tilts `a`
    (rad): the gradients (Å) of α⁴/4, α³cos(φ−φ21)/3 and α³cos 3(φ−φ23)/3."""
    ax, ay = a[:, 0], a[:, 1]
    a2 = ax**2 + ay**2
    z2 = (ax + 1j * ay) ** 2
    cols = [(a2 * ax, a2 * ay),                                   # C30
            ((2 * ax * ax + a2) / 3, 2 * ax * ay / 3),            # C21 along x
            (2 * ax * ay / 3, (2 * ay * ay + a2) / 3),            # C21 along y
            (z2.real, -z2.imag), (z2.imag, z2.real)]              # C23 cos, sin
    return np.stack([np.stack(c, axis=1) for c in cols], axis=2)


def higher_orders(kappa: np.ndarray, shifts: np.ndarray, geometry: dict) -> dict:
    """A diagnostic, not a correction: the shifts of the groups `geometry` kept, fitted with a
    general linear term plus the gradients of C30, C21 and C23 in its scan frame.

    At low spatial frequency a vBF image moves by ∇χ(k)/2π whatever χ is, so this measures how far
    the surface departs from a quadratic one (probe aberrations, or a radial distortion of the
    diffraction pattern on the detector, which looks the same). Parallax cannot use it: it keeps
    χ(k ± q) to first order in q, and only for quadratic χ is the dropped second-order term,
    ½ qᵀH q, the same for every bright-field pixel. With C30, C21 or C23 the Hessian H(k) varies
    over the disk, so no shift aligns the vBF images at higher q (Varnavides et al., Microsc.
    Microanal. 32(2) ozaf139, eq. 9 and Fig. 2); direct ptychography handles them."""
    keep = geometry["inliers"]
    b = _rot(geometry["rotation_deg"]) @ np.diag([1.0, -1.0 if geometry["flip_cols"] else 1.0])
    a = kappa[keep] @ b.T
    g = len(a)
    cols = []
    for i in range(2):                                             # general linear (4)
        for j in range(2):
            c = np.zeros((g, 2))
            c[:, i] = a[:, j]
            cols.append(c)
    cols += list(np.moveaxis(_higher_order_basis(a), 2, 0))         # 5 higher-order fields
    cols += [np.tile([1.0, 0.0], (g, 1)), np.tile([0.0, 1.0], (g, 1))]   # translation
    design = np.stack([c.reshape(-1) for c in cols], axis=1)
    x, *_ = np.linalg.lstsq(design, shifts[keep].reshape(-1), rcond=None)
    resid = (shifts[keep].reshape(-1) - design @ x).reshape(-1, 2)
    lin = decompose(x[:4].reshape(2, 2) @ b, geometry["rotation_deg"])
    c30, c21x, c21y, c23c, c23s = x[4:9]
    return {"c10": lin["c10"], "c12": lin["c12"], "phi12_deg": lin["phi12_deg"],
            "rotation_deg": lin["rotation_deg"], "flip_cols": lin["flip_cols"],
            "c30": float(c30), "c21": float(math.hypot(c21x, c21y)),
            "phi21_deg": math.degrees(math.atan2(c21y, c21x)),
            "c23": float(math.hypot(c23c, c23s)),
            "phi23_deg": math.degrees(math.atan2(c23s, c23c)) / 3,
            "resid": resid}


def model_shifts(ab: dict, kappa: np.ndarray, **override) -> np.ndarray:
    """The shift field (Å, no translation) of `ab` (a `fit_geometry` or `higher_orders` result;
    any entry overridden by keyword) at raw-frame tilts `kappa`: S·a plus any higher orders at a,
    with a = R(θ)·diag(1, ±1)·κ the tilt in the scan frame."""
    ab = {**ab, **override}
    a = kappa @ (_rot(ab["rotation_deg"]) @ np.diag([1.0, -1.0 if ab["flip_cols"] else 1.0])).T
    p = 2 * math.radians(ab["phi12_deg"])
    sym = ab["c10"] * np.eye(2) + ab["c12"] * np.array([[math.cos(p), math.sin(p)],
                                                      [math.sin(p), -math.cos(p)]])
    p21, p23 = math.radians(ab.get("phi21_deg", 0.0)), 3 * math.radians(ab.get("phi23_deg", 0.0))
    c21, c23 = ab.get("c21", 0.0), ab.get("c23", 0.0)
    coef = np.array([ab.get("c30", 0.0), c21 * math.cos(p21), c21 * math.sin(p21),
                     c23 * math.cos(p23), c23 * math.sin(p23)])
    return a @ sym.T + _higher_order_basis(a) @ coef


# ── 5. the variance loss ───────────────────────────────────────────────────────

def variance_loss(images: torch.Tensor, shift_fields: list[np.ndarray], scan_sampling: float,
                  lowpass: float | None = None, spec: torch.Tensor | None = None) -> np.ndarray:
    """For each (G, 2) field of engine shifts (Å): the spread across the vBF images moved by them
    (sub-pixel, in Fourier space; low-passed at `lowpass` 1/Å), relative to their total variance,
    over the part of the scan that no shift wraps into. Smallest where the shifts align the
    images; 1 − loss is the "bright-field consistency" (quantem's `residual_loss`)."""
    g, ny, nx = images.shape
    dev = images.device
    qy, qx = _freqs((ny, nx), dev)
    spec = torch.fft.fft2(images) if spec is None else spec
    if lowpass:
        sigma = 0.5 * lowpass * scan_sampling                   # cycles per scan px
        spec = spec * torch.exp(-0.5 * (qy**2 + qx**2) / sigma**2).to(torch.float32)
    out = []
    for field_ in shift_fields:
        sh = np.asarray(field_) / scan_sampling
        sh = sh - sh.mean(0)
        mrg = int(math.ceil(np.abs(sh).max())) + 2
        if 2 * mrg >= min(ny, nx) - 4:
            out.append(np.nan)
            continue
        moved = torch.fft.ifft2(spec * _ramp(torch.as_tensor(-sh, device=dev), qy, qx)).real
        a = moved[:, mrg:ny - mrg, mrg:nx - mrg]
        out.append(float(a.var(dim=0, unbiased=False).mean() / a.var(unbiased=False)))
    return np.asarray(out)


# ── the whole calibration ──────────────────────────────────────────────────────

_GEOMETRY = ("c10", "c12", "phi12_deg", "rotation_deg", "flip_cols", "M")
_HIGHER = ("c10", "c12", "phi12_deg", "rotation_deg", "flip_cols", "c30", "c21", "phi21_deg",
           "c23", "phi23_deg")


@dataclass
class Calibration:
    disk: Disk
    scan: tuple[int, int]
    energy: float
    semiangle: float
    angular_sampling: float
    scan_sampling: float
    geometry: dict          # C10, C12/φ12, θ, flip (and M): the parallax geometry run.py takes
    higher_orders: dict     # diagnostic only (`higher_orders`): C10 with C30, C21/φ21, C23/φ23
    sigma: dict             # bootstrap standard errors over the groups
    shift_rms_px: float     # rms residual of the quadratic model over the kept groups, scan px
    higher_order_rms_px: float   # the same with the higher orders
    n_groups: int           # groups whose shifts were measured
    outliers: int           # of those, left out of the fit (correlation peak lost in the noise)
    group_size: int
    frames: int
    lowpass: float          # 1/Å, the highest frequency the shifts are measured from
    margin_px: float        # pixels closer than this to the disk edge are not used
    iterations: int
    loss: dict = field(default_factory=dict)
    refine: dict = field(default_factory=dict)    # C10 on the lattice (`refine_c10`)
    timings: dict = field(default_factory=dict)

    @property
    def detector_shape(self) -> tuple[int, int]:
        return self.disk.shape

    @property
    def depth_of_field(self) -> float:
        return _electron_wavelength(self.energy) / (self.semiangle * 1e-3) ** 2

    def astigmatism_significant(self) -> bool:
        return self.geometry["c12"] > 3 * self.sigma.get("c12", 0.0)

    @property
    def c10(self) -> float:
        """The C10 `run.py` gets: refined on the lattice when that found a clear maximum, else
        the shift fit's."""
        return self.refine["best_c10"] if self.refine.get("used") else self.geometry["c10"]

    def flags(self) -> list[str]:
        """The `run.py` flags: the acquisition values given plus the geometry measured."""
        det = self.detector_shape
        g = self.geometry
        f = ["--binned"] if tuple(det) == (96, 96) else ["--detector", str(det[0]), str(det[1])]
        f += ["--scan", str(self.scan[0]), str(self.scan[1]), "--energy", f"{self.energy:g}",
              "--semiangle", f"{self.semiangle:g}", "--angular-sampling",
              f"{self.angular_sampling:g}", "--scan-sampling", f"{self.scan_sampling:g}",
              f"--defocus={self.c10:.0f}", f"--rotation={g['rotation_deg']:.2f}",
              "--center", f"{self.disk.center[0]:.2f}", f"{self.disk.center[1]:.2f}"]
        if g["flip_cols"]:
            f.append("--flip-cols")
        if self.astigmatism_significant():
            f += [f"--c12={g['c12']:.0f}", f"--phi12={g['phi12_deg']:.1f}"]
        return f


def calibrate(batches: Batches, scan: tuple[int, int], energy: float, semiangle: float,
              angular_sampling: float, scan_sampling: float, *,
              rotation_prior: float | None = None, group: int | None = None,
              mean_every: int = 8, lowpass: float | None = None, tol: float = 0.01,
              max_iterations: int = 30, loss: bool = True, bootstrap: int = 200,
              refine: bool = True, refine_band: tuple[float, float] | None = None,
              refine_span: float | None = None, device: str = "cpu", log=print) -> Calibration:
    """The whole calibration (module docstring). `lowpass` (1/Å) limits the spatial frequencies
    the shifts are measured from, and with them the band at the disk edge whose pixels are not
    used (`bf_groups`); by default 15% of the disk radius, at most the scan's Nyquist. `refine`:
    C10 on the lattice (`refine_c10`, with `refine_band` and `refine_span`)."""
    t0 = time.perf_counter()
    mean, n_mean, top = mean_pattern(batches, mean_every)
    disk = fit_disk(mean, top, n_mean)
    t1 = time.perf_counter()
    dk = angular_sampling * 1e-3 / _electron_wavelength(energy)      # 1/Å per detector px
    if lowpass is None:
        lowpass = 0.15 * disk.radius * dk
    lowpass = min(lowpass, 0.5 / scan_sampling)
    margin = 1.0 + lowpass / dk
    # by default about 256 groups and, for counted data, each collecting at least ~1e6 electrons
    # over the scan, enough for a clean correlation peak (binned Gd2O3 scan: 3x3 px, ~2e6)
    min_px = 1
    if np.isfinite(top):
        per_px = float(np.median(mean[_disk_mask(disk)])) * scan[0] * scan[1]
        min_px = math.ceil(1e6 / max(per_px, 1.0))
    groups = bf_groups(disk, group, margin=margin, min_px=min_px)
    if len(groups.starts) < 12:
        raise ValueError(f"only {len(groups.starts)} groups lie {margin:.1f} px inside the disk "
                         f"edge: lower --lowpass or --group")
    stack, placed = vbf_stack(batches, groups, tuple(scan))
    if placed < 0.9 * scan[0] * scan[1]:
        log(f"[calibrate] WARNING: only {placed} of {scan[0] * scan[1]} scan positions have a frame")
    t2 = time.perf_counter()
    images = normalized(stack, device)
    kappa = groups.offsets * angular_sampling * 1e-3                 # rad
    d, iterations = measure_shifts(images, kappa, lowpass=lowpass * scan_sampling, tol=tol,
                                   max_iterations=max_iterations)
    shifts = -d * scan_sampling                                      # engine shifts, Å
    geo = fit_geometry(kappa, shifts, rotation_prior)
    ho = higher_orders(kappa, shifts, geo)
    # bootstrap over the groups (each resample keeps the twin of the chosen rotation)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(bootstrap):
        i = rng.integers(0, len(kappa), len(kappa))
        b = fit_geometry(kappa[i], shifts[i], geo["rotation_deg"])
        boot.append({**{k: b[k] for k in ("c10", "c12", "rotation_deg")},
                     "c30": higher_orders(kappa[i], shifts[i], b)["c30"]})
    sigma = {k: float(np.std([b[k] for b in boot])) for k in boot[0]} if boot else {}
    t3 = time.perf_counter()
    keep = geo["inliers"]
    rms = lambda r: float(np.sqrt((r ** 2).sum(1).mean()) / scan_sampling)   # noqa: E731
    cal = Calibration(disk=disk, scan=tuple(scan), energy=energy, semiangle=semiangle,
                      angular_sampling=angular_sampling, scan_sampling=scan_sampling,
                      geometry={k: geo[k] for k in _GEOMETRY},
                      higher_orders={k: ho[k] for k in _HIGHER}, sigma=sigma,
                      shift_rms_px=rms(geo["resid"][keep]), higher_order_rms_px=rms(ho["resid"]),
                      n_groups=len(kappa), outliers=int((~keep).sum()), group_size=groups.size,
                      frames=placed, lowpass=lowpass, margin_px=margin, iterations=iterations)
    if loss:
        cal.loss = loss_scan(cal, images, kappa)
    t4 = time.perf_counter()
    if refine:
        cal.refine = refine_c10(batches, cal, refine_band, refine_span, device=device)
    cal.timings = {"mean_pattern_s": t1 - t0, "vbf_stack_s": t2 - t1, "shifts_fit_s": t3 - t2,
                   "loss_s": t4 - t3, "refine_s": time.perf_counter() - t4}
    cal._mean, cal._images, cal._kappa, cal._shifts, cal._keep = mean, images, kappa, shifts, keep
    cal._t = shifts[keep].mean(0) - model_shifts(cal.geometry, kappa[keep]).mean(0)
    return cal


def loss_scan(cal: Calibration, images: torch.Tensor, kappa: np.ndarray) -> dict:
    """An image-based check of the measurement: the variance loss of the groups (low-passed like
    the shifts) for the geometry along C10 and θ, for the other handedness at every rotation,
    without shifts, and with the higher orders. The minima along C10 and θ should sit at the
    geometry measured."""
    spec = torch.fft.fft2(images)
    g = cal.geometry
    span = max(6 * cal.depth_of_field, 0.5 * abs(g["c10"]))
    c10s = g["c10"] + np.linspace(-span, span, 31)
    rots = g["rotation_deg"] + np.linspace(-8, 8, 33)
    other = np.arange(-180, 180, 10.0)
    vl = lambda fields: variance_loss(images, fields, cal.scan_sampling,   # noqa: E731
                                      cal.lowpass, spec)
    l_c10 = vl([model_shifts(g, kappa, c10=c) for c in c10s])
    l_rot = vl([model_shifts(g, kappa, rotation_deg=r) for r in rots])
    l_other = vl([model_shifts(g, kappa, rotation_deg=r, flip_cols=not g["flip_cols"])
                  for r in other])
    fit, higher, none = vl([model_shifts(g, kappa), model_shifts(cal.higher_orders, kappa),
                            np.zeros_like(kappa)])
    return {"geometry": float(fit), "with_higher_orders": float(higher), "no_shift": float(none),
            "c10": c10s.tolist(), "loss_c10": l_c10.tolist(),
            "rotation": rots.tolist(), "loss_rotation": l_rot.tolist(),
            "other_rotation": other.tolist(), "loss_other_handedness": l_other.tolist(),
            "best_c10": float(c10s[np.nanargmin(l_c10)]),
            "best_rotation": float(rots[np.nanargmin(l_rot)]),
            "best_other_handedness": float(np.nanmin(l_other))}


# ── 7. C10 on the lattice ──────────────────────────────────────────────────────

def engine_geometry(cal: Calibration, c10: float | None = None, upsampling: int = 1):
    """The engine's `ParallaxGeometry` for the calibrated geometry, as its `run.py` flags build it
    (C12 only when significant), with C10 replaced by `c10`."""
    g = cal.geometry
    c12 = g["c12"] if cal.astigmatism_significant() else 0.0
    return preprocess_geometry(
        det_shape=tuple(cal.detector_shape), scan_shape=tuple(cal.scan),
        scan_sampling=(cal.scan_sampling, cal.scan_sampling), energy=cal.energy,
        semiangle_cutoff=cal.semiangle,
        angular_sampling=(cal.angular_sampling, cal.angular_sampling),
        aberration_coefs={"C10": g["c10"] if c10 is None else c10, "C12": c12,
                          "phi12": math.radians(g["phi12_deg"])},
        rotation_angle=math.radians(g["rotation_deg"]), upsampling_factor=upsampling,
        probe_center=tuple(cal.disk.center), flip_cols=g["flip_cols"])


def bright_field_values(batches: Batches, pixels: np.ndarray, scan: tuple[int, int],
                        edge: int = 128) -> tuple[tuple[int, int], np.ndarray]:
    """The bright-field `pixels` (flat indices into a frame) of the frames in the central
    ≤ edge × edge positions of the scan, each pixel's mean over them removed (the readout's DC
    correction on a complete scan): ((ny, nx) of the region, (ny·nx, M) float32)."""
    ny, nx = scan
    edge = max(16, min(edge, int(math.sqrt(5e7 / max(len(pixels), 1)))))   # ≤ 200 MB
    cy, cx = min(ny, edge), min(nx, edge)
    y0, x0 = (ny - cy) // 2, (nx - cx) // 2
    vals = np.zeros((cy * cx, len(pixels)), np.float32)
    seen = np.zeros(cy * cx, bool)
    for ids, frames in batches(1):
        keep = ids < ny * nx
        iy, ix = np.divmod(ids[keep], nx)
        inside = (iy >= y0) & (iy < y0 + cy) & (ix >= x0) & (ix < x0 + cx)
        if inside.any():
            loc = (iy[inside] - y0) * cx + (ix[inside] - x0)
            vals[loc] = frames.reshape(len(frames), -1)[keep][inside][:, pixels]
            seen[loc] = True
    vals[seen] -= vals[seen].mean(0)
    return (cy, cx), vals


def _deposit(vals: torch.Tensor, region: tuple[int, int], shifts: np.ndarray, upsampling: int,
             chunk: int = 2048) -> torch.Tensor:
    """The engine's deposit: each position's bright-field pixels `vals` at their integer
    `shifts` on the ×`upsampling` grid, wrapped on the region's torus."""
    cy, cx = region
    nyu, nxu = cy * upsampling, cx * upsampling
    dev = vals.device
    sy = torch.as_tensor(shifts[:, 0], dtype=torch.long, device=dev)
    sx = torch.as_tensor(shifts[:, 1], dtype=torch.long, device=dev)
    out = torch.zeros(nyu * nxu, dtype=torch.float32, device=dev)
    for s in range(0, cy * cx, chunk):
        pos = torch.arange(s, min(s + chunk, cy * cx), device=dev)
        oy = ((pos // cx) * upsampling)[:, None] + sy[None, :]
        ox = ((pos % cx) * upsampling)[:, None] + sx[None, :]
        out.index_add_(0, ((oy % nyu) * nxu + (ox % nxu)).reshape(-1),
                       vals[s:s + len(pos)].reshape(-1))
    return out.view(nyu, nxu)


def _lattice_prominence(vals: torch.Tensor, region: tuple[int, int], shifts: np.ndarray,
                        upsampling: int, window: torch.Tensor, rings: torch.Tensor,
                        n_rings: int, top: float = 1e-3) -> float:
    """How far the sharpest spots of the reconstruction's power spectrum stand above its diffuse
    background in the band: the mean of the top `top` fraction of the band's pixels, each divided
    by its ring's median power (`rings`: ring index per pixel, −1 outside the band).

    Diffuse power (shot noise, amorphous detail, aliases of the low frequencies left by
    misalignment) has speckle statistics, whose top/median ratio is the same at any level: only
    sharp spots, a lattice's reflections, raise it. Each output pixel is divided by the number of
    bright-field pixels in its sub-pixel phase class (the U×U classes the shifts fall in modulo U),
    so uneven coverage of the classes (e.g. all in one at C10 ≈ 0) adds no comb. NaN when a class
    is empty."""
    u = upsampling
    cy, cx = region
    count = np.bincount((shifts[:, 0] % u) * u + (shifts[:, 1] % u), minlength=u * u)
    if (count == 0).any():
        return float("nan")
    yy, xx = np.indices((cy * u, cx * u))
    norm = torch.as_tensor((1.0 / count[(yy % u) * u + (xx % u)]).astype(np.float32),
                           device=vals.device)
    spec = torch.fft.fft2(_deposit(vals, region, shifts, u) * norm * window)
    p = (spec.real ** 2 + spec.imag ** 2)[rings >= 0]
    r = rings[rings >= 0]
    med = torch.stack([p[r == k].median() for k in range(n_rings)])
    rel = p / med[r].clamp_min(1e-30)
    k = max(8, int(top * rel.numel()))
    return float(torch.topk(rel, k).values.mean())


def refine_c10(batches: Batches, cal: Calibration, band: tuple[float, float] | None = None,
               span: float | None = None, upsampling: int = 4, edge: int = 128,
               device: str = "cpu") -> dict:
    """C10 where a lattice is sharpest. The shifts are measured below `cal.lowpass`, where
    whatever dominates the low frequencies sets the focus: on a support film with particles, the
    film. Particles sit about a radius off it, and their lattice is what upsampling is for. So:
    reconstruct the central ≤ edge² positions at ×`upsampling` (the engine's own geometry and
    rounding) for C10 over the fit ± `span` (on the fit's side of zero), and keep the C10 whose
    power spectrum shows the most prominent lattice spots (`_lattice_prominence`) in `band` (1/Å;
    default from the scan's Nyquist frequency to 0.95× the scan frequency, what only upsampling
    recovers), less disks about the scan lattice's reciprocal points (±1/step), where the
    strongest aliases of the low frequencies land. Coarse steps of one depth of field λ/α², then
    quarter steps about the best.

    What it maximizes is the lattice contrast of the reconstruction. Reconstructed at C10′, the
    image transfers frequency q about as sin(πλC10′q²), so the best C10′ for a lattice can lie
    up to ~half a depth of field from its plane, on the side where its reflections transfer
    better.

    `used` is false when the maximum lies at the end of the range (widen `span`) or no C10 stands
    out (no lattice above the diffuse background, e.g. an amorphous specimen): `--defocus` then
    keeps the shift fit."""
    g = cal.geometry
    step = cal.scan_sampling
    lo, hi = band if band is not None else (0.5 / step, 0.95 / step)
    dof = cal.depth_of_field
    fit = g["c10"]
    span = span if span is not None else max(0.5 * abs(fit), 10 * dof)
    base = engine_geometry(cal, upsampling=upsampling)
    region, vals = bright_field_values(batches, base.bf_flat_inds, cal.scan, edge)
    vals = torch.as_tensor(vals, device=device)
    nyu, nxu = region[0] * upsampling, region[1] * upsampling
    fy = np.fft.fftfreq(nyu, step / upsampling)[:, None]
    fx = np.fft.fftfreq(nxu, step / upsampling)[None, :]
    q = np.hypot(fy, fx)
    keep = (q >= lo) & (q < hi)
    r_alias = max(cal.lowpass, 0.15 / step)
    for ky in (-1, 0, 1):
        for kx in (-1, 0, 1):
            if ky or kx:
                keep &= np.hypot(fy - ky / step, fx - kx / step) > r_alias
    n_rings = 16
    ring = np.minimum(((q - lo) / (hi - lo) * n_rings).astype(int), n_rings - 1)
    ring = np.where(keep, ring, -1)
    present = np.unique(ring[ring >= 0])
    ring = np.where(ring >= 0, np.searchsorted(present, ring), -1)     # drop rings left empty
    rings = torch.as_tensor(ring, device=device)
    window = torch.as_tensor(np.outer(np.hanning(nyu), np.hanning(nxu)).astype(np.float32),
                             device=device)

    prominence: dict[float, float] = {}

    def at(c: float) -> float:
        c = round(float(c), 3)
        if c not in prominence:
            geo = engine_geometry(cal, c, upsampling)
            assert np.array_equal(geo.bf_flat_inds, base.bf_flat_inds)
            prominence[c] = _lattice_prominence(vals, region, geo.shifts, upsampling, window,
                                                rings, len(present))
        return prominence[c]

    n = int(math.ceil(span / dof))
    coarse = fit + dof * np.arange(-n, n + 1)
    coarse = coarse[coarse * np.sign(fit) > dof]                  # the fit's side of zero
    values = np.array([at(c) for c in coarse])
    best = coarse[int(np.nanargmax(values))]
    for c in best + dof / 4 * np.arange(-4, 5):
        at(c)
    cs = np.array(sorted(prominence))
    p = np.array([prominence[c] for c in cs])
    best = float(cs[np.nanargmax(p)])
    at_edge = best < coarse.min() + 0.5 * dof or best > coarse.max() - 0.5 * dof
    # stands out: the best prominence is well above the curve's typical (speckle-only) level
    clear = bool(np.nanmax(p) > 1.5 * np.nanmedian(p))
    at_fit = at(fit)
    return {"band": [lo, hi], "upsampling": upsampling, "region": list(region),
            "alias_radius": r_alias, "c10": cs.tolist(), "prominence": p.tolist(),
            "best_c10": best, "fit_c10": fit, "gain": float(np.nanmax(p) / at_fit),
            "at_edge": bool(at_edge), "clear": clear, "used": bool(clear and not at_edge)}


# ── report ─────────────────────────────────────────────────────────────────────

def report(cal: Calibration) -> list[str]:
    d, g, ho, sig = cal.disk, cal.geometry, cal.higher_orders, cal.sigma
    radius_mrad = d.radius * cal.angular_sampling
    edge_c30 = abs(ho["c30"]) * (cal.semiangle * 1e-3) ** 3              # Å at the disk edge
    lines = [
        f"[calibrate] {cal.frames} frames, detector {d.shape[0]}x{d.shape[1]}, scan "
        f"{cal.scan[0]}x{cal.scan[1]}",
        f"[calibrate] disk: centre ({d.center[0]:.2f}, {d.center[1]:.2f}) px, radius "
        f"{d.radius:.2f} px = {radius_mrad:.2f} mrad at {cal.angular_sampling:g} mrad/px "
        f"(--semiangle {cal.semiangle:g}: {100 * (radius_mrad / cal.semiangle - 1):+.1f}%)",
    ]
    for name, pixels in (("flagged (dtype maximum)", d.flagged), ("hot", d.hot)):
        if pixels:
            where = ", ".join(f"({q[0]},{q[1]}) {'IN the disk' if d.inside(q) else 'outside'}"
                              for q in pixels[:8])
            lines.append(f"[calibrate] {len(pixels)} {name} pixel(s): {where}"
                         + (" …" if len(pixels) > 8 else ""))
    lines += [
        f"[calibrate] shifts of {cal.n_groups} groups of {cal.group_size}x{cal.group_size} px at "
        f"least {cal.margin_px:.1f} px inside the edge, below {cal.lowpass:.3f} 1/Å "
        f"({1 / cal.lowpass:.1f} Å), {cal.iterations} iterations, {cal.outliers} outliers left "
        f"out; rms residual {cal.shift_rms_px:.3f} scan px",
        f"[calibrate] parallax geometry: C10 {g['c10']:.1f} ± {sig.get('c10', 0):.1f} Å, C12 "
        f"{g['c12']:.1f} ± {sig.get('c12', 0):.1f} Å at {g['phi12_deg']:.1f}°, rotation "
        f"{g['rotation_deg']:.2f} ± {sig.get('rotation_deg', 0):.2f}°, columns "
        f"{'flipped' if g['flip_cols'] else 'not flipped'} (depth of field λ/α² = "
        f"{cal.depth_of_field:.0f} Å)",
        f"[calibrate] the twin (C10 {-g['c10']:.1f} Å, rotation "
        f"{_wrap_deg(g['rotation_deg'] + 180):.2f}°) gives the same shifts",
        f"[calibrate] beyond quadratic (diagnostic; parallax corrects quadratic aberrations "
        f"only): with C30 {ho['c30'] / 1e4:.2f} ± {sig.get('c30', 0) / 1e4:.2f} µm "
        f"({edge_c30 / cal.scan_sampling:.2f} scan px of shift at {cal.semiangle:g} mrad), C21 "
        f"{ho['c21']:.0f} Å at {ho['phi21_deg']:.0f}°, C23 {ho['c23']:.0f} Å at "
        f"{ho['phi23_deg']:.0f}° and C10 {ho['c10']:.1f} Å the rms residual drops to "
        f"{cal.higher_order_rms_px:.3f} px (probe aberrations, or radial distortion of the "
        f"pattern on the detector)",
    ]
    if cal.loss:
        lo = cal.loss
        lines.append(
            f"[calibrate] variance loss of the groups: geometry {lo['geometry']:.4f}, no shifts "
            f"{lo['no_shift']:.4f}, other handedness at best {lo['best_other_handedness']:.4f}, "
            f"with the higher orders {lo['with_higher_orders']:.4f}; minimum along C10 at "
            f"{lo['best_c10']:.0f} Å, along θ at {lo['best_rotation']:.2f}°")
    if cal.refine:
        rf = cal.refine
        where = (f"C10 {rf['best_c10']:.0f} Å, {rf['gain']:.2f}× the lattice prominence at the "
                 f"shift fit's {rf['fit_c10']:.0f} Å")
        if rf["used"]:
            verdict = "--defocus uses it"
            if abs(rf["best_c10"] - rf["fit_c10"]) > 2 * cal.depth_of_field:
                verdict += (" (the low-frequency contrast the shifts measure, e.g. a support film, "
                            "focuses elsewhere)")
        elif rf["at_edge"]:
            verdict = "at the end of the range: --defocus keeps the shift fit; widen --refine-span"
        else:
            verdict = ("no C10 stands out (no lattice above the diffuse background): --defocus "
                       "keeps the shift fit")
        lines.append(
            f"[calibrate] focus on the lattice ({rf['band'][0]:.3f}–{rf['band'][1]:.3f} 1/Å, "
            f"×{rf['upsampling']} over {rf['region'][0]}x{rf['region'][1]} positions): {where}; "
            f"{verdict}")
    return lines


def plot(cal: Calibration, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4, figsize=(20, 9.5))
    d, g, ho = cal.disk, cal.geometry, cal.higher_orders
    ax[0, 0].imshow(np.log1p(np.clip(cal._mean, 0, None)), cmap="gray")
    ax[0, 0].add_patch(plt.Circle(d.center[::-1], d.radius, fill=False, color="tab:orange"))
    ax[0, 0].add_patch(plt.Circle(d.center[::-1], d.radius - cal.margin_px, fill=False,
                                  color="tab:orange", ls="--"))
    ax[0, 0].plot(d.center[1], d.center[0], "+", color="tab:orange")
    for q in d.flagged + d.hot:
        ax[0, 0].plot(q[1], q[0], "x", color="tab:red")
    ax[0, 0].set_title(f"mean frame (log); r = {d.radius:.2f} px, groups inside the dashed circle")

    keep = cal._keep
    kap = cal._kappa[keep]
    s = cal._shifts[keep] - cal._t                                    # without the translation
    fit = model_shifts(g, kap)
    off = kap / (cal.angular_sampling * 1e-3)
    ax[0, 1].quiver(off[:, 1], off[:, 0], s[:, 1], s[:, 0], color="k", angles="xy",
                    label="measured")
    ax[0, 1].quiver(off[:, 1], off[:, 0], fit[:, 1], fit[:, 0], color="tab:blue", angles="xy",
                    alpha=0.6, label="geometry")
    ax[0, 1].legend(loc="upper right")
    ax[0, 1].set_title("vBF shifts (Å) per group (raw-frame px)")

    r = (s - fit) / cal.scan_sampling
    sc = ax[0, 2].scatter(off[:, 1], off[:, 0], c=np.hypot(r[:, 0], r[:, 1]), cmap="viridis")
    fig.colorbar(sc, ax=ax[0, 2], label="scan px")
    ax[0, 2].set_title(f"residual of the geometry (rms {cal.shift_rms_px:.3f} px)")
    for a in (ax[0, 1], ax[0, 2]):
        a.invert_yaxis()
        a.set_aspect("equal")

    # radial shift per radian of tilt, s·a / |a|²: flat for a quadratic surface
    b = _rot(g["rotation_deg"]) @ np.diag([1.0, -1.0 if g["flip_cols"] else 1.0])
    a = kap @ b.T
    a2 = (a ** 2).sum(1)
    ok = a2 > (2 * cal.angular_sampling * 1e-3) ** 2
    ax[0, 3].plot(np.sqrt(a2[ok]) * 1e3, (s[ok] * a[ok]).sum(1) / a2[ok], ".", ms=3, color="k",
                  label="measured")
    al = np.linspace(0, cal.semiangle, 50)
    ax[0, 3].axhline(g["c10"], color="tab:blue", ls="--", label="geometry C10")
    ax[0, 3].plot(al, ho["c10"] + ho["c30"] * (al * 1e-3) ** 2, color="tab:orange",
                  label="C10 + C30 α² (diagnostic)")
    ax[0, 3].set_xlabel("tilt α (mrad)")
    ax[0, 3].set_ylabel("s·α / α² (Å)")
    ax[0, 3].legend()
    ax[0, 3].set_title("radial shift per tilt")

    if cal.loss:
        lo = cal.loss
        ax[1, 0].plot(lo["c10"], lo["loss_c10"], ".-")
        ax[1, 0].axvline(g["c10"], color="tab:blue", ls="--")
        ax[1, 0].set_xlabel("C10 (Å)")
        ax[1, 0].set_ylabel("variance loss (groups)")
        ax[1, 0].set_title("along C10")
        ax[1, 1].plot(lo["rotation"], lo["loss_rotation"], ".-")
        ax[1, 1].axvline(g["rotation_deg"], color="tab:blue", ls="--")
        ax[1, 1].set_xlabel("rotation (°)")
        ax[1, 1].set_title("along θ")
        ax[1, 2].plot(lo["other_rotation"], lo["loss_other_handedness"], ".-", color="gray")
        ax[1, 2].axhline(lo["geometry"], color="tab:blue", ls="--", label="this handedness")
        ax[1, 2].legend()
        ax[1, 2].set_xlabel("rotation (°)")
        ax[1, 2].set_title("the other handedness, all rotations")
    if cal.refine:
        rf = cal.refine
        a = ax[1, 0].twinx() if cal.loss else ax[1, 0]
        a.plot(rf["c10"], rf["prominence"], ".-", color="tab:orange")
        a.axvline(rf["best_c10"], color="tab:orange", ls="--")
        a.set_ylabel(f"lattice prominence {rf['band'][0]:.2f}–{rf['band'][1]:.2f} 1/Å",
                     color="tab:orange")
        ax[1, 0].set_xlabel("C10 (Å)")
        ax[1, 0].set_title("along C10: loss (blue) and lattice prominence (orange)")

    images = cal._images
    qy, qx = _freqs(images.shape[1:], images.device)
    sh = torch.as_tensor(-model_shifts(g, cal._kappa) / cal.scan_sampling, device=images.device)
    aligned = torch.fft.ifft2(torch.fft.fft2(images) * _ramp(sh, qy, qx)).real.mean(0)
    both = np.concatenate([images.mean(0).cpu().numpy(), aligned.cpu().numpy()], axis=1)
    lo_, hi_ = np.percentile(both, [0.5, 99.5])
    ax[1, 3].imshow(both, cmap="gray", vmin=lo_, vmax=hi_)
    ax[1, 3].set_title("vBF sum: unshifted | geometry shifts")
    for a in (ax[0, 0], ax[1, 3]):
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"C10 {g['c10']:.0f} Å (shift fit; --defocus {cal.c10:.0f} Å), "
                 f"θ {g['rotation_deg']:.2f}°, "
                 f"{'flipped' if g['flip_cols'] else 'unflipped'} columns, C12 {g['c12']:.0f} Å  "
                 f"|  diagnostic: C30 {ho['c30'] / 1e4:.2f} µm")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python -m tools.calibrate_parallax", description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("recording", type=Path,
                   help="A recorded scan: .dstream (tools/record_stream.py) or a .npy of frames "
                        "(..., H, W) in raster order.")
    req = p.add_argument_group("acquisition values (required, no defaults)")
    req.add_argument("--scan", type=int, nargs=2, required=True, metavar=("NY", "NX"),
                     help="Scan grid.")
    req.add_argument("--energy", type=float, required=True, metavar="EV", help="Beam energy (eV).")
    req.add_argument("--semiangle", type=float, required=True, metavar="MRAD",
                     help="Probe convergence semiangle (mrad).")
    req.add_argument("--angular-sampling", type=float, required=True, metavar="MRAD_PX",
                     help="Detector angular sampling of the recorded mode (mrad/px); scales the "
                          "measured C10 and C12.")
    req.add_argument("--scan-sampling", type=float, required=True, metavar="ANG",
                     help="Scan step (Å).")
    p.add_argument("--rotation-prior", type=float, default=None, metavar="DEG",
                   help="Report the twin solution whose rotation is nearest this (e.g. the "
                        "rotation used live). Default: the twin with |rotation| <= 90°.")
    p.add_argument("--group", type=int, default=None, metavar="PX",
                   help="Edge of the square groups of bright-field pixels whose vBF images are "
                        "aligned (default: about 256 groups).")
    p.add_argument("--mean-every", type=int, default=8, metavar="K",
                   help="Average every K-th frame for the mean diffraction pattern.")
    p.add_argument("--lowpass", type=float, default=None, metavar="Q",
                   help="Highest spatial frequency (1/Å) the shifts are measured from; pixels "
                        "within lowpass / Δk of the disk edge are not fitted. Default: 15%% of the "
                        "disk radius in 1/Å, at most the scan's Nyquist frequency.")
    p.add_argument("--tol", type=float, default=0.01, metavar="PX",
                   help="Stop the shift iterations when no shift changes by more than this.")
    p.add_argument("--max-iterations", type=int, default=30,
                   help="At most this many shift iterations.")
    p.add_argument("--no-loss", dest="loss", action="store_false",
                   help="Skip the variance-loss check along C10, rotation and handedness.")
    p.add_argument("--no-refine", dest="refine", action="store_false",
                   help="Keep the shift fit's C10 in --defocus: skip the refinement on the "
                        "lattice (the C10 that sharpens a particle's lattice rather than its "
                        "support film).")
    p.add_argument("--refine-band", type=float, nargs=2, default=None, metavar=("QMIN", "QMAX"),
                   help="Spatial frequencies (1/Å) where the refinement looks for lattice spots. "
                        "Default: "
                        "from the scan's Nyquist frequency to 0.95× the scan frequency (what only "
                        "upsampling recovers); e.g. a band about a lattice's reflections.")
    p.add_argument("--refine-span", type=float, default=None, metavar="ANG",
                   help="The refinement searches C10 within ± this of the shift fit (Å). "
                        "Default: half the fitted |C10|, at least 10 depths of field.")
    p.add_argument("--device", default="cpu",
                   help="Torch device for the cross-correlations, the loss and the refinement.")
    p.add_argument("--out", type=Path, default=None, metavar="PREFIX",
                   help="Write <prefix>.flags (run.py flags), <prefix>.json (all results) and "
                        "<prefix>.png (diagnostics, needs matplotlib).")
    return p


def main(argv: list[str] | None = None) -> Calibration:
    from realtime_stem.processing.parallax.config import join_negative_floats

    args = build_parser().parse_args(join_negative_floats(sys.argv[1:] if argv is None else argv))
    batches = file_batches(args.recording)
    cal = calibrate(batches, tuple(args.scan), args.energy, args.semiangle,
                    args.angular_sampling, args.scan_sampling,
                    rotation_prior=args.rotation_prior, group=args.group,
                    mean_every=args.mean_every, lowpass=args.lowpass, tol=args.tol,
                    max_iterations=args.max_iterations, loss=args.loss, refine=args.refine,
                    refine_band=tuple(args.refine_band) if args.refine_band else None,
                    refine_span=args.refine_span, device=args.device)
    for line in report(cal):
        print(line, flush=True)
    flags = " ".join(cal.flags())
    print(f"[calibrate] run.py flags: {flags}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".flags").write_text(flags + "\n")
        out = asdict(cal)
        out.update(recording=str(args.recording), flags=cal.flags(),
                   depth_of_field=cal.depth_of_field)
        args.out.with_suffix(".json").write_text(json.dumps(out, indent=2))
        try:
            plot(cal, args.out.with_suffix(".png"))
        except ImportError:
            print("[calibrate] matplotlib not installed: no plot", flush=True)
        print(f"[calibrate] wrote {args.out}.flags / .json / .png", flush=True)
    return cal


if __name__ == "__main__":
    main()
