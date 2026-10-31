"""Optimum bright field (OBF) direct ptychography, frame by frame, on the parallax geometry.

A port of the correlation route of quantem's `DirectPtychographyMontage`
(`reconstruct(deconvolution_kernel="obf", convolution_mode="correlation")`) for the analytic
probe, one slice and a pure-phase object. quantem stays the offline oracle; nothing here imports it.

OBF's estimate at spatial frequency q is −i·b(q) / √(Gram(q)·W), W the summed bright-field
intensity. The numerator b(q) = Σ_k c_k*(q) G_k(q) is a double sum over scan positions and
bright-field pixels. The FFT route transforms each bright-field image over the scan and contracts
over k afterwards. Here the contraction comes first, frame by frame: c_k*(q) is a product of the
probe at k and at k ∓ q, so Σ_k c_k*(q) V_R(k) is a cross-correlation over the detector, which a
few padded detector FFTs give at every q at once. The work per frame therefore does not grow with
the scan, which is what makes OBF streamable.

- The frames land on a canvas aligned with the detector, a whole number m of probe fields of view
  (1/Δk) per axis, so that its lattice is the detector's refined m times. A frame reaches it either
  on the lattice (m_r·m_c ≤ 16: every sub-lattice correlated, the frame entering through its exact
  plane wave) or as a patch (otherwise, the electron case): synthesized on a window of two probe
  fields of view about its position, shifted by its sub-pixel part in Fourier space and added with
  `index_add_`. One canvas FFT at readout gives the numerator. The window misses the aperture's
  diffraction tails, about 1% of the numerator.
- A patch holds only the band the output grid can show (its Nyquist box turned into the detector
  frame, plus `BAND_MARGIN` pixels per field of view), not OBF's whole band, unless
  `margin=None` (quantem's `patch_margin`, the same rule). The patches, the detector-side
  correlations, the canvas and the Gram shrink with it. On the binned Gd2O3 scan at x2 that is
  3.8x faster per frame and 0.3% from the whole band. Each patch's spectrum is then cut at its
  edge, which is all the difference.
- The Gram Σ_k |c_k(q)|² does not depend on the data. Its direct terms are FFT autocorrelations;
  its cross term has no FFT form and is a per-pixel pass inside the probe's support radius. It is
  built once per geometry, in one call (`ObfSetup.build`) or a unit of device work at a time
  (`ObfSetup.build_steps`), so that a live callback can spread it over its time budget.
- The readout solves per q, transforms back and reads the detector-aligned image onto the parallax
  output grid, (Ny·U, Nx·U) at the upsampled scan step, through a refine-2 cubic B-spline.

Conventions: the probe is built on parallax's k-grid (the rotation, the column flip and the disk
centre of `ParallaxGeometry`, χ of `readout.quadratic_aberration_surface`, the abTEM sign: C10 < 0
is underfocus), so OBF and parallax agree in orientation by construction. In quantem's terms the
frames are the column-flipped ("canonical") ones, its origin the disk centre after the flip, its
`rotation_angle` the rotation in degrees and its C10 the `--defocus` value.

Each bright-field pixel is normalized as quantem does, V = I/m(k) − 1 with m(k) its mean over the
positions held (`normalize`). The departures from quantem are deliberate and small:
- the detector-aligned canvas is always the rotated one, also for a rotation of zero, so the image
  is always read through the spline (quantem skips it when the requested canvas already spans
  whole fields of view, unrotated);
- the probe is evaluated in float64 on the host (quantem: float32, stored as complex64);
- the probe is evaluated at each bright-field pixel's true k, from the unrounded disk centre,
  where quantem resamples the frames onto the centre instead (`subpixel_center`);
- the per-frequency solve runs in the working precision on the device (quantem: complex128 on the
  CPU); with one column it is exact either way;
- `fft_sizes=True` rounds the canvas's pixels per field of view up to a size the FFTs handle
  quickly (a finer canvas, the same band).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from live4d.algorithms.parallax.readout import quadratic_aberration_surface

__all__ = ["ObfSetup", "bf_offsets", "bright_field_mean", "normalize", "reconstruct_once"]

#: canvas sub-lattices (fields of view along rows times along columns) up to which a frame is
#: placed on the whole canvas in Fourier space, exactly; beyond it, as a patch
_LATTICE_SUBLATTICES = 16
#: probe fields of view per axis spanned by a patch, centred on its position
_PATCH_FIELDS = 2
#: where the Gram falls below this fraction of its peak a patch's numerator is only its error
_PATCH_FLOOR = 1e-6
#: (bright-field pixel, frequency) pairs per batch in the Gram's per-pixel pass
_GRAM_POINTS_PER_BATCH = 4_194_304
#: host points (probe values, operand entries) prepared between two yields of `build_steps`, so
#: that a live callback advancing the build stays within its budget (about 10 ms of numpy each)
_HOST_POINTS_PER_STEP = 131_072
#: complex values per contraction sub-batch (quantem's `_DEFAULT_POINTS_PER_BATCH`)
_POINTS_PER_BATCH = 4_194_304
#: an overlap below this fraction of the peak, or a power below its square, is round-off
OVERLAP_FLOOR = 1e-6
#: pixels per probe field of view a patch holds beyond the output grid's Nyquist box (turned into
#: the detector frame), rounded up to an FFT size. On the binned Gd2O3 scan it is 3.8x faster per
#: frame than the whole band at x2 (2.4x at x4) and 0.3% from it (0.2% at x4); None: the band.
BAND_MARGIN = 4


# ── small helpers ──────────────────────────────────────────────────────────────

def signed_index(n: int) -> np.ndarray:
    """The signed lattice index of each corner-centred position, `fftfreq(n) * n`."""
    return np.round(np.fft.fftfreq(n, 1 / n)).astype(np.int64)


def smooth_fft_size(n: int) -> int:
    """The smallest 2^a 3^b 5^c at least `n`."""
    best = None
    power2 = 1
    while power2 < 2 * n:
        power3 = power2
        while power3 < 2 * n:
            size = power3
            while size < n:
                size *= 5
            best = size if best is None else min(best, size)
            power3 *= 3
        power2 *= 2
    return int(best)


def _reflect(x: torch.Tensor) -> torch.Tensor:
    """`x(-n)` on a periodic grid, over the last two axes."""
    return torch.roll(torch.flip(x, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1))


def _dev(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A host array on `device` as `dtype`, converted on the host first (MPS takes no float64)."""
    return torch.as_tensor(array).to(dtype).to(device)


def _dev_async(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """`_dev` without waiting for the stream's queued work (CUDA: from pinned memory)."""
    host = torch.as_tensor(array).to(dtype)
    if device.type == "cuda":
        return host.pin_memory().to(device, non_blocking=True)
    return host.to(device)


def _sub_batch(n: int, cap: int) -> int:
    """The size a deposit's sub-batch of `n` frames is padded to: the next power of two, at most
    `cap` (the setup's sub-batch)."""
    return min(int(cap), 1 << (int(n) - 1).bit_length()) if n > 0 else 0


def _run(steps):
    """Drive a generator of the stepped build to its end; its return value."""
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def _working_dtype(device: torch.device, dtype: torch.dtype | None) -> torch.dtype:
    """float32 unless asked otherwise; MPS has no float64."""
    if device.type == "mps" or dtype is None:
        return torch.float32
    return dtype


def bf_offsets(geometry) -> np.ndarray:
    """(M, 2) signed detector offsets (row, col) of the bright-field pixels from k = 0, in the
    canonical (column-unflipped) frame, in the order of `geometry.bf_flat_inds`: the inverse of
    the placement in `preprocess_geometry`."""
    h, w = (int(n) for n in geometry.detector_shape)
    det_i, det_j = np.divmod(np.asarray(geometry.bf_flat_inds, dtype=np.int64), w)
    center = geometry.probe_center
    ci = h // 2 if center is None else int(round(center[0]))
    if center is not None:
        cj = int(round(center[1]))
    else:
        cj = w - 1 - w // 2 if geometry.flip_cols else w // 2
    off_j = cj - det_j if geometry.flip_cols else det_j - cj
    return np.stack((det_i - ci, off_j), axis=-1)


def center_fraction(geometry) -> np.ndarray:
    """(2,) the true k of offset 0, in detector pixels of the canonical frame: `preprocess_geometry`
    rounds the disk centre to a pixel, so each bright-field pixel sits this far from where the
    rounding puts it."""
    center = geometry.probe_center
    if center is None:
        return np.zeros(2)
    frac = np.asarray(center, dtype=np.float64) - np.round(center)
    # a raw column right of the centre is a canonical column left of it
    return np.array([-frac[0], frac[1] if geometry.flip_cols else -frac[1]])


def _raw_float(raw: torch.Tensor, kind: str) -> torch.Tensor:
    """Stored (gathered) bright-field values → float32, undoing the unsigned bit-views of
    `ParallaxAccumulator._gather`."""
    g = raw.float()
    if kind == "u16":
        g = torch.where(g < 0, g + 65536.0, g)
    elif kind == "u32":
        g = torch.where(g < 0, g + 4294967296.0, g)
    return g


def bright_field_mean(store: torch.Tensor, kind: str, held: torch.Tensor,
                      chunk: int = 4096) -> torch.Tensor:
    """(M,) mean of each bright-field pixel over the positions `held` of a `latest` store."""
    total = torch.zeros(store.shape[1], dtype=torch.float64 if store.device.type != "mps"
                        else torch.float32, device=store.device)
    for pos in held.split(chunk):
        total += _raw_float(store[pos], kind).sum(0).to(total.dtype)
    return (total / max(int(held.numel()), 1)).float()


def normalize(values: torch.Tensor, mean: torch.Tensor,
              subtract_frame_mean: bool = False) -> torch.Tensor:
    """quantem's V = I/m − 1 per bright-field pixel (zero where m is), then, with
    `subtract_frame_mean`, each frame's mean over its bright-field pixels removed."""
    safe = torch.where(mean > 0, mean, torch.ones_like(mean))
    v = torch.where(mean > 0, values / safe - 1.0, torch.zeros_like(values))
    if subtract_frame_mean:
        v = v - v.mean(dim=1, keepdim=True)
    return v


# ── the probe and the canvas layout (host, NumPy) ─────────────────────────────

def _probe(shape, step, wavelength, semiangle, angular_sampling, coefs, rotation,
           shift=(0.0, 0.0)) -> np.ndarray:
    """ψ(k) = A(k)·exp(−iχ(k)) on a detector-aligned lattice of `shape` points `step` 1/Å apart,
    corner-centred, read at the scan-frame frequency R·k of each: quantem's analytic
    `FourierProbe.at` with its soft-edged aperture (a pixel wide at the detector's own
    `angular_sampling`, mrad/px). `shift` (1/Å, detector frame) is the true k of lattice point 0,
    for a disk centre between pixels."""
    return _run(_probe_steps(shape, step, wavelength, semiangle, angular_sampling, coefs,
                             rotation, shift))


def _probe_steps(shape, step, wavelength, semiangle, angular_sampling, coefs, rotation,
                 shift=(0.0, 0.0)):
    """`_probe` a block of rows at a time: yields between blocks of about
    `_HOST_POINTS_PER_STEP` points and returns the probe."""
    shape = tuple(int(n) for n in shape)
    kx = signed_index(shape[0]) * step[0] + shift[0]
    ky = signed_index(shape[1]) * step[1] + shift[1]
    out = np.empty(shape, dtype=np.complex128)
    rows = max(1, _HOST_POINTS_PER_STEP // max(shape[1], 1))
    for r0 in range(0, shape[0], rows):
        out[r0:r0 + rows] = _probe_rows(kx[r0:r0 + rows], ky, wavelength, semiangle,
                                        angular_sampling, coefs, rotation)
        yield
    return out


def _probe_rows(kx, ky, wavelength, semiangle, angular_sampling, coefs, rotation) -> np.ndarray:
    """The probe at the frequencies kx (rows) × ky (columns), 1/Å, detector frame."""
    kxa, kya = np.meshgrid(kx, ky, indexing="ij")
    c, s = math.cos(rotation), math.sin(rotation)
    kxa, kya = c * kxa - s * kya, s * kxa + c * kya
    alpha = np.hypot(kxa, kya) * wavelength
    phi = np.arctan2(kya, kxa)
    width = np.sqrt((np.cos(phi) * angular_sampling[0] * 1e-3) ** 2
                    + (np.sin(phi) * angular_sampling[1] * 1e-3) ** 2)
    aperture = np.clip((semiangle * 1e-3 - alpha) / width + 0.5, 0.0, 1.0)
    chi = quadratic_aberration_surface(alpha, phi, wavelength, coefs)
    return aperture * np.exp(-1j * chi)


def _support(probe: np.ndarray) -> np.ndarray:
    """Where |ψ|² exceeds round-off, OVERLAP_FLOOR² of its peak."""
    power = np.abs(probe) ** 2
    return power > OVERLAP_FLOOR**2 * power.max()


def _layout(canvas_shape, sampling, fov, reach, rotation, margin=None):
    """Where the frames land (quantem's `_correlation_layout`, its rotated branch): `multiples`
    probe fields of view per axis span a canvas of `per_fov` pixels to a field of view, covering
    the requested canvas turned into the detector frame, with room per side for half a field of
    view (lattice) or half a patch, so nothing wraps onto it. `_finish_layout` adds the rest.

    A patch holds OBF's whole band if `margin` is None. Otherwise it holds only the requested
    canvas's Nyquist box turned into the detector frame and `margin` pixels per field of view
    beyond it, so the patches and the detector-side correlations shrink, but never fewer pixels
    than the requested canvas has (quantem's `patch_margin`)."""
    sampling = np.asarray(sampling, dtype=np.float64)
    band = 4 * reach + 1                      # pixels per field of view holding the whole band
    native = fov / sampling
    rot = np.array([[math.cos(rotation), -math.sin(rotation)],
                    [math.sin(rotation), math.cos(rotation)]])
    extent = np.asarray(canvas_shape, dtype=np.float64) * sampling
    corners = np.array([[0, 0], [extent[0], 0], [0, extent[1]], extent]) @ rot
    span = corners.max(0) - corners.min(0)
    multiples = np.ceil(span / fov + 1 - 1e-9).astype(int)
    lattice = int(np.prod(multiples)) <= _LATTICE_SUBLATTICES
    if not lattice:
        multiples = np.ceil(span / fov + _PATCH_FIELDS - 1e-9).astype(int)
    # pixels no coarser than the requested ones, fine enough for the requested canvas's Nyquist
    # box turned into this frame, never beyond the band (except that a patch needs the band)
    turned = fov * (np.abs(rot[0, :]) / sampling[0] + np.abs(rot[1, :]) / sampling[1])
    per_fov = np.maximum(np.ceil(native - 1e-6), np.minimum(np.ceil(turned - 1e-6), band))
    if not lattice:
        per_fov = np.maximum(per_fov, band if margin is None else np.minimum(
            [smooth_fft_size(int(v)) for v in per_fov + int(margin)], band))
    return dict(placement="lattice" if lattice else "patch", multiples=multiples,
                per_fov=per_fov.astype(int), rotation=rot, corners=corners, span=span)


def _finish_layout(layout, fov, positions_A):
    """The canvas `shape` of pixels `step` Å wide, its corner `origin` in Å in the detector
    frame, and the scan `positions` (Å) in its pixels, once `per_fov` is settled."""
    multiples, per_fov, span = layout["multiples"], layout["per_fov"], layout["span"]
    step = fov / per_fov
    # centred to a whole pixel, so an unrotated canvas with the same pixels is read at whole pixels
    origin = layout["corners"].min(0) - np.round((multiples * fov - span) / 2 / step) * step
    layout.update(shape=tuple(int(v) for v in multiples * per_fov), step=step, origin=origin,
                  positions=(positions_A @ layout["rotation"] - origin) / step)
    return layout


# ── the correlation operands ───────────────────────────────────────────────────

@dataclass
class _Operands:
    """A correlation onto a grid whose lattice is `sub` times the detector's: the detector
    grid `padded` so no lag up to `lags` wraps, the probe shifted by s/sub of a pixel per
    sub-lattice s (`minus`: ψ(k − sΔk/sub), `plus`: ψ(k + sΔk/sub), (S, *padded)), and the map
    of per-sub-lattice correlations onto the grid (zero beyond `lags`)."""
    padded: tuple
    sub: tuple
    minus: torch.Tensor
    plus: torch.Tensor
    row_lookup: torch.Tensor
    col_lookup: torch.Tensor
    kept: torch.Tensor | None

    @property
    def psi(self) -> torch.Tensor:
        return self.minus[0]

    def to_grid(self, stack: torch.Tensor) -> torch.Tensor:
        """(S, *lead, *padded) per-sub-lattice correlations → (*lead, *grid)."""
        s0, s1 = self.sub
        p0, p1 = self.padded
        lead = stack.shape[1:-2]
        stack = stack.reshape(s0, s1, *lead, p0, p1).movedim((0, 1), (-4, -2))
        stack = stack.reshape(*lead, s0 * p0, s1 * p1)
        out = stack.index_select(-2, self.row_lookup).index_select(-1, self.col_lookup)
        return out if self.kept is None else out * self.kept


def _operands(refined: np.ndarray, gpts, sub, grid_shape, lags, device, cdtype) -> _Operands:
    """quantem's `_lattice_operands` + `_sublattice_probe` for one coherent mode."""
    return _run(_operands_steps(refined, gpts, sub, grid_shape, lags, device, cdtype))


def _operands_steps(refined: np.ndarray, gpts, sub, grid_shape, lags, device, cdtype):
    """`_operands` a group of sub-lattices at a time: yields after each group's shifted probes
    (about `_HOST_POINTS_PER_STEP` values) are filled and sent to the device."""
    padded = tuple(smooth_fft_size(int(gpts[a]) + int(lags[a]) + 1) for a in (0, 1))
    rows = signed_index(int(gpts[0])) % padded[0]
    cols = signed_index(int(gpts[1])) % padded[1]
    pairs = [(a, b) for a in range(int(sub[0])) for b in range(int(sub[1]))]
    group = max(1, _HOST_POINTS_PER_STEP // (int(gpts[0]) * int(gpts[1])))
    out = {}
    for sign in (-1, 1):
        stack = torch.zeros((len(pairs), *padded), dtype=cdtype, device=device)
        for g0 in range(0, len(pairs), group):
            block = np.zeros((len(pairs[g0:g0 + group]), *padded), dtype=np.complex128)
            for n, (a, b) in enumerate(pairs[g0:g0 + group]):
                picks = []
                for axis, s in ((0, a), (1, b)):
                    size, factor = int(gpts[axis]), int(sub[axis])
                    index = factor * signed_index(size) + sign * s
                    valid = (index >= -(size * factor // 2)) & (index < (size * factor + 1) // 2)
                    picks.append((index % (size * factor), valid))
                (ri, rv), (ci, cv) = picks
                block[n][rows[:, None], cols[None, :]] = refined[ri[:, None], ci[None, :]] * (
                    rv[:, None] & cv[None, :])
            stack[g0:g0 + len(block)] = _dev_async(block, device, cdtype)
            yield
        out[sign] = stack

    def lookup(n, factor, size, protected):
        index = signed_index(n)
        lag = index // factor
        flat = (index % factor) * size + lag % size
        return (torch.as_tensor(flat, device=device),
                torch.as_tensor(np.abs(lag) <= protected, device=device))

    row_lookup, row_kept = lookup(int(grid_shape[0]), int(sub[0]), padded[0], int(lags[0]))
    col_lookup, col_kept = lookup(int(grid_shape[1]), int(sub[1]), padded[1], int(lags[1]))
    kept = row_kept[:, None] & col_kept[None, :]
    return _Operands(padded=padded, sub=(int(sub[0]), int(sub[1])), minus=out[-1], plus=out[1],
                     row_lookup=row_lookup, col_lookup=col_lookup,
                     kept=None if bool(kept.all()) else kept)


@dataclass
class _PatchProducts:
    """The patch grid's contraction as two matrix products (`ObfSetup._contract`).

    Per frame the FFT route forms, for every sub-lattice s, the spectrum Z_s = Â_s* ⊙ F − B̂_s ⊙ G
    over the padded detector grid (F = fft2(V·ψ), G = conj(F) for real V), transforms it back in
    full and keeps only the lags the patch grid holds (`_Operands.to_grid`). Those lags are a
    partial inverse DFT, E0 · Z_s · E1ᵀ, and the elementwise products fold into its first factor:
    T_s[m, v] = Σ_u (KA − KB)[m, u, v] Re F[u, v] + i (KA + KB)[m, u, v] Im F[u, v], with
    KA = E0[m, u] Â_s*[u, v] and KB = E0[m, u] B̂_s[u, v]. So a frame needs one detector FFT, one
    real matrix product batched over v (`w`) and one complex one (`e1`), and Z_s is never formed.
    """
    w: torch.Tensor          # (p1, 2·p0, s1·s0·n0·2): rows (u, Re/Im F), columns (a', a, m, Re/Im)
    e1: torch.Tensor         # (s1, n1, p1): the partial inverse DFT along v, per column group
    flat: torch.Tensor       # (g0·g1,): grid pixel → its entry in a frame's (s1, n1, s0·n0) block
    bf_t: torch.Tensor       # bright-field pixels in the transposed padded frame
    psi_bf: torch.Tensor     # ψ at the bright-field pixels


def _patch_products(ops: _Operands, conj_a_hat: torch.Tensor, b_hat: torch.Tensor, bf_flat,
                    grid_shape, device, rdtype, cdtype) -> _PatchProducts:
    p0, p1 = ops.padded
    s0, s1 = ops.sub
    g0, g1 = (int(n) for n in grid_shape)
    n0, n1 = g0 // s0, g1 // s1
    row_flat, col_flat = ops.row_lookup.cpu().numpy(), ops.col_lookup.cpu().numpy()
    kept = None if ops.kept is None else ops.kept.cpu().numpy()
    row_kept = np.ones(g0, bool) if kept is None else kept.any(1)
    col_kept = np.ones(g1, bool) if kept is None else kept.any(0)
    a_row, r_row = row_flat // p0, row_flat % p0     # each grid row's sub-lattice and padded row
    a_col, c_col = col_flat // p1, col_flat % p1
    m_row, n_col = np.empty(g0, np.int64), np.empty(g1, np.int64)
    e0 = np.zeros((s0, n0, p0), np.complex128)
    e1 = np.zeros((s1, n1, p1), np.complex128)
    for a in range(s0):
        rows = np.flatnonzero(a_row == a)
        m_row[rows] = np.arange(n0)
        e0[a] = (np.exp(2j * np.pi * np.outer(r_row[rows], np.arange(p0)) / p0) / p0
                 * row_kept[rows][:, None])          # torch's ifft convention, 1/p per axis
    for a in range(s1):
        cols = np.flatnonzero(a_col == a)
        n_col[cols] = np.arange(n1)
        e1[a] = (np.exp(2j * np.pi * np.outer(c_col[cols], np.arange(p1)) / p1) / p1
                 * col_kept[cols][:, None])
    e0_s = _dev(e0, device, cdtype)[torch.arange(s0 * s1, device=device) // s1]   # (S, n0, p0)
    ka = torch.einsum("smu,suv->vusm", e0_s, conj_a_hat.resolve_conj())
    kb = torch.einsum("smu,suv->vusm", e0_s, b_hat)
    parts = torch.stack((torch.view_as_real(ka - kb), torch.view_as_real(1j * (ka + kb))), 2)
    # (v, u, Re/Im F, S = (a, a'), m, Re/Im) → columns grouped by a' then a
    w = parts.view(p1, p0, 2, s0, s1, n0, 2).permute(0, 1, 2, 4, 3, 5, 6)
    w = w.reshape(p1, 2 * p0, s1 * s0 * n0 * 2).contiguous().to(rdtype)
    flat = (a_col[None, :] * n1 + n_col[None, :]) * (s0 * n0) + a_row[:, None] * n0 + m_row[:, None]
    bf = bf_flat.cpu().numpy()
    return _PatchProducts(
        w=w, e1=_dev(e1, device, cdtype), flat=torch.as_tensor(flat.reshape(-1), device=device),
        bf_t=torch.as_tensor((bf % p1) * p0 + bf // p1, device=device),
        psi_bf=ops.psi.reshape(-1)[bf_flat])


# ── the Gram ───────────────────────────────────────────────────────────────────

def _gram_steps(refined: np.ndarray, operands: _Operands, offsets: np.ndarray, gpts, multiples,
                grid_shape, dk, device, rdtype, cdtype):
    """Σ_k |c_k(q)|² on the canvas lattice (quantem's `_correlation_gram`, one mode).

    |c|² = |t−|² + |t+|² − 2 Re(t− t+*). Summed over k the first two are autocorrelations of
    |ψ|², one FFT pair per sub-lattice. The cross term Σ_k ψ(k − q) ψ(k + q) ψ*(k)² has no FFT
    form; it is evaluated per bright-field pixel where it can be nonzero, |q| within the probe's
    support radius, and |c_k(−q)| = |c_k(q)| halves that. The Gram must be exact on the grid it
    is used on: it falls steeply at the band's edge, where OBF divides by its square root.

    A generator: it yields after the direct terms, after each block of the cross term's host
    arrays and after each of its batches, and returns the Gram."""
    psi = operands.psi
    padded = operands.padded
    # the probe's support radius, read before the direct terms are queued, so that reading it
    # waits for little queued work
    support = psi.abs().square() > OVERLAP_FLOOR**2 * psi.abs().square().max()
    sr, sc = signed_index(int(gpts[0])), signed_index(int(gpts[1]))
    on_det = support[torch.as_tensor(sr % padded[0], device=device)[:, None],
                     torch.as_tensor(sc % padded[1], device=device)[None, :]].cpu().numpy()
    radius = float(np.hypot(sr[:, None] * dk[0], sc[None, :] * dk[1])[on_det].max())
    p0 = torch.fft.fft2(psi.abs().square().to(cdtype))
    minus = torch.fft.fft2(operands.minus.abs().square().to(cdtype)).conj() * p0
    plus = torch.fft.fft2(operands.plus.abs().square().to(cdtype)).conj() * p0
    gram = (operands.to_grid(torch.fft.ifft2(minus))
            + operands.to_grid(_reflect(torch.fft.ifft2(plus)))).real.to(rdtype)
    del minus, plus
    yield

    # the cross term, per bright-field pixel, where it can live: within the radius, and the
    # upper half (the rest mirrors it), a block of the grid's rows at a time
    g0, g1 = (int(n) for n in grid_shape)
    row_index, col_index = signed_index(g0), signed_index(g1)
    m0, m1 = (int(v) for v in multiples)
    computed = np.empty((g0, g1), dtype=bool)
    mirror = np.empty((g0, g1), dtype=bool)
    block = max(1, _HOST_POINTS_PER_STEP // g1)
    for r0 in range(0, g0, block):
        rows, cols = row_index[r0:r0 + block, None], col_index[None, :]
        inside = np.hypot(rows * dk[0] / m0, cols * dk[1] / m1) <= radius * (1 + 1e-6)
        upper = (rows >= 0) | (rows == -(g0 // 2)) | (cols == -(g1 // 2))
        computed[r0:r0 + block] = inside & upper
        mirror[r0:r0 + block] = inside & ~upper
        yield
    if computed.any():
        r_index, c_index = np.nonzero(computed)          # row-major, as boolean indexing
        q_rows, q_cols = row_index[r_index], col_index[c_index]
        q_row = _dev_async(q_rows, device, torch.int64)
        q_col = _dev_async(q_cols, device, torch.int64)
        k_row = torch.as_tensor(offsets[:, 0] * m0, device=device)
        k_col = torch.as_tensor(offsets[:, 1] * m1, device=device)
        # quantem's LatticeOverlap: the refined probe fftshifted into a zero-padded flat array
        # wide enough that every k ∓ q lands inside it, so a batch is one add and one gather
        n0, n1 = refined.shape
        reach0, reach1 = int(np.abs(q_rows).max()) + 1, int(np.abs(q_cols).max()) + 1
        width = n1 + 2 * reach1
        flat = torch.zeros((n0 + 2 * reach0) * width, dtype=cdtype, device=device)
        padded_view = flat.view(n0 + 2 * reach0, width)
        block = max(1, _HOST_POINTS_PER_STEP // n1)
        for r0 in range(0, n0, block):                  # fftshift(refined), a block of rows a step
            r1 = min(n0, r0 + block)
            shifted = np.roll(refined[(np.arange(r0, r1) - n0 // 2) % n0], n1 // 2, axis=1)
            padded_view[reach0 + r0:reach0 + r1, reach1:reach1 + n1] = _dev_async(
                shifted, device, cdtype)
            yield
        k_base = (k_row + reach0 + n0 // 2) * width + (k_col + reach1 + n1 // 2)
        q_offset = q_row * width + q_col
        exact = torch.zeros(q_offset.numel(), dtype=rdtype, device=device)
        batch = max(1, _GRAM_POINTS_PER_BATCH // q_offset.numel())
        for lo in range(0, k_base.numel(), batch):
            base = k_base[lo:lo + batch, None]
            psi_k = flat[base]
            t_minus = flat[base - q_offset[None]] * psi_k.conj()
            t_plus = flat[base + q_offset[None]].conj() * psi_k
            exact += (t_minus - t_plus).abs().square().sum(0).to(rdtype)
            yield
        gram[_dev_async(computed, device, torch.bool)] = exact
        m_rows, m_cols = np.nonzero(mirror)
        src = ((-row_index[m_rows]) % g0, (-col_index[m_cols]) % g1)
        gram[_dev_async(mirror, device, torch.bool)] = gram[
            _dev_async(src[0], device, torch.int64), _dev_async(src[1], device, torch.int64)]
    return gram


# ── reading the detector-aligned image onto the parallax grid ──────────────────

class _CanvasReader:
    """quantem's `_read_onto_canvas`, with everything that does not depend on the image built
    once: the image's spectrum is cut to the requested canvas's Nyquist box turned into the
    detector frame, refined `refine` times by zero-padding, divided by the transform of the
    sampled cubic B-spline, and the spline read at each requested pixel."""

    def __init__(self, inner_shape, step, origin, rotation, canvas_shape, sampling, device,
                 rdtype, cdtype, refine: int = 2):
        # formed on `device` in float64 (the same elementwise arithmetic as NumPy), so a live
        # callback building it queues the work instead of making and uploading canvas-sized arrays
        self.device, self.rdtype, self.cdtype = device, rdtype, cdtype
        f64 = dict(device=device, dtype=torch.float64)
        n_rows, n_cols = inner_shape
        q_rows = torch.as_tensor(np.fft.fftfreq(n_rows, step[0]), **f64)[:, None]
        q_cols = torch.as_tensor(np.fft.fftfreq(n_cols, step[1]), **f64)[None, :]
        along_rows = q_rows * float(rotation[0, 0]) + q_cols * float(rotation[0, 1])
        along_cols = q_rows * float(rotation[1, 0]) + q_cols * float(rotation[1, 1])
        keep = ((along_rows.abs() <= 0.5 / float(sampling[0]) * (1 + 1e-9))
                & (along_cols.abs() <= 0.5 / float(sampling[1]) * (1 + 1e-9)))
        # an even size's Nyquist bin has no partner on the finer grid
        if n_rows % 2 == 0:
            keep[n_rows // 2] = False
        if n_cols % 2 == 0:
            keep[:, n_cols // 2] = False
        self.keep = keep
        self.fine = (refine * n_rows, refine * n_cols)
        self.rows = torch.as_tensor(signed_index(n_rows) % self.fine[0], device=device)
        self.cols = torch.as_tensor(signed_index(n_cols) % self.fine[1], device=device)
        self.spline = [_dev((2 + np.cos(2 * np.pi * np.fft.fftfreq(n))) / 3, device, rdtype)
                       for n in self.fine]
        self.scale = float(refine * refine)
        i = torch.arange(canvas_shape[0], **f64)[:, None] * float(sampling[0])
        j = torch.arange(canvas_shape[1], **f64)[None, :] * float(sampling[1])
        u = (i * float(rotation[0, 0]) + j * float(rotation[1, 0]) - float(origin[0])) \
            / (float(step[0]) / refine)
        v = (i * float(rotation[0, 1]) + j * float(rotation[1, 1]) - float(origin[1])) \
            / (float(step[1]) / refine)
        self.taps = []
        for coordinate, n in ((u, self.fine[0]), (v, self.fine[1])):
            base = torch.floor(coordinate)
            t = (coordinate - base).to(rdtype)
            weights = ((1 - t) ** 3 / 6, (3 * t**3 - 6 * t**2 + 4) / 6,
                       (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6, t**3 / 6)
            index = [(base.long() + o) % n for o in (-1, 0, 1, 2)]
            self.taps.append((weights, index))

    def to(self, device: torch.device) -> "_CanvasReader":
        new = object.__new__(_CanvasReader)
        new.__dict__.update(self.__dict__)
        new.device = device
        new.keep, new.rows, new.cols = (t.to(device) for t in (self.keep, self.rows, self.cols))
        new.spline = [t.to(device) for t in self.spline]
        new.taps = [(tuple(w.to(device) for w in weights), [i.to(device) for i in index])
                    for weights, index in self.taps]
        return new

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(image.to(self.cdtype)) * self.keep
        refined = torch.zeros(self.fine, device=self.device, dtype=self.cdtype)
        refined[self.rows[:, None], self.cols[None, :]] = spectrum
        refined /= self.spline[0][:, None] * self.spline[1][None, :]
        coefficients = torch.fft.ifft2(refined).real * self.scale
        (rw, ri), (cw, ci) = self.taps
        out = torch.zeros(rw[0].shape, device=self.device, dtype=self.rdtype)
        for a in range(4):
            for b in range(4):
                out += rw[a] * cw[b] * coefficients[ri[a], ci[b]]
        return out


# ── the setup: everything that does not depend on the data ─────────────────────

def geometry_key(geometry) -> tuple:
    """What an `ObfSetup` depends on: the probe, the bright-field pixels, the rotation and the
    output grid. Equal keys share a setup (and its Gram)."""
    coefs = geometry.aberration_coefs
    return (tuple(geometry.detector_shape), tuple(geometry.scan_shape),
            tuple(geometry.upsampled_scan_gpts), tuple(round(float(v), 9)
                                                       for v in geometry.upsampled_scan_sampling),
            round(float(geometry.wavelength), 12), round(float(geometry.semiangle_cutoff), 9),
            tuple(round(float(v), 12) for v in geometry.sampling),
            round(float(coefs.get("C10", 0.0)), 6), round(float(coefs.get("C12", 0.0)), 6),
            round(float(coefs.get("phi12", 0.0)), 9),
            round(float(geometry.rotation_angle or 0.0), 12),
            np.asarray(geometry.bf_flat_inds).tobytes(), bool(geometry.flip_cols),
            None if geometry.probe_center is None else tuple(geometry.probe_center))


@dataclass
class ObfSetup:
    """The probe, the canvas layout, the correlation operands and the Gram for one geometry.

    `buffer()` makes an empty numerator; `deposit(buffer, values, pos)` adds frames (normalized
    bright-field values (B, M), scan positions (B,)); `image(buffer)` reads the OBF image onto
    the parallax output grid. Deposits are linear in the values and independent of order, so a
    numerator can be split across buffers and summed (`latest`: deposit new − old)."""

    key: tuple
    device: torch.device
    rdtype: torch.dtype
    cdtype: torch.dtype
    placement: str
    multiples: tuple
    per_fov: tuple
    inner_shape: tuple
    grid_shape: tuple
    canvas_shape: tuple
    n_bf: int
    bf_weights: float
    gram: torch.Tensor
    build_s: float = 0.0
    _ops: _Operands | None = field(default=None, repr=False)
    _bf_flat: torch.Tensor | None = field(default=None, repr=False)
    _conj_a_hat: torch.Tensor | None = field(default=None, repr=False)
    _b_hat: torch.Tensor | None = field(default=None, repr=False)
    _q: tuple | None = field(default=None, repr=False)
    _place: tuple | None = field(default=None, repr=False)
    _offsets: tuple | None = field(default=None, repr=False)
    _operator: torch.Tensor | None = field(default=None, repr=False)
    _numerator_mask: torch.Tensor | None = field(default=None, repr=False)
    _reader: _CanvasReader | None = field(default=None, repr=False)
    _products: _PatchProducts | None = field(default=None, repr=False)
    _scale: float = 1.0
    _batch: int = 1

    @classmethod
    def build(cls, geometry, device=None, dtype: torch.dtype | None = None,
              fft_sizes: bool = False, subpixel_center: bool = True,
              margin: int | None = BAND_MARGIN) -> "ObfSetup":
        """Everything OBF needs for `geometry` (a `ParallaxGeometry`), on `device`, in `dtype`
        (float32 by default; float64 for checks off MPS), in one call: `build_steps` run to the
        end, then synchronized so that `build_s` is the time it took. Takes seconds (the Gram).
        `subpixel_center` evaluates the probe at each bright-field pixel's true k, from the
        unrounded disk centre (`center_fraction`); off, the probe sits on the rounded centre as
        parallax's shifts do. On the Gd2O3 recordings (0.14 and 0.41 px off a pixel) it takes the
        interior correlation with quantem's own OBF from 0.988 to 0.999. `margin`: a patch holds
        only the output grid's band and this many pixels per field of view beyond it
        (`BAND_MARGIN`); None, OBF's whole band (quantem's `patch_margin=None`)."""
        import time
        t0 = time.perf_counter()
        steps = cls.build_steps(geometry, device, dtype, fft_sizes, subpixel_center, margin)
        while True:
            try:
                next(steps)
            except StopIteration as done:
                setup = done.value
                break
        if setup.device.type == "cuda":
            torch.cuda.synchronize(setup.device)
        elif setup.device.type == "mps":
            torch.mps.synchronize()
        setup.build_s = time.perf_counter() - t0
        return setup

    @classmethod
    def build_steps(cls, geometry, device=None, dtype: torch.dtype | None = None,
                    fft_sizes: bool = False, subpixel_center: bool = True,
                    margin: int | None = BAND_MARGIN):
        """`build` a unit of work at a time: a generator that yields between units (blocks of the
        probes' and operands' host arrays, `_HOST_POINTS_PER_STEP` values each, and their uploads;
        each batch of the Gram's cross term) and returns the `ObfSetup`. Its device work runs on
        the caller's current stream, unsynchronized but for two scalar reads, so a live callback
        can advance it within a time budget and use the setup on the same stream."""
        import time
        t0 = time.perf_counter()
        device = torch.device(device) if device is not None else torch.device("cpu")
        rdtype = _working_dtype(device, dtype)
        cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
        offsets = bf_offsets(geometry)
        reach_bf = np.abs(offsets).max(0)
        gpts = tuple(int(2 * a + 3) for a in reach_bf)          # the bright-field crop, ±1 px
        dk = np.array([1.0 / (geometry.detector_shape[a] * geometry.sampling[a]) for a in (0, 1)])
        fov = 1.0 / dk
        wavelength = float(geometry.wavelength)
        angular = dk * wavelength * 1e3                         # mrad/px
        rotation = float(geometry.rotation_angle or 0.0)
        coefs = dict(geometry.aberration_coefs)
        shift = center_fraction(geometry) * dk if subpixel_center else np.zeros(2)

        def refined(m):                                 # a generator: `yield from` it
            m = np.asarray(m, dtype=int)
            return _probe_steps(tuple(int(g * f) for g, f in zip(gpts, m)), dk / m, wavelength,
                                float(geometry.semiangle_cutoff), angular, coefs, rotation, shift)

        fields = np.full(2, _PATCH_FIELDS)
        patch_probe = yield from refined(fields)
        support = _support(patch_probe)
        reach = []
        for axis in (0, 1):
            index = signed_index(gpts[axis] * int(fields[axis]))
            line = support.any(axis=1 - axis)
            reach.append(max(math.ceil(float(np.abs(index[line]).max()) / fields[axis]),
                             int(reach_bf[axis])))
        reach = np.asarray(reach)

        sampling = np.asarray(geometry.upsampled_scan_sampling, dtype=np.float64)
        scan_sampling = sampling * int(geometry.upsampling_factor)
        ny, nx = (int(n) for n in geometry.scan_shape)
        iy, ix = np.divmod(np.arange(ny * nx), nx)
        positions_A = np.stack((iy * scan_sampling[0], ix * scan_sampling[1]), -1)
        canvas_shape = tuple(int(n) for n in geometry.upsampled_scan_gpts)
        layout = _layout(canvas_shape, sampling, fov, reach, rotation, margin)
        if fft_sizes:
            layout["per_fov"] = np.asarray([smooth_fft_size(int(v)) for v in layout["per_fov"]])
            if layout["placement"] == "patch":
                layout["multiples"] = np.asarray(
                    [smooth_fft_size(int(v)) for v in layout["multiples"]])
        layout = _finish_layout(layout, fov, positions_A)
        multiples, per_fov, step = layout["multiples"], layout["per_fov"], layout["step"]
        inner = layout["shape"]

        # the Gram on the canvas's own lattice, from the probe refined `multiples` times
        canvas_lags = tuple(int(n) // 2 // int(m) + 1 for n, m in zip(inner, multiples))
        yield
        gram_probe = yield from refined(multiples)
        ops = yield from _operands_steps(gram_probe, gpts, multiples, inner, canvas_lags, device,
                                         cdtype)
        gram = yield from _gram_steps(gram_probe, ops, offsets, gpts, multiples, inner, dk,
                                      device, rdtype, cdtype)
        del ops, gram_probe

        if layout["placement"] == "lattice":
            sub, grid_shape, lags = multiples, inner, canvas_lags
            probe = yield from refined(multiples)
        else:
            sub, grid_shape = fields, tuple(int(v) for v in fields * per_fov)
            # a correlation vanishes beyond 2·reach lags, and a sub-lattice adds under one
            lags = tuple(int(v) for v in np.minimum(per_fov // 2 + 1, 2 * reach + 2))
            probe = patch_probe
        ops = yield from _operands_steps(probe, gpts, sub, grid_shape, lags, device, cdtype)
        padded = ops.padded
        bf_rows = offsets[:, 0] % padded[0]
        bf_cols = offsets[:, 1] % padded[1]
        bf_flat = torch.as_tensor(bf_rows * padded[1] + bf_cols, device=device)
        psi = ops.psi
        bf_weights = float(psi.abs().square().reshape(-1)[bf_flat].sum())
        conj_a_hat = torch.fft.fft2(ops.minus).conj()
        b_hat = torch.fft.fft2(ops.plus)
        yield
        products = None
        if layout["placement"] == "patch":
            products = _patch_products(ops, conj_a_hat, b_hat, bf_flat, grid_shape, device,
                                       rdtype, cdtype)
        yield
        q_rows = _dev(np.fft.fftfreq(int(grid_shape[0]), step[0]), device, rdtype)
        q_cols = _dev(np.fft.fftfreq(int(grid_shape[1]), step[1]), device, rdtype)

        positions = layout["positions"]
        if layout["placement"] == "lattice":
            place = (_dev(positions * step, device, rdtype),)
            offs = None
        else:
            base = np.round(positions)
            fraction = _dev((positions - base) * step, device, rdtype)
            place = (torch.as_tensor(base.astype(np.int64), device=device), fraction)
            offs = (torch.as_tensor(signed_index(int(grid_shape[0])), device=device),
                    torch.as_tensor(signed_index(int(grid_shape[1])), device=device))

        # the one-column solve, −i·b / √(Gram·W), as a multiplier on the numerator
        peak = float(gram.max())
        live = gram > OVERLAP_FLOOR**2 * peak
        safe = torch.where(live, gram, torch.ones_like(gram))
        operator = torch.where(live, torch.rsqrt(safe * bf_weights), torch.zeros_like(gram))
        operator = operator.to(cdtype) * (-1j)
        numerator_mask = None
        if layout["placement"] == "patch":
            numerator_mask = gram > _PATCH_FLOOR * peak
        yield

        reader = _CanvasReader(inner, step, layout["origin"], layout["rotation"], canvas_shape,
                               sampling, device, rdtype, cdtype)
        sublattice_points = len(ops.minus) * padded[0] * padded[1]
        grid_points = int(grid_shape[0]) * int(grid_shape[1])
        batch = max(1, min(_POINTS_PER_BATCH * 4 // sublattice_points,
                           _POINTS_PER_BATCH * (8 if layout["placement"] == "lattice" else 2)
                           // grid_points))
        return cls(
            key=geometry_key(geometry), device=device, rdtype=rdtype, cdtype=cdtype,
            placement=layout["placement"], multiples=tuple(int(v) for v in multiples),
            per_fov=tuple(int(v) for v in per_fov), inner_shape=tuple(inner),
            grid_shape=tuple(int(v) for v in grid_shape), canvas_shape=canvas_shape,
            n_bf=int(offsets.shape[0]), bf_weights=bf_weights, gram=gram,
            build_s=time.perf_counter() - t0, _ops=ops, _bf_flat=bf_flat,
            _conj_a_hat=conj_a_hat, _b_hat=b_hat, _q=(q_rows, q_cols), _place=place,
            _offsets=offs, _operator=operator, _numerator_mask=numerator_mask, _reader=reader,
            _products=products, _scale=float(np.prod(sampling) / np.prod(step)), _batch=batch)

    # ── the numerator ──

    def buffer(self) -> torch.Tensor:
        """An empty numerator: the real-space canvas as (pixels, 2) real/imaginary pairs (patch),
        or the numerator's spectrum itself (lattice)."""
        if self.placement == "patch":
            n = self.inner_shape[0] * self.inner_shape[1]
            return torch.zeros((n, 2), dtype=self.rdtype, device=self.device)
        return torch.zeros(self.inner_shape, dtype=self.cdtype, device=self.device)

    def _contract(self, values: torch.Tensor) -> torch.Tensor:
        """Σ_k c_k*(q) V_R(k) on the patch (or canvas) grid, (B, *grid), for values (B, M)."""
        if self._products is not None:
            return self._contract_patch(values)
        ops = self._ops
        b = values.shape[0]
        p0, p1 = ops.padded
        frames = torch.zeros((b, p0 * p1), dtype=self.cdtype, device=self.device)
        frames[:, self._bf_flat] = values.to(self.cdtype)
        f_hat = torch.fft.fft2(frames.view(b, p0, p1) * ops.psi)
        # B(q) = Σ_k ψ(k + q) ψ*(k) V(k) reads the spectrum of V·ψ* at −ν, which for real V is
        # the conjugate of V·ψ's at ν
        spectra = self._conj_a_hat[:, None] * f_hat[None] - self._b_hat[:, None] * f_hat.conj()[None]
        return ops.to_grid(torch.fft.ifft2(spectra))

    def _contract_patch(self, values: torch.Tensor) -> torch.Tensor:
        """`_contract` on the patch grid as two matrix products (`_PatchProducts`). The frames are
        laid out transposed, so that the FFT's output feeds the first product without a copy."""
        pr = self._products
        b = values.shape[0]
        p0, p1 = self._ops.padded
        s0, s1 = self._ops.sub
        n0 = self.grid_shape[0] // s0
        frames = torch.zeros((b, p1 * p0), dtype=self.cdtype, device=self.device)
        frames[:, pr.bf_t] = values.to(self.cdtype) * pr.psi_bf
        f_t = torch.fft.fft2(frames.view(b, p1, p0))                   # F transposed: (B, v, u)
        xy = torch.view_as_real(f_t).reshape(b, p1, 2 * p0).transpose(0, 1)   # (v, B, (u, Re/Im))
        t = torch.view_as_complex(torch.bmm(xy, pr.w).view(p1, b, s1, s0 * n0, 2))
        y = torch.matmul(pr.e1, t.permute(1, 2, 0, 3))                 # (B, a', n1, (a, m))
        return y.reshape(b, -1)[:, pr.flat].view(b, *self.grid_shape)

    def _ramps(self, frequencies: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return torch.exp(-2j * math.pi * (offsets[:, None] * frequencies[None, :]).to(self.cdtype))

    def deposit(self, buffer: torch.Tensor, values: torch.Tensor, pos: torch.Tensor) -> None:
        """Add the frames at scan positions `pos` (B,), with normalized bright-field values
        `values` (B, M), to `buffer`. Each sub-batch is padded with zero frames, which add
        nothing, to a power of two (`_sub_batch`), so that its batched FFTs see a few sizes: a
        cuFFT plan for a new one takes tens of ms to make, in the middle of a live step."""
        values = values.to(self.device, self.rdtype)
        pos = pos.to(self.device, torch.long)
        q_rows, q_cols = self._q
        for lo in range(0, values.shape[0], self._batch):
            v, p = values[lo:lo + self._batch], pos[lo:lo + self._batch]
            pad = _sub_batch(v.shape[0], self._batch) - v.shape[0]
            if pad:
                v = torch.cat((v, v.new_zeros((pad, v.shape[1]))))
                p = torch.cat((p, p.new_zeros(pad)))
            if self.placement == "lattice":
                where = self._place[0][p]
                placed = self._contract(v) * self._ramps(q_rows, where[:, 0])[:, :, None]
                buffer += (placed * self._ramps(q_cols, where[:, 1])[:, None, :]).sum(0)
                continue
            base, fraction = self._place[0][p], self._place[1][p]
            # the sub-pixel part of each position, as a phase on the patch's own lattice
            patch = self._contract(v) * self._ramps(q_rows, fraction[:, 0])[:, :, None]
            patch = torch.fft.ifft2(patch * self._ramps(q_cols, fraction[:, 1])[:, None, :])
            i0, i1 = self.inner_shape
            rows = (base[:, 0:1] + self._offsets[0][None, :]) % i0
            cols = (base[:, 1:2] + self._offsets[1][None, :]) % i1
            flat = rows[:, :, None] * i1 + cols[:, None, :]
            buffer.index_add_(0, flat.reshape(-1), torch.view_as_real(patch).reshape(-1, 2))

    def numerator(self, buffer: torch.Tensor) -> torch.Tensor:
        """b(q) on the canvas lattice, from a buffer."""
        if self.placement == "lattice":
            return buffer
        canvas = torch.view_as_complex(buffer.view(*self.inner_shape, 2).contiguous())
        return torch.where(self._numerator_mask, torch.fft.fft2(canvas), 0)

    def image(self, buffer: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        """The OBF phase image, (Ny·U, Nx·U) on the device, zero-mean, optionally multiplied in
        Fourier space by `base` (the parallax readout's Butterworth / Nyquist filters)."""
        obj = torch.fft.ifft2(self.numerator(buffer) * self._operator).real
        # the inverse transform divides by the pixel count, so the same coefficients give an
        # image proportional to the pixel area; the requested canvas holds only part of the inner
        # one, whose mean is zero, so its own mean is set to zero as the solve sets every other's
        obj = self._reader(obj) * self._scale
        obj = obj - obj.mean()
        if base is not None:
            obj = torch.fft.ifft2(torch.fft.fft2(obj) * base).real
        return obj

    def to(self, device) -> "ObfSetup":
        """The same setup on another device."""
        device = torch.device(device)

        def move(t):
            return None if t is None else t.to(device)

        pr = self._products
        if pr is not None:
            pr = _PatchProducts(w=move(pr.w), e1=move(pr.e1), flat=move(pr.flat),
                                bf_t=move(pr.bf_t), psi_bf=move(pr.psi_bf))

        ops = self._ops
        ops = _Operands(padded=ops.padded, sub=ops.sub, minus=move(ops.minus), plus=move(ops.plus),
                        row_lookup=move(ops.row_lookup), col_lookup=move(ops.col_lookup),
                        kept=move(ops.kept))
        return dataclasses.replace(
            self, device=device, gram=move(self.gram), _ops=ops, _bf_flat=move(self._bf_flat),
            _conj_a_hat=move(self._conj_a_hat), _b_hat=move(self._b_hat),
            _q=tuple(move(t) for t in self._q), _place=tuple(move(t) for t in self._place),
            _offsets=None if self._offsets is None else tuple(move(t) for t in self._offsets),
            _operator=move(self._operator), _numerator_mask=move(self._numerator_mask),
            _reader=self._reader.to(device), _products=pr)

    def nbytes(self) -> int:
        """Device memory the setup holds (roughly)."""
        tensors = [self.gram, self._ops.minus, self._ops.plus, self._conj_a_hat, self._b_hat,
                   self._operator, *self._place]
        if self._products is not None:
            tensors.append(self._products.w)
        return int(sum(t.numel() * t.element_size() for t in tensors))


def reconstruct_once(store: torch.Tensor, visits: torch.Tensor, kind: str, geometry, *,
                     setup: ObfSetup | None = None, subtract_frame_mean: bool = False,
                     device=None, dtype=None, chunk: int = 4096,
                     base: torch.Tensor | None = None) -> torch.Tensor:
    """OBF of every position a `latest` store holds (`visits > 0`), in one call: the oracle of
    the streaming paths, and the offline entry point."""
    if setup is None:
        setup = ObfSetup.build(geometry, device=device if device is not None else store.device,
                               dtype=dtype)
    held = torch.nonzero(visits > 0).squeeze(1).to(store.device)
    mean = bright_field_mean(store, kind, held)
    buffer = setup.buffer()
    for pos in held.split(chunk):
        values = normalize(_raw_float(store[pos], kind), mean, subtract_frame_mean)
        setup.deposit(buffer, values.to(setup.device), pos.to(setup.device))
    return setup.image(buffer, base)
