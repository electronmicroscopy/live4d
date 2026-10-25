"""Parallax / tcBF engine: build the acquisition geometry (host, NumPy) and accumulate the
tilt-corrected bright-field reconstruction on the device (PyTorch), casting integer frames to
float on the GPU.

Two pieces, a clean host-prep / device-accumulate split:

- `preprocess_geometry(...)` — pure NumPy, no GPU, off the hot path. From the acquisition
  geometry (energy, convergence angle, detector/scan sampling) and the aberration coefficients
  (defocus C10, astigmatism C12/phi12) it builds (a) `bf_flat_inds` — the flat indices of the
  bright-field-disk pixels in a *center-centered* detector frame — and (b) `shifts` — the integer
  parallax shift (in upsampled-scan pixels) each BF pixel's tilted view must be moved by to
  realign it. The shift is `∇χ / 2π` in Ångstroms (χ = the aberration phase surface), i.e. for
  pure defocus simply `tilt × C10`. Fully unit-testable.

- `ParallaxAccumulator` — the hot path. Each batch of *integer* frames is uploaded as-is (or
  arrives already on the device from a Holoscan DLPack sink), cast to float on the GPU, reduced to
  its M bright-field pixels, and **scatter-added** into the reconstruction with each BF pixel
  toroidally shifted by its precomputed parallax shift. This scatter-add — one `index_add_` over
  B·M contributions per batch — is the heavy GPU compute. With `subtract_frame_mean=True` each
  frame's mean over its BF pixels is removed first, as the reference does: that also removes the
  amplitude contrast (the bright-field intensity at each position) below about one over the
  probe size, so it is off by default.

The math is ported from Georgios Varnavides's streaming-parallax reference; the `spatial_frequencies`
kx/ky naming and the `(dy, dx) = shift` unpacking are preserved so correctness can be pinned by
matching a float64 transcription of that reference in the tests (and the engine, with
`subtract_frame_mean=True`, reproduces the reference's own output on the public apoferritin dataset,
Zenodo 10.5281/zenodo.18346854, to float32 precision). Three deliberate departures from the
reference script:

- The per-frame mean is kept unless `subtract_frame_mean=True` (see above).

- The scatter uses independent `% Ny` / `% Nx` wraps — the correct toroidal boundary (the
  reference's phase-flip kernel, not ported, has a flat-wrap bug).
- The reference's `n_rot` block — which moves every BF-disk index by one detector pixel whenever
  the rotation *rounds* to a quarter turn, i.e. for any |rotation| ≥ 45° (e.g. 77.8°), not only
  for exact quarter turns — is omitted. It is even-grid bookkeeping for quarter-turn-rotated image
  data; here the disk sits at the measured `probe_center` and is never moved. For |rotation| < 45°
  the two are identical; beyond that they differ by exactly that one-pixel disk offset.

Frames are assumed **center-centered** (BF disk at `probe_center`, default the detector center),
matching real detector frames and the fftshift correction baked into `bf_flat_inds`.

Detector orientation follows LiberTEM-parallax: a transposed or reversed detector is described by
`rotation_angle` plus `flip_cols` (LiberTEM's `detector_flip_cols`, i.e. `frames[..., ::-1]`). The
flip costs nothing per frame here: it only mirrors the column of each BF pixel in `bf_flat_inds`.
For a flipped or transposed layout give `probe_center` as measured on the raw frames.

Besides the reconstruction itself the accumulator keeps two small side accumulators (the visits per
scan position and the running per-BF-pixel sum) from which `snapshot()` derives the live channels
— per-BF scan-mean removal, the phase-flipped image, coverage — see `readout.py`.

Repeated scans (the microscope scanning continuously while the operator tunes it) are handled by
two options. `wrap_ids` maps image ids past the end of the scan back onto it (id % positions), so
one long series, or several series via `start_series()`, cover the scan again and again. And
`accumulate` chooses what the image holds:

- "all": the sum of every frame received (one scan, or an average over repeated ones);
- "latest": only the most recent frame at each scan position. The bright-field values of the
  newest frame per position are kept on the device; when a position is revisited, its new
  contribution minus its stored one is deposited at the same (unchanged) output pixels, so the
  image is always exactly the reconstruction of the last full scan pass, and follows changes on
  the microscope within one pass. Costs an (positions × M) store and one extra gather per batch.
- "reset": each pass starts a fresh image (pass k = ids k·positions … (k+1)·positions − 1); the
  live view is the current pass filling in, the saved image the last completed pass. No store.
- "swap": like "reset", but the live view shows the current pass only on the output rows it has
  completed and the last completed pass elsewhere, so it is never partial or seamed by more than
  the parallax shifts. With `pass_average` β < 1 the completed passes are blended,
  front ← (1 − β)·front + β·pass. Costs one more image, not a store.

Pass-aware modes split each batch at pass boundaries (one host sync per batch) and call
`on_pass_complete(k)` when pass k is complete: the first frame of pass k + 1 arrived, or its series
ended on its last position (`end_series()`). Frames that arrive after their pass was closed (a
connection's stragglers) go into the completed pass ("swap"), are dropped ("reset"), or are
deposited as usual; `late_frames` counts them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from realtime_stem.processing.parallax.readout import Readout, ReadoutConfig

__all__ = ["ParallaxGeometry", "preprocess_geometry", "ParallaxAccumulator"]

_NO_ID = 2 ** 62          # "no frame yet" for the smallest-id trackers


# ── geometry (host NumPy, ported from the streaming reference) ─────────────────

def _electron_wavelength(energy: float) -> float:
    """Relativistic de Broglie wavelength in Ångstroms for an electron of `energy` eV."""
    m, e, c, h = 9.109383e-31, 1.602177e-19, 299792458.0, 6.62607e-34
    lam = h / np.sqrt(2 * m * e * energy) / np.sqrt(1 + e * energy / 2 / m / c**2)
    return lam * 1e10


def _spatial_frequencies(gpts, sampling, rotation_angle=None):
    """fftfreq grids for the detector, with an optional scan/detector rotation applied."""
    ny, nx = gpts
    sy, sx = sampling
    kx = np.fft.fftfreq(ny, sy)
    ky = np.fft.fftfreq(nx, sx)
    kxa, kya = np.meshgrid(kx, ky, indexing="ij")
    if rotation_angle is not None:
        cos_t, sin_t = np.cos(rotation_angle), np.sin(rotation_angle)
        kxa, kya = cos_t * kxa - sin_t * kya, sin_t * kxa + cos_t * kya
    return kxa, kya


def _polar_coordinates(kx, ky):
    return np.sqrt(kx**2 + ky**2), np.arctan2(ky, kx)


def _quadratic_aberration_cartesian_gradients(alpha, phi, aberration_coefs):
    """∇χ in Cartesian (x, y) for the C10/C12 aberration surface — already ×2π (see reference)."""
    C10 = aberration_coefs.get("C10", 0.0)
    C12 = aberration_coefs.get("C12", 0.0)
    phi12 = aberration_coefs.get("phi12", 0.0)

    cos2, sin2 = np.cos(2.0 * (phi - phi12)), np.sin(2.0 * (phi - phi12))
    scale = 2 * np.pi
    dchi_dk = scale * alpha * (C10 + C12 * cos2)
    dchi_dphi = -scale * alpha * (C12 * sin2)

    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    dchi_dx = cos_phi * dchi_dk - sin_phi * dchi_dphi
    dchi_dy = sin_phi * dchi_dk + cos_phi * dchi_dphi
    return dchi_dx, dchi_dy


@dataclass(frozen=True)
class ParallaxGeometry:
    """Preprocessed acquisition geometry for the parallax accumulator (all host-side).

    `bf_flat_inds` and `shifts` are the only things the hot path needs; the rest is metadata
    kept for readability, the result plot, and later online aberration refinement.
    """
    bf_flat_inds: np.ndarray            # (M,) int — flat idx into a center-centered (H*W) frame
    shifts: np.ndarray                  # (M, 2) int — (dy, dx) parallax shift in upsampled px
    detector_shape: tuple               # (H, W)
    scan_shape: tuple                   # (Ny, Nx) original scan grid
    upsampling_factor: int
    upsampled_scan_gpts: tuple          # (Ny*U, Nx*U) — output reconstruction shape
    wavelength: float = 0.0             # Å
    sampling: tuple = (0.0, 0.0)        # detector real-space sampling (Å)
    upsampled_scan_sampling: tuple = (0.0, 0.0)
    aberration_coefs: dict = field(default_factory=dict)
    rotation_angle: float | None = None # rad
    semiangle_cutoff: float = 0.0       # mrad (the readout band limit)
    flip_cols: bool = False
    probe_center: tuple | None = None   # (row, col) px on the raw frame; None → frame center
    n_clipped: int = 0                  # BF-disk pixels that fall outside the frame (dropped)

    @property
    def n_bf(self) -> int:
        return int(self.bf_flat_inds.shape[0])


def preprocess_geometry(
    det_shape: tuple[int, int],
    scan_shape: tuple[int, int],
    scan_sampling: tuple[float, float],
    energy: float,
    semiangle_cutoff: float,
    angular_sampling: tuple[float, float] | None = None,
    reciprocal_sampling: tuple[float, float] | None = None,
    aberration_coefs: dict[str, float] | None = None,
    rotation_angle: float | None = None,
    upsampling_factor: int = 1,
    probe_center: tuple[float, float] | None = None,
    flip_cols: bool = False,
) -> ParallaxGeometry:
    """Build BF-pixel indices + integer parallax shifts from the acquisition geometry.

    Args:
        det_shape: (H, W) detector pixels.
        scan_shape: (Ny, Nx) raster scan grid.
        scan_sampling: (sy, sx) scan step in Ångstroms.
        energy: beam energy in eV.
        semiangle_cutoff: probe convergence semiangle in mrad (sets the BF-disk radius).
        angular_sampling | reciprocal_sampling: detector pixel size, in mrad/px or 1/Å/px
            respectively — supply exactly one (angular is the usual microscope readout).
        aberration_coefs: {"C10", "C12", "phi12"} in Å / Å / rad. Empty → in-focus (no shifts).
        rotation_angle: scan/detector relative rotation in radians (None → none).
        upsampling_factor: integer output-grid upsampling (finer shifts); 1 → scan grid.
        probe_center: BF-disk center on the detector, (row, col) px — py4DSTEM's
            (Center_X, Center_Y) from `get_probe_size`, measured on the raw frames. None → the
            frame center (for `flip_cols`, the mirror of the canonical center, col W-1-W//2).
        flip_cols: the detector's columns are reversed relative to the canonical (Q_rows, Q_cols)
            layout — LiberTEM-parallax's `detector_flip_cols`. Combined with `rotation_angle` this
            covers every transpose/flip (see the table in docs/GUIDE.md).

    Returns:
        ParallaxGeometry with `bf_flat_inds` (M,) and `shifts` (M, 2), plus metadata.
    """
    if (angular_sampling is None) == (reciprocal_sampling is None):
        raise ValueError("Specify exactly one of `angular_sampling` or `reciprocal_sampling`.")
    if aberration_coefs is None:
        aberration_coefs = {}

    wavelength = _electron_wavelength(energy)
    if reciprocal_sampling is None:
        reciprocal_sampling = (angular_sampling[0] / wavelength / 1e3,
                               angular_sampling[1] / wavelength / 1e3)

    gpts = (int(det_shape[0]), int(det_shape[1]))
    sampling = (1.0 / reciprocal_sampling[0] / gpts[0],
                1.0 / reciprocal_sampling[1] / gpts[1])
    upsampled_scan_gpts = (scan_shape[0] * upsampling_factor,
                           scan_shape[1] * upsampling_factor)
    upsampled_scan_sampling = (scan_sampling[0] / upsampling_factor,
                               scan_sampling[1] / upsampling_factor)

    kxa, kya = _spatial_frequencies(gpts, sampling, rotation_angle)
    k, phi = _polar_coordinates(kxa, kya)

    # BF disk: tilt angle α = k·λ (rad) ≤ cutoff (mrad → ×1e3). where() gives fft-order indices;
    # each maps to a signed tilt offset (px from the disk center), and the disk is placed at
    # `probe_center` in the real frame layout (None → frame center, i.e. a centered detector).
    bf_mask = k * wavelength * 1e3 <= semiangle_cutoff
    inds_i, inds_j = np.where(bf_mask)
    off_i = ((inds_i + gpts[0] // 2) % gpts[0]) - gpts[0] // 2
    off_j = ((inds_j + gpts[1] // 2) % gpts[1]) - gpts[1] // 2
    # A column flip mirrors the tilt offsets about the center: the raw frame is `canonical[:, ::-1]`,
    # so canonical column cj_c + off sits at raw column W-1-(cj_c+off) = cj_raw - off.
    ci = gpts[0] // 2 if probe_center is None else int(round(probe_center[0]))
    if probe_center is not None:
        cj = int(round(probe_center[1]))
    else:
        cj = gpts[1] - 1 - gpts[1] // 2 if flip_cols else gpts[1] // 2
    det_i, det_j = ci + off_i, (cj - off_j if flip_cols else cj + off_j)
    ok = (det_i >= 0) & (det_i < gpts[0]) & (det_j >= 0) & (det_j < gpts[1])
    n_clipped = int((~ok).sum())
    if not ok.all():  # disk clips the frame edge — those tilted views fall off the detector
        inds_i, inds_j, det_i, det_j = inds_i[ok], inds_j[ok], det_i[ok], det_j[ok]
    bf_flat_inds = (det_i * gpts[1] + det_j).astype(np.int32)

    dchi_dx, dchi_dy = _quadratic_aberration_cartesian_gradients(k * wavelength, phi, aberration_coefs)
    grad_k = np.stack((dchi_dx[inds_i, inds_j], dchi_dy[inds_i, inds_j]), axis=-1)
    shifts = np.round(grad_k / (2 * np.pi) / upsampled_scan_sampling).astype(np.int32)

    return ParallaxGeometry(
        bf_flat_inds=bf_flat_inds, shifts=shifts, detector_shape=gpts,
        scan_shape=(int(scan_shape[0]), int(scan_shape[1])),
        upsampling_factor=int(upsampling_factor), upsampled_scan_gpts=upsampled_scan_gpts,
        wavelength=wavelength, sampling=sampling,
        upsampled_scan_sampling=upsampled_scan_sampling, aberration_coefs=dict(aberration_coefs),
        rotation_angle=rotation_angle, semiangle_cutoff=float(semiangle_cutoff),
        flip_cols=bool(flip_cols),
        probe_center=None if probe_center is None else (float(probe_center[0]),
                                                        float(probe_center[1])),
        n_clipped=n_clipped,
    )


# ── accumulator (device torch) ─────────────────────────────────────────────────

def _pick_device(device: str | None) -> torch.device:
    """Default to CUDA when available (the edge box), then Apple's MPS, else CPU."""
    if device is not None and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ParallaxAccumulator:
    """Accumulate the tcBF reconstruction on the device via a batched shifted scatter-add.

    Usage:
        geom = preprocess_geometry(...)
        acc = ParallaxAccumulator(geom)
        acc.process(frames, ids)     # frames: (B, H, W) int/float; ids: (B,) scan indices
        recon = acc.result()         # (Ny*U, Nx*U) host array — the raw shifted sum
        stack, names, meta = acc.snapshot()   # live channels, see readout.py

    `process` accepts either host integer ndarrays or an integer/float torch tensor already on the
    device (GPU decode path, or a Holoscan DLPack sink) — the `Accumulator` contract shared by
    every algorithm.
    """

    def __init__(self, geometry: ParallaxGeometry, device: str | None = None,
                 measure: bool = False, readout: ReadoutConfig | None = None,
                 accumulate: str = "all", wrap_ids: bool = False,
                 subtract_frame_mean: bool = False, track_passes: bool = False,
                 pass_average: float = 1.0):
        if accumulate not in ("all", "latest", "reset", "swap"):
            raise ValueError(f"accumulate must be 'all', 'latest', 'reset' or 'swap', not "
                             f"{accumulate!r}")
        if not 0.0 < pass_average <= 1.0:
            raise ValueError(f"pass_average must be in (0, 1], not {pass_average}")
        self.device = _pick_device(device)
        self.geometry = geometry
        self.subtract_frame_mean = bool(subtract_frame_mean)
        self.detector_shape = tuple(geometry.detector_shape)
        self.scan_shape = tuple(geometry.scan_shape)
        self.upsampled_scan_gpts = tuple(geometry.upsampled_scan_gpts)

        self._nx = self.scan_shape[1]                       # original scan width (ids → coords)
        self._u = int(geometry.upsampling_factor)
        self._nyu, self._nxu = self.upsampled_scan_gpts

        # Device-resident geometry: BF indices + the two shift columns (dy, dx per the reference).
        self._bf_inds = torch.as_tensor(np.asarray(geometry.bf_flat_inds), dtype=torch.long,
                                        device=self.device)
        shifts = np.asarray(geometry.shifts)
        self._sy = torch.as_tensor(shifts[:, 0], dtype=torch.long, device=self.device)
        self._sx = torch.as_tensor(shifts[:, 1], dtype=torch.long, device=self.device)

        # Flat reconstruction buffer on the device; reshaped to 2D only on readout. Flat makes the
        # shifted scatter a single index_add_.
        n = self._nyu * self._nxu
        self.out = torch.zeros(n, dtype=torch.float32, device=self.device)

        # Side accumulators for the readout (O(B·M) adds per batch, no extra scatter): how many
        # frames each scan position contributes (0/1 for "latest"), and each BF pixel's running
        # sum of its deposited values. The readout operators are built lazily, on the
        # first snapshot.
        self.n_positions = self.scan_shape[0] * self.scan_shape[1]
        self._visits = torch.zeros(self.n_positions, dtype=torch.float32, device=self.device)
        self._bf_sum = torch.zeros(geometry.n_bf, dtype=torch.float32, device=self.device)
        self._n_frames = 0                                  # frames in the current image
        self._n_total = 0                                   # frames received

        # Repeated scans: ids past the scan wrap onto it; `_id_offset` continues the count across
        # series whose ids restart at 0; `_max_id` (a device scalar, no per-batch sync) gives the
        # scan pass and the cursor position for the live metadata.
        self.accumulate = accumulate
        self.wrap_ids = bool(wrap_ids)
        self._id_offset = 0
        self._max_id = torch.full((), -1, dtype=torch.long, device=self.device)
        # The smallest id deposited since the current image was last emptied (built, a pass
        # restarted, a retune restarted it): where it began filling in, for `complete_rows`.
        self._fill_min = torch.full((), _NO_ID, dtype=torch.long, device=self.device)
        self._store: torch.Tensor | None = None     # "latest": newest raw BF values per position
        self._store_kind: str | None = None

        # Passes (one scan of the area each, in raster order): "reset" / "swap" restart the image
        # at each pass and keep the last completed one as the front (`_close_pass`);
        # `track_passes` reports completed passes in any mode (`on_pass_complete(k)`).
        self._track = accumulate in ("reset", "swap") or bool(track_passes)
        self.pass_average = float(pass_average)
        self.on_pass_complete = None
        self.passes_completed = 0
        self.late_frames = 0
        self._pass = -1
        self._pass_closed = False
        self._front: dict | None = None
        self._row_source = self._rows_from(shifts)
        self._sources = self._sources_from(shifts)
        self.readout_config = readout if readout is not None else ReadoutConfig()
        self._readout: Readout | None = None

        # Instrumentation (off by default): host prep, integer H2D copy, GPU cast+scatter, with
        # CUDA events so measurement doesn't perturb the run.
        self.measure = measure
        self._cuda = self.device.type == "cuda"
        self._host_prep_s = 0.0
        self._compute_s = 0.0
        self._events: list[tuple] = []

    def _to_int_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """Wrap integer frames as a torch tensor for upload, avoiding a host float cast.

        Widths take a small *integer* widening where torch lacks the unsigned dtype's kernels; the
        float conversion itself happens on the device. This host path is not the live hot path
        (the Holoscan ingest hands over device tensors), so an integer copy is fine.
        """
        a = np.ascontiguousarray(frames)
        dt = a.dtype
        if dt == np.uint32:
            # int64 widening, NOT an int32 bit-view: a bit-view turns values >= 2^31 negative,
            # and real ARINA uint32 frames carry a dtype-max (4294967295) pixel-mask sentinel.
            return torch.from_numpy(a.astype(np.int64))
        if dt == np.float64 and self.device.type == "mps":
            return torch.from_numpy(a.astype(np.float32))   # MPS has no float64
        if dt == np.uint8 or dt in (np.int8, np.int16, np.int32, np.float32, np.float64):
            return torch.from_numpy(a)
        return torch.from_numpy(a.astype(np.int32))  # e.g. uint16: safe widening

    def process(self, frames, ids) -> None:
        """Upload a batch (integer), cast to float + shifted scatter-add on the GPU, by scan id.

        `ids` may be a host array-like or a torch tensor (any device) — the Holoscan sink
        passes the ingest's device tensor straight through to avoid a per-batch D2H sync."""
        if tuple(frames.shape[1:]) != self.detector_shape:
            raise ValueError(
                f"frame detector shape {tuple(frames.shape[1:])} != configured "
                f"{self.detector_shape} — the detector doesn't match the geometry."
            )

        # Already a device tensor (GPU decode / Holoscan DLPack): no host prep, no H2D here.
        if isinstance(frames, torch.Tensor):
            if frames.dtype == torch.float64 and self.device.type == "mps":
                frames = frames.float()                       # MPS has no float64
            f = frames if frames.device == self.device else frames.to(self.device)
            if not self.measure:
                self._accumulate_into(f, ids)
                return
            if self._cuda:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                self._accumulate_into(f, ids)
                e1.record()
                self._events.append((e0, e0, e1))
            else:
                t1 = time.perf_counter()
                self._accumulate_into(f, ids)
                self._compute_s += time.perf_counter() - t1
            return

        if not self.measure:
            f_int = self._to_int_tensor(frames).to(self.device, non_blocking=True)
            self._accumulate_into(f_int, ids)
            return

        # Measured path.
        t0 = time.perf_counter()
        t_int = self._to_int_tensor(frames)
        self._host_prep_s += time.perf_counter() - t0

        if self._cuda:
            e_copy0 = torch.cuda.Event(enable_timing=True)
            e_copy1 = torch.cuda.Event(enable_timing=True)
            e_comp1 = torch.cuda.Event(enable_timing=True)
            e_copy0.record()
            g = t_int.to(self.device, non_blocking=True)
            e_copy1.record()
            self._accumulate_into(g, ids)
            e_comp1.record()
            self._events.append((e_copy0, e_copy1, e_comp1))
        else:
            t1 = time.perf_counter()
            self._accumulate_into(t_int, ids)
            self._compute_s += time.perf_counter() - t1

    def _gather(self, flat_f: torch.Tensor) -> tuple[torch.Tensor, str]:
        """The (B, M) bright-field pixels of flat frames, still integer, and how to float them.

        Gather BEFORE casting to float: only M of the H·W pixels survive, so casting the whole
        batch first would allocate and write B·H·W floats per batch for nothing.
        """
        if flat_f.dtype == torch.uint16:
            # uint16 is a second-class torch dtype with no CUDA gather kernel: reinterpret the
            # bits as int16 (zero-copy view), gather the M columns, then recover the exact
            # unsigned values on the small (B, M) slice (`_as_float`). float32 holds ≤ 2^24
            # exactly, so 0..65535 round-trips exactly.
            return flat_f.view(torch.int16)[:, self._bf_inds], "u16"
        if flat_f.dtype == torch.uint32:
            # Same trick for uint32 (what the ingest emits for the DCU's long-dwell modes, and
            # equally second-class in torch): int32 bit-view, gather, undo the wrap on the small
            # (B, M) slice. Values >= 2^24 round in float32 exactly as a direct cast would.
            return flat_f.view(torch.int32)[:, self._bf_inds], "u32"
        return flat_f[:, self._bf_inds], "plain"

    def _as_float(self, raw: torch.Tensor, kind: str) -> torch.Tensor:
        """Gathered BF pixels → float32, the per-frame mean removed if `subtract_frame_mean`."""
        g = raw.float()
        if kind == "u16":
            g = torch.where(g < 0, g + 65536.0, g)
        elif kind == "u32":
            g = torch.where(g < 0, g + 4294967296.0, g)
        if self.subtract_frame_mean:
            g = g - g.mean(dim=1, keepdim=True)
        return g

    def _ids(self, ids) -> torch.Tensor:
        """The batch's image ids on the device, continued across series (`start_series`); also
        advances the scan cursor (`_max_id`, a device scalar: no sync)."""
        if isinstance(ids, torch.Tensor):
            # Device ids from the Holoscan sink — keep them on-device (no numpy round trip).
            idx = ids.to(device=self.device, dtype=torch.long, non_blocking=True)
        else:
            idx = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.device)
        if self.wrap_ids:
            idx = idx + self._id_offset
        self._max_id = torch.maximum(self._max_id, idx.max())
        return idx

    def _accumulate_into(self, f: torch.Tensor, ids) -> None:
        # Gather the M bright-field pixels of every frame (minus the frame's mean over them, if
        # `subtract_frame_mean`), then scatter-add each contribution to its parallax-shifted,
        # toroidally-wrapped output position. Independent % Ny / % Nx wraps = the correct scan-grid torus.
        b = f.shape[0]
        raw, kind = self._gather(f.reshape(b, -1))
        idx = self._ids(ids)
        self._n_total += b
        if not self._track:
            self._deposit(raw, kind, idx)
            return
        # Pass-aware: split the batch at pass boundaries (one host sync per batch; the sink makes
        # one anyway). The first frame of a newer pass closes the current one.
        passes = idx // self.n_positions
        lo, hi = (int(v) for v in torch.stack((passes.min(), passes.max())).tolist())
        for k in range(lo, hi + 1):
            if lo == hi:
                r, i = raw, idx
            else:
                sel = passes == k
                r, i = raw[sel], idx[sel]
                if i.numel() == 0:
                    continue
            if k > self._pass:
                if self._pass >= 0 and not self._pass_closed:
                    self._close_pass()
                self._pass, self._pass_closed = k, False
            if k < self._pass or self._pass_closed:
                self._late(r, kind, i)
            else:
                self._deposit(r, kind, i)

    def _deposit(self, raw: torch.Tensor, kind: str, idx: torch.Tensor,
                 into: dict | None = None) -> None:
        """Fold gathered bright-field pixels `raw` (B, M) of the frames at absolute ids `idx` into
        the current image, or into the buffers `into` (`out`, `visits`, `bf_sum`, `n`)."""
        i_bf = self._as_float(raw, kind)                       # (B, M)
        pos = idx % self.n_positions if self.wrap_ids else idx
        if into is None:
            out, visits, bf_sum = self.out, self._visits, self._bf_sum
            self._n_frames += raw.shape[0]
            self._fill_min = torch.minimum(self._fill_min, idx.min())
        else:
            out, visits, bf_sum = into["out"], into["visits"], into["bf_sum"]
            into["n"] += raw.shape[0]

        if self.accumulate == "latest":
            # Replace each position's previous frame: deposit (new − old) at the same pixels.
            # The store keeps the raw gathered integers, so the old contribution is recomputed
            # exactly as it was deposited (never-visited positions hold zeros → contribute 0).
            # Assumes a batch holds each position at most once (batch ≤ scan positions).
            if self._store is None:
                self._store = torch.zeros((self.n_positions, raw.shape[1]), dtype=raw.dtype,
                                          device=self.device)
                self._store_kind = kind
            elif kind != self._store_kind or raw.dtype != self._store.dtype:
                raise ValueError(f"frame dtype changed mid-stream ({raw.dtype}, {kind}) — "
                                 f"the stored frames are {self._store.dtype}")
            i_bf = i_bf - self._as_float(self._store[pos], kind)
            self._store[pos] = raw
            visits[pos] = 1.0
        else:
            visits.index_add_(0, pos, torch.ones_like(pos, dtype=torch.float32))
        bf_sum += i_bf.sum(dim=0)

        self._scatter(i_bf, pos, out)

    def _scatter(self, i_bf: torch.Tensor, pos: torch.Tensor, out: torch.Tensor) -> None:
        """Add each position's bright-field values (B, M) at its parallax-shifted output pixels,
        wrapped on the output torus."""
        yt = (pos // self._nx) * self._u                       # (B,)
        xt = (pos % self._nx) * self._u
        oy = (yt.view(-1, 1) + self._sy.view(1, -1)) % self._nyu   # (B, M)
        ox = (xt.view(-1, 1) + self._sx.view(1, -1)) % self._nxu
        flat = (oy * self._nxu + ox).reshape(-1)               # (B*M,)
        out.index_add_(0, flat, i_bf.reshape(-1))

    def _rows_from(self, shifts: np.ndarray) -> np.ndarray:
        """"swap": for each output row, the last scan row whose shifted deposits land on it (the
        torus wrap included; -1: none). The row is complete once the cursor is past that row."""
        dys = np.unique(np.asarray(shifts)[:, 0])
        y = np.arange(self._nyu)[:, None]
        src = (y - dys[None, :]) % self._nyu
        return np.where(src % self._u == 0, src // self._u, -1).max(axis=1)

    def _sources_from(self, shifts: np.ndarray) -> np.ndarray:
        """(output rows, scan rows) bool: which scan rows' shifted deposits land on each output
        row, the torus wrap included."""
        dys = np.unique(np.asarray(shifts)[:, 0])
        src = (np.arange(self._nyu)[:, None] - dys[None, :]) % self._nyu
        y, k = np.nonzero(src % self._u == 0)
        m = np.zeros((self._nyu, self.scan_shape[0]), dtype=bool)
        m[y, src[y, k] // self._u] = True
        return m

    def set_geometry(self, geometry: ParallaxGeometry, chunk: int = 4096) -> str:
        """Re-shift with another geometry while running: new C10, C12 or rotation, with the same
        bright-field pixels and output grid. "latest" re-deposits every frame it holds, so the
        image is at once what the new geometry gives for them; the other modes keep no frames,
        so their image (and "reset" / "swap"'s last completed pass) restarts. Returns which."""
        if (not np.array_equal(np.asarray(geometry.bf_flat_inds),
                               np.asarray(self.geometry.bf_flat_inds))
                or tuple(geometry.upsampled_scan_gpts) != self.upsampled_scan_gpts):
            raise ValueError("a new geometry must keep the bright-field pixels and the output "
                             "grid: only C10, C12 and the rotation can change while running")
        shifts = np.asarray(geometry.shifts)
        self.geometry = geometry
        self._sy = torch.as_tensor(shifts[:, 0], dtype=torch.long, device=self.device)
        self._sx = torch.as_tensor(shifts[:, 1], dtype=torch.long, device=self.device)
        self._row_source = self._rows_from(shifts)
        self._sources = self._sources_from(shifts)
        self._readout = None                                   # χ of the phase flip
        self.out.zero_()
        if self.accumulate == "latest" and self._store is not None:
            held = torch.nonzero(self._visits > 0).squeeze(1)
            for pos in held.split(chunk):
                self._scatter(self._as_float(self._store[pos], self._store_kind), pos, self.out)
            return f"re-rendered the {held.numel():,} frames held"
        self._visits.zero_()
        self._bf_sum.zero_()
        self._n_frames = 0
        self._fill_min.fill_(_NO_ID)
        self._front = None
        return "restarted the image (only --accumulate latest keeps the frames to re-render)"

    def _late(self, raw: torch.Tensor, kind: str, idx: torch.Tensor) -> None:
        """Frames of a pass already closed (a connection's stragglers past the boundary): into
        the completed pass for "swap", dropped for "reset" (its image is the current pass),
        deposited as usual for "latest" / "all"."""
        self.late_frames += int(idx.numel())
        if self.accumulate in ("reset", "swap"):
            if self.accumulate == "swap" and self._front is not None:
                self._deposit(raw, kind, idx, into=self._front)
            return
        self._deposit(raw, kind, idx)

    def _close_pass(self) -> None:
        """Pass `_pass` is complete (a newer one started, or its series ended on its last
        position). "reset" / "swap": it becomes the front image (for "swap" with
        `pass_average` β < 1, front ← (1 − β)·front + β·pass) and the current image restarts.
        Then `on_pass_complete(k)` is called, while `images()` holds pass k."""
        k = self._pass
        if self.accumulate in ("reset", "swap"):
            beta = self.pass_average if self.accumulate == "swap" else 1.0
            if self._front is None or beta >= 1.0:
                self._front = {"out": self.out.clone(), "visits": self._visits.clone(),
                               "bf_sum": self._bf_sum.clone(), "n": self._n_frames}
            else:
                for key, cur in (("out", self.out), ("visits", self._visits),
                                 ("bf_sum", self._bf_sum)):
                    self._front[key].mul_(1.0 - beta).add_(cur, alpha=beta)
                self._front["n"] = (1.0 - beta) * self._front["n"] + beta * self._n_frames
            self.out.zero_()
            self._visits.zero_()
            self._bf_sum.zero_()
            self._n_frames = 0
            self._fill_min.fill_(_NO_ID)
        self._pass_closed = True
        self.passes_completed += 1
        if self.on_pass_complete is not None:
            self.on_pass_complete(k)

    def end_series(self) -> None:
        """The detector series ended (the runner calls this after each stream). In a pass-aware
        mode a pass whose last position arrived is closed now, not when the next pass starts."""
        if not self._track or self._pass < 0 or self._pass_closed:
            return
        max_id = int(self._max_id)
        if max_id // self.n_positions == self._pass and \
                max_id % self.n_positions == self.n_positions - 1:
            self._close_pass()

    def result(self) -> np.ndarray:
        """The raw parallax / tcBF sum as a (Ny*U, Nx*U) host array (no readout processing): for
        "reset" / "swap" the last completed pass, once there is one."""
        out = self._front["out"] if self._front is not None else self.out
        return out.reshape(self.upsampled_scan_gpts).cpu().numpy()

    def start_series(self) -> None:
        """A new detector series begins (the runner calls this per round). With `wrap_ids`, its
        ids — which restart at 0 — continue after the last pass seen instead of overlapping it."""
        if self.wrap_ids:
            max_id = int(self._max_id)
            if max_id >= 0:
                self._id_offset = (max_id // self.n_positions + 1) * self.n_positions

    def store_bytes(self, itemsize: int = 2) -> int:
        """Device memory the "latest" store needs (or holds), in bytes."""
        if self._store is not None:
            return self._store.numel() * self._store.element_size()
        return self.n_positions * self.geometry.n_bf * itemsize if self.accumulate == "latest" else 0

    def _window_frames(self) -> int:
        """Frames currently in the image: all received, or one per visited position."""
        if self.accumulate == "latest":
            return int(self._visits.sum())
        return self._n_frames

    def _channels_of(self, out, visits, bf_sum, n_frames) -> dict[str, torch.Tensor]:
        if self._readout is None:
            self._readout = Readout(self.geometry, self.readout_config, self.device)
        return self._readout.compute(out, visits, bf_sum, n_frames)

    def _rows_done(self) -> int:
        """Scan rows the current pass has completed (frames arrive in raster order)."""
        max_id = int(self._max_id)
        if max_id < 0 or max_id // self.n_positions != self._pass:
            return 0
        return (max_id % self.n_positions + 1) // self._nx

    def complete_rows(self) -> list[int] | None:
        """[r0, r1): the output rows of the live image that hold a finished image while it fills
        in from empty, i.e. the longest run of rows whose depositing scan rows (`_sources`, the
        torus wrap included) have all been received since the image was emptied. That is a
        scan's first pass, every pass of "reset", and any image started mid-run: a mode switched
        to, or restarted by a retune. The rest is blank or partly deposited. None once every row
        is complete ("swap" shows the last completed pass on the others)."""
        if self._front is not None and (self.accumulate == "swap" or self._n_frames == 0):
            return None
        max_id, start = int(self._max_id), int(self._fill_min)
        if max_id < 0 or start > max_id:
            return [0, 0]
        span = max_id - start + 1                    # positions since the first, in raster order
        if span >= self.n_positions:
            return None
        nx = self._nx
        covered = (np.arange(self.scan_shape[0]) * nx - start) % self.n_positions + nx <= span
        done = ~(self._sources & ~covered[None, :]).any(axis=1)
        if done.all():
            return None
        edges = np.flatnonzero(np.diff(np.concatenate(([0], done.astype(np.int8), [0]))))
        if edges.size == 0:
            return [0, 0]
        starts, ends = edges[::2], edges[1::2]
        k = int(np.argmax(ends - starts))
        return [int(starts[k]), int(ends[k])]

    def channels(self) -> dict[str, torch.Tensor]:
        """The live readout channels (`parallax`, `phase_flipped`, …) as device tensors.

        "reset": the current pass alone. "swap": the current pass on the output rows it has
        completed (every scan row whose shifted deposits land there is done, the torus wrap
        included) and the last completed pass elsewhere, each read out on its own. Both show the
        last completed pass while the current one is empty."""
        current = (self.out, self._visits, self._bf_sum, self._window_frames())
        if self._front is None:
            return self._channels_of(*current)
        F = self._front
        if self._n_frames == 0:
            return self._channels_of(F["out"], F["visits"], F["bf_sum"], F["n"])
        if self.accumulate == "reset":
            return self._channels_of(*current)
        back = self._channels_of(*current)
        front = self._channels_of(F["out"], F["visits"], F["bf_sum"], F["n"])
        done = torch.as_tensor(self._row_source < self._rows_done(), device=self.device)
        return {k: torch.where(done.view(-1, 1), back[k], front[k]) for k in back}

    def completed_channels(self) -> dict[str, torch.Tensor]:
        """The channels of the last completed pass ("reset" / "swap"), else the live ones."""
        if self._front is None:
            return self.channels()
        F = self._front
        return self._channels_of(F["out"], F["visits"], F["bf_sum"], F["n"])

    def snapshot(self) -> tuple[np.ndarray, list[str], dict]:
        """The live channels as one (C, Ny*U, Nx*U) float32 host stack, their names, and metadata.

        The optional `Accumulator` snapshot hook: the ingest sink publishes this when present.
        """
        ch = self.channels()
        names = list(ch)
        stack = torch.stack([ch[k] for k in names]).float().cpu().numpy()
        geo = self.geometry
        max_id = int(self._max_id)
        meta = {
            "C10": float(geo.aberration_coefs.get("C10", 0.0)),
            "rotation_deg": (0.0 if geo.rotation_angle is None
                             else round(float(np.rad2deg(geo.rotation_angle)), 6)),
            "C12": float(geo.aberration_coefs.get("C12", 0.0)),
            "phi12_deg": round(float(np.rad2deg(geo.aberration_coefs.get("phi12", 0.0))), 6),
            "upsampling": self._u,
            # output pixel size (y, x) in Å: scale bars and d-spacings in the viewers
            "sampling_A": [float(v) for v in geo.upsampled_scan_sampling],
            "frames": self._n_total,
            "coverage": float((self._visits > 0).float().mean()),
            "accumulate": self.accumulate,
            "subtract_frame_mean": self.subtract_frame_mean,
            # the scan cursor: which pass over the scan (0-based) and how far into it
            "pass": max(max_id, 0) // self.n_positions,
            "progress": 0.0 if max_id < 0 else (max_id % self.n_positions + 1) / self.n_positions,
        }
        rows = self.complete_rows()
        if rows is not None:              # a partly filled image: e.g. the viewers' FFT uses these
            meta["complete_rows"] = rows
        if self._track:
            meta["passes_completed"] = self.passes_completed
            meta["late_frames"] = self.late_frames
        return stack, names, meta

    def images(self) -> dict[str, np.ndarray]:
        """Every saved image as host arrays: the raw sum plus the readout channels (for "reset" /
        "swap", of the last completed pass once there is one)."""
        out = {"raw": self.result()}
        out.update({k: v.float().cpu().numpy() for k, v in self.completed_channels().items()})
        return out

    def perf_stats(self) -> dict | None:
        """Per-stage timing for the run, or None if measurement was off."""
        if not self.measure:
            return None
        out: dict = {"host_prep_s": self._host_prep_s}
        if self._cuda and self._events:
            torch.cuda.synchronize()
            out["gpu_copy_s"] = sum(a.elapsed_time(b) for a, b, _ in self._events) / 1000.0
            out["gpu_compute_s"] = sum(b.elapsed_time(c) for _, b, c in self._events) / 1000.0
            out["n_batches"] = len(self._events)
        else:
            out["compute_s"] = self._compute_s
        return out
