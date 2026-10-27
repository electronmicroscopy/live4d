"""Readout-time processing of the parallax accumulator: everything that is linear and
shift-invariant over the scan, so it can be applied to the accumulated image instead of per frame.

The hot path (`ParallaxAccumulator._accumulate_into`) only deposits shifted bright-field values
(minus each frame's mean with `subtract_frame_mean`). Every live snapshot then derives, from that one buffer plus two small side
accumulators (the visited-position mask and the per-BF-pixel running sum):

- **phase flip**: `sign(sin χ(q))` on the scan-space frequencies. The streamed reference
  (`parallax_phase_flip_accumulate_cpu`, a grouped kernel scattered per frame) is the same operator
  applied frame by frame; by linearity filtering the partial sum gives exactly the streamed
  phase-flipped image at that moment, mid-scan included, for one FFT pair per snapshot instead of a
  U×M scatter per frame. Also Butterworth low/high-pass, Nyquist suppression and the parallax band
  limit (`2α/λ`), following quantem's `DirectPtychographyMontage._apply_fourier_filters`.
- **per-BF scan-mean removal**: subtracting each bright-field pixel's scan mean from every deposit
  (quantem's `_preprocess`) is a convolution of the comb of position visits with the per-pixel means
  placed at their shifts, so it is applied once here rather than per frame. Together with the
  per-frame mean (`subtract_frame_mean`) this is the double-centering quantem does (the two
  commute). On a fully-scanned, unupsampled wrap (where every output pixel receives each BF pixel
  once) it only removes a constant; mid-scan and with upsampling it removes the static
  bright-field pattern deposited at the scan front and on the upsampled sub-lattice. It keeps the
  amplitude contrast, which the per-frame mean removes.
- **coverage**: the number of deposits each output pixel has received, likewise a convolution of
  the comb with the shift histogram, for normalizing a partial scan (`mean * support`, quantem's
  `_moments_from_buffers`).

All convolutions are circular on the (upsampled) scan torus, matching the accumulator's independent
`% Ny` / `% Nx` wraps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["ReadoutConfig", "Readout", "quadratic_aberration_surface", "fourier_filters"]


@dataclass(frozen=True)
class ReadoutConfig:
    """What a snapshot derives from the accumulator. Defaults are the live-preview defaults."""

    phase_flip: bool = True                 # add a `phase_flipped` channel
    dc_correct: bool = True                 # remove each BF pixel's (running) scan mean
    normalize: bool = False                 # divide by coverage, taper below weight_threshold
    weight_threshold: float = 1e-2          # fraction of peak coverage below which to taper
    coverage_channel: bool = False          # add the coverage map as a channel
    q_lowpass: float | None = None          # Butterworth low-pass cutoff, 1/Å
    q_highpass: float | None = None         # Butterworth high-pass cutoff, 1/Å
    butterworth_order: int = 12
    suppress_nyquist: bool | None = None    # None → on when upsampling_factor > 1
    band_limit: bool = False                # clamp the low-pass at 2α/λ (needs semiangle_cutoff)


def quadratic_aberration_surface(alpha, phi, wavelength: float, aberration_coefs: dict):
    """χ(α, φ) for C10/C12 in abTEM's polar convention (the reference's own definition)."""
    C10 = aberration_coefs.get("C10", 0.0)
    C12 = aberration_coefs.get("C12", 0.0)
    phi12 = aberration_coefs.get("phi12", 0.0)
    return np.pi / wavelength * alpha**2 * (C10 + C12 * np.cos(2.0 * (phi - phi12)))


def fourier_filters(gpts, sampling, wavelength: float, aberration_coefs: dict, *,
                    q_lowpass=None, q_highpass=None, butterworth_order: int = 12,
                    suppress_nyquist: bool = False, band_limit_q: float | None = None):
    """`(base, flip)` real Fourier multipliers on a `gpts` grid with real-space `sampling` (Å).

    `base` holds the Butterworth / band-limit / Nyquist terms (None when it would be all ones);
    `flip` is `sign(sin χ(q))`. Built in float64 on the host: χ(q) reaches tens of radians, so the
    sign is ill-conditioned at its zero crossings and should not depend on the device precision.
    Frequencies are unrotated: the scan/detector rotation belongs to the detector k-grid only.
    """
    # the reference's `spatial_frequencies` naming: "qx" runs along axis 0, "qy" along axis 1
    ny, nx = gpts
    qxa, qya = np.meshgrid(np.fft.fftfreq(ny, sampling[0]), np.fft.fftfreq(nx, sampling[1]),
                           indexing="ij")
    q = np.sqrt(qxa**2 + qya**2)
    theta = np.arctan2(qya, qxa)

    flip = np.sign(np.sin(quadratic_aberration_surface(q * wavelength, theta, wavelength,
                                                       aberration_coefs)))

    base = np.ones((ny, nx))
    if band_limit_q is not None:
        q_lowpass = band_limit_q if q_lowpass is None else min(q_lowpass, band_limit_q)
    if q_lowpass:
        base = base / (1 + (q / q_lowpass) ** (2 * butterworth_order))
    if q_highpass:
        base = base * (1 - 1 / (1 + (q / q_highpass) ** (2 * butterworth_order)))
    if suppress_nyquist:
        if ny % 2 == 0:
            base[ny // 2, :] = 0.0
        if nx % 2 == 0:
            base[:, nx // 2] = 0.0
    if np.all(base == 1.0):
        base = None
    return base, flip


class Readout:
    """Device-resident readout operators for one `ParallaxGeometry`."""

    def __init__(self, geometry, config: ReadoutConfig, device: torch.device):
        self.config = config
        self.device = device
        self._nyu, self._nxu = (int(n) for n in geometry.upsampled_scan_gpts)
        self._ny, self._nx = (int(n) for n in geometry.scan_shape)
        u = int(geometry.upsampling_factor)

        # visited mask (Ny*Nx) → comb on the upsampled torus
        iy, ix = np.divmod(np.arange(self._ny * self._nx), self._nx)
        self._comb = torch.as_tensor((iy * u) * self._nxu + ix * u, dtype=torch.long, device=device)

        # each BF pixel's deposit offset on the torus
        shifts = np.asarray(geometry.shifts)
        flat = (shifts[:, 0] % self._nyu) * self._nxu + (shifts[:, 1] % self._nxu)
        self._shift_flat = torch.as_tensor(flat, dtype=torch.long, device=device)
        hist = np.bincount(flat, minlength=self._nyu * self._nxu).reshape(self._nyu, self._nxu)
        self._hist_f = torch.fft.fft2(torch.as_tensor(hist, dtype=torch.float32, device=device))

        suppress = (u > 1) if config.suppress_nyquist is None else config.suppress_nyquist
        band_q = None
        if config.band_limit and getattr(geometry, "semiangle_cutoff", 0.0):
            band_q = 2 * geometry.semiangle_cutoff * 1e-3 / geometry.wavelength
        base, flip = fourier_filters(
            geometry.upsampled_scan_gpts, geometry.upsampled_scan_sampling, geometry.wavelength,
            geometry.aberration_coefs, q_lowpass=config.q_lowpass, q_highpass=config.q_highpass,
            butterworth_order=config.butterworth_order, suppress_nyquist=suppress,
            band_limit_q=band_q)
        self._base = None if base is None else torch.as_tensor(base, dtype=torch.float32,
                                                               device=device)
        if config.phase_flip:
            full = flip if base is None else flip * base
            self._flip = torch.as_tensor(full, dtype=torch.float32, device=device)
        else:
            self._flip = None

    def _visits_comb_f(self, visits: torch.Tensor) -> torch.Tensor:
        comb = torch.zeros(self._nyu * self._nxu, dtype=torch.float32, device=self.device)
        comb[self._comb] = visits
        return torch.fft.fft2(comb.view(self._nyu, self._nxu))

    def coverage(self, visits: torch.Tensor) -> torch.Tensor:
        """(NyU, NxU) number of deposits each output pixel has received."""
        cov = torch.fft.ifft2(self._visits_comb_f(visits) * self._hist_f).real
        return torch.round(cov).clamp_min_(0)

    def background(self, visits: torch.Tensor, bf_mean: torch.Tensor) -> torch.Tensor:
        """(NyU, NxU) the per-BF-pixel means deposited once per visit of each scan position."""
        h = torch.zeros(self._nyu * self._nxu, dtype=torch.float32, device=self.device)
        h.index_add_(0, self._shift_flat, bf_mean.to(torch.float32))
        h_f = torch.fft.fft2(h.view(self._nyu, self._nxu))
        return torch.fft.ifft2(self._visits_comb_f(visits) * h_f).real

    @staticmethod
    def _filter(img: torch.Tensor, mult: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(torch.fft.fft2(img) * mult).real

    def compute(self, out: torch.Tensor, visits: torch.Tensor, bf_sum: torch.Tensor,
                n_frames: int) -> dict[str, torch.Tensor]:
        """The snapshot channels from the raw flat accumulator `out` and the side accumulators
        (`visits`: frames per scan position in the image; `n_frames`: their total)."""
        cfg = self.config
        img = out.view(self._nyu, self._nxu)
        if cfg.dc_correct and n_frames > 0:
            img = img - self.background(visits, bf_sum / n_frames)

        cov = None
        if cfg.normalize or cfg.coverage_channel:
            cov = self.coverage(visits)
        if cfg.normalize:
            peak = cov.max()
            support = (cov / (cfg.weight_threshold * peak).clamp_min(1e-12)).clamp(max=1.0)
            img = img / cov.clamp_min(1.0) * support

        channels = {"parallax": img if self._base is None else self._filter(img, self._base)}
        if self._flip is not None:
            channels["phase_flipped"] = self._filter(img, self._flip)
        if cfg.coverage_channel:
            channels["coverage"] = cov
        return channels
