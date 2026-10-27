"""Linear sample drift from a 0°/90° pair of 4D-STEM scans (quantem's affine drift correction).

Two scans of one field, the second with the scan rotated by 90°, see the same drift along swapped fast
and slow axes. quantem's `DriftCorrection.correct_affine()` finds the one linear drift that makes their
virtual images (the mean over the whole detector) agree. This module holds no Holoscan state: it is
called with two finished scans (`Dataset4dstemGPU` as `quantem.gpu.io.load` or the ANS encoder gives them, or arrays).

    rate = drift_0_90(scan_0, scan_90, sampling_nm=0.29771, line_time_s=512 * 8.33e-6)
    rate.nm_per_s        # (row, col) drift in the 0° scan frame, for the stage or piezo
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScanDrift:
    """Affine drift of one 0°/90° pair, (row, col) in the 0° scan frame."""

    px_per_line: tuple[float, float]
    nm_per_line: tuple[float, float] | None
    nm_per_s: tuple[float, float] | None
    correction: object                      # the solved quantem DriftCorrection, for reports and merging

    def summary(self) -> dict:
        return dict(px_per_line=self.px_per_line, nm_per_line=self.nm_per_line, nm_per_s=self.nm_per_s)


def drift_0_90(scan_0, scan_90, *, sampling_nm: float | None = None, line_time_s: float | None = None,
               device: str | None = None, verbose: bool = False) -> ScanDrift:
    """Affine drift rate of a 0°/90° pair.

    sampling_nm: scan step; without it only px/line is returned. line_time_s: time of one scan line
    (columns x frame time); without it nm/s is None.
    """
    from quantem.imaging import DriftCorrection

    if tuple(scan_0.shape) != tuple(scan_90.shape):
        raise ValueError(f"the 0° and 90° scans differ in shape: {tuple(scan_0.shape)} vs {tuple(scan_90.shape)}")
    correction = DriftCorrection.from_images(virtual_image(scan_0), virtual_image(scan_90),
                                             scan_direction_degrees=(0, 90), device=device)
    correction.correct_affine(show_combined=False, show_scans=False, verbose=verbose)
    px = tuple(float(v) for v in correction.drift_rate)
    nm = None if sampling_nm is None else (px[0] * sampling_nm, px[1] * sampling_nm)
    per_s = None if nm is None or not line_time_s else (nm[0] / line_time_s, nm[1] / line_time_s)
    return ScanDrift(px, nm, per_s, correction)


def virtual_image(scan) -> np.ndarray:
    """Mean over the detector, (rows, cols) float32: from the ANS counts for a `Dataset4dstemGPU`
    (no decode of the scan), else from an array. A 2D image (a DirectPtycho phase) is used as it is."""
    if len(scan.shape) == 2:
        image = scan
    elif hasattr(scan, "mean") and hasattr(scan, "representation"):
        image = scan.mean(dim=(2, 3))
    else:
        image = scan.mean(axis=(2, 3)) if isinstance(scan, np.ndarray) else scan.float().mean(dim=(2, 3))
    if hasattr(image, "get"):                  # CuPy
        image = image.get()
    elif hasattr(image, "detach"):             # torch
        image = image.detach().cpu().numpy()
    return np.asarray(image, dtype=np.float32)
