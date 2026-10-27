"""Detector orientation: the 8 ways a detector's axes can be transposed/reversed relative to the
canonical (Q_rows, Q_cols) layout, and how each maps onto the geometry's two knobs.

Following LiberTEM-parallax, every layout is a rotation of the detector k-grid plus an optional
column flip (`flip_cols`); row flips are a π rotation combined with a column flip. A layout name
spells what the raw frame's (axis 0, axis 1) hold, e.g. "cols,-rows" = axis 0 runs along canonical
columns, axis 1 along reversed canonical rows (`raw = canonical.T[:, ::-1]`).

Unlike LiberTEM-parallax this needs no even-grid parity correction (its `n_rot` block): the BF disk
is placed at a center given in *raw* frame coordinates — measured, or mapped from the canonical
center by `layout_center` — so the one-pixel shift a reversed even axis introduces is carried by
the center instead.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["DETECTOR_LAYOUTS", "layout_frames", "layout_center", "layout_shape"]

# layout → (rotation to add, in radians; flip_cols)   (LiberTEM-parallax's table)
DETECTOR_LAYOUTS: dict[str, tuple[float, bool]] = {
    "rows,cols": (0.0, False),
    "rows,-cols": (0.0, True),
    "-rows,cols": (math.pi, True),
    "-rows,-cols": (math.pi, False),
    "cols,rows": (math.pi / 2, True),
    "cols,-rows": (math.pi / 2, False),
    "-cols,rows": (-math.pi / 2, False),
    "-cols,-rows": (-math.pi / 2, True),
}


def _parse(layout: str):
    if layout not in DETECTOR_LAYOUTS:
        raise ValueError(f"unknown detector layout {layout!r}; one of {', '.join(DETECTOR_LAYOUTS)}")
    a0, a1 = layout.split(",")
    return a0.lstrip("-") == "cols", a0.startswith("-"), a1.startswith("-")


def layout_frames(canonical: np.ndarray, layout: str) -> np.ndarray:
    """Re-lay canonical (..., H, W) frames as a detector with `layout` would deliver them."""
    transpose, rev0, rev1 = _parse(layout)
    out = np.swapaxes(canonical, -1, -2) if transpose else canonical
    if rev0:
        out = out[..., ::-1, :]
    if rev1:
        out = out[..., :, ::-1]
    return out


def layout_shape(canonical_shape: tuple[int, int], layout: str) -> tuple[int, int]:
    """The raw frame shape for a canonical (H, W) under `layout`."""
    transpose, _, _ = _parse(layout)
    h, w = canonical_shape
    return (w, h) if transpose else (h, w)


def layout_center(raw_shape: tuple[int, int], layout: str,
                  canonical_center: tuple[float, float] | None = None) -> tuple[float, float]:
    """Where the canonical BF-disk center lands on the raw frame (row, col).

    `canonical_center` defaults to the canonical frame center (H//2, W//2), the geometry's own
    default for an unflipped detector.
    """
    transpose, rev0, rev1 = _parse(layout)
    hc, wc = (raw_shape[1], raw_shape[0]) if transpose else raw_shape
    c0, c1 = (hc // 2, wc // 2) if canonical_center is None else canonical_center
    r0, r1 = (c1, c0) if transpose else (c0, c1)
    if rev0:
        r0 = raw_shape[0] - 1 - r0
    if rev1:
        r1 = raw_shape[1] - 1 - r1
    return float(r0), float(r1)
