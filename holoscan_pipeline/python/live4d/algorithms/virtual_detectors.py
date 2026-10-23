"""Integrate bright-field and annular dark-field signals from diffraction frames."""

import cupy as cp


def build_masks(bf_radius, adf_inner, adf_outer):
    """Build float32 BF and ADF masks for 96 by 96 diffraction frames."""
    y, x = cp.mgrid[:96, :96].astype(cp.float32)
    radius = cp.hypot(y - 47.5, x - 47.5)
    return cp.stack([radius <= bf_radius,
                     (radius >= adf_inner) & (radius <= adf_outer)]
                    ).reshape(2, -1).T.astype(cp.float32)


def integrate(frames, masks):
    """Return float32 BF and ADF values for each frame."""
    return frames.reshape(frames.shape[0], -1).astype(cp.float32) @ masks
