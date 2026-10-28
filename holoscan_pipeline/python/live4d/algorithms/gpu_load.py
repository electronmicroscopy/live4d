"""Synthetic GPU load: repeated unitary 2-D FFTs of diffraction frames."""

import cupy as cp


def fft_load(frames, repeats=1):
    """Apply ``repeats`` unitary 2-D FFTs to each frame and return the summed magnitude."""
    data = frames.astype(cp.complex64)
    for _ in range(repeats):
        data = cp.fft.fft2(data, norm="ortho")
    return cp.abs(data).sum(dtype=cp.float64)
