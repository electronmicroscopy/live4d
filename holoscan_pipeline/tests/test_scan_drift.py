"""live4d.algorithms.scan_drift: the affine drift of a 0°/90° pair is recovered from synthetic scans.

One smooth random object is sampled along a drifting raster twice: the 0° scan rows-fast-columns, the
90° scan with the scan rotated (its image (i, j) sees object (j, n - 1 - i)). The drift, in pixels per
scan line of the 0° frame, must come back from quantem's affine search.
"""
import numpy as np
import pytest

pytest.importorskip("quantem.imaging")
from scipy.ndimage import gaussian_filter, map_coordinates

from live4d.algorithms.scan_drift import drift_0_90

N = 128


def scan_pair(drift, n=N, detector=(4, 4)):
    rng = np.random.default_rng(3)
    sample = gaussian_filter(rng.normal(size=(3 * n, 3 * n)), 2.0).astype(np.float32)
    i, j = np.indices((n, n), dtype=np.float64)
    lines = i + j / n                                  # time, in scan lines
    moved = (drift[0] * lines, drift[1] * lines)
    image_0 = map_coordinates(sample, [n + i + moved[0], n + j + moved[1]], order=3)
    image_90 = map_coordinates(sample, [n + j + moved[0], n + (n - 1 - i) + moved[1]], order=3)
    pattern = np.arange(np.prod(detector), dtype=np.float32).reshape(detector)
    return ((image_0[..., None, None] + pattern).astype(np.float32),
            (image_90[..., None, None] + pattern).astype(np.float32))


@pytest.mark.parametrize("drift", [(0.0, 0.0), (0.05, 0.0), (0.0, 0.05), (-0.04, 0.03)])
def test_drift_0_90_recovers_a_linear_drift(drift):
    scan_0, scan_90 = scan_pair(drift)
    rate = drift_0_90(scan_0, scan_90, sampling_nm=0.3, line_time_s=N * 8.33e-6)
    np.testing.assert_allclose(rate.px_per_line, drift, atol=0.005)
    np.testing.assert_allclose(rate.nm_per_line, np.multiply(rate.px_per_line, 0.3))
    np.testing.assert_allclose(rate.nm_per_s, np.divide(rate.nm_per_line, N * 8.33e-6))


def test_drift_0_90_rejects_scans_of_different_shapes():
    scan_0, scan_90 = scan_pair((0.0, 0.0))
    with pytest.raises(ValueError, match="differ in shape"):
        drift_0_90(scan_0, scan_90[:-1])
