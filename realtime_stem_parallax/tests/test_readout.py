"""
Tests for the readout channels derived from the parallax accumulator (`readout.py`).

Oracles, all float64 NumPy transcriptions:
  • phase flip: the streaming reference's grouped phase-flip kernel
    (`prepare_grouped_phase_flipping_kernel` + `parallax_phase_flip_accumulate_cpu`), scattered per
    frame, with the scatter wrapped per axis (`% Ny`, `% Nx`). The reference wraps the flat index
    `% (Ny*Nx)` instead, which carries a column overflow into the next row; that variant is checked
    to differ, so the test documents the bug rather than inheriting it.
  • per-BF scan-mean removal: subtract each BF pixel's mean over the frames seen so far from every
    deposit (quantem's `_preprocess`; double-centered when the per-frame mean is subtracted too).

The phase flip and the scan-mean removal are checked with and without `subtract_frame_mean`.
  • coverage: count the deposits landing on each output pixel.

Each is checked mid-scan (the live case) and on a complete scan, at upsampling 1 and 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_stem.processing.parallax.engine import ParallaxAccumulator, preprocess_geometry
from realtime_stem.processing.parallax.readout import ReadoutConfig, quadratic_aberration_surface

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (
    ["cuda"] if torch.cuda.is_available() else [])

_SCAN = (12, 10)


def _geometry(upsampling_factor=1, rotation_deg=-15.0):
    return preprocess_geometry(
        det_shape=(32, 32), scan_shape=_SCAN, scan_sampling=(256 / 12, 256 / 12), energy=300e3,
        semiangle_cutoff=4.0, angular_sampling=(0.307617, 0.307617),
        aberration_coefs={"C10": -1.5e4}, rotation_angle=np.deg2rad(rotation_deg),
        upsampling_factor=upsampling_factor)


def _frames(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60000, size=(n, 32, 32), dtype=np.uint16)


def _bf_values(frames, geo, subtract_frame_mean=False):
    """(T, M) BF values as deposited, float64 (minus each frame's mean if asked)."""
    v = frames.reshape(len(frames), -1)[:, geo.bf_flat_inds].astype(np.float64)
    return v - v.mean(axis=1, keepdims=True) if subtract_frame_mean else v


def _deposit(values, ids, geo):
    """Shifted scatter of (T, M) values, float64, independent per-axis wraps."""
    nyu, nxu = geo.upsampled_scan_gpts
    u = geo.upsampling_factor
    yt = (ids // _SCAN[1]) * u
    xt = (ids % _SCAN[1]) * u
    out = np.zeros((nyu, nxu))
    np.add.at(out, ((yt[:, None] + geo.shifts[None, :, 0]) % nyu,
                    (xt[:, None] + geo.shifts[None, :, 1]) % nxu), values)
    return out


def _reference_phase_flip(frames, ids, geo, flat_wrap=False, subtract_frame_mean=False):
    """The streaming reference's grouped phase-flip kernel, scattered frame by frame."""
    nyu, nxu = geo.upsampled_scan_gpts
    sy, sx = geo.upsampled_scan_sampling
    qxa, qya = np.meshgrid(np.fft.fftfreq(nyu, sy), np.fft.fftfreq(nxu, sx), indexing="ij")
    q, theta = np.sqrt(qxa**2 + qya**2), np.arctan2(qya, qxa)
    kernel_f = np.sign(np.sin(quadratic_aberration_surface(q * geo.wavelength, theta,
                                                           geo.wavelength, geo.aberration_coefs)))
    kernel_f[nyu // 2, :] = 0.0
    kernel_f[:, nxu // 2] = 0.0
    kernel = np.fft.ifft2(kernel_f).real

    # prepare_grouped_phase_flipping_kernel, verbatim math
    h, w = kernel.shape
    m_count = geo.shifts.shape[0]
    dy_rep = np.tile(np.repeat(np.arange(h), w), m_count)
    dx_rep = np.tile(np.tile(np.arange(w), h), m_count)
    s_my = np.repeat(geo.shifts[:, 0], h * w)
    s_mx = np.repeat(geo.shifts[:, 1], h * w)
    offsets = ((dy_rep + s_my) % nyu) * nxu + ((dx_rep + s_mx) % nxu)
    unique_offsets, inv = np.unique(offsets, return_inverse=True)
    grouped = np.zeros((unique_offsets.size, m_count))
    np.add.at(grouped, (inv, np.repeat(np.arange(m_count), h * w)), np.tile(kernel.ravel(), m_count))

    off_y, off_x = np.divmod(unique_offsets, nxu)
    out = np.zeros((nyu, nxu))
    u = geo.upsampling_factor
    for t, idx in enumerate(ids):
        vals = grouped @ _bf_values(frames[t:t + 1], geo, subtract_frame_mean)[0]
        yt, xt = (idx // _SCAN[1]) * u, (idx % _SCAN[1]) * u
        if flat_wrap:   # the reference: flat index wrap, a column overflow carries into the next row
            np.add.at(out.reshape(-1), (yt * nxu + xt + unique_offsets) % (nyu * nxu), vals)
        else:
            np.add.at(out, ((yt + off_y) % nyu, (xt + off_x) % nxu), vals)
    return out


def _accumulate(frames, ids, geo, device, subtract_frame_mean=False, **readout):
    acc = ParallaxAccumulator(geo, device=device, readout=ReadoutConfig(**readout),
                              subtract_frame_mean=subtract_frame_mean)
    acc.process(frames, ids)
    return acc


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("upsampling_factor", [1, 2])
@pytest.mark.parametrize("n_frames", [37, 120])          # mid-scan, complete scan
@pytest.mark.parametrize("subtract", [False, True])
def test_phase_flip_matches_streamed_grouped_kernel(device, upsampling_factor, n_frames, subtract):
    geo = _geometry(upsampling_factor)
    frames = _frames(n_frames)
    ids = np.arange(n_frames)
    acc = _accumulate(frames, ids, geo, device, subtract_frame_mean=subtract, dc_correct=False,
                      suppress_nyquist=True)
    got = acc.channels()["phase_flipped"].cpu().numpy()

    exp = _reference_phase_flip(frames, ids, geo, subtract_frame_mean=subtract)
    scale = np.abs(exp).max()
    assert np.allclose(got, exp, atol=2e-4 * scale), np.abs(got - exp).max() / scale

    # the reference's flat wrap is a different (wrong) operator, not a rounding difference
    buggy = _reference_phase_flip(frames, ids, geo, flat_wrap=True, subtract_frame_mean=subtract)
    assert np.abs(buggy - exp).max() > 0.1 * scale


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("upsampling_factor", [1, 2])
@pytest.mark.parametrize("n_frames", [37, 120])
@pytest.mark.parametrize("subtract", [False, True])
def test_dc_correction_matches_double_centering(device, upsampling_factor, n_frames, subtract):
    geo = _geometry(upsampling_factor)
    frames = _frames(n_frames, seed=1)
    ids = np.random.default_rng(2).permutation(np.prod(_SCAN))[:n_frames]   # any arrival order
    acc = ParallaxAccumulator(geo, device=device, subtract_frame_mean=subtract,
                              readout=ReadoutConfig(phase_flip=False, suppress_nyquist=False))
    acc.process(frames[: n_frames // 2], ids[: n_frames // 2])            # two batches
    acc.process(frames[n_frames // 2:], ids[n_frames // 2:])
    got = acc.channels()["parallax"].cpu().numpy()

    v = _bf_values(frames, geo, subtract)
    exp = _deposit(v - v.mean(axis=0, keepdims=True), ids, geo)
    raw = acc.result()
    # float32 sums of the raw deposits: the error scales with the raw image, not the result
    tol = 2e-4 * np.abs(exp).max() + 2e-6 * np.abs(raw).max()
    assert np.allclose(got, exp, atol=tol), np.abs(got - exp).max()

    diff = raw - exp
    if upsampling_factor == 1 and n_frames == np.prod(_SCAN):
        # complete, unupsampled wrap: each output pixel receives every BF pixel once, so the
        # correction only removes a constant (zero when the frame mean is subtracted too)
        assert np.allclose(diff, diff.mean(), atol=tol)
        if subtract:
            assert abs(diff.mean()) < tol
    else:
        assert np.abs(diff - diff.mean()).max() > 1e-2 * np.abs(exp).max()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("upsampling_factor", [1, 2])
def test_coverage_counts_deposits(device, upsampling_factor):
    geo = _geometry(upsampling_factor)
    ids = np.random.default_rng(3).permutation(np.prod(_SCAN))[:53]
    acc = _accumulate(_frames(53), ids, geo, device, coverage_channel=True)
    got = acc.channels()["coverage"].cpu().numpy()
    exp = _deposit(np.ones((len(ids), geo.n_bf)), ids, geo)
    assert np.array_equal(got, exp)


@pytest.mark.parametrize("device", DEVICES)
def test_normalize_is_mean_per_deposit(device):
    """normalize → (Σ deposits / coverage) · support, quantem's `mean * support`."""
    geo = _geometry(1)
    frames = _frames(53, seed=4)
    ids = np.arange(53)
    acc = _accumulate(frames, ids, geo, device, normalize=True, dc_correct=False,
                      phase_flip=False)
    got = acc.channels()["parallax"].cpu().numpy()
    cov = _deposit(np.ones((53, geo.n_bf)), ids, geo)
    support = np.clip(cov / (1e-2 * cov.max()), None, 1.0)
    exp = _deposit(_bf_values(frames, geo), ids, geo) / np.maximum(cov, 1) * support
    assert np.allclose(got, exp, atol=1e-4 * np.abs(exp).max())


def test_snapshot_stack_and_meta():
    geo = _geometry(1)
    acc = _accumulate(_frames(30), np.arange(30), geo, "cpu", coverage_channel=True)
    stack, names, meta = acc.snapshot()
    assert names == ["parallax", "phase_flipped", "coverage"]
    assert stack.shape == (3, *geo.upsampled_scan_gpts) and stack.dtype == np.float32
    assert meta["frames"] == 30 and np.isclose(meta["coverage"], 30 / np.prod(_SCAN))
    assert np.isclose(meta["C10"], -1.5e4) and np.isclose(meta["rotation_deg"], -15.0)
    images = acc.images()
    assert set(images) == {"raw", "parallax", "phase_flipped", "coverage"}
    assert np.array_equal(images["raw"], acc.result())


def test_filters_lowpass_and_nyquist():
    """Butterworth low-pass removes high frequencies; Nyquist suppression zeroes those bins."""
    geo = _geometry(2)
    frames = _frames(120, seed=5)
    acc = _accumulate(frames, np.arange(120), geo, "cpu", phase_flip=False, dc_correct=False,
                      q_lowpass=0.02, suppress_nyquist=True)
    img = acc.channels()["parallax"].cpu().numpy()
    spec = np.abs(np.fft.fft2(img))
    ny, nx = img.shape
    assert spec[ny // 2, :].max() < 1e-3 * spec.max() and spec[:, nx // 2].max() < 1e-3 * spec.max()
    sy, sx = geo.upsampled_scan_sampling
    q = np.hypot(*np.meshgrid(np.fft.fftfreq(ny, sy), np.fft.fftfreq(nx, sx), indexing="ij"))
    raw_spec = np.abs(np.fft.fft2(acc.result()))
    assert spec[q > 0.04].sum() < 1e-3 * raw_spec[q > 0.04].sum()
