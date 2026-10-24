"""
Tests for the parallax / tcBF engine.

The oracle is direct math: a faithful **float64 NumPy transcription of Georgios Varnavides's
streaming-parallax reference** — `preprocess_geometry` (BF indices + integer parallax shifts) and
`parallax_accumulate_cpu` (the per-frame gather → mean-subtract → toroidal shifted scatter-add).
"Engine == oracle" therefore literally means "the equations were ported correctly." The engine
keeps the frame mean by default; the oracle drops its mean-subtract step for that case.

Checks:
  • Geometry: our `preprocess_geometry` bf-indices and integer shifts match the reference exactly
    (int equality), across geometry variants (rotation on/off, C10/C12), and honor an off-center
    probe.
  • Accumulation: the batched-torch `ParallaxAccumulator` reconstruction matches the reference
    float64 loop on synthetic frames — arbitrary id order, split across process() calls, for
    uint8/uint16/uint32; upsampling; toroidal wrap; with and without the frame mean.
  • Analytic (physics, transcription-independent): pure-defocus shifts are radial and scale
    linearly with C10; zero aberration → the mean-subtracted contributions cancel → zero output,
    and with the mean kept the in-focus image is the virtual bright-field image.
  • Contract/CLI: device-tensor input (Holoscan DLPack path) == ndarray input; the `Accumulator`
    protocol holds; the tuning-knob CLI maps to the config with the right units; GPU == CPU.

Run from the repo root (a torch env; on a CUDA host the GPU-parity test runs, on a CPU-only host
it skips cleanly):
    python -m pytest tests/test_parallax.py
    python tests/test_parallax.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_stem.processing.parallax.engine import (
    ParallaxAccumulator,
    ParallaxGeometry,
    preprocess_geometry,
)


# ── oracle: faithful float64 transcription of the streaming reference ──────────
# Verbatim math (names/conventions preserved, incl. the spatial_frequencies kx/ky naming and the
# `dy, dx = shifts[m]` unpacking).

def _electron_wavelength(energy: float) -> float:
    m, e, c, h = 9.109383e-31, 1.602177e-19, 299792458.0, 6.62607e-34
    lam = h / np.sqrt(2 * m * e * energy) / np.sqrt(1 + e * energy / 2 / m / c**2)
    return lam * 1e10  # Angstroms


def _spatial_frequencies(gpts, sampling, rotation_angle=None):
    ny, nx = gpts
    sy, sx = sampling
    kx = np.fft.fftfreq(ny, sy)
    ky = np.fft.fftfreq(nx, sx)
    kxa, kya = np.meshgrid(kx, ky, indexing="ij")
    if rotation_angle is not None:
        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        kxa, kya = c * kxa - s * kya, s * kxa + c * kya
    return kxa, kya


def _polar(kx, ky):
    return np.sqrt(kx**2 + ky**2), np.arctan2(ky, kx)


def _quadratic_gradients(alpha, phi, coefs):
    C10 = coefs.get("C10", 0.0)
    C12 = coefs.get("C12", 0.0)
    phi12 = coefs.get("phi12", 0.0)
    cos2, sin2 = np.cos(2.0 * (phi - phi12)), np.sin(2.0 * (phi - phi12))
    scale = 2 * np.pi
    dchi_dk = scale * alpha * (C10 + C12 * cos2)
    dchi_dphi = -scale * alpha * (C12 * sin2)
    cphi, sphi = np.cos(phi), np.sin(phi)
    dchi_dx = cphi * dchi_dk - sphi * dchi_dphi
    dchi_dy = sphi * dchi_dk + cphi * dchi_dphi
    return dchi_dx, dchi_dy


def _reference_geometry(det_shape, scan_sampling, energy, semiangle_cutoff, angular_sampling,
                        aberration_coefs, rotation_angle=None, upsampling_factor=1):
    """Returns (bf_flat_inds int32 (M,), shifts int32 (M,2)) — the reference preprocess_geometry."""
    wavelength = _electron_wavelength(energy)
    reciprocal_sampling = (angular_sampling[0] / wavelength / 1e3,
                           angular_sampling[1] / wavelength / 1e3)
    gpts = (det_shape[0], det_shape[1])
    sampling = (1.0 / reciprocal_sampling[0] / gpts[0],
                1.0 / reciprocal_sampling[1] / gpts[1])
    upsampled_scan_sampling = (scan_sampling[0] / upsampling_factor,
                               scan_sampling[1] / upsampling_factor)

    kxa, kya = _spatial_frequencies(gpts, sampling, rotation_angle)
    k, phi = _polar(kxa, kya)

    bf_mask = k * wavelength * 1e3 <= semiangle_cutoff
    inds_i, inds_j = np.where(bf_mask)
    inds_i_fft = (inds_i - gpts[0] // 2) % gpts[0]
    inds_j_fft = (inds_j - gpts[1] // 2) % gpts[1]
    bf_flat_inds = (inds_i_fft * gpts[1] + inds_j_fft).astype(np.int32)

    dx, dy = _quadratic_gradients(k * wavelength, phi, aberration_coefs)
    grad_k = np.stack((dx[inds_i, inds_j], dy[inds_i, inds_j]), axis=-1)
    shifts = np.round(grad_k / (2 * np.pi) / upsampled_scan_sampling).astype(np.int32)
    return bf_flat_inds, shifts


def _reference_accumulate(frames, bf_flat_inds, shifts, coords, out_shape, subtract_mean=True):
    """The reference parallax_accumulate_cpu, pure float64 (no numba). Returns out (Ny, Nx).
    `subtract_mean=False` leaves out its per-frame mean (the engine's default)."""
    out = np.zeros(out_shape, dtype=np.float64)
    T = frames.shape[0]
    M = shifts.shape[0]
    Ny, Nx = out_shape
    sx = frames.shape[2]
    ff = frames.astype(np.float64)
    for t in range(T):
        frame = ff[t]
        yt, xt = coords[t]
        mean = 0.0
        for m in range(M):
            fi = bf_flat_inds[m]
            mean += frame[fi // sx, fi % sx]
        mean = mean / M if subtract_mean else 0.0
        for m in range(M):
            fi = bf_flat_inds[m]
            val = frame[fi // sx, fi % sx] - mean
            dy, dx = shifts[m]
            out[(yt + dy) % Ny, (xt + dx) % Nx] += val
    return out


# ── shared geometry knobs for tests (small detector) ───────────────────────────

_GEO = dict(
    det_shape=(32, 32),
    scan_sampling=(256 / 12, 256 / 12),     # Angstroms
    energy=300e3,                           # eV
    semiangle_cutoff=4.0,                   # mrad
    angular_sampling=(0.307617, 0.307617),  # mrad
)


def _coords(ids, nx, upsampling_factor):
    return np.stack([(ids // nx) * upsampling_factor, (ids % nx) * upsampling_factor], axis=1)


def _defocus_geo(c10, upsampling_factor=1):
    return preprocess_geometry(
        det_shape=_GEO["det_shape"], scan_shape=(12, 12),
        scan_sampling=_GEO["scan_sampling"], energy=_GEO["energy"],
        semiangle_cutoff=_GEO["semiangle_cutoff"], angular_sampling=_GEO["angular_sampling"],
        aberration_coefs={"C10": c10}, upsampling_factor=upsampling_factor,
    )


def _parallax_config(argv, base=None):
    """Parse parallax CLI argv exactly as the runner does (the algorithm's add_cli_args +
    preprocess_argv), then map to a config — pins the live tuning knobs end to end."""
    from realtime_stem.processing.parallax.algorithm import ParallaxAlgorithm
    from realtime_stem.processing.parallax.config import ParallaxConfig, config_from_args
    algo = ParallaxAlgorithm()
    p = argparse.ArgumentParser()
    algo.add_cli_args(p)
    args = p.parse_args(algo.preprocess_argv(argv))
    return config_from_args(args, base or ParallaxConfig())


# ── geometry: our preprocess_geometry == reference (exact) ─────────────────────

def test_geometry_matches_reference():
    """BF indices and integer shifts must match the reference exactly."""
    scan_shape = (12, 12)
    for rotation_angle, coefs in [
        (None, {"C10": -1.5e4}),
        (np.deg2rad(-15), {"C10": -1.5e4}),
        (None, {"C10": -1.0e4, "C12": 2.0e3, "phi12": 0.3}),
    ]:
        geo = preprocess_geometry(
            det_shape=_GEO["det_shape"], scan_shape=scan_shape,
            scan_sampling=_GEO["scan_sampling"], energy=_GEO["energy"],
            semiangle_cutoff=_GEO["semiangle_cutoff"],
            angular_sampling=_GEO["angular_sampling"],
            aberration_coefs=coefs, rotation_angle=rotation_angle,
        )
        exp_bf, exp_shifts = _reference_geometry(
            _GEO["det_shape"], _GEO["scan_sampling"], _GEO["energy"],
            _GEO["semiangle_cutoff"], _GEO["angular_sampling"], coefs, rotation_angle,
        )
        assert np.array_equal(geo.bf_flat_inds, exp_bf), (rotation_angle, coefs, "bf inds")
        assert np.array_equal(geo.shifts, exp_shifts), (rotation_angle, coefs, "shifts")
    print("test_geometry_matches_reference: PASS (bf inds + shifts exact, rot/C12 variants)")


# ── accumulation: batched torch == reference float64 loop ──────────────────────

@pytest.mark.parametrize("subtract", [True, False])
def test_accumulate_matches_reference(subtract):
    """Synthetic frames, arbitrary id order, split across two process() calls — all dtypes."""
    rng = np.random.default_rng(0)
    scan_shape = (12, 12)
    coefs = {"C10": -1.5e4}
    bf, shifts = _reference_geometry(
        _GEO["det_shape"], _GEO["scan_sampling"], _GEO["energy"],
        _GEO["semiangle_cutoff"], _GEO["angular_sampling"], coefs,
    )
    geom = ParallaxGeometry(
        bf_flat_inds=bf, shifts=shifts, detector_shape=_GEO["det_shape"],
        scan_shape=scan_shape, upsampling_factor=1, upsampled_scan_gpts=scan_shape,
    )
    n = 90
    ids = rng.permutation(scan_shape[0] * scan_shape[1])[:n]
    coords = _coords(ids, scan_shape[1], 1)

    for dtype, hi in [(np.uint32, 100000), (np.uint16, 60000), (np.uint8, 255)]:
        frames = rng.integers(0, hi, size=(n, *_GEO["det_shape"]), dtype=dtype)

        acc = ParallaxAccumulator(geom, device="cpu", subtract_frame_mean=subtract)
        acc.process(frames[:40], ids[:40])           # two calls → tests accumulation
        acc.process(frames[40:], ids[40:])
        got = acc.result()

        exp = _reference_accumulate(frames, bf, shifts, coords, scan_shape, subtract)
        scale = max(np.abs(exp).max(), 1.0)
        assert np.allclose(got, exp, rtol=1e-4, atol=1e-4 * scale), (
            dtype, np.abs(got - exp).max(), scale)
    print("test_accumulate_matches_reference: PASS (uint32/uint16/uint8, arbitrary ids, 2 calls)")


@pytest.mark.parametrize("subtract", [True, False])
def test_accumulate_upsampled_matches_reference(subtract):
    """upsampling_factor>1 (finer output grid) must still match the reference loop."""
    rng = np.random.default_rng(3)
    scan_shape = (10, 10)
    u = 2
    geo = preprocess_geometry(
        det_shape=_GEO["det_shape"], scan_shape=scan_shape,
        scan_sampling=_GEO["scan_sampling"], energy=_GEO["energy"],
        semiangle_cutoff=_GEO["semiangle_cutoff"], angular_sampling=_GEO["angular_sampling"],
        aberration_coefs={"C10": -1.5e4}, upsampling_factor=u,
    )
    n = 70
    ids = rng.permutation(scan_shape[0] * scan_shape[1])[:n]
    coords = _coords(ids, scan_shape[1], u)
    frames = rng.integers(0, 60000, size=(n, *_GEO["det_shape"]), dtype=np.uint16)

    acc = ParallaxAccumulator(geo, device="cpu", subtract_frame_mean=subtract)
    acc.process(frames, ids)
    got = acc.result()
    exp = _reference_accumulate(frames, geo.bf_flat_inds, geo.shifts, coords,
                                geo.upsampled_scan_gpts, subtract)
    assert got.shape == geo.upsampled_scan_gpts == (scan_shape[0] * u, scan_shape[1] * u)
    scale = max(np.abs(exp).max(), 1.0)
    assert np.allclose(got, exp, rtol=1e-4, atol=1e-4 * scale), np.abs(got - exp).max()
    print("test_accumulate_upsampled_matches_reference: PASS (upsampling_factor=2)")


# ── analytic physics (transcription-independent) ───────────────────────────────

def test_zero_aberration_cancels_to_zero():
    """In-focus (no aberration) → shifts all zero → every frame's mean-subtracted BF pixels pile
    on one output pixel and cancel (Σ(I−mean)=0). So the reconstruction is ~zero everywhere."""
    geo = preprocess_geometry(
        det_shape=_GEO["det_shape"], scan_shape=(12, 12),
        scan_sampling=_GEO["scan_sampling"], energy=_GEO["energy"],
        semiangle_cutoff=_GEO["semiangle_cutoff"], angular_sampling=_GEO["angular_sampling"],
        aberration_coefs={},
    )
    assert np.all(geo.shifts == 0), "no aberration → no parallax shifts"
    rng = np.random.default_rng(4)
    ids = np.arange(12 * 12)
    frames = rng.integers(0, 255, size=(len(ids), *_GEO["det_shape"]), dtype=np.uint8)
    acc = ParallaxAccumulator(geo, device="cpu", subtract_frame_mean=True)
    acc.process(frames, ids)
    recon = acc.result()
    assert np.isfinite(recon).all()
    assert np.abs(recon).max() < 1e-1, f"DC removal should cancel in-focus recon, got {np.abs(recon).max()}"

    # Mean kept (the default): in focus, every BF pixel lands on its own scan position, so the
    # image is the virtual bright-field image (the BF disk's counts per position).
    kept = ParallaxAccumulator(geo, device="cpu")
    kept.process(frames, ids)
    vbf = frames.reshape(len(ids), -1)[:, geo.bf_flat_inds].sum(1).astype(np.float64)
    assert np.allclose(kept.result().ravel(), vbf, rtol=1e-6)
    print("test_zero_aberration_cancels_to_zero: PASS (in-focus → ~zero via DC removal)")


def test_frame_mean_is_kept_by_default():
    """No silent per-frame mean subtraction: it removes the amplitude contrast, so it is opt-in.
    A uniformly brighter frame raises the image; with the subtraction it changes nothing."""
    geo = _defocus_geo(-1.5e4)
    frames = np.full((4, *_GEO["det_shape"]), 10, dtype=np.uint16)
    ids = np.arange(4)
    default = ParallaxAccumulator(geo, device="cpu")
    assert not default.subtract_frame_mean
    default.process(frames, ids)
    assert default.result().sum() == pytest.approx(4 * 10 * geo.n_bf)
    sub = ParallaxAccumulator(geo, device="cpu", subtract_frame_mean=True)
    sub.process(frames, ids)
    assert np.abs(sub.result()).max() < 1e-4


def test_defocus_shifts_radial_and_scaling():
    """Pure-defocus shifts: (1) the untilted (DC) view is unshifted; (2) shifts flip sign with C10
    (exact under symmetric rounding); (3) the edge magnitude ≈ cutoff·|C10|/sampling; (4) doubling
    |C10| ~doubles the max shift. All independent of the reference transcription."""
    h, w = _GEO["det_shape"]
    geo = _defocus_geo(-1.5e4)
    center_flat = (h // 2) * w + (w // 2)
    at_center = np.where(geo.bf_flat_inds == center_flat)[0]
    assert at_center.size == 1, "the untilted BF pixel (detector center) must be in the disk"
    assert tuple(geo.shifts[at_center[0]]) == (0, 0), "DC view has zero parallax shift"

    geo_neg = _defocus_geo(+1.5e4)
    assert np.array_equal(geo.shifts, -geo_neg.shifts), "shift field flips sign with C10"

    # edge magnitude: max tilt α_max = cutoff (mrad→rad); shift_Å = α·|C10|; /sampling → px.
    cutoff_rad = _GEO["semiangle_cutoff"] * 1e-3
    expected_edge_px = cutoff_rad * 1.5e4 / _GEO["scan_sampling"][0]
    max_shift = np.abs(geo.shifts).max()
    assert 0.6 * expected_edge_px <= max_shift <= expected_edge_px + 1, (
        max_shift, expected_edge_px)

    geo_2x = _defocus_geo(-3.0e4)
    assert 1.8 <= np.abs(geo_2x.shifts).max() / max_shift <= 2.2, "doubling C10 ~doubles shifts"
    print("test_defocus_shifts_radial_and_scaling: PASS (DC unshifted, sign flip, edge mag, scaling)")


def test_off_center_probe_geometry():
    """Real frames often have the BF disk off detector center. With `probe_center` given,
    bf_flat_inds must select the pixels around THAT point, the untilted (given-center) pixel must
    have zero shift, and the default (None) must stay byte-identical to the centered behavior."""
    h, w = 96, 96
    center = (43.95, 49.67)                      # (row, col) px, py4DSTEM Center_X/Center_Y
    kw = dict(det_shape=(h, w), scan_shape=(32, 32), scan_sampling=(68.0, 68.0),
              energy=200e3, semiangle_cutoff=12.0, angular_sampling=(0.588, 0.588),
              aberration_coefs={"C10": 1.97e5})
    geo = preprocess_geometry(**kw, probe_center=center)

    rows, cols = np.divmod(geo.bf_flat_inds, w)
    r_px = 12.0 / 0.588                          # disk radius in px = cutoff / sampling
    dist = np.hypot(rows - round(center[0]), cols - round(center[1]))
    assert dist.max() <= r_px + 1.0, "BF pixels must lie in the disk around the GIVEN center"
    area = np.pi * r_px**2
    assert abs(len(rows) - area) < 0.15 * area, "full (unclipped) disk expected inside the frame"

    at_center = np.where((rows == round(center[0])) & (cols == round(center[1])))[0]
    assert at_center.size == 1, "the untilted BF pixel (the given center) must be in the disk"
    assert tuple(geo.shifts[at_center[0]]) == (0, 0), "untilted view has zero parallax shift"

    # shifts are a function of tilt only — same set as the centered geometry, same fft order
    geo_default = preprocess_geometry(**kw)
    assert np.array_equal(geo.shifts, geo_default.shifts)

    # None must stay byte-identical to the old behavior == explicit frame center
    geo_explicit = preprocess_geometry(**kw, probe_center=(h // 2, w // 2))
    assert np.array_equal(geo_default.bf_flat_inds, geo_explicit.bf_flat_inds)
    assert np.array_equal(geo_default.shifts, geo_explicit.shifts)

    # a disk clipping the frame edge drops the off-detector tilts instead of wrapping
    geo_edge = preprocess_geometry(**kw, probe_center=(5.0, 49.67))
    er, ec = np.divmod(geo_edge.bf_flat_inds, w)
    assert er.min() >= 0 and len(er) < len(rows), "clipped disk → fewer BF pixels, none wrapped"
    edist = np.hypot(er - 5.0, ec - 49.67)
    assert edist.max() <= r_px + 1.0, "surviving pixels still belong to the disk"

    # CLI plumb: --center ROW COL lands in the config
    cfg = _parallax_config(["--center", "43.95", "49.67"])
    assert cfg.probe_center == (43.95, 49.67)
    from realtime_stem.processing.parallax.config import ParallaxConfig
    assert ParallaxConfig().probe_center is None, "default stays centered"
    print("test_off_center_probe_geometry: PASS (mask recentered, DC shift 0, default identical, edge clip, CLI)")


# ── mechanics: toroidal wrap, input parity, shape guard, protocol ──────────────

def test_toroidal_wrap():
    """A single-BF-pixel geometry with a shift past the grid edge must wrap around the torus."""
    h, w = 5, 5
    bf = np.array([2 * w + 2, 0 * w + 0], dtype=np.int32)      # center-ish + corner pixel
    shifts = np.array([[-1, 3], [0, 0]], dtype=np.int32)       # first wraps in y (and x edge)
    geo = ParallaxGeometry(bf_flat_inds=bf, shifts=shifts, detector_shape=(h, w),
                           scan_shape=(4, 5), upsampling_factor=1, upsampled_scan_gpts=(4, 5))
    frame = np.zeros((1, h, w), dtype=np.uint16)
    frame[0, 2, 2] = 12                                        # first BF pixel hot
    ids = np.array([1 * 5 + 4])                                # scan (yt=1, xt=4) → x wraps too
    acc = ParallaxAccumulator(geo, device="cpu", subtract_frame_mean=True)
    acc.process(frame, ids)
    recon = acc.result()

    exp = _reference_accumulate(frame, bf, shifts, _coords(ids, 5, 1), (4, 5))
    assert np.allclose(recon, exp, atol=1e-4), np.abs(recon - exp).max()
    # explicit wrap check: pixel 0 (val 12−6=6) at (yt+dy, xt+dx)=((1-1)%4,(4+3)%5)=(0,2)
    assert np.isclose(recon[0, 2], 6.0, atol=1e-4), recon[0, 2]
    # pixel 1 (val 0−6=−6) at ((1+0)%4,(4+0)%5)=(1,4)
    assert np.isclose(recon[1, 4], -6.0, atol=1e-4), recon[1, 4]
    print("test_toroidal_wrap: PASS (independent %Ny/%Nx wrap, matches reference)")


def test_torch_tensor_input_matches_numpy_input():
    """The GPU-decode / Holoscan-DLPack path hands us a torch tensor — must equal the ndarray path."""
    rng = np.random.default_rng(5)
    geo = _defocus_geo(-1.5e4)
    frames = rng.integers(0, 60000, size=(20, *_GEO["det_shape"]), dtype=np.uint16)
    ids = np.arange(20)

    a_np = ParallaxAccumulator(geo, device="cpu")
    a_np.process(frames, ids)
    a_t = ParallaxAccumulator(geo, device="cpu")
    a_t.process(torch.from_numpy(frames.astype(np.int32)), ids)
    assert np.allclose(a_np.result(), a_t.result(), atol=1e-5)
    print("test_torch_tensor_input_matches_numpy_input: PASS")


def test_ids_as_torch_tensor_matches_numpy_ids():
    """The Holoscan sink hands ids as a *device* tensor (skipping a per-batch D2H round trip) —
    process() must accept torch ids on any device and produce exactly the numpy-ids result."""
    rng = np.random.default_rng(7)
    geo = _defocus_geo(-1.5e4)
    frames = rng.integers(0, 60000, size=(20, *_GEO["det_shape"]), dtype=np.uint16)
    ids = np.arange(20)[::-1].copy()            # non-sequential, in scan range

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = ParallaxAccumulator(geo, device=device)
    ref.process(frames, ids)
    acc = ParallaxAccumulator(geo, device=device)
    acc.process(frames, torch.as_tensor(ids, device=device))
    a, b = ref.result(), acc.result()
    scale = max(np.abs(a).max(), 1.0)
    assert np.allclose(a, b, rtol=1e-5, atol=1e-4 * scale), np.abs(a - b).max()
    print(f"test_ids_as_torch_tensor_matches_numpy_ids: PASS ({device})")


def test_uint16_torch_tensor_high_values_exact():
    """The live stream hands uint16 frames as torch tensors, and defect pixels sit at 65535.
    uint16 is a second-class torch dtype, so the hot path reinterprets bits — values above the
    int16 range must still come out exact (pinned against the numpy-uint16 path)."""
    rng = np.random.default_rng(8)
    geo = _defocus_geo(-1.5e4)
    frames = rng.integers(0, 65536, size=(12, *_GEO["det_shape"]), dtype=np.uint16)
    frames[0, 0, 0] = 65535
    frames[3, :, :] = 65535                     # a whole saturated frame
    ids = np.arange(12)

    ref = ParallaxAccumulator(geo, device="cpu")
    ref.process(frames, ids)                    # numpy path: exact int32 widening

    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc = ParallaxAccumulator(geo, device=device)
    acc.process(torch.from_numpy(frames).to(device), ids)   # torch.uint16 tensor
    a, b = ref.result(), acc.result()
    scale = max(np.abs(a).max(), 1.0)
    assert np.allclose(a, b, rtol=1e-4, atol=1e-3 * scale), np.abs(a - b).max()
    print(f"test_uint16_torch_tensor_high_values_exact: PASS ({device})")


def test_uint32_high_values_exact():
    """ARINA uint32 (long-dwell) frames carry a dtype-max pixel-mask sentinel (4294967295), and
    counts may exceed 2^31. The numpy path and the torch.uint32 tensor path (what the Holoscan
    ingest hands over for --dtype uint32) must both agree with a float64 truth — an int32
    bit-reinterpretation would turn those pixels negative."""
    rng = np.random.default_rng(9)
    geo = _defocus_geo(-1.5e4)
    frames = rng.integers(0, 2**32 - 1, size=(10, *_GEO["det_shape"]), dtype=np.uint32)
    frames[0, 0, 0] = 2**32 - 1
    frames[2, :, :] = 2**32 - 1                       # a whole sentinel frame
    ids = np.arange(10)

    truth = ParallaxAccumulator(geo, device="cpu")
    truth.process(frames.astype(np.float64), ids)     # float input: no integer tricks at all
    exp = truth.result()
    scale = max(np.abs(exp).max(), 1.0)

    a = ParallaxAccumulator(geo, device="cpu")
    a.process(frames, ids)                            # numpy uint32 path
    assert np.allclose(a.result(), exp, rtol=1e-4, atol=1e-3 * scale), np.abs(a.result() - exp).max()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = ParallaxAccumulator(geo, device=device)
    b.process(torch.from_numpy(frames.astype(np.int64)).to(torch.uint32).to(device), ids)
    assert np.allclose(b.result(), exp, rtol=1e-4, atol=1e-3 * scale), np.abs(b.result() - exp).max()
    print(f"test_uint32_high_values_exact: PASS ({device})")


def test_shape_guard():
    geo = _defocus_geo(-1.5e4)                                 # detector 32x32
    acc = ParallaxAccumulator(geo, device="cpu")
    try:
        acc.process(np.zeros((1, 16, 16), np.uint32), np.array([0]))
        raise AssertionError("shape guard should have raised")
    except ValueError:
        pass
    acc.process(np.zeros((1, 32, 32), np.uint32), np.array([0]))   # correct shape ok
    print("test_shape_guard: PASS")


def test_kept_frame_mean_keeps_amplitude_contrast():
    """A pure amplitude object (every BF pixel sees the same image) cancels exactly under the
    per-frame mean removal, and survives as the shifted BF sum when the mean is kept (default)."""
    geo = _defocus_geo(-1.5e4)
    rng = np.random.default_rng(0)
    amplitude = rng.integers(50, 200, size=144)                    # one level per scan position
    frames = np.broadcast_to(amplitude[:, None, None], (144, 32, 32)).astype(np.uint32)
    ids = np.arange(144)

    removed = ParallaxAccumulator(geo, device="cpu", subtract_frame_mean=True)
    removed.process(frames, ids)
    assert np.abs(removed.result()).max() < 1e-3

    kept = ParallaxAccumulator(geo, device="cpu")
    kept.process(frames, ids)
    # each scan position deposits its level at every BF pixel's shifted output position
    expected = np.zeros(144)
    oy = (ids[:, None] // 12 + geo.shifts[None, :, 0]) % 12
    ox = (ids[:, None] % 12 + geo.shifts[None, :, 1]) % 12
    np.add.at(expected, (oy * 12 + ox).ravel(), np.repeat(amplitude.astype(float), geo.n_bf))
    np.testing.assert_allclose(kept.result().ravel(), expected, rtol=1e-5)
    print("test_kept_frame_mean_keeps_amplitude_contrast: PASS")


def test_accumulator_satisfies_protocol():
    """ParallaxAccumulator must satisfy the `Accumulator` contract the ingest sink drives."""
    from realtime_stem.processing.base import Accumulator
    acc = ParallaxAccumulator(_defocus_geo(-1.5e4), device="cpu")
    assert isinstance(acc, Accumulator)
    assert acc.detector_shape == (32, 32)
    assert acc.scan_shape == (12, 12)
    print("test_accumulator_satisfies_protocol: PASS")


def test_registry_has_parallax():
    """The runner discovers algorithms through the registry."""
    from realtime_stem.processing import registry
    from realtime_stem.processing.base import Algorithm
    algo = registry.get("parallax")
    assert isinstance(algo, Algorithm)
    assert algo.name == "parallax"
    assert "parallax" in registry.names()
    print("test_registry_has_parallax: PASS")


# ── CLI → config mapping (the live tuning knobs for a microscope run) ─────────
# A silent unit slip here (deg vs rad, doubled sampling) wrecks a live run while every oracle
# test still passes, so the mapping is pinned explicitly.

def test_cli_config_overrides():
    """Every live control knob must land in the config with the right units/shape."""
    from realtime_stem.processing.parallax.config import ParallaxConfig
    base = ParallaxConfig()
    assert _parallax_config([], base) == base, "no flags → config unchanged"

    cfg = _parallax_config([
        "--defocus", "-2e4", "--rotation", "-15", "--c12", "2e3", "--phi12", "30",
        "--energy", "200e3", "--semiangle", "22.1",
        "--angular-sampling", "0.61", "--scan-sampling", "0.39",
        "--scan", "512", "512", "--upsampling", "2",
    ], base)
    assert cfg.aberration_coefs["C10"] == -2e4
    assert cfg.aberration_coefs["C12"] == 2e3
    assert np.isclose(cfg.aberration_coefs["phi12"], np.deg2rad(30)), "phi12 CLI is degrees"
    assert np.isclose(cfg.rotation_angle, np.deg2rad(-15)), "rotation CLI is degrees"
    assert cfg.energy == 200e3
    assert cfg.semiangle_cutoff == 22.1
    assert cfg.angular_sampling == (0.61, 0.61)
    assert cfg.scan_sampling == (0.39, 0.39)
    assert cfg.scan_shape == (512, 512)
    assert cfg.upsampling_factor == 2
    print("test_cli_config_overrides: PASS (defocus/rotation/astig/energy/samplings/scan)")


def test_negative_scientific_notation_parses():
    """`--defocus -1.5e4` must parse (argparse reads sci-notation negatives as flags otherwise);
    the algorithm handles it via preprocess_argv (join_negative_floats)."""
    cfg = _parallax_config(["--defocus", "-1.5e4", "--rotation", "-15", "--upsampling", "2",
                            "--scan", "512", "512"])
    assert cfg.aberration_coefs["C10"] == -1.5e4
    assert np.isclose(cfg.rotation_angle, np.deg2rad(-15))
    assert cfg.upsampling_factor == 2
    assert cfg.scan_shape == (512, 512)
    print("test_negative_scientific_notation_parses: PASS")


def test_cli_binned_with_explicit_angular_sampling():
    """--binned doubles the config-default angular sampling; an explicit --angular-sampling
    describes the actual mode being run and must win verbatim (no double-doubling)."""
    from realtime_stem.processing.parallax.config import ParallaxConfig
    base = ParallaxConfig()
    cfg = _parallax_config(["--binned"], base)
    assert cfg.detector_shape == (96, 96)
    assert cfg.angular_sampling == (base.angular_sampling[0] * 2, base.angular_sampling[1] * 2)

    cfg = _parallax_config(["--binned", "--angular-sampling", "0.7"], base)
    assert cfg.detector_shape == (96, 96)
    assert cfg.angular_sampling == (0.7, 0.7)
    print("test_cli_binned_with_explicit_angular_sampling: PASS")


# ── GPU parity: CUDA result must equal CPU result (GPU host only; skips without CUDA) ──

def test_gpu_matches_cpu():
    if not torch.cuda.is_available():
        print("test_gpu_matches_cpu: SKIP (no CUDA)")
        return
    rng = np.random.default_rng(6)
    geo = _defocus_geo(-1.5e4)
    frames = rng.integers(0, 60000, size=(64, *_GEO["det_shape"]), dtype=np.uint16)
    ids = np.arange(64)
    cpu = ParallaxAccumulator(geo, device="cpu")
    cpu.process(frames, ids)
    gpu = ParallaxAccumulator(geo, device="cuda")
    gpu.process(frames, ids)
    a, b = cpu.result(), gpu.result()
    scale = max(np.abs(a).max(), 1.0)
    assert np.allclose(a, b, rtol=1e-4, atol=1e-3 * scale), np.abs(a - b).max()
    print("test_gpu_matches_cpu: PASS")


if __name__ == "__main__":
    test_geometry_matches_reference()
    test_accumulate_matches_reference()
    test_accumulate_upsampled_matches_reference()
    test_zero_aberration_cancels_to_zero()
    test_defocus_shifts_radial_and_scaling()
    test_off_center_probe_geometry()
    test_toroidal_wrap()
    test_torch_tensor_input_matches_numpy_input()
    test_ids_as_torch_tensor_matches_numpy_ids()
    test_uint16_torch_tensor_high_values_exact()
    test_uint32_high_values_exact()
    test_shape_guard()
    test_kept_frame_mean_keeps_amplitude_contrast()
    test_accumulator_satisfies_protocol()
    test_registry_has_parallax()
    test_cli_config_overrides()
    test_negative_scientific_notation_parses()
    test_cli_binned_with_explicit_angular_sampling()
    test_gpu_matches_cpu()
    print("\nALL PARALLAX TESTS PASSED")
