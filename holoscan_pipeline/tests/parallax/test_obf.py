"""Tests for live OBF (`live4d.algorithms.obf`, `live4d.parallax.obf`).

The port itself is checked against quantem offline, on identical inputs, to 2e-6 relative on the
Gd2O3 recordings (EM_Realtime_Processing, `sessions/2026-10-20_Gd2O3/analysis/obf_vs_quantem.py`).
Here the oracle is `reconstruct_once`, the whole store in one call, and the physics a simulated
weak phase object:

  • geometry: `bf_offsets` inverts `preprocess_geometry`'s placement, and a column-flipped detector
    gives the same offsets and sub-pixel centre as its canonical layout;
  • the numerator is linear and order-free: deposits in any chunks and order give one image, and
    `latest` replacements (new − old) give the image of the frames held;
  • the setup built a unit at a time is the setup built in one call;
  • the tracker, stepped after batches with revisits and out of order, ends exactly on
    `reconstruct_once` of the final store; a retune rebuilds it; off and on again; a slow build is
    spread over steps that each keep to their budget;
  • a simulated weak phase object comes out with the right sign and closer to the truth than the
    phase-flipped parallax image, on both placements (lattice and patch).
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from live4d.algorithms.obf.core import (
    ObfSetup,
    bf_offsets,
    center_fraction,
    normalize,
    reconstruct_once,
)
from live4d.algorithms.parallax.engine import ParallaxAccumulator, preprocess_geometry
from live4d.algorithms.parallax.readout import fourier_filters
from live4d.parallax.ingest.sink import ACC_LOCK
from live4d.parallax.obf import _SETUPS, ObfTracker

REPO = Path(__file__).resolve().parents[2] / "python/live4d/parallax"

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])   # OBF does not run on mps

_SCAN = (12, 10)
_NPOS = _SCAN[0] * _SCAN[1]


def _geometry(c10=-1.5e4, rotation=-15.0, upsampling=2, center=(16.3, 15.6), flip=True, scan=_SCAN):
    return preprocess_geometry(
        (32, 32), scan, (256 / 12, 256 / 12), 300e3, 4.0, (0.307617, 0.307617),
        aberration_coefs={"C10": c10, "C12": 300.0, "phi12": 0.4},
        rotation_angle=math.radians(rotation), upsampling_factor=upsampling,
        probe_center=center, flip_cols=flip)


def _frames(n, seed=0):
    return np.random.default_rng(seed).integers(800, 1200, size=(n, 32, 32), dtype=np.uint16)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _tol(device):
    return 1e-9 if device == "cpu" else 2e-5


def _dtype(device):
    return torch.float64 if device == "cpu" else None


@pytest.fixture(autouse=True)
def _fresh_setups():
    _SETUPS.clear()
    yield
    _SETUPS.clear()


# ── geometry ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("flip", [False, True])
def test_bf_offsets_invert_the_placement(flip):
    geo = _geometry(center=(15.7, 17.2), flip=flip)
    off = bf_offsets(geo)
    canonical = _geometry(center=(15.7, 17.2) if not flip else (15.7, 31 - 17.2), flip=False)
    np.testing.assert_array_equal(off, bf_offsets(canonical))
    np.testing.assert_allclose(center_fraction(geo), center_fraction(canonical))
    # every offset is inside the semiangle, and offset 0 is the rounded centre's pixel
    assert (np.hypot(*off.T) <= 4.0 / 0.307617 + 1e-9).all()
    np.testing.assert_allclose(center_fraction(_geometry(center=(15.7, 17.2), flip=False)),
                               [0.3, -0.2], atol=1e-12)


# ── the setup and the numerator ────────────────────────────────────────────────

def test_the_setup_built_a_unit_at_a_time_is_the_setup_built_at_once():
    geo = _geometry()
    whole = ObfSetup.build(geo, device="cpu", dtype=torch.float64)
    steps, units = ObfSetup.build_steps(geo, device="cpu", dtype=torch.float64), 0
    while True:
        try:
            next(steps)
            units += 1
        except StopIteration as done:
            stepped = done.value
            break
    assert units > 4                                   # the probes, the operands, the Gram's batches
    assert stepped.key == whole.key and stepped.placement == whole.placement
    torch.testing.assert_close(stepped.gram, whole.gram, rtol=0, atol=0)
    torch.testing.assert_close(stepped._operator, whole._operator, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("upsampling", [1, 2])
def test_the_patch_products_are_the_fft_route(device, upsampling):
    """The patch grid's contraction as two matrix products equals the route it replaces, written
    out here: per sub-lattice the spectra Â_s* ⊙ F − B̂_s ⊙ reflect(fft2(V·ψ*)), inverse FFTs over
    the padded detector grid, and the lags the patch grid keeps."""
    from live4d.algorithms.obf.core import _reflect

    geo = _geometry(upsampling=upsampling)
    setup = ObfSetup.build(geo, device=device, dtype=_dtype(device))
    assert setup.placement == "patch" and setup._products is not None
    rng = np.random.default_rng(4)
    values = torch.as_tensor(rng.standard_normal((37, geo.n_bf))).to(setup.rdtype).to(device)
    ops = setup._ops
    p0, p1 = ops.padded
    frames = torch.zeros((37, p0 * p1), dtype=setup.cdtype, device=device)
    frames[:, setup._bf_flat] = values.to(setup.cdtype)
    frames = frames.view(37, p0, p1)
    f_hat = torch.fft.fft2(frames * ops.psi)
    g_hat = _reflect(torch.fft.fft2(frames * ops.psi.conj()))
    spectra = setup._conj_a_hat[:, None] * f_hat[None] - setup._b_hat[:, None] * g_hat[None]
    expected = ops.to_grid(torch.fft.ifft2(spectra))
    got = setup._contract(values)
    assert got.shape == expected.shape == (37, *setup.grid_shape)
    diff = torch.linalg.vector_norm(got - expected) / torch.linalg.vector_norm(expected)
    assert diff.item() < 10 * _tol(device)


@pytest.mark.parametrize("device", DEVICES)
def test_deposit_is_linear_and_order_free(device):
    geo = _geometry()
    setup = ObfSetup.build(geo, device=device, dtype=_dtype(device))
    rng = np.random.default_rng(1)
    values = torch.as_tensor(rng.standard_normal((_NPOS, geo.n_bf))).to(setup.rdtype).to(device)
    pos = torch.arange(_NPOS, device=device)
    whole = setup.buffer()
    setup.deposit(whole, values, pos)
    parts = setup.buffer()
    order = torch.as_tensor(rng.permutation(_NPOS), device=device)
    for chunk in order.split(17):
        setup.deposit(parts, values[chunk], pos[chunk])
    a, b = setup.image(whole).cpu().numpy(), setup.image(parts).cpu().numpy()
    assert _rel(b, a) < 10 * _tol(device)
    doubled = setup.buffer()
    setup.deposit(doubled, 2 * values, pos)
    assert _rel(setup.image(doubled).cpu().numpy(), 2 * a) < 10 * _tol(device)


@pytest.mark.parametrize("device", DEVICES)
def test_a_sub_batch_padded_with_zero_frames_deposits_the_same(device):
    """Deposits pad each sub-batch to a power of two with zero frames (few FFT sizes, few cuFFT
    plans): 3, 5 and 7 frames at once deposit what they deposit one at a time."""
    from live4d.algorithms.obf.core import _sub_batch

    assert [_sub_batch(n, 655) for n in (0, 1, 3, 5, 8, 300, 600, 655)] == \
        [0, 1, 4, 8, 8, 512, 655, 655]
    geo = _geometry()
    setup = ObfSetup.build(geo, device=device, dtype=_dtype(device))
    rng = np.random.default_rng(3)
    values = torch.as_tensor(rng.standard_normal((7, geo.n_bf))).to(setup.rdtype).to(device)
    pos = torch.as_tensor(rng.choice(_NPOS, 7, replace=False), device=device)
    for n in (3, 5, 7):
        together, apart = setup.buffer(), setup.buffer()
        setup.deposit(together, values[:n], pos[:n])
        for i in range(n):
            setup.deposit(apart, values[i:i + 1], pos[i:i + 1])
        torch.testing.assert_close(together, apart, rtol=0, atol=10 * _tol(device) * float(apart.abs().max()))


@pytest.mark.parametrize("device", DEVICES)
def test_latest_replacements_are_the_frames_held(device):
    geo = _geometry()
    setup = ObfSetup.build(geo, device=device, dtype=_dtype(device))
    rdtype = setup.rdtype
    first = torch.as_tensor(_frames(_NPOS, 0).reshape(_NPOS, -1)[:, geo.bf_flat_inds]).to(rdtype)
    second = torch.as_tensor(_frames(_NPOS, 1).reshape(_NPOS, -1)[:, geo.bf_flat_inds]).to(rdtype)
    first, second = first.to(device), second.to(device)
    mean = first.mean(0)
    pos = torch.arange(_NPOS, device=device)
    replaced = torch.as_tensor(np.random.default_rng(2).permutation(_NPOS)[:50], device=device)
    buffer = setup.buffer()
    setup.deposit(buffer, normalize(first, mean), pos)
    setup.deposit(buffer, normalize(second[replaced], mean) - normalize(first[replaced], mean),
                  replaced)
    held = first.clone()
    held[replaced] = second[replaced]
    fresh = setup.buffer()
    setup.deposit(fresh, normalize(held, mean), pos)
    assert _rel(setup.image(buffer).cpu().numpy(), setup.image(fresh).cpu().numpy()) < \
        10 * _tol(device)


# ── the tracker ────────────────────────────────────────────────────────────────

def _feed(acc, passes, seed=0, batch=32, shuffle=True, budget=0.002):
    """Passes of frames in batches, under ACC_LOCK, each followed by an OBF step as the sink
    does; ids out of order within a pass when `shuffle`."""
    rng = np.random.default_rng(seed)
    for p in range(passes):
        ids = rng.permutation(_NPOS) if shuffle else np.arange(_NPOS)
        frames = _frames(_NPOS, seed + p)
        for lo in range(0, _NPOS, batch):
            sel = ids[lo:lo + batch]
            with ACC_LOCK:
                acc.process(frames[sel], sel + p * _NPOS)
                if acc.obf is not None:
                    acc.obf.step(budget)


def _expected(acc, geo):
    """`reconstruct_once` of the store, through the readout's Nyquist filter as the tracker."""
    with ACC_LOCK:
        img = reconstruct_once(acc._store, acc._visits, acc._store_kind, geo).cpu().numpy()
    base, _ = fourier_filters(geo.upsampled_scan_gpts, geo.upsampled_scan_sampling,
                              geo.wavelength, geo.aberration_coefs,
                              suppress_nyquist=geo.upsampling_factor > 1)
    return img if base is None else np.fft.ifft2(np.fft.fft2(img) * base).real


def _step_until(tracker, cond, budget=0.05, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        with ACC_LOCK:
            tracker.step(budget)
    return cond()


@pytest.mark.parametrize("device", DEVICES)
def test_tracker_ends_on_the_frames_held(device):
    geo = _geometry()
    acc = ParallaxAccumulator(geo, device=device, accumulate="latest", wrap_ids=True)
    tracker = ObfTracker(acc, live_every=0.02)
    try:
        _feed(acc, passes=1)
        with ACC_LOCK:                         # a partial second pass: some positions replaced
            acc.process(_frames(40, 7), np.arange(40) + 3 * _NPOS // 2)
            image = tracker.finalize(timeout_s=60)
        assert image is not None and tracker.state == "live"
        assert _rel(image, _expected(acc, geo)) < 1e-4
        with ACC_LOCK:
            stack, names, meta = acc.snapshot()
        assert names[-1] == "obf" and stack.shape == (len(names), *geo.upsampled_scan_gpts)
        assert meta["obf_state"] == "live" and meta["obf_coverage"] == 1.0
        with ACC_LOCK:
            assert "obf" in acc.images()
    finally:
        acc.close()


def test_steps_alone_cover_the_frames_held():
    """Without `finalize`: steps of a small budget build the weights, then cover every position
    held; with the means taken over all of them, the image is that of the store."""
    geo = _geometry()
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    _feed(acc, passes=1)
    tracker = ObfTracker(acc, live_every=0.0)
    try:
        assert _step_until(tracker, lambda: tracker.state == "live", budget=0.005)
        assert tracker.coverage == 1.0
        assert _rel(tracker.image(), _expected(acc, geo)) < 1e-4
    finally:
        acc.close()


def test_tracker_starts_on_a_store_that_is_already_full():
    geo = _geometry()
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    _feed(acc, passes=2, shuffle=False)
    tracker = ObfTracker(acc, live_every=0.02)
    try:
        assert tracker._held.all() and tracker._max_id == 2 * _NPOS - 1
        with ACC_LOCK:
            assert _rel(tracker.finalize(timeout_s=60), _expected(acc, geo)) < 1e-4
    finally:
        acc.close()


def test_tracker_follows_a_retune_and_can_be_switched_off():
    geo = _geometry()
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    tracker = ObfTracker(acc, live_every=0.02)
    try:
        _feed(acc, passes=1)
        with ACC_LOCK:
            before = tracker.finalize(timeout_s=60)
        retuned = _geometry(c10=-1.3e4, rotation=-14.0)
        with ACC_LOCK:
            acc.set_geometry(retuned)
            tracker.step(0.002)
        assert tracker.weights_state() == "building" or tracker.state == "filling"
        with ACC_LOCK:
            after = tracker.finalize(timeout_s=60)
        assert _rel(after, _expected(acc, retuned)) < 1e-4
        assert _rel(after, before) > 0.05             # the geometry changed the image
        with ACC_LOCK:
            tracker.set_enabled(False)
            tracker.step(0.002)
            _, names, meta = acc.snapshot()
        assert tracker.state == "off" and tracker._shadow is None
        assert "obf" not in names and meta["obf_state"] == "off"
        with ACC_LOCK:
            tracker.set_enabled(True)                 # the store is swept again from the start
            assert _rel(tracker.finalize(timeout_s=60), _expected(acc, retuned)) < 1e-4
    finally:
        acc.close()


def test_the_channel_is_obf_where_covered_and_the_phase_flipped_stand_in_elsewhere():
    geo = _geometry()
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    tracker = ObfTracker(acc, enabled=False)
    try:
        rng = np.random.default_rng(3)
        ny, nx = geo.upsampled_scan_gpts
        image = rng.standard_normal((ny, nx)).astype(np.float32) * 0.1
        standin = rng.standard_normal((ny, nx)).astype(np.float32) * 500 + 3000
        np.testing.assert_array_equal(tracker.compose(standin)[0], standin)  # nothing yet
        done = np.zeros(ny, bool)
        done[6:15] = True
        tracker.image_np, tracker.done_np = image, done
        out, rows = tracker.compose(standin)
        assert rows == [6, 15]
        np.testing.assert_array_equal(out[done], image[done])
        scale = image[done].std() / standin[done].std()
        expected = scale * (standin - standin[done].mean()) + image[done].mean()
        np.testing.assert_allclose(out[~done], expected[~done], rtol=1e-4, atol=1e-6)
        tracker.done_np = None                                   # every frame covered: OBF alone
        np.testing.assert_array_equal(tracker.compose(standin)[0], image)
    finally:
        acc.close()


def test_an_output_row_shows_obf_once_the_frames_within_the_probe_are_covered():
    geo = _geometry()                                         # ×2, 12 x 10 scan
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    tracker = ObfTracker(acc, enabled=False)
    try:
        u = geo.upsampling_factor
        reach = int(np.ceil(np.hypot(*np.asarray(geo.shifts, float).T).max() / u)) + 2
        tracker._seen[:] = True
        tracker._seen.reshape(_SCAN)[0, 3] = False             # one frame of scan row 0 missing
        done = tracker._done_rows()
        scan_row = np.arange(_SCAN[0] * u) // u
        np.testing.assert_array_equal(done, scan_row > reach)
    finally:
        acc.close()


def test_a_slow_build_is_spread_over_steps_within_their_budget():
    """Each unit of the weights' build takes 20 ms more here: a step of a 10 ms budget takes one
    unit (the floor), never most of the build, and the batches keep their pace meanwhile."""
    unit_s = 0.02

    def slow(*args, **kwargs):
        steps = ObfSetup.build_steps(*args, **kwargs)
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value
            time.sleep(unit_s)
            yield

    geo = _geometry()
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    tracker = ObfTracker(acc, live_every=0.02, builder=slow)
    try:
        frames = _frames(_NPOS)
        worst, total, steps = 0.0, 0.0, 0
        while tracker.weights_state() != "ready" and steps < 2000:
            t0 = time.perf_counter()
            with ACC_LOCK:
                acc.process(frames[:30], np.arange(30))
                tracker.step(0.01)
            took = time.perf_counter() - t0
            worst, total, steps = max(worst, took), total + took, steps + 1
        assert tracker.weights_state() == "ready" and steps > 4
        assert worst < total / 2                      # no step held most of the build
        with ACC_LOCK:
            assert tracker.finalize(timeout_s=60) is not None
    finally:
        acc.close()


def test_the_build_s_host_work_comes_in_small_pieces(monkeypatch):
    """The probes', the operands' and the cross term's host arrays are made a block at a time:
    smaller blocks give more units and the same setup."""
    from live4d.algorithms.obf import core

    def build(points):
        monkeypatch.setattr(core, "_HOST_POINTS_PER_STEP", points)
        steps, units = ObfSetup.build_steps(_geometry(), device="cpu", dtype=torch.float64), 0
        while True:
            try:
                next(steps)
                units += 1
            except StopIteration as done:
                return done.value, units

    whole, coarse = build(1 << 30)
    fine, many = build(256)
    assert many > coarse + 10
    for name in ("gram", "_operator", "_conj_a_hat", "_b_hat"):
        torch.testing.assert_close(getattr(fine, name), getattr(whole, name), rtol=1e-14, atol=0)


def test_obf_yields_to_a_busy_branch():
    """OBF's budget is zero while the branch is behind, on the batch that publishes a snapshot and
    at a zero scale; its share halves while the branch's batches take most of their intervals,
    down to zero, and grows back once they don't; the shares are saved up into steps."""
    from live4d.parallax.ingest.sink import LOAD_WINDOW_S, OBF_STEP_S, BatchSink

    acc = ParallaxAccumulator(_geometry(), device="cpu", accumulate="latest", wrap_ids=True)
    tracker = ObfTracker(acc, live_every=0.02)
    sink = BatchSink(acc, obf_share=0.4)
    steps = []
    tracker.step = lambda budget: steps.append(budget) or False
    try:
        assert sink._obf_budget(0.01, 0.0, False) == pytest.approx(0.004)
        assert sink._obf_budget(0.01, 0.02, False) == 0               # behind
        assert sink._obf_budget(0.01, 0.0, True) == 0                 # publishing
        windows = int(math.ceil(LOAD_WINDOW_S / 0.01))
        scales = []
        for _ in range(6):                                            # 90% busy
            for _ in range(windows):
                sink._pace_load(0.009, 0.01)
            scales.append(sink.obf_scale)
        assert scales[:4] == [0.5, 0.25, 0.125, 0.0625] and scales[-1] == 0
        frames = _frames(30)
        sink.feed(frames, np.arange(30), interval_s=0.01, queued_s=0.0)
        assert steps == [] and tracker.share_scale == 0               # no OBF at all
        for _ in range(30):                                           # 10% busy
            for _ in range(windows):
                sink._pace_load(0.001, 0.01)
        assert sink.obf_scale == 1
        sink.feed(frames, np.arange(30, 60), interval_s=0.01, queued_s=0.0)
        assert steps == []                                            # 4 ms saved up
        sink.feed(frames, np.arange(60, 90), interval_s=0.01, queued_s=0.0)
        assert steps == [pytest.approx(0.008)] and 0.008 >= OBF_STEP_S
    finally:
        acc.close()


def test_a_continuous_run_saves_obf_between_series_without_waiting(tmp_path):
    """Between the series of a continuous run (a tag) the latest OBF image is saved as it is."""
    from live4d.parallax.algorithm import ParallaxAlgorithm

    acc = ParallaxAccumulator(_geometry(), device="cpu", accumulate="latest", wrap_ids=True)
    tracker = ObfTracker(acc, live_every=0.02)
    try:
        _feed(acc, passes=1)
        with ACC_LOCK:
            tracker.finalize(timeout_s=60)
        ParallaxAlgorithm().save(acc, tmp_path, tag="scan002")
        assert (tmp_path / "parallax_obf_scan002.npy").exists()
    finally:
        acc.close()


def test_offline_replay_saves_obf(tmp_path):
    """The offline replay with `compute_obf` on a capture of a pass and a bit: at the end of the
    finite acquisition parallax_obf.npy is the OBF of the frames held."""
    from .helpers import offline_args

    frames = _frames(_NPOS + 30, seed=9)
    np.save(tmp_path / "scan.npy", frames)
    flags = ["--scan", *map(str, _SCAN), "--energy", "300e3",
             "--semiangle", "4", "--angular-sampling", "0.307617", "--scan-sampling",
             f"{256 / 12}", "--defocus=-1.5e4", "--rotation=-15", "--c12=300",
             f"--phi12={math.degrees(0.4)}", "--center", "16.3", "15.6", "--flip-cols",
             "--upsampling", "2"]
    argv = offline_args(tmp_path, flags, tmp_path / "scan.npy", accumulate="latest",
                        compute_obf=True)
    out = subprocess.run([sys.executable, "-m", "live4d.parallax.offline", *argv], cwd=REPO,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "finishing the image of every frame held" in out.stdout
    acc = ParallaxAccumulator(_geometry(), device="cpu", accumulate="latest", wrap_ids=True)
    acc.process(frames[:_NPOS], np.arange(_NPOS))
    acc.process(frames[_NPOS:], np.arange(_NPOS, _NPOS + 30))
    saved = np.load(tmp_path / "out" / "parallax_obf.npy")
    assert _rel(saved, _expected(acc, _geometry())) < 1e-4


# ── physics: a simulated weak phase object ─────────────────────────────────────

@pytest.mark.parametrize("scan, placement", [((48, 48), "lattice"), ((64, 64), "patch")])
def test_weak_phase_object(scan, placement):
    """Noiseless frames of a weak phase object: OBF has the sign of the phase and is closer to the
    (band-limited) truth than the phase-flipped parallax image."""
    from live4d.parallax.tools.simulate_4dstem import SimParams, Simulation

    p = SimParams(scan=scan, detector=(48, 48), semiangle=10.0, angular_sampling=0.5,
                  scan_step=2.0, aberrations={"C10": -2000.0}, rotation_deg=-15.0, dose=0,
                  n_particles=6, seed=3)
    sim = Simulation(p, device="cpu")
    frames = np.concatenate([f for _, f in sim.batches(512)])
    g = sim.grid
    geo = preprocess_geometry((48, 48), scan, (g.scan_step,) * 2, p.energy, p.semiangle,
                              (p.angular_sampling,) * 2, aberration_coefs={"C10": -2000.0},
                              rotation_angle=math.radians(-15.0))
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    acc.process(frames, np.arange(len(frames)))
    setup = ObfSetup.build(geo, device="cpu", dtype=torch.float64)
    assert setup.placement == placement
    obf = reconstruct_once(acc._store, acc._visits, acc._store_kind, geo, setup=setup).numpy()
    flipped = acc.channels()["phase_flipped"].numpy()
    q = np.hypot(*np.meshgrid(np.fft.fftfreq(scan[0], g.scan_step),
                              np.fft.fftfreq(scan[1], g.scan_step), indexing="ij"))
    band = q < 2 * p.semiangle * 1e-3 / g.wavelength
    truth = np.fft.ifft2(np.fft.fft2(sim.scan_phase()) * band).real

    def corr(a):
        return np.corrcoef(a.ravel(), truth.ravel())[0, 1]

    assert corr(obf) > 0.7
    assert corr(obf) > corr(flipped) + 0.05
