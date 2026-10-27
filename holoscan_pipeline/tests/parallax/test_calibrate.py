"""Tests for `tools/calibrate_parallax.py`: the geometry measured from a simulated scan is the one
it was simulated with, for every detector layout, and its `run.py` flags rebuild the simulator's
own reconstruction geometry. The simulation is quadratic (C10, C12), so the higher-order
diagnostic must find nothing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


from live4d.parallax.algorithm import ParallaxAlgorithm  # noqa: E402
from live4d.algorithms.parallax.config import config_from_args  # noqa: E402
from live4d.algorithms.parallax.engine import preprocess_geometry  # noqa: E402
from live4d.algorithms.parallax.orientation import (  # noqa: E402
    DETECTOR_LAYOUTS,
    layout_frames,
)
from live4d.parallax.tools.calibrate_parallax import (  # noqa: E402
    _wrap_deg,
    array_batches,
    calibrate,
    fit_disk,
    shift_matrix,
)
from live4d.parallax.tools.simulate_4dstem import SimParams, Simulation, reconstruct_flags  # noqa: E402

C10, C12, PHI12, ROT = -1000.0, 120.0, 25.0, -15.0


@pytest.fixture(scope="module")
def sim():
    """A noiseless canonical scan; every layout is a re-lay of the same frames."""
    p = SimParams(scan=(48, 48), detector=(64, 64), semiangle=10, angular_sampling=0.5,
                  scan_step=2.0, aberrations={"C10": C10, "C12": C12, "phi12": np.deg2rad(PHI12)},
                  rotation_deg=ROT, dose=0, n_particles=8, seed=11)
    s = Simulation(p, device="cpu")
    return p, s.grid, np.concatenate([f for _, f in s.batches(512)])


def _calibrate(p, g, frames, **kw):
    kw.setdefault("loss", False)
    kw.setdefault("refine", False)
    kw.setdefault("bootstrap", 20)
    return calibrate(array_batches(frames), p.scan, p.energy, p.semiangle, p.angular_sampling,
                     g.scan_step, log=lambda *a: None, **kw)


@pytest.mark.parametrize("layout", list(DETECTOR_LAYOUTS))
def test_recovers_every_layout(sim, layout):
    p, g, frames = sim
    rot_add, flip = DETECTOR_LAYOUTS[layout]
    true_rot = _wrap_deg(ROT + np.rad2deg(rot_add))
    cal = _calibrate(p, g, np.ascontiguousarray(layout_frames(frames, layout)),
                     rotation_prior=true_rot)
    g = cal.geometry
    assert g["flip_cols"] == flip
    assert abs(_wrap_deg(g["rotation_deg"] - true_rot)) < 0.2
    assert g["c10"] == pytest.approx(C10, rel=0.01)
    assert g["c12"] == pytest.approx(C12, abs=3)
    assert abs(_wrap_deg(2 * (g["phi12_deg"] - PHI12)) / 2) < 2
    assert cal.shift_rms_px < 0.05 and cal.outliers == 0
    assert abs(cal.higher_orders["c30"]) < 1e4        # none simulated: < 1 µm


def test_twin_without_prior(sim):
    """A detector turned by 180° (total rotation 165°) without a prior: the twin with |θ| ≤ 90°,
    i.e. the opposite C10, describes the same shifts."""
    p, g, frames = sim
    cal = _calibrate(p, g, np.ascontiguousarray(layout_frames(frames, "-rows,-cols")))
    g = cal.geometry
    assert abs(g["rotation_deg"]) <= 90
    assert g["c10"] == pytest.approx(-C10, rel=0.01)
    np.testing.assert_allclose(
        shift_matrix(g["c10"], g["rotation_deg"], g["flip_cols"], g["c12"], g["phi12_deg"]),
        shift_matrix(C10, 165.0, False, C12, PHI12), rtol=0, atol=0.02 * abs(C10))


def test_loss_minimum_at_the_fit_and_handedness(sim):
    p, g, frames = sim
    cal = _calibrate(p, g, np.ascontiguousarray(layout_frames(frames, "cols,-rows")),
                     rotation_prior=75.0, loss=True)
    lo = cal.loss
    assert lo["best_c10"] == pytest.approx(cal.geometry["c10"], abs=0.05 * abs(C10))
    assert abs(lo["best_rotation"] - cal.geometry["rotation_deg"]) < 1.0
    assert lo["geometry"] < 0.5 * lo["best_other_handedness"]
    assert lo["geometry"] < 0.5 * lo["no_shift"]


def _geometry(flags):
    algo = ParallaxAlgorithm()
    parser = argparse.ArgumentParser()
    algo.add_cli_args(parser)
    args = parser.parse_args(algo.preprocess_argv(flags))
    assert not algo.check_args(args)
    cfg = config_from_args(args)
    return preprocess_geometry(cfg.detector_shape, cfg.scan_shape, cfg.scan_sampling, cfg.energy,
                               cfg.semiangle_cutoff, angular_sampling=cfg.angular_sampling,
                               aberration_coefs=cfg.aberration_coefs,
                               rotation_angle=cfg.rotation_angle, probe_center=cfg.probe_center,
                               flip_cols=cfg.flip_cols)


def test_flags_rebuild_the_simulators_geometry(sim):
    """The measured flags (--rotation + --flip-cols, measured --center) and the simulator's own
    (--detector-layout) select the same bright-field pixels with the same shifts."""
    p, g, frames = sim
    layout = "cols,-rows"
    cal = _calibrate(p, g, np.ascontiguousarray(layout_frames(frames, layout)), rotation_prior=75)
    p_layout = SimParams(**{**p.__dict__, "detector_layout": layout})
    ours, theirs = _geometry(cal.flags()), _geometry(reconstruct_flags(p_layout, g))
    order_o, order_t = np.argsort(ours.bf_flat_inds), np.argsort(theirs.bf_flat_inds)
    np.testing.assert_array_equal(ours.bf_flat_inds[order_o], theirs.bf_flat_inds[order_t])
    diff = np.abs(ours.shifts[order_o] - theirs.shifts[order_t])
    assert diff.max() <= 1 and diff.mean() < 0.05


_NOISY = dict(scan=(48, 48), detector=(64, 64), semiangle=10, angular_sampling=0.5, scan_step=2.0,
              rotation_deg=ROT, dose=20000, n_particles=8)


def _frames(sims):
    """The frames of several simulations of one scan, summed: in the weak-phase limit, one
    specimen holding all their objects, each at its own defocus."""
    return sum(np.concatenate([f for _, f in s.batches(512)]).astype(np.float32) for s in sims)


def test_refinement_keeps_the_fit_without_a_lattice():
    """Amorphous-like particles only: no C10 makes lattice spots stand out of the shot noise's
    speckle, so --defocus keeps the shift fit."""
    s = Simulation(SimParams(**_NOISY, seed=11, aberrations={"C10": C10}), device="cpu")
    cal = _calibrate(s.params, s.grid, _frames([s]), refine=True)
    rf = cal.refine
    assert rf["band"] == pytest.approx([0.5 / s.grid.scan_step, 0.95 / s.grid.scan_step])
    assert not rf["clear"] and not rf["used"]
    assert cal.c10 == cal.geometry["c10"] == pytest.approx(C10, rel=0.02)
    assert f"--defocus={cal.geometry['c10']:.0f}" in cal.flags()


def test_refinement_finds_the_plane_of_the_lattice():
    """Particles (low-frequency contrast) at C10 −1000 Å and a 3 Å lattice (beyond the ~3.7 Å
    Nyquist period of the scan) 500 Å closer to focus: the shift fit measures the particles' plane,
    the refinement the lattice's. Reconstructed at C10′ the lattice transfers about as
    sin(πλC10′q²), near a zero at −500 Å for 3 Å, so the best C10′ sits a little beyond it."""
    sims = [Simulation(SimParams(**_NOISY, seed=11 + i, aberrations={"C10": c10}), device="cpu")
            for i, c10 in enumerate((-1000.0, -500.0))]
    lattice = sims[1]
    ny, nx = lattice.grid.object_shape
    y, x = np.mgrid[:ny, :nx] / np.array([ny, nx])[:, None, None]
    n = round(ny * lattice.grid.dx / 3.0)                  # whole periods: 3 Å on the object
    lattice.phase = 0.15 * (np.cos(2 * np.pi * n * x) + np.cos(2 * np.pi * n * y))
    lattice._trans = torch.as_tensor(np.exp(1j * lattice.phase).astype(np.complex64))
    p, g = sims[0].params, sims[0].grid
    cal = _calibrate(p, g, _frames(sims), refine=True)
    assert cal.geometry["c10"] == pytest.approx(-1000, rel=0.03)
    assert cal.refine["used"] and cal.refine["gain"] > 10
    assert abs(cal.c10 - (-500)) < 0.5 * cal.depth_of_field
    assert f"--defocus={cal.c10:.0f}" in cal.flags()


def test_disk_and_flagged_pixels():
    yy, xx = np.indices((96, 96))
    mean = np.where(np.hypot(yy - 47.3, xx - 49.6) <= 28.0, 3.0, 0.05)
    mean[61, 90] = 1600.0                    # hot, outside the disk
    mean[50, 50] = 65535.0                   # the detector's defective-pixel flag, inside
    disk = fit_disk(mean, top=65535.0)
    assert disk.center == pytest.approx((47.3, 49.6), abs=0.1)
    assert disk.radius == pytest.approx(28.0, abs=0.3)
    assert disk.flagged == [(50, 50)] and disk.inside((50, 50))
    assert (61, 90) in disk.hot and not disk.inside((61, 90))


def test_cli_on_a_recording(tmp_path):
    """End to end from a .dstream: the written flags parse with run.py's own CLI."""
    pytest.importorskip("cbor2")
    from live4d.parallax.tools import calibrate_parallax
    from live4d.parallax.tools.simulate_4dstem import write

    p = SimParams(scan=(48, 48), detector=(64, 64), semiangle=10, angular_sampling=0.5,
                  scan_step=2.0, aberrations={"C10": -800.0}, rotation_deg=20,
                  detector_layout="rows,-cols", dose=20000, n_particles=8, seed=4)
    meta = write(p, tmp_path / "s", formats=("dstream",), device="cpu")
    step = meta["grid"]["scan_step"]
    cal = calibrate_parallax.main([str(tmp_path / "s.dstream"), "--scan", "48", "48",
                                   "--energy", "300000", "--semiangle", "10",
                                   "--angular-sampling", "0.5", "--scan-sampling", str(step),
                                   "--rotation-prior", "20", "--no-loss",
                                   "--out", str(tmp_path / "cal")])
    assert cal.geometry["flip_cols"]
    assert abs(cal.geometry["rotation_deg"] - 20) < 1
    assert cal.geometry["c10"] == pytest.approx(-800, rel=0.05)
    flags = (tmp_path / "cal.flags").read_text().split()
    assert "--flip-cols" in flags
    _geometry(flags)
    assert (tmp_path / "cal.json").exists()
