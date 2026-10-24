"""Tests for the 4D-STEM simulator (`tools/simulate_4dstem.py`).

The simulator and the reconstruction share conventions (probe exp(-iχ), axis-0 frequencies, the
active detector rotation, the detector layouts) and these tests pin that they agree, independently
of either implementation's internals:
  • parallax criterion: with the bright-field images shifted by the geometry's shifts, the spread
    across them at each output pixel (quantem's variance / residual loss) is smallest at the
    simulated C10 and rotation — not at the opposite sign, nor a few degrees off;
  • the reconstruction flags the simulator writes, parsed by the runner's own CLI, reproduce the
    canonical reconstruction for a transposed/reversed detector layout;
  • the .dstream it writes decodes to exactly the .npy it writes; the dose sets the counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("cbor2")

from realtime_stem.ingest.python_ingest import PythonIngest  # noqa: E402
from realtime_stem.processing.parallax.engine import ParallaxAccumulator, preprocess_geometry  # noqa: E402
from tools.simulate_4dstem import SimParams, Simulation, reconstruct_flags, write  # noqa: E402


def _frames(sim):
    return np.concatenate([f for _, f in sim.batches(512)])


def _variance_loss(frames, p, g, c10, rotation_deg, inner=0.8):
    """Weighted per-pixel variance across the shifted BF images / their total power."""
    det = frames.shape[-2:]
    geo = preprocess_geometry(det, p.scan, (g.scan_step,) * 2, p.energy, p.semiangle * inner,
                              angular_sampling=(p.angular_sampling,) * 2,
                              aberration_coefs={"C10": c10}, rotation_angle=np.deg2rad(rotation_deg))
    v = frames.reshape(len(frames), -1)[:, geo.bf_flat_inds].astype(np.float64)
    v = v - v.mean(axis=1, keepdims=True)
    v = v - v.mean(axis=0, keepdims=True)
    ny, nx = p.scan
    ids = np.arange(ny * nx)
    oy = ((ids // nx)[:, None] + geo.shifts[None, :, 0]) % ny
    ox = ((ids % nx)[:, None] + geo.shifts[None, :, 1]) % nx
    flat = (oy * nx + ox).ravel()
    w = np.bincount(flat, minlength=ny * nx)
    s1 = np.bincount(flat, v.ravel(), ny * nx)
    s2 = np.bincount(flat, (v**2).ravel(), ny * nx)
    var = s2 / np.maximum(w, 1) - (s1 / np.maximum(w, 1)) ** 2
    return (var * w).sum() / s2.sum()


@pytest.fixture(scope="module")
def noiseless():
    p = SimParams(scan=(48, 48), dose=0, n_particles=6, seed=3)
    sim = Simulation(p, device="cpu")
    return p, sim.grid, _frames(sim)


def test_frames_are_normalized(noiseless):
    _, _, frames = noiseless
    np.testing.assert_allclose(frames.sum(axis=(1, 2)), 1.0, rtol=1e-3)


def test_parallax_criterion_minimal_at_simulated_defocus(noiseless):
    p, g, frames = noiseless
    true_c10 = p.aberrations["C10"]
    losses = {c: _variance_loss(frames, p, g, c, p.rotation_deg)
              for c in (true_c10 * 1.25, true_c10, true_c10 * 0.75, -true_c10)}
    assert min(losses, key=losses.get) == true_c10, losses
    assert losses[true_c10] < 0.5 * min(v for c, v in losses.items() if c != true_c10)


def test_parallax_criterion_minimal_at_simulated_rotation(noiseless):
    p, g, frames = noiseless
    rot = p.rotation_deg
    losses = {r: _variance_loss(frames, p, g, p.aberrations["C10"], r)
              for r in (rot - 10, rot, rot + 10, -rot)}
    assert min(losses, key=losses.get) == rot, losses


def test_layout_flags_reproduce_canonical():
    """A transposed+reversed detector, reconstructed from the simulator's own flags via the
    runner's CLI, equals the canonical-layout reconstruction of the same (seeded) data."""
    from realtime_stem.processing.parallax.algorithm import ParallaxAlgorithm
    from realtime_stem.processing.parallax.config import config_from_args

    algo = ParallaxAlgorithm()
    recon = {}
    for layout in ("rows,cols", "cols,-rows"):
        p = SimParams(scan=(16, 16), detector=(96, 80), dose=500, detector_layout=layout, seed=5)
        sim = Simulation(p, device="cpu")
        frames = _frames(sim)
        parser = argparse.ArgumentParser()
        algo.add_cli_args(parser)
        cfg = config_from_args(parser.parse_args(algo.preprocess_argv(reconstruct_flags(p, sim.grid))))
        geo = preprocess_geometry(cfg.detector_shape, cfg.scan_shape, cfg.scan_sampling, cfg.energy,
                                  cfg.semiangle_cutoff, angular_sampling=cfg.angular_sampling,
                                  aberration_coefs=cfg.aberration_coefs,
                                  rotation_angle=cfg.rotation_angle, probe_center=cfg.probe_center,
                                  flip_cols=cfg.flip_cols)
        assert tuple(frames.shape[1:]) == cfg.detector_shape
        acc = ParallaxAccumulator(geo, device="cpu")
        acc.process(frames, np.arange(len(frames)))
        recon[layout] = acc.result()
    ref = recon["rows,cols"]
    np.testing.assert_allclose(recon["cols,-rows"], ref, rtol=0, atol=1e-5 * np.abs(ref).max())


def test_written_files_agree(tmp_path):
    p = SimParams(scan=(8, 8), dose=800, seed=7)
    meta = write(p, tmp_path / "sim", formats=("npy", "dstream"), device="cpu")
    frames = np.load(tmp_path / "sim.npy")
    assert frames.shape == (8, 8, 96, 96) and frames.dtype == np.uint16
    electrons = p.dose * meta["grid"]["scan_step"] ** 2
    assert abs(frames.sum(axis=(2, 3)).mean() / electrons - 1) < 0.05
    assert np.load(tmp_path / "sim_phase.npy").shape == (8, 8)
    assert (tmp_path / "sim.flags").read_text().split() == meta["reconstruct_flags"]

    geo = preprocess_geometry((96, 96), p.scan, (meta["grid"]["scan_step"],) * 2, p.energy,
                              p.semiangle, angular_sampling=(p.angular_sampling,) * 2,
                              aberration_coefs=p.aberrations,
                              rotation_angle=np.deg2rad(p.rotation_deg))
    from_npy = ParallaxAccumulator(geo, device="cpu")
    PythonIngest(from_npy, source=tmp_path / "sim.npy").run()
    from_dstream = ParallaxAccumulator(geo, device="cpu")
    PythonIngest(from_dstream, source=tmp_path / "sim.dstream").run()
    np.testing.assert_array_equal(from_dstream.result(), from_npy.result())
