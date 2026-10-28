"""Detector orientation: every transpose/flip layout, described by `--detector-layout` (a rotation
offset plus `flip_cols`) and its raw-frame center, must reconstruct exactly what the canonical
layout does — the LiberTEM-parallax table, without its even-grid parity correction.

The equivalence is purely geometric, so random frames pin it exactly; the apoferritin capture (if
present next to this repository) checks it on real data with an off-diagonal rotation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


from live4d.algorithms.parallax.engine import ParallaxAccumulator, preprocess_geometry
from live4d.algorithms.parallax.orientation import (
    DETECTOR_LAYOUTS,
    layout_center,
    layout_frames,
    layout_shape,
)

APOF = (Path(__file__).resolve().parents[2] / "quantem-tutorials" / "data"
        / "apoF_4mrad_1.5um-df_3A-step_30eA2_binary_uint8.npy")


def _recon(frames, scan, layout, rotation_deg, canonical_center=None, **geo):
    rot_adjust, flip = DETECTOR_LAYOUTS[layout]
    raw = np.ascontiguousarray(layout_frames(frames, layout))
    center = layout_center(raw.shape[-2:], layout, canonical_center)
    g = preprocess_geometry(det_shape=raw.shape[-2:], scan_shape=scan,
                            rotation_angle=np.deg2rad(rotation_deg) + rot_adjust,
                            flip_cols=flip, probe_center=center, **geo)
    acc = ParallaxAccumulator(g, device="cpu")
    acc.process(raw, np.arange(len(raw)))
    return acc.result(), g


_SMALL = dict(scan_sampling=(256 / 12, 256 / 12), energy=300e3, semiangle_cutoff=4.0,
              angular_sampling=(0.307617, 0.307617), aberration_coefs={"C10": -1.5e4})


@pytest.mark.parametrize("layout", list(DETECTOR_LAYOUTS))
@pytest.mark.parametrize("det_shape,center", [((32, 32), None), ((32, 28), (15.3, 13.8)),
                                              ((31, 29), None)])
def test_layouts_reproduce_canonical(layout, det_shape, center):
    rng = np.random.default_rng(0)
    scan = (12, 10)
    frames = rng.integers(0, 60000, size=(scan[0] * scan[1], *det_shape), dtype=np.uint16)
    ref, ref_geo = _recon(frames, scan, "rows,cols", -15.0, center, **_SMALL)
    got, geo = _recon(frames, scan, layout, -15.0, center, **_SMALL)
    assert geo.detector_shape == layout_shape(det_shape, layout)
    assert geo.n_bf == ref_geo.n_bf
    scale = np.abs(ref).max()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5 * scale)


@pytest.mark.skipif(not APOF.exists(), reason="apoferritin capture not present")
@pytest.mark.parametrize("layout", list(DETECTOR_LAYOUTS))
def test_layouts_on_apoferritin(layout):
    data = np.load(APOF)
    frames = data.reshape(-1, *data.shape[-2:])
    geo = dict(scan_sampling=(256 / 72, 256 / 72), energy=300e3, semiangle_cutoff=4.0,
               angular_sampling=(0.307617, 0.307617), aberration_coefs={"C10": -1.5e4})
    ref, _ = _recon(frames, data.shape[:2], "rows,cols", -15.0, **geo)
    got, _ = _recon(frames, data.shape[:2], layout, -15.0, **geo)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5 * np.abs(ref).max())


def test_flip_cols_default_center_matches_libertem():
    """flip_cols with no center == LiberTEM's `frames[..., ::-1]` on the canonical geometry."""
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 60000, size=(120, 32, 32), dtype=np.uint16)
    common = dict(det_shape=(32, 32), scan_shape=(12, 10), rotation_angle=np.deg2rad(20), **_SMALL)
    a = ParallaxAccumulator(preprocess_geometry(**common), device="cpu")
    a.process(frames, np.arange(120))
    b = ParallaxAccumulator(preprocess_geometry(flip_cols=True, **common), device="cpu")
    b.process(np.ascontiguousarray(frames[..., ::-1]), np.arange(120))
    np.testing.assert_allclose(b.result(), a.result(), rtol=0, atol=1e-5 * np.abs(a.result()).max())


def test_detector_layout_cli():
    """--detector-layout sets the rotation offset, the flip and the mapped default center."""
    import argparse

    from live4d.parallax.algorithm import ParallaxAlgorithm
    from live4d.algorithms.parallax.config import config_from_args

    algo = ParallaxAlgorithm()
    p = argparse.ArgumentParser()
    algo.add_cli_args(p)
    args = p.parse_args(algo.preprocess_argv(
        ["--detector", "32", "28", "--rotation", "-15", "--detector-layout", "cols,-rows"]))
    cfg = config_from_args(args)
    assert np.isclose(cfg.rotation_angle, np.deg2rad(-15) + np.pi / 2)
    assert cfg.flip_cols is False
    assert cfg.probe_center == layout_center((32, 28), "cols,-rows")
    with pytest.raises(SystemExit):
        config_from_args(p.parse_args(["--detector-layout", "rows,-cols", "--flip-cols"]))
