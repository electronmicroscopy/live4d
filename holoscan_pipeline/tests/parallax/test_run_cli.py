"""Parallax acquisition validation and saved channel names."""
import argparse
from types import SimpleNamespace
import numpy as np
from live4d.parallax import registry

ACQUISITION = ["--energy", "300000", "--semiangle", "20", "--angular-sampling", "1.1",
               "--scan-sampling", "1.0", "--scan", "64", "64", "--defocus", "-1.5e4",
               "--rotation", "-15", "--center", "48", "48"]

def _args(argv):
    algo = registry.get("parallax")
    parser = argparse.ArgumentParser()
    algo.add_cli_args(parser)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(algo.preprocess_argv(argv))

def test_continuous_save_keeps_a_numbered_copy(tmp_path):
    """In continuous mode every scan also gets its own file, so a session keeps all its scans;
    the plain name always holds the latest one."""
    from live4d.parallax.algorithm import ParallaxAlgorithm
    algo = ParallaxAlgorithm()
    for k, tag in enumerate(("scan001", "scan002")):
        acc = SimpleNamespace(result=lambda k=k: np.full((4, 4), k, dtype=np.float32),
                              geometry=SimpleNamespace(shifts=np.zeros((3, 2), np.int32)))
        algo.save(acc, tmp_path, tag=tag)
    names = sorted(f.name for f in tmp_path.iterdir())
    assert names == ["parallax_recon.npy", "parallax_recon_scan001.npy",
                     "parallax_recon_scan002.npy", "parallax_shifts.npy"]
    assert np.load(tmp_path / "parallax_recon_scan001.npy").max() == 0
    assert np.load(tmp_path / "parallax_recon.npy").max() == 1          # the latest scan


def test_continuous_save_tags_every_channel(tmp_path):
    """With readout channels, the numbered copy is kept for each channel."""
    from live4d.parallax.algorithm import ParallaxAlgorithm
    acc = SimpleNamespace(images=lambda: {"raw": np.zeros((4, 4), np.float32),
                                          "phase_flipped": np.ones((4, 4), np.float32)},
                          geometry=SimpleNamespace(shifts=np.zeros((3, 2), np.int32)))
    ParallaxAlgorithm().save(acc, tmp_path, tag="scan001")
    names = sorted(f.name for f in tmp_path.iterdir())
    assert names == ["parallax_phase_flipped.npy", "parallax_phase_flipped_scan001.npy",
                     "parallax_recon.npy", "parallax_recon_scan001.npy", "parallax_shifts.npy"]


def test_readout_flags_map_to_config():
    from live4d.algorithms.parallax.config import readout_from_args
    cfg = readout_from_args(_args([]))
    assert cfg.phase_flip and cfg.dc_correct and not cfg.normalize and cfg.suppress_nyquist is None
    cfg = readout_from_args(_args(["--no-phase-flip", "--no-dc-correct", "--normalize",
                                   "--coverage", "--lowpass", "0.2", "--highpass", "0.01",
                                   "--suppress-nyquist", "off", "--band-limit"]))
    assert not cfg.phase_flip and not cfg.dc_correct and cfg.normalize and cfg.coverage_channel
    assert cfg.q_lowpass == 0.2 and cfg.q_highpass == 0.01
    assert cfg.suppress_nyquist is False and cfg.band_limit


def test_frame_mean_is_kept_unless_asked():
    """The per-frame mean subtraction removes amplitude contrast: off unless the flag is given."""
    algo = registry.get("parallax")
    assert not algo.build(_args(["--binned", *ACQUISITION, "--device", "cpu"])).subtract_frame_mean
    acc = algo.build(_args(["--binned", *ACQUISITION, "--device", "cpu", "--subtract-frame-mean"]))
    assert acc.subtract_frame_mean
