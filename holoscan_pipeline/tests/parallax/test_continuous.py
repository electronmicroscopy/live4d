"""Continuous scanning: the reconstruction across repeated scans (`--accumulate latest | all`).

The use case is live tuning: the microscope scans the same area over and over while the operator
adjusts focus / stigmation until the image matches the algorithm's target aberrations. So:

  • "latest" must equal, at every moment, a fresh reconstruction of the newest frame at each scan
    position — exactly the last full pass once one has completed, a mix of the current and the
    previous pass mid-scan — however the passes are cut into batches, for one long series (ids
    running past the scan) or one series per pass (ids restarting at 0);
  • it must not drift over long runs (it adds and removes the same deposits for hours);
  • "all" must equal the sum over every pass;
  • the readout channels (DC correction, coverage) stay exact in both modes;
  • end to end through the runner (file and ZMQ), and on simulated data where the microscope's
    defocus walks towards the target over the passes, the live image converges to the
    reconstruction at the target, and is exactly it once a full pass was taken there.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = (Path(__file__).resolve().parents[2] / "python/live4d/parallax")

from live4d.algorithms.parallax.engine import ParallaxAccumulator, preprocess_geometry  # noqa: E402
from live4d.algorithms.parallax.readout import ReadoutConfig  # noqa: E402

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (
    ["cuda"] if torch.cuda.is_available() else [])

_SCAN = (12, 10)
_NPOS = _SCAN[0] * _SCAN[1]


def _geometry(upsampling_factor=1):
    return preprocess_geometry(
        det_shape=(32, 32), scan_shape=_SCAN, scan_sampling=(256 / 12, 256 / 12), energy=300e3,
        semiangle_cutoff=4.0, angular_sampling=(0.307617, 0.307617),
        aberration_coefs={"C10": -1.5e4}, rotation_angle=np.deg2rad(-15),
        upsampling_factor=upsampling_factor)


def _passes(n_passes, seed=0):
    """Frames for `n_passes` scans, each pass different data. (n_passes·positions, 32, 32)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60000, size=(n_passes * _NPOS, 32, 32), dtype=np.uint16)


def _feed(acc, frames, ids, batch):
    for s in range(0, len(frames), batch):
        acc.process(frames[s:s + batch], ids[s:s + batch])


def _newest_per_position(frames, ids):
    """The newest frame of every visited position (raster position = id % positions)."""
    newest = {}
    for i, f in zip(ids, frames):
        newest[int(i) % _NPOS] = f
    pos = np.array(sorted(newest))
    return np.stack([newest[p] for p in pos]), pos


def _fresh(geo, frames, ids, device, **readout):
    acc = ParallaxAccumulator(geo, device=device, readout=ReadoutConfig(**readout))
    acc.process(frames, ids)
    return acc


def _close(a, b, rel=2e-5):
    scale = max(np.abs(b).max(), 1e-12)
    np.testing.assert_allclose(a, b, rtol=0, atol=rel * scale)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_frames", [3 * _NPOS, 2 * _NPOS + 47])   # whole passes, mid-pass
@pytest.mark.parametrize("upsampling_factor", [1, 2])
def test_latest_is_newest_frame_per_position(device, n_frames, upsampling_factor):
    geo = _geometry(upsampling_factor)
    frames = _passes(3)[:n_frames]
    ids = np.arange(n_frames)                           # one long series: ids run past the scan
    acc = ParallaxAccumulator(geo, device=device, accumulate="latest", wrap_ids=True)
    _feed(acc, frames, ids, batch=37)                   # batches straddle the pass boundaries

    newest, pos = _newest_per_position(frames, ids)
    ref = _fresh(geo, newest, pos, device)
    # replaced deposits are subtracted in float32, in the device's (on mps varying) order: the
    # rounding left over reaches ~2e-5 of the maximum after a few passes, as in the drift test
    _close(acc.result(), ref.result(), rel=1e-4)
    for name, img in acc.channels().items():            # DC correction / phase flip stay exact
        _close(img.cpu().numpy(), ref.channels()[name].cpu().numpy(), rel=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_latest_does_not_drift(device):
    """Forty passes of adding and removing deposits leave exactly the last pass."""
    geo = _geometry()
    frames = _passes(40, seed=1)
    acc = ParallaxAccumulator(geo, device=device, accumulate="latest", wrap_ids=True)
    _feed(acc, frames, np.arange(len(frames)), batch=64)
    ref = _fresh(geo, frames[-_NPOS:], np.arange(_NPOS), device)
    _close(acc.result(), ref.result(), rel=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_all_sums_every_pass(device):
    geo = _geometry()
    frames = _passes(3, seed=2)
    ids = np.arange(len(frames))
    acc = ParallaxAccumulator(geo, device=device, accumulate="all", wrap_ids=True,
                              readout=ReadoutConfig(coverage_channel=True))
    _feed(acc, frames, ids, batch=50)
    ref = _fresh(geo, frames, ids % _NPOS, device, coverage_channel=True)   # same frames, positions
    _close(acc.result(), ref.result())
    # the channels subtract the scan mean from the raw float32 sum (which keeps the frame mean):
    # the rounding error scales with the raw image, not with the channel
    raw_scale = np.abs(ref.result()).max()
    for name, img in acc.channels().items():
        want = ref.channels()[name].cpu().numpy()
        np.testing.assert_allclose(img.cpu().numpy(), want, rtol=0,
                                   atol=2e-5 * max(np.abs(want).max(), raw_scale))
    assert np.all(acc.channels()["coverage"].cpu().numpy() == 3 * geo.n_bf)   # 3 deposits × M


def test_series_restarting_at_zero():
    """One series per pass (ids restart at 0): `start_series` continues the pass count."""
    geo = _geometry()
    frames = _passes(3, seed=3)
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    for k in range(3):
        acc.start_series()
        n = _NPOS if k < 2 else 70                        # the last series stops mid-scan
        _feed(acc, frames[k * _NPOS:k * _NPOS + n], np.arange(n), batch=33)
    newest, pos = _newest_per_position(
        np.concatenate([frames[:2 * _NPOS], frames[2 * _NPOS:2 * _NPOS + 70]]),
        np.arange(2 * _NPOS + 70))
    _close(acc.result(), _fresh(geo, newest, pos, "cpu").result())
    meta = acc.snapshot()[2]
    assert meta["pass"] == 2 and np.isclose(meta["progress"], 70 / _NPOS)
    assert meta["accumulate"] == "latest" and meta["coverage"] == 1.0


def test_scan_mode_still_rejects_ids_past_the_scan():
    from live4d.parallax.ingest.sink import BatchSink

    acc = ParallaxAccumulator(_geometry(), device="cpu")
    with pytest.raises(ValueError, match="--accumulate"):
        BatchSink(acc).feed(_passes(1)[:2], np.array([0, _NPOS]))
    wrapping = ParallaxAccumulator(_geometry(), device="cpu", accumulate="latest", wrap_ids=True)
    BatchSink(wrapping).feed(_passes(1)[:2], np.array([0, _NPOS]))   # accepted, wraps


# ── end to end ──────────────────────────────────────────────────────────────────

def _small_sim(passes, schedule, scan=(24, 24), dose=0, seed=4):
    from live4d.parallax.tools.simulate_4dstem import SimParams, Simulation
    p = SimParams(scan=scan, passes=passes, defocus_schedule=schedule, dose=dose,
                  n_particles=4, seed=seed)
    sim = Simulation(p, device="cpu")
    frames = np.concatenate([f for _, f in sim.batches(256)])
    return p, sim.grid, frames


def _sim_geometry(p, g, c10):
    return preprocess_geometry((96, 96), p.scan, (g.scan_step,) * 2, p.energy, p.semiangle,
                               angular_sampling=(p.angular_sampling,) * 2,
                               aberration_coefs={"C10": c10},
                               rotation_angle=np.deg2rad(p.rotation_deg))


def test_tuning_converges_to_the_target_reconstruction():
    """The microscope's defocus walks −2600 → −2000 Å over the passes; the algorithm's target is
    −2000 Å. The live image (snapshotted at the end of each pass) gets closer to the target-focus
    reconstruction pass by pass, and equals it once a full pass was taken in focus."""
    target = -2000.0
    schedule = (-2600.0, -2400.0, -2200.0, -2000.0)
    p, g, frames = _small_sim(len(schedule), schedule)
    n_pos = p.scan[0] * p.scan[1]
    geo = _sim_geometry(p, g, target)

    reference = _fresh(geo, frames[-n_pos:], np.arange(n_pos), "cpu").channels()["phase_flipped"]
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    corr = []
    for k in range(len(schedule)):
        _feed(acc, frames[k * n_pos:(k + 1) * n_pos], np.arange(k * n_pos, (k + 1) * n_pos), 100)
        live = acc.channels()["phase_flipped"].numpy().ravel()
        corr.append(np.corrcoef(live, reference.numpy().ravel())[0, 1])
    assert all(a < b for a, b in zip(corr, corr[1:])), corr
    # 30% off in defocus mostly costs the high frequencies: clearly worse, not unrecognizable
    assert corr[0] < 0.98 and corr[-1] > 1 - 1e-6, corr


def test_runner_file_replay_of_repeated_scans(tmp_path):
    """`run.py --accumulate latest` on a multi-pass capture saves the last pass's reconstruction."""
    from live4d.parallax.tools.simulate_4dstem import SimParams, write

    p = SimParams(scan=(16, 16), passes=3, defocus_schedule=(-2500.0, -2000.0, -2000.0),
                  dose=600, n_particles=3, seed=6)
    meta = write(p, tmp_path / "rep", formats=("npy", "dstream"), device="cpu")
    flags = meta["reconstruct_flags"]
    assert flags[-2:] == ["--accumulate", "latest"]
    out = tmp_path / "out"
    from .helpers import offline_args
    subprocess.run([sys.executable, "-m", "live4d.parallax.offline",
                    *offline_args(tmp_path, flags, tmp_path / "rep.dstream", batch_size=100,
                                  out_dir=out)], cwd=REPO, check=True, capture_output=True)


    frames = np.load(tmp_path / "rep.npy").reshape(-1, 96, 96)
    n_pos = 16 * 16
    geo = _sim_geometry(p, type("G", (), meta["grid"]), -2000.0)
    ref = _fresh(geo, frames[-n_pos:], np.arange(n_pos), "cpu")
    _close(np.load(out / "parallax_recon.npy"), ref.result())


def test_replay_series_per_pass_keeps_one_image(tmp_path):
    from live4d.config import load_config
    from live4d.parallax.runtime import ParallaxRuntime
    from live4d.parallax.offline import replay
    from live4d.parallax.ingest import codec
    from live4d.parallax.ingest.dstream import write_magic, write_record
    from .helpers import offline_args
    geo = _geometry()
    frames = _passes(3, seed=7)
    lengths = (_NPOS, _NPOS, 50)
    source = tmp_path / "multi.dstream"
    with source.open("wb") as stream:
        write_magic(stream)
        for k, count in enumerate(lengths):
            write_record(stream, codec.encode_start_message((32, 32), count, series_id=k))
            for i in range(count):
                write_record(stream, codec.encode_image_message(i, frames[k * _NPOS + i], series_id=k))
            write_record(stream, codec.encode_end_message(series_id=k))
    flags = ["--scan", *map(str, _SCAN), "--energy", "300e3", "--semiangle", "4",
             "--angular-sampling", "0.307617", "--scan-sampling", str(256 / 12),
             "--defocus=-1.5e4", "--rotation=-15", "--center", "16", "16",
             "--accumulate", "latest"]
    argv = offline_args(tmp_path, flags, source)
    runtime = ParallaxRuntime(load_config(Path(argv[1])), device="cpu")
    try:
        replay(runtime, source)
        assert runtime.rounds == 3
        assert not runtime.sink.completeness()[0]
    finally:
        runtime.close()
    seen = np.concatenate([frames[k * _NPOS:k * _NPOS + n] for k, n in enumerate(lengths)])
    newest, pos = _newest_per_position(seen, np.concatenate([np.arange(n) for n in lengths]))
    _close(np.load(tmp_path / "out/parallax_recon.npy"), _fresh(geo, newest, pos, "cpu").result())
