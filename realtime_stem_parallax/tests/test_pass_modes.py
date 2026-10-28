"""Pass-aware continuous modes: `--accumulate reset | swap`, `--pass-average`, `--save-passes`.

A pass is one scan of the area; with ids running on, pass k holds ids k·positions …
(k+1)·positions − 1. The modes must, however the passes are cut into batches:

  • "reset": show the current pass alone, and keep (save) the last completed pass exactly;
  • "swap": show the current pass on the output rows it has completed — exactly, because every
    deposit landing on such a row comes from a finished scan row — and the last completed pass
    elsewhere; with `pass_average` β, keep the running blend of completed passes;
  • call `on_pass_complete(k)` once per pass, also when each pass is its own series;
  • put a closed pass's stragglers into the completed pass ("swap") or drop them ("reset");
  • and through the runner, `--save-passes` writes every pass's channels.
(`tools/repeat_dstream.py`, which loops a recording into passes: tests/test_repeat_dstream.py.)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from realtime_stem.processing.parallax.engine import ParallaxAccumulator, preprocess_geometry  # noqa: E402

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (
    ["cuda"] if torch.cuda.is_available() else [])
_SCAN = (12, 10)
_NPOS = _SCAN[0] * _SCAN[1]
_FLAGS = ["--detector", "32", "32", "--scan", *map(str, _SCAN), "--energy", "300e3",
          "--semiangle", "4", "--angular-sampling", "0.307617", "--scan-sampling", f"{256 / 12}",
          "--defocus=-1.5e4", "--rotation=-15", "--center", "16", "16"]


def _geometry(upsampling_factor=1, c10=-1.5e4):
    return preprocess_geometry(
        det_shape=(32, 32), scan_shape=_SCAN, scan_sampling=(256 / 12, 256 / 12), energy=300e3,
        semiangle_cutoff=4.0, angular_sampling=(0.307617, 0.307617),
        aberration_coefs={"C10": c10}, rotation_angle=np.deg2rad(-15),
        upsampling_factor=upsampling_factor)


def _passes(n_passes, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60000, size=(n_passes * _NPOS, 32, 32), dtype=np.uint16)


def _feed(acc, frames, ids, batch):
    for s in range(0, len(frames), batch):
        acc.process(frames[s:s + batch], ids[s:s + batch])


def _pass_ref(geo, frames, k, n=_NPOS, device="cpu"):
    """A fresh reconstruction of the first `n` frames of pass `k`."""
    acc = ParallaxAccumulator(geo, device=device)
    acc.process(frames[k * _NPOS:k * _NPOS + n], np.arange(n))
    return acc


def _close(a, b, rel=2e-5):
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=0, atol=rel * max(np.abs(b).max(), 1e-12))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [7, 50, _NPOS + 13])
def test_reset_shows_the_current_pass_and_keeps_the_last_complete(device, batch):
    geo = _geometry()
    frames = _passes(3, seed=11)
    n = 2 * _NPOS + 47
    acc = ParallaxAccumulator(geo, device=device, accumulate="reset", wrap_ids=True)
    done = []
    acc.on_pass_complete = lambda k: done.append((k, acc.result().copy()))
    _feed(acc, frames[:n], np.arange(n), batch)
    assert [k for k, _ in done] == [0, 1]
    for k, img in done:
        _close(img, _pass_ref(geo, frames, k, device=device).result())
    _close(acc.result(), _pass_ref(geo, frames, 1, device=device).result())
    live = _pass_ref(geo, frames, 2, 47, device).channels()
    for name, img in acc.channels().items():
        _close(img.cpu().numpy(), live[name].cpu().numpy())
    assert acc.snapshot()[2]["passes_completed"] == 2


@pytest.mark.parametrize("upsampling_factor", [1, 2])
def test_swap_composite_is_exact_on_the_completed_rows(upsampling_factor):
    geo = _geometry(upsampling_factor)
    frames = _passes(2, seed=12)
    rows = 9                                              # shifts span ±3 rows on 12, wrapping
    cut = rows * _SCAN[1] + 4                             # pass 1 is 4 frames into its 10th row
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="swap", wrap_ids=True)
    _feed(acc, frames[:_NPOS + cut], np.arange(_NPOS + cut), 23)
    done = acc._row_source < rows
    assert done.any() and (~done).any()
    # the rows called complete receive nothing from the scan rows still to come
    _close(_pass_ref(geo, frames, 1).result()[done],
           _pass_ref(geo, frames, 1, rows * _SCAN[1]).result()[done])
    front, back = _pass_ref(geo, frames, 0).channels(), _pass_ref(geo, frames, 1, cut).channels()
    for name, img in acc.channels().items():
        img = img.numpy()
        _close(img[done], back[name].numpy()[done])
        _close(img[~done], front[name].numpy()[~done])
    _close(acc.result(), _pass_ref(geo, frames, 0).result())    # saved: the completed pass


def test_swap_pass_average_blends_the_completed_passes():
    geo = _geometry()
    frames = _passes(3, seed=13)
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="swap", wrap_ids=True,
                              pass_average=0.5)
    _feed(acc, frames, np.arange(3 * _NPOS), 40)
    acc.end_series()                                      # the last pass ended with the series
    p = [_pass_ref(geo, frames, k).result() for k in range(3)]
    _close(acc.result(), 0.25 * p[0] + 0.25 * p[1] + 0.5 * p[2])
    assert acc.passes_completed == 3
    with pytest.raises(ValueError):
        ParallaxAccumulator(geo, device="cpu", accumulate="swap", pass_average=0.0)


@pytest.mark.parametrize("mode", ["reset", "swap"])
def test_stragglers_of_a_closed_pass(mode):
    geo = _geometry()
    frames = _passes(2, seed=14)
    held = 37                                             # pass 0's frame 37 arrives late
    order = [i for i in range(_NPOS) if i != held] + list(range(_NPOS, _NPOS + 20)) + [held]
    acc = ParallaxAccumulator(geo, device="cpu", accumulate=mode, wrap_ids=True)
    for i in order:
        acc.process(frames[i:i + 1], np.array([i]))
    assert acc.late_frames == 1
    kept = [i for i in range(_NPOS) if i != held]
    without = ParallaxAccumulator(geo, device="cpu")
    without.process(frames[kept], np.array(kept))
    want = _pass_ref(geo, frames, 0).result() if mode == "swap" else without.result()
    _close(acc.result(), want)


def test_a_series_per_pass_closes_each_pass_once():
    geo = _geometry()
    frames = _passes(3, seed=15)
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="swap", wrap_ids=True)
    done = []
    acc.on_pass_complete = done.append
    for k in range(3):
        acc.start_series()
        _feed(acc, frames[k * _NPOS:(k + 1) * _NPOS], np.arange(_NPOS), 33)
        acc.end_series()
    assert done == [0, 1, 2]
    _close(acc.result(), _pass_ref(geo, frames, 2).result())
    # a series cut short leaves its pass open
    acc.start_series()
    _feed(acc, frames[:50], np.arange(50), 33)
    acc.end_series()
    assert done == [0, 1, 2]


@pytest.mark.parametrize("upsampling_factor", [1, 2])
def test_complete_rows_of_an_image_filling_in(upsampling_factor):
    geo = _geometry(upsampling_factor)
    frames = _passes(1, seed=17)
    full = _pass_ref(geo, frames, 0).result()
    cut = 9 * _SCAN[1] + 4                                # 4 frames into the 10th scan row
    acc = ParallaxAccumulator(geo, device="cpu")          # a single scan
    assert acc.complete_rows() == [0, 0]
    acc.process(frames[:cut], np.arange(cut))
    r0, r1 = acc.complete_rows()
    assert acc.snapshot()[2]["complete_rows"] == [r0, r1]
    assert 0 < r1 - r0 < full.shape[0]
    part = acc.result()
    _close(part[r0:r1], full[r0:r1])                      # those rows are final …
    for r in (r0 - 1, r1):                                # … the ones next to them are not
        if 0 <= r < full.shape[0]:
            assert not np.allclose(part[r], full[r])
    acc.process(frames[cut:], np.arange(cut, _NPOS))
    assert acc.complete_rows() is None and "complete_rows" not in acc.snapshot()[2]


@pytest.mark.parametrize("mode", ["reset", "swap", "latest"])
def test_complete_rows_in_the_continuous_modes(mode):
    geo = _geometry()
    frames = _passes(2, seed=18)
    cut = 9 * _SCAN[1] + 4
    acc = ParallaxAccumulator(geo, device="cpu", accumulate=mode, wrap_ids=True)
    acc.process(frames[:cut], np.arange(cut))
    first = acc.complete_rows()
    assert first is not None and first[1] > first[0]      # every mode's first pass fills in
    acc.process(frames[cut:_NPOS + cut], np.arange(cut, _NPOS + cut))
    # on the next pass "reset" fills in again; "swap" and "latest" show a whole image
    assert acc.complete_rows() == (first if mode == "reset" else None)


@pytest.mark.parametrize("mode", ["swap", "reset", "all", "latest"])
def test_complete_rows_of_an_image_started_mid_run(mode):
    """A mode switched to mid-run gets a new accumulator whose first ids are passes past the
    scan, here also mid-pass: it fills in from there, and only the rows it has completed count."""
    geo = _geometry()
    frames = _passes(1, seed=19)
    full = _pass_ref(geo, frames, 0).result()
    a, b = 3 * _SCAN[1] + 2, 11 * _SCAN[1] + 5            # scan rows 4–10 whole
    acc = ParallaxAccumulator(geo, device="cpu", accumulate=mode, wrap_ids=True)
    acc.process(frames[a:b], 3 * _NPOS + np.arange(a, b))
    r0, r1 = acc.complete_rows()
    assert 0 < r1 - r0 < full.shape[0]
    _close(acc.result()[r0:r1], full[r0:r1])
    for r in (r0 - 1, r1):
        if 0 <= r < full.shape[0]:
            assert not np.allclose(acc.result()[r], full[r])


@pytest.mark.parametrize("mode", ["swap", "all"])
def test_complete_rows_after_a_retune_restarts_the_image(mode):
    geo, new = _geometry(), _geometry(c10=-1.2e4)
    frames = _passes(2, seed=20)
    acc = ParallaxAccumulator(geo, device="cpu", accumulate=mode, wrap_ids=True)
    acc.process(frames[:_NPOS + 4 * _SCAN[1]], np.arange(_NPOS + 4 * _SCAN[1]))   # a pass and a bit
    acc.set_geometry(new)                                     # the image restarts
    a, b = _NPOS + 4 * _SCAN[1], _NPOS + 11 * _SCAN[1] + 3   # scan rows 4–10 whole, after the restart
    acc.process(frames[a:b], np.arange(a, b))
    r0, r1 = acc.complete_rows()
    full = _pass_ref(new, frames, 1).result()
    assert 0 < r1 - r0 < full.shape[0]
    _close(acc.result()[r0:r1], full[r0:r1])


@pytest.mark.parametrize("mode", ["latest", "all"])
def test_track_passes_in_the_other_modes(mode):
    geo = _geometry()
    frames = _passes(2, seed=16)
    acc = ParallaxAccumulator(geo, device="cpu", accumulate=mode, wrap_ids=True,
                              track_passes=True)
    seen = []
    acc.on_pass_complete = lambda k: seen.append((k, acc.result().copy()))
    _feed(acc, frames, np.arange(2 * _NPOS), 50)
    acc.end_series()
    assert [k for k, _ in seen] == [0, 1]
    _close(seen[0][1], _pass_ref(geo, frames, 0).result())   # pass 0 completes alone in both


def _run(args, tmp_path, check=True):
    return subprocess.run([sys.executable, "-m", "realtime_stem.apps.run", "--device", "cpu",
                           "--out-dir", str(tmp_path / "out"), *_FLAGS, *args], cwd=REPO,
                          check=check, capture_output=True, text=True)


def test_runner_saves_every_pass(tmp_path):
    frames = _passes(3, seed=17)
    np.save(tmp_path / "rep.npy", frames.reshape(3, *_SCAN, 32, 32))
    _run(["--source", str(tmp_path / "rep.npy"), "--batch-size", "40", "--accumulate", "swap",
          "--save-passes"], tmp_path)
    geo = _geometry()
    out = tmp_path / "out"
    for k in range(3):
        want = _pass_ref(geo, frames, k).channels()["phase_flipped"].numpy()
        _close(np.load(out / "passes" / f"parallax_phase_flipped_pass{k:04d}.npy"), want)
    _close(np.load(out / "parallax_recon.npy"), _pass_ref(geo, frames, 2).result())


def test_runner_rejects_inconsistent_pass_flags(tmp_path):
    for args, msg in ((["--pass-average", "0.5", "--accumulate", "reset"], "--pass-average needs"),
                      (["--save-passes"], "--save-passes needs repeated scans")):
        proc = _run(["--source", "nothing.npy", *args], tmp_path, check=False)
        assert proc.returncode != 0 and msg in proc.stderr, proc.stderr
