# From a recording to a live demo

How to watch the live image in a browser, measure a scan's geometry from a recording,
reconstruct it offline, replay it as a continuous acquisition, and turn the focus knob on it for
a demo. Every command works on a laptop without a detector or an NVIDIA GPU, using
`--ingest python` (Apple `mps`, or `cpu`). The worked examples are the sessions in
[`sessions/`](https://github.com/kucukogluberk/EM_Realtime_Processing/tree/v2.2.0/sessions) of EM_Realtime_Processing: Gd₂O₃ (2026-10-20) and an Au grating (2026-10-23).
`scripts/session_report.py` runs the standard analysis of a session (calibration,
reconstructions, figures) in one go. Every flag is in [CLI.md](CLI.md); the background is in
[GUIDE.md](GUIDE.md).

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[local,viz,replay]"   # once
```

Below, `scan.dstream` is a recording (`tools/record_stream.py`) and `scan.flags` its `run.py`
flags.

## Watch in a browser

![The browser viewer served by run.py --viewer-port: the Reconstruction section's sliders, the parallax image phase-flipped, high- and low-passed, and its Fourier transform with the filters' cutoff circles](images/web_viewer.jpg)

Start the viewer, subscribed to the pipeline's live stream (`run.py --live-endpoint …`), and
open http://127.0.0.1:8050:

```bash
python -m realtime_stem.streaming.web tcp://127.0.0.1:5601
```

Or give `run.py` `--viewer-port 8050`: the pipeline serves the page itself, and the page can also
tune the reconstruction (see [Tune it live](#tune-it-live)).

- **Navigation:** one image, the parallax reconstruction. Wheel to zoom, drag to pan,
  double-click or `f` to fit, `1` for native pixels.
- **Filters:** the phase flip, and Butterworth high- and low-pass set by their cutoff period in
  Å (logarithmic sliders; the high-pass reaches the field of view), order 12 by default. The
  Fourier transform shows them too, at once, with a circle at each cutoff.
- **Fourier transform** (optional panel) of the visible square, with q and the d-spacing under
  the cursor.
  - A dashed outline on the image marks the square.
  - On a partly scanned image (a scan's first pass, every pass of `reset`, an image restarted
    by a mode switch or a retune) it uses only the finished rows.
- **Contrast:** limits from the whole frame's histogram, from what is on screen, or manual.
  - Drag the lines on the histogram to set them; manual starts from the limits shown.
  - Gamma, colour map and invert.
- **Scale bars** on the image and the Fourier transform, the pass number and a progress bar.
- **History:** pause (space) and a slider over past snapshots (`--history-mb`).
- **Export:** one PNG of the view (image and Fourier transform), and `.npy` of the shown channel.
- **Record:** a video at 15 frames/s by default (5–30): `.mkv` (H.264) in Chrome, `.webm` in
  Firefox.
  - **page** (the default) records the whole page with its controls; the browser asks to share
    the tab.
  - **view** records the image and Fourier transform as the PNG shows them.
- **URL presets:** e.g. `?flip=1&fft=1&lp=1.5` for the phase-flipped image with its Fourier
  transform, or `?cmap=inferno&view=1to1`. `?snapshot` shows one snapshot without the live
  stream.
- **Another computer:** run the viewer next to the pipeline and `ssh -L 8050:127.0.0.1:8050
  <server>`. Only screen-sized images cross the network, so a 4096² image stays smooth.

`python examples/viewer.py tcp://127.0.0.1:5601` (matplotlib) and the notebook view still work.

## Measure the geometry of a recording

Record one scan at a known defocus, and give the acquisition values you know. The rotation
prior is the rotation you used live; it picks between C10 and the twin −C10 at θ + 180°.

```bash
python -m tools.calibrate_parallax scan.dstream --scan 256 256 --energy 200000 --semiangle 30 \
    --angular-sampling 1.066 --scan-sampling 1.9766 --rotation-prior 3.75 --out scan
```

It writes:
- `scan.flags`, the `run.py` flags: `--center`, `--flip-cols`, `--rotation`, `--defocus` and
  `--c12/--phi12`;
- `scan.json`, everything it measured;
- `scan.png`, the diagnostics.

It also reports flagged and hot pixels and whether they fall in the disk.

Departures from a quadratic aberration surface (C30, C21, C23) are reported as a diagnostic only.
Parallax corrects quadratic aberrations alone.

**Focus on the lattice, not the support.**
- **Why:** the shifts are measured at low spatial frequencies, so the fitted C10 focuses whatever
  dominates there, often the support film. Particles sit about one radius off the support. On
  the Au grating of 2026-10-23, the islands' lattice was sharpest 120–150 Å closer to focus than
  the fit.
- **What the tool does:** it then reconstructs ×4 images over a range of C10. It keeps the C10
  where lattice spots beyond the scan's Nyquist frequency stand out most from the diffuse
  background, and writes that C10 as `--defocus`. The fitted C10 is still reported.
- **Without a lattice:** with no lattice above the noise, e.g. an amorphous specimen, it keeps
  the fit.
- **Options:** `--refine-band QMIN QMAX` sets the frequencies (e.g. a band about a lattice's
  reflections), `--refine-span` the range, and `--no-refine` skips the step.

## Reconstruct it offline

```bash
python -m realtime_stem.apps.run --ingest python --device mps --source scan.dstream \
    $(cat scan.flags) --upsampling 4 --out-dir out/scan
```

- **Output:** `parallax_recon.npy` (the raw sum), `parallax_corrected.npy` and
  `parallax_phase_flipped.npy`.
- **Upsampling:** `--upsampling 2–8` recovers spatial frequencies past the scan's Nyquist, up to
  2α/λ.
- **Frame mean:** `--subtract-frame-mean` removes the amplitude contrast; the default keeps it.
- **DC correction:** `--no-dc-correct` turns it off. It matters mid-scan and on upsampled grids.
- **Comparing with quantem:** the Gd₂O₃ analysis does it with the same geometry
  ([`sessions/2026-10-20_Gd2O3/analysis/quantem_recons.py`](https://github.com/kucukogluberk/EM_Realtime_Processing/tree/v2.2.0/sessions/2026-10-20_Gd2O3/analysis/quantem_recons.py)). quantem has no column flip, so flip
  the frames first.

## Replay it live

Run the pipeline with a live stream, paced like a scan, and watch it in the browser viewer:

```bash
python -m realtime_stem.apps.run --ingest python --device mps --source scan.dstream $(cat scan.flags) \
    --upsampling 4 --source-rate 20000 --live-endpoint tcp://127.0.0.1:5601 --out-dir out/replay
```

Over ZeroMQ, as the DCU sends it, start the pipeline first:

```bash
python -m realtime_stem.apps.run --ingest python --device mps --endpoint tcp://127.0.0.1:5555 \
    $(cat scan.flags) --live-endpoint tcp://127.0.0.1:5601 --out-dir out/replay
```

then the feeder:

```bash
python -m tools.replay_to_zmq scan.dstream --endpoint tcp://127.0.0.1:5555 --rate 20000
```

## Tune it live

`--viewer-port` puts a Reconstruction section on the page. It holds C10, the rotation and C12 as
stigmator-like a/b components, each with a slider about its command-line value, plus the
`--accumulate` mode of a repeated scan.

```bash
python -m realtime_stem.apps.run --ingest python --device mps --endpoint tcp://127.0.0.1:5555 \
    $(cat scan.flags) --upsampling 4 --accumulate latest --continuous --viewer-port 8050
```

- **Applying changes:** with `latest` a change re-renders the frames held at once (~0.2 s for a
  256² scan at ×4 on an M4 Pro); the other modes restart their image.
- **Keeping the result:** "copy flags" gives the tuned `run.py` flags. The command line's values
  remain the target for the microscope's knobs; tune here what the knobs can't easily reach,
  such as the rotation.
- **Offline:** on a recording, `--source scan.dstream --accumulate latest --viewer-port 8050`
  keeps the viewer up after the replay. Tune, then Ctrl-C saves the tuned image and prints its
  flags.

## Repeated scans: loop a recording

A continuously scanning microscope sends the same area pass after pass. To get that from one
recording, loop it:

```bash
python -m tools.repeat_dstream scan.dstream loop.dstream --passes 10 --scan 256 256 --drift 1 2
```

- **The file:** one series with ids running on, as from one arm for many passes. Use
  `--series per-pass` for a series per pass, played with `--continuous`.
- **Drift:** `--drift DY DX` moves the specimen by whole scan pixels per pass.
- **The payloads** are copied byte for byte; only `image_id` changes.

```bash
python -m realtime_stem.apps.run --ingest python --device mps --source loop.dstream $(cat scan.flags) \
    --upsampling 4 --accumulate swap --source-rate 20000 --live-endpoint tcp://127.0.0.1:5601 \
    --save-passes --out-dir out/loop
```

| `--accumulate` | the live image | extra device memory |
|---|---|---|
| `latest` | the newest frame at every position: the current pass above the cursor, the previous one below | positions × bright-field pixels |
| `swap` | the current pass on the rows it has completed, the last completed pass elsewhere; with `--pass-average β`, a running average of passes | one image |
| `reset` | the current pass alone, filling in | one image |
| `all` | every frame of every pass | — |

- **Saving passes:** `--save-passes` writes each completed pass to `out/loop/passes/`, a movie of
  the session.
- **Comparing the modes:** `scripts/compare_accumulate.py loop.dstream $(cat scan.flags)
  --out cmp.png` shows each mode's live image part-way through each pass, against the true pass
  images.
- To keep a loop running, replay it to `run.py --continuous` in a loop:
  `while true; do python -m tools.replay_to_zmq loop.dstream --endpoint … --rate 20000; done`.

## A focus-walk demo

The microscope's focus can be turned on a recording too. Under the parallax model, a defocus
change only moves each bright-field pixel's image by ΔC10·α along its tilt. So each pass is
rewritten as the scan would look at another defocus. Reconstructing with the recording's own
flags, the image sharpens as the "knob" reaches focus:

```bash
python -m tools.refocus_dstream scan.dstream walk.dstream --flags scan.flags \
    --defocus-schedule -500 -450 -400 -350 -300 -316 -316 -316
```

The schedule above walks the Gd₂O₃ binned scan, recorded at −316 Å, from −500 Å into focus. A
pass written at −500 Å measures −501 Å with `tools.calibrate_parallax`. The moves are whole scan
pixels, so steps smaller than about one pixel at the disk edge don't show.

For the demo, start the viewer, then the pipeline (`--continuous`), then the looping feeder, each
in its own terminal:

```bash
python -m realtime_stem.streaming.web tcp://127.0.0.1:5601        # http://127.0.0.1:8050
```

```bash
python -m realtime_stem.apps.run --ingest python --device mps --endpoint tcp://127.0.0.1:5555 \
    $(cat scan.flags) --upsampling 4 --accumulate latest --continuous \
    --live-endpoint tcp://127.0.0.1:5601 --out-dir out/demo
```

```bash
while true; do python -m tools.replay_to_zmq walk.dstream --endpoint tcp://127.0.0.1:5555 --rate 15000; done
```

At 15 kHz each pass of 65,536 frames takes about 4 s. `?flip=1` gives one large
panel for recording.
