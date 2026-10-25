# Changelog

This folder holds realtime-stem 2.2.0. Each release is a git tag in
[EM_Realtime_Processing](https://github.com/kucukogluberk/EM_Realtime_Processing) (e.g. `v2.2.0`), where its history is; the recorded sessions
named below are there too, in [`sessions/`](https://github.com/kucukogluberk/EM_Realtime_Processing/tree/v2.2.0/sessions).

## 2.2.0

New in this release: a browser viewer served by `run.py --viewer-port` with live tuning of
C10, the rotation, C12 and the accumulate mode; recording the viewer; the geometry calibration
from a recording (C10 refined on the lattice); the continuous modes `reset` and `swap`; tools to
loop and refocus a recording; one layout for recorded sessions. The C++/CUDA operators are
unchanged (still 2.0.1), so an image built for 2.1.0 runs it.

**Record the viewer.** Save → "● record" writes a video in one of two kinds.

- **page** (the default): the whole page with its controls, through the browser's tab capture,
  e.g. a supplementary video of tuning a reconstruction. The browser asks to share the tab.
- **view:** the composition of the PNG export, redrawn at a fixed rate: the image, the Fourier
  transform if open, scale bars, cutoff circles and the caption.
- **Rate:** 15 frames/s by default, 5–30 in a dropdown.
- **Format:** H.264 `.mkv` in Chrome and Edge, VP9 `.webm` in Firefox, `.mp4` in Safari.
  - A 6 s recording of the view at 1286×802 is about 2 MB.
  - The page adds the duration that `MediaRecorder` leaves out, so players show the length and
    can seek.

**Tune the reconstruction from the browser: `run.py --viewer-port 8050`.** The pipeline serves
the viewer itself (one command instead of two), and the page's Reconstruction section fine-tunes
the running reconstruction.

- **What it tunes:** C10, the rotation, C12 (as Cartesian a, b) and the `--accumulate` mode. The
  command line's values are the starting point and remain the target for the microscope's
  knobs.
- **How a change lands:** between batches, under a lock shared with the ingest, and into the
  run's arguments, so later scans and the saved results use it. A snapshot is published at once.
  - `latest` re-deposits the frames it holds with the new shifts
    (`ParallaxAccumulator.set_geometry`), so the image is exact at once: ~0.2 s for a 256² scan
    at ×4 on an M4 Pro.
  - The other modes restart their image.
- **Reset and copy:** back to the command line's values, and "copy flags" for the tuned ones.
- **After the run** the viewer stays up until Ctrl-C, which saves the tuned results and prints
  the flags. This tunes a recording offline (`--source … --accumulate latest`).
- **Algorithm-agnostic runner:** an `Algorithm` declares its tunable parameters (`live_params`,
  `retune`, `tuned_flags`); `apps/control.py` holds the rest.
- **Snapshot metadata:** it now also carries C12 and φ12.

**Sessions: one layout and one standard analysis for every recorded session.**

- **Layout:** `sessions/<date>_<specimen>/`, with `session.json` (the recordings, their
  acquisition values, the live flags), `data/` and `recon/` (not in git), `calibration/`,
  `figures/` and `analysis/`. It is described in `sessions/README.md`.
- **Standard analysis:** `scripts/session_report.py <session>` produces, for each recording:
  - the calibration, with C10 refined on the lattice;
  - the reconstructions at ×1/2/4/8, at the support's focus and with the live flags;
  - the same figures for every session: an overview of live, support and lattice focus with the
    spectrum and the calibration along C10, and an upsampling series with a crop where the
    lattice is strongest;
  - `summary.json`.
- **Sessions so far:**
  - `out_gd2o3_30mrad/` is now `sessions/2026-10-20_Gd2O3/`. It was re-run with the refined
    calibration (binned C10 −281 Å, against the fit's −316 Å), quantem included.
  - The Au-grating session of 2026-10-23 is `sessions/2026-10-23_Au-grating/`, with its own
    checks of the disk size, the scan step and the CTF zeros.

**Continuous modes `--accumulate reset | swap`, movies with `--save-passes`, and looping a
recording.**

- **`reset`** starts a fresh image with each pass, so the live view is the current pass filling
  in.
- **`swap`** shows the current pass on the rows it has completed and the last completed pass
  elsewhere, never partial. `--pass-average β` makes it a running average of passes.
- **Saving and memory:** both save the last completed pass and need one more image rather than
  `latest`'s store (1 MB against 649 MB for the binned Gd₂O₃ scan at ×2).
- **Pass boundaries:** batches are split at pass boundaries, and stragglers of a closed pass are
  counted.
- **`--save-passes`** writes each completed pass's channels to `<out-dir>/passes/`, a movie of the
  session, from a background thread.
- **`tools/repeat_dstream.py`** loops a recorded scan into N passes: one series or one per pass,
  optionally with drift. Only `image_id` is rewritten, and the receiver's C parser reads the
  patched ids.
- **`tools/refocus_dstream.py`** turns the focus knob on a recording. Pass k is the scan as at
  microscope defocus C10_k: each bright-field pixel's image is moved by the parallax shift of the
  change. Use it for demos of live tuning on real data.
- **`scripts/compare_accumulate.py`** compares the modes on a looped capture.
  `scripts/bench_accumulator.py` times `reset` / `swap` too: 985 kHz against 1,123 kHz for `all`
  on `mps` at M = 1249.

**Browser viewer: `python -m realtime_stem.streaming.web tcp://<host>:5601`.** A page on localhost
replaces the matplotlib window as the recommended viewer.

- **Smooth at large sizes.** The page asks only for the region it shows, at the size it shows it,
  so a 4096² snapshot is as smooth as a 512² one. Contrast, gamma and the colour map are applied
  on the GPU (WebGL2).
- **Controls:**
  - **The image:** one panel, the parallax reconstruction.
  - **Filters,** shown in the Fourier transform too:
    - the phase flip (the pipeline's `phase_flipped` channel);
    - Butterworth high- and low-pass (order 12 by default, 2–16 in a menu), set by their
      cutoff period in Å on logarithmic sliders (the high-pass up to the field of view). The
      page applies them to the spectrum itself, so they don't refetch the Fourier transform,
      and marks each cutoff with a circle.
  - **Fourier transform:** an optional panel of the visible square at native pixels (≤ 2048²),
    with d-spacings, a scale bar in Å⁻¹, and an outline of its square on the image.
    - It is fetched once the view stops moving, at the panel's size.
    - The server computes it as a float32 real FFT.
  - **Contrast:** auto (frame or view) or manual limits, with a histogram whose lines can be
    dragged. Manual starts from the limits shown.
  - **Always on:** scale bars.
  - **History:** pause and a history slider.
  - **Export:** one PNG of the view (image and Fourier transform, scale bars, caption) and
    `.npy`.
  - **URL presets.**
- **Dependencies:** the standard library, numpy and pyzmq; the page loads nothing from the
  internet. `examples/viewer.py` and the notebook view are unchanged.
- **Snapshot metadata:** it now carries `sampling_A`, the output pixel size. While the image is
  filling in (a scan's first pass, every pass of `reset`, an image restarted mid-run by a mode
  switch or a retune), it also carries `complete_rows`, the band of output rows already
  finished. The Fourier transform and the frame's histogram use only that band, so the blank
  part, and the phase flip's ringing off the edge of the scan, don't show in them.

**Calibration from a recording: `tools/calibrate_parallax.py`.** It takes one recorded scan (a
`.dstream` or `.npy`) and the acquisition values, and measures the parallax geometry.

- **What it measures:**
  - the disk centre on the raw frames, and the flagged / hot pixels, with whether they fall in the
    disk;
  - the column flip (handedness), the rotation, C10 and C12.
- **How:** it cross-correlates the virtual bright-field images of small groups of bright-field
  pixels, then fits the engine's own shift model. The fit is checked with the variance loss, along
  C10 and the rotation and against the other handedness.
- **Output:** `<out>.flags` for `run.py`, `<out>.json`, and a diagnostic plot. `--rotation-prior`
  picks between the twins (C10, θ) and (−C10, θ + 180°).
- **Two measurement details:**
  - Shifts are measured below a low-pass, from pixels clear of the disk edge by that frequency.
    Near the edge only one sideband is transferred, which moves the correlation peak outwards.
  - The estimates are refined against an affine-regularized reference until they converge.
- **Diagnostic only:** departures from a quadratic aberration surface (C30, C21, C23, or radial
  distortion of the pattern). Parallax keeps χ(k ± q) to first order, and only for quadratic χ is
  the dropped term common to all bright-field pixels.
- **C10 on the lattice.** The shifts are measured at low frequencies, so the fit focuses the
  support film. The tool then reconstructs ×4 images over a range of C10 and keeps the one at
  which lattice spots beyond the scan's Nyquist frequency stand out most from the diffuse
  background. That value goes into `--defocus`.
  - **The score:** a top-pixel/median ratio, ring by ring. Noise, amorphous contrast and aliases
    have the same ratio at any level, so only spots raise it.
  - **Sub-pixel phase classes:** each output pixel is normalized by its class count, so uneven
    coverage adds no comb.
  - **On the Au grating:** the lattice was sharpest 136–148 Å closer to focus than the fit,
    with 24–32× the prominence.
  - **Without a lattice:** it keeps the fit.
  - **Options:** `--refine-band`, `--refine-span`, `--no-refine`.
- **Tests:** all 8 layouts, the twin and the flags round trip on simulated data. It is GUIDE
  commissioning step 4.

## 2.1.0

New in this release: a live phase-flipped image, any detector orientation, continuous scanning
for live tuning, and a pure-Python ingest that runs the same processing without an NVIDIA GPU,
with a simulator and a notebook view. The C++/CUDA operators are unchanged (still 2.0.1). The
per-frame mean is now kept by default (see below); with `--subtract-frame-mean`,
`parallax_recon.npy` is the same image as in 2.0.2.

**Live channels** (`processing/parallax/readout.py`), computed from the accumulated sum at each
snapshot, with no per-frame cost:

- `phase_flipped`: the parallax image filtered by `sign(sin χ(q))`, identical mid-scan to
  applying the phase flip frame by frame, at one FFT pair per snapshot. On by default
  (`--no-phase-flip`).
- Per-bright-field-pixel scan-mean removal (`--no-dc-correct` to switch off): removes the static
  bright-field pattern at the scan front and on upsampled grids.
- `--coverage`, `--normalize`, Butterworth `--lowpass` / `--highpass`, `--suppress-nyquist`,
  `--band-limit`.
- The viewer shows one panel per channel; snapshots carry the channel names and metadata
  (C10, rotation, frames, scan pass). Use the viewer of this release: a 2.0.x viewer expects a
  single image.
- Results add `parallax_corrected.npy` and `parallax_phase_flipped.npy` next to
  `parallax_recon.npy`; with `--continuous`, each also gets its `_scanNNN` copy.

**Detector orientation.** `--detector-layout` names one of the 8 transpose/flip layouts (e.g.
`cols,-rows`), adds the matching rotation and maps the default disk centre onto the raw frame.
`--flip-cols` mirrors the detector columns. A mirrored detector no longer needs a change on the
scan generator.

**Continuous scanning: `--accumulate {scan,latest,all}`.**

- `latest` keeps the newest frame per scan position, so while the microscope scans repeatedly
  the image is the last full pass and follows focus and stigmator changes within one pass. It
  stores positions × bright-field pixels × 2 bytes on the GPU (160 MB at 256², 2.6 GB at 1024²).
- `all` sums every repeat.
- Image ids past the scan wrap onto it, so one long series works; with `--continuous`, a series
  per pass also works. The end-of-scan report then gives the positions covered and the passes.
- `--live-every-rows N` publishes a snapshot every N scan rows instead of every `--live-every`
  seconds.

**Run without the container or an NVIDIA GPU.**

- `--ingest python`: decodes the DCU's Stream2 messages on the CPU (about 80k messages/s per
  core) from ZeroMQ, or from a file with `--source capture.dstream | frames.npy`
  (`--source-rate` paces it). `--ingest auto` (the default) takes the Holoscan ingest where it is
  available.
- `--device {auto,cuda,mps,cpu}` for the reconstruction. The Holoscan ingest needs `cuda`.
- `scripts/run_local_demo.sh`: simulate, replay and watch a scan with no Docker.
- Both ingests share one sink (`ingest/sink.py`): the same throughput prints, live snapshots and
  `COMPLETE` / `INCOMPLETE SCAN` report.

**Notebook view.** `realtime_stem.streaming.notebook.LiveView` subscribes to the live stream
from Jupyter (`examples/live_view.ipynb`), keeps a history of snapshots and writes animations.
The desktop viewer and the notebook draw the same panels (`streaming/display.py`).

**Simulator.** `tools/simulate_4dstem.py` makes weak-phase-object 4D-STEM data of any size with
a known ground truth, scan/detector rotation, detector layouts and Poisson dose, as `.npy` or
`.dstream`, plus the exact `run.py` flags. `--passes` with `--defocus-schedule` /
`--c12-schedule` simulates an operator tuning the microscope.

**Tools and docs.** `tools/npy_to_dstream.py` converts a 4D `.npy` into a recording;
`tools/record_dstream.py` records a stream from Python; `tools/replay_to_zmq.py --rate` paces a
replay. `docs/CLI.md` lists every parameter (generated by `scripts/gen_cli_reference.py`).
`scripts/bench_accumulator.py` times the reconstruction step per device and mode.

**Fixed.** The final live snapshot could be lost when the run ended (the publisher now lingers up
to 1 s).

**Tests:** about 200, among them the live channels against the frame-by-frame phase flip, all 8
detector layouts (also on the public apoferritin data), the decoder against the DECTRIS and
bitshuffle implementations, the Python ingest over ZeroMQ, continuous scanning and the simulator.
Checked end to end on CPU and Apple silicon, including a recorded 1024×1024 ARINA scan (complete).
For the full rate on a given GPU, replay a recording with `scripts/check_install.sh`.

Contributed by Georgios Varnavides.

**Setup checks.**

- The DCU address is checked first: the pipeline and `tools/record_stream.py` warn at start-up,
  and every 30 s, while nothing accepts connections at the stream address, instead of waiting
  silently.
- The run prints the GPU it uses (name, PCI bus), warns when several GPUs are visible, and notes
  when the GPU is already busy.
- `scripts/check_install.sh`: the DCU check comes first (`DCU=<ip>` is enough) and tells a wrong
  address from a closed port. `GPU=` is required on machines with several GPUs and is compared
  with the GPU the container sees. A replay no longer hangs when the pipeline stops early or a
  leftover container holds the replay port, and the feeder's log is kept. The check ends with the
  `--gpus` and `--endpoint` values for the run command.
- The run commands in the docs use `--gpus "device=N"` instead of `--gpus all`.
- The acquisition values have no defaults any more: `--energy`, `--semiangle`,
  `--angular-sampling`, `--scan-sampling`, `--scan`, `--defocus` and `--rotation` are required,
  and a run without one of them stops before it starts, listing what is missing (a silently used
  default gave a plausible but wrong image). `--center` still defaults to the frame center. `scripts/check_install.sh` takes the recording's
  values in `FLAGS`. The simulator's `.flags` include `--center`.
- The start-up summary prints the bright-field disk's radius (semiangle ÷ angular sampling) and
  warns when part of the disk falls outside the frame (those pixels were dropped silently).
- **The frame mean is kept by default.** Up to now each frame's mean over its bright-field
  pixels was subtracted before the frame was added in. That mean is the bright-field intensity at
  the scan position, so the subtraction removed the amplitude contrast (mass-thickness,
  diffraction contrast) over regions larger than about the probe, and the images looked flat. On
  a recorded ARINA scan it lowered the image's standard deviation by a quarter and removed nearly
  all of the contrast below 6 cycles per field of view. `--subtract-frame-mean` restores the
  previous behaviour (identical images); it also removes frame-to-frame changes in beam
  intensity. The start-up summary says which is used.
- `tools/detector_setup.py`: `check` (read only) and `apply` the DCU settings the pipeline needs,
  `arm`, `disarm`, and `scan` (arm, a site's own `start_scan()`, disarm). The scan generator is
  left to each site.

## 2.0.2

- The default scheduler is Holoscan's single-threaded one (`--scheduler-threads 0`). The
  4-thread `MultiThreadScheduler` of earlier versions lets idle workers sleep up to 5 ms between
  checks for work, which costs 5–7 ms per batch: at a paced 120 kHz it ended a 1024² scan about
  0.5 s late (a backlog of about 60k frames) and fell further behind on a busy machine. The
  single-threaded scheduler kept up in every test and uses less CPU. `--scheduler-threads 4`
  restores the old behaviour. No change to the operators (still 2.0.1), the image or the
  reconstruction.
- Ctrl-C and `docker stop` end a `--continuous` session with either scheduler, also between two
  scans. Holoscan stops the running graph by itself on these signals and `app.run()` returns as
  if the scan had ended; the runner now records the signal instead of relying on a
  KeyboardInterrupt (which the single-threaded scheduler swallows, and which `docker stop` never
  raised: in 2.0.1 and earlier, `docker stop` of a `--continuous` session armed the next scan and
  was killed 10 s later, after the results were saved).
- `tools/record_stream.py` records a live scan to a `.dstream` file, message for message, for
  replay and for tuning the parameters offline (3 connections by default).

## 2.0.1

The reconstruction is unchanged.

- **Fix: about one frame per million was skipped.** The receiver finds a frame's compressed data
  by searching the message for the three bytes of the DECTRIS compression tag. The DCU's
  timestamps contain the same three bytes about once per million frames; the receiver then
  rejected that frame ("could not read a bslz4-compressed frame") and the scan was reported
  incomplete by one position. A match that is not followed by a valid payload is now skipped and
  the search goes on (found by checking a recorded DCU stream frame by frame; also in 1.0.0 and
  2.0.0). The
  operators are now 2.0.1.
- `--n-connections` defaults to 3. The DCU paces each connection to about 60k beam-on binned
  frames per second, so the former default of 2 capped a live scan near 112 kHz.
- `docker/Dockerfile` pins the base image (by digest) and the pip packages to the versions 2.0.0
  was tested with, so a later build gives the same image.
- `scripts/check_install.sh` also reports the speed of the network link towards the DCU (step 6).
- Docs: the GPU architectures compiled for; the network and connection requirements; the decode packages for
  `tools/make_dstream.py` are host-side (the image does not carry them).

## 2.0.0

Robustness at the edges of a scan. The reconstruction itself is unchanged: on the same data,
2.0.0 and 1.0.0 give the same image (float32 rounding).

**End of scan**

- The last, partial batch of a scan is now processed. Any batch size works, and scans whose size
  is not a multiple of the batch size (e.g. 500×500) are complete.
- After the `end` message the receiver keeps reading until every image the `start` message
  announced is in, or until the series has been quiet for 100 ms. With several connections, `end`
  can overtake frames still queued on another connection; 1.0.0 lost them (up to one batch,
  typically on the first scan after start-up).
- Only the receiver stops at the end of a scan; the decompressor and the reconstruction finish
  everything already received before the run ends.
- The scheduler gets a 500 ms grace period before it decides that the pipeline has nothing left
  to do. 1.0.0 could end a run in the middle of a scan, without an error, when the machine was
  busy.

**Scans in quick succession (`--continuous`)**

- The connection to the DCU stays open from one scan to the next. 1.0.0 closed it after every
  scan and threw away whatever had already arrived for the next scan, so a scan that started
  within about a second of the previous one lost its first frames. Messages of the next scan that
  arrive while the current one is still finishing are kept and processed by the next round.
  Tested with scans back to back at 120 kHz with no gap at all: every scan complete.

**Ctrl-C and stopping**

- The receiver waits for messages with a timeout, so Ctrl-C and `docker stop` work at any time,
  also while waiting for a scan or when a scan never sends its `end`. The image received so far is
  saved.

**Series handling**

- The receiver follows the stream's `series_id`. Frames of another series are never mixed into
  the current scan. A series that never sends its `end` is ended when the next series begins,
  with a warning.
- An `end` message left over from an earlier run (before any start or frame) is ignored instead
  of ending the run with nothing.
- Frames that arrive without their series' `start` (left over from a crashed run, or the
  pipeline was started during a scan) are reported with a warning, and the run reports the scan
  as incomplete.
- The `start` message is checked: a warning when the detector announces a different number of
  images than `--scan` has positions, or a different frame size than configured.

**Integrity**

- Every image's compressed data is checked on the GPU (number of LZ4 blocks, and nvcomp's status
  for every block). Corrupt data now stops the run with an error instead of giving a silently
  wrong image.
- Image messages that cannot be decoded (no bslz4 payload, no `image_id`) are skipped and
  counted; the first one is reported with a hint. 1.0.0 placed a frame without `image_id` at scan
  position 0.
- The message type is recognised wherever the `type` key sits in a small message.

**Reporting and results**

- After every scan the runner reports whether every scan position received exactly one frame:
  `COMPLETE`, or `WARNING: INCOMPLETE SCAN` with the number of missing positions and duplicate
  frames.
- `--continuous` also saves every scan as `parallax_recon_scanNNN.npy`; `parallax_recon.npy` is
  always the latest scan.
- If the run stops with an error (e.g. a frame id beyond `--scan`), the image so far is saved
  first.
- At start-up the runner prints its version and the operators' version, and warns when the DCU
  address does not accept connections.

## 1.0.0

First release: parallax / tcBF on the Holoscan ingest, live stream-out, replay tools, and the
C++/CUDA operators in `operators/`. The operators are those in use since July 2026, apart from the
batch ceiling (2048).
