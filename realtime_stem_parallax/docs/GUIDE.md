# realtime-stem — technical guide

The full description: architecture, build, usage, live tuning, commissioning, known issues. For
the short version (what is where, install, run) see the [README](../README.md); every
command-line parameter is in [CLI.md](CLI.md); what changed and when is in the
[CHANGELOG](../CHANGELOG.md).

Real-time STEM data processing at full detector frame rate. Detector frames stream from a DECTRIS
detector control unit (DCU), are GPU-decompressed on an NVIDIA Holoscan ingest (or decoded on the
CPU by a pure-Python ingest, anywhere), folded into a reconstruction as they arrive, and streamed
out live over ZeroMQ — so the reconstruction fills in *during* the scan rather than only at the
end, and keeps following the microscope while it scans continuously.

> **Holoscan 5 (3.0.0).** This guide describes the Holoscan 4 image (`docker/Dockerfile`, ZMQ)
> in detail. The Holoscan 5 image (`docker/Dockerfile.hsdk5`) runs the same processing,
> streaming and viewer, with the repository's shared receivers over ZMQ or RDMA: how to build it,
> choose a transport, set up the DCU and test it without a detector is in the
> [README](../README.md) ("Which image, which transport" onwards). Where this guide names
> `operators/`, `kMaxBatchSize` (2048) or `--scheduler-threads`, the Holoscan 5 equivalents are
> `../holoscan_pipeline/`, a batch limit of 1000, and the Holoscan 5 runtime's own scheduling.

The worked example is **parallax / tilt-corrected bright field (tcBF)**, running at 120 kHz on
binned ARINA data. The pipeline is deliberately modular so a new algorithm is a drop-in.

## Architecture

Three stages with narrow interfaces, so each can be understood and changed on its own:

```
   DCU ─(ZeroMQ)─►  ingest  ─(batches)─►  processing  ─(reconstruction)─►  stream-out  ─(ZeroMQ)─►  subscribers
                 ZmqRxOp →              Accumulator                     LivePublisher            viewer / probe / …
                 DecompressGpuOp        .process(frames, ids)
                 (C++/CUDA)             .result() / .snapshot()
             or  PythonIngest
                 (CPU decode; also
                  .dstream / .npy files)
```

- **ingest** (`realtime_stem/ingest/`) — two interchangeable front ends feeding one `BatchSink`
  (`sink.py`: batch → `accumulator.process`, throughput, throttled live snapshots).
  The Holoscan graph (`holoscan_graph.py`) receives raw frames and bslz4-decompresses them on the
  GPU using two compiled C++ operators, handing each batch over as a zero-copy CUDA tensor: the
  full-rate path. The pure-Python ingest (`python_ingest.py`, `codec.py`) decodes the same
  messages on the CPU — from ZeroMQ or straight from a `.dstream`/`.npy` file — for machines with
  no NVIDIA GPU, at tens of kHz (see "Running without the container"). Algorithm-agnostic.
- **processing** (`realtime_stem/processing/`) — turns batches into a reconstruction behind two
  small contracts (`processing/base.py`): `Accumulator` (the hot path) and `Algorithm` (the
  wiring). Parallax lives in `processing/parallax/`.
- **stream-out** (`realtime_stem/streaming/`) — publishes the reconstruction-so-far over ZeroMQ
  (PUB, lossy, non-blocking). An accumulator with a `snapshot()` hook publishes several named
  channels per frame (parallax: the corrected image and the phase-flipped one) with metadata in
  the header. A matplotlib viewer is just one example subscriber; the product is the stream.

`apps/run.py` wires the three together and owns only transport flags (`--ingest`, `--device`,
`--source`, live cadence, …); it never mentions parallax.

## Prerequisites

The pipeline runs inside a Docker image that carries Holoscan, a CUDA build of PyTorch, nvcomp
and the two compiled C++/CUDA Holoscan operators `ZmqRxOp` and `DecompressGpuOp`. **This
repository contains everything needed to build it**: the operator sources are in `operators/`
(see `operators/README.md`) and `docker/Dockerfile` builds the complete image on top of NVIDIA's
public base image (`nvcr.io/nvidia/clara-holoscan/holoscan:v4.4.0-cuda12-dgpu`, which supplies
Holoscan 4.4 and PyTorch 2.11). From the repository root:

```bash
docker build -t realtime-stem:latest -f docker/Dockerfile .
```

The build compiles the CUDA kernels, so it takes a few minutes.

### What the operators do

They originate in the public, MIT-licensed
[electronmicroscopy/live4d](https://github.com/electronmicroscopy/live4d) repository. The version
in `operators/` extends it with what this pipeline needs (the upstream build alone is hard-wired to
96×96 uint16 frames and a single TCP connection):

1. runtime `detector_width` / `detector_height` / `dtype` parameters, validated against the stream
   (a mismatch stops with an error instead of emitting wrong-sized garbage);
2. the bslz4 block size read from each frame's header (required for uint32 streams);
3. an `n_connections` parameter (the DCU paces each TCP connection, so 120 kHz needs 3);
4. a clean end of scan (since 2.0.0): after `end` the receiver drains all connections, flushes
   the last partial batch and stops only itself, so everything received is processed; the
   connection to the DCU stays open across the scans of a `--continuous` session;
5. series handling and checks (since 2.0.0): the `series_id` keeps one run to one series, the
   `start` message is compared with the configuration (`expected_images` = Ny×Nx), every image's
   compressed data is checked on the GPU, and the receiver waits with a 100 ms timeout so the
   application can always be stopped.

`realtime_stem/ingest/holoscan_graph.py` passes those parameters and refuses to start on an image
built from the upstream operators.

### Host requirements

`run.py` prints the operators' version at start-up (`operators 2.0.1`). "before 2.0" means the
image was built from older sources and lacks the fixes listed in `CHANGELOG.md`.

- Linux, Docker, the NVIDIA container toolkit (`--gpus all`), and an NVIDIA driver that supports
  CUDA 12.x (the `cuda12-dgpu` base image; a `cuda13` base exists for newer driver stacks). The
  `dgpu` base covers x86_64 and aarch64 hosts with a discrete GPU (IGX Orin dGPU, GH200). Jetson
  integrated GPUs would need the `igpu` base and are untested here.
- **GPU architecture.** The CUDA kernels are compiled for the architectures named at image build
  time. The default in `docker/Dockerfile` is `80;86;89;90`: A100 (sm_80), RTX 30xx / A6000
  (sm_86), RTX 40xx / RTX 6000 Ada / L4 / L40 (sm_89) and H100 / H200 (sm_90). Newer GPUs (Blackwell) run
  via PTX just-in-time compilation from compute_90 (slower first launch, otherwise fine). **Older
  GPUs (V100 sm_70, T4 sm_75) fail with "no kernel image is available"** unless their
  architecture is added with `--build-arg CUDA_ARCHITECTURES=...`.
- Validated on an NVIDIA IGX Orin (aarch64) with an RTX 6000 Ada. On x86_64 the image builds
  (kernels for sm_80 to sm_90), its operators import and the unit tests pass; a run on an x86_64
  GPU is checked with `scripts/check_install.sh`. NVIDIA's CUDA compiler does not run under
  emulation, so build the x86_64 image on an x86_64 machine.

The processing engine and streaming transport are pure PyTorch / numpy / pyzmq; PyTorch and the
Holoscan ops come from the container, so nothing GPU-specific is pip-installed here.

## Quick start — offline replay demo (no detector)

Replay a bundled synthetic capture through the whole pipeline and watch the reconstruction wipe in:

```bash
scripts/run_replay_demo.sh
```

It starts `run.py` in the container, a viewer, and a replay feeder that streams
`examples/data/sample.dstream` (a 32×32 scan of 96×96 uint16 frames, 1024 frames, real bslz4
compression in the real DECTRIS message framing) over localhost. The viewer and the feeder run on
the host's `python3`, which needs numpy, pyzmq, matplotlib and cbor2:

```bash
pip install -e ".[viz,replay]"
```

Make your own test capture with `python -m tools.make_dstream 96 96 uint16 1024 --out
examples/data/sample.dstream` (needs the decode deps, `pip install -e ".[decode]"`, i.e. cbor2,
bitshuffle and dectris-compression, in a host environment; the image does not carry them).

## Live usage (against a detector)

Two values come from the site, not from this repository: the DCU's stream address (the IP of its
data interface on the network to this machine; it can differ from the address of the DCU's web
page) and the GPU (its number in `nvidia-smi -L`). Check both first:
`GPU=<N> DCU=<DCU_IP> scripts/check_install.sh`.

The DCU must be set up for the stream: stream `enabled`, format `cbor`, compression `bslz4`,
binning `2x2` for `--binned`, trigger mode `exts` (one trigger per frame from the scan
generator), `ntrigger` = Ny×Nx and `nimages` = 1. `tools/detector_setup.py` checks and sets these
over the DCU's REST API (standard library only):

| step | command | what it does |
|---|---|---|
| check | `python -m tools.detector_setup check --ip <DCU_IP> --scan 256 256 --binned` | compares the settings, changes nothing; prints the matching `run.py` flags |
| apply | `… apply --ip <DCU_IP> --scan 256 256 --binned --frame-time 8.334e-6` | sets what differs (frame time = the scan generator's dwell time) |
| arm | `… arm --ip <DCU_IP>` | arms; start the pipeline first, the series' `start` message is sent now |
| disarm | `… disarm --ip <DCU_IP>` | ends the series (the pipeline then reports `COMPLETE`) and shows the DCU's dropped frames |
| scan | `… scan --ip <DCU_IP> --scan 256 256 --binned` | arm, `start_scan()`, disarm |

The scan generator is not driven by this repository. `start_scan()` in `tools/detector_setup.py`
is an empty slot, with pseudo-code, for a site's own code; until it is filled in, `scan` asks you
to start the scan by hand and disarms when you press Enter. `--ip` is the DCU's REST address,
which can differ from the stream address used for `--endpoint`. Settings missing on a DCU model
are skipped; `--api` sets the SIMPLON API version if the DCU does not report it.

Run the pipeline inside the container, pointing it at the DCU's ZMQ stream:

```bash
docker run --rm --network host --gpus "device=<N>" --ulimit stack=33554432:33554432 \
  -v "$PWD":/work -v "$PWD/out":/out -w /work --entrypoint python3 \
  realtime-stem:latest -m realtime_stem.apps.run \
    --algorithm parallax \
    --endpoint tcp://<DCU_IP>:31001 \
    --binned --scan 256 256 --batch-size 2048 --n-connections 3 \
    --energy <eV> --semiangle <mrad> --angular-sampling <mrad/px> --scan-sampling <Å> \
    --defocus <Å> --rotation <degrees> --center <row> <col> --upsampling 2 \
    --live-endpoint tcp://127.0.0.1:5601 --continuous
```

On the host, watch it fill in: run the browser viewer and open http://127.0.0.1:8050 (see
"Watching in a browser"):

```bash
python -m realtime_stem.streaming.web tcp://127.0.0.1:5601
```

`--continuous` re-arms after each scan's end (one container for a whole session); to tune the
microscope while it scans continuously, use `--accumulate latest` instead (see "Continuous
scanning"). Results
(`parallax_recon.npy`, `parallax_shifts.npy`) land in `out/`; plot a saved one with
`python examples/plot_result.py out/parallax_recon.npy`. See [CLI.md](CLI.md) (or `--help`) for
all flags; the
parallax tuning knobs (`--defocus`, `--rotation`, `--energy`, `--semiangle`, `--angular-sampling`,
`--scan-sampling`, `--center`, …) use the same units live as offline.

None of the acquisition values comes from the stream, and they have no defaults: `--energy` (eV,
300 kV = 300000), `--semiangle` (mrad), `--angular-sampling` (mrad/px of the mode running, e.g.
2 × semiangle ÷ the bright-field disk's diameter in pixels), `--scan-sampling` (Å per step),
`--scan`, `--defocus` and `--rotation`. A run without one of them stops before it starts and lists
the missing ones; a default that is silently wrong would give a plausible but wrong image.
`--center` (the disk centre in pixels on a raw frame) defaults to the frame center.
Keep a session's values in a file and pass them as `$(cat site.flags)`. The disk's radius in
pixels is not an input: it follows from `--semiangle` ÷ `--angular-sampling`, and the run prints it
at start-up to compare with the disk on a frame. If part of the disk falls outside the frame
(wrong centre or sampling), the run warns and says how many pixels it leaves out.

Three parameters decide whether the image is sharp, and none of them comes from the stream:
`--defocus` (C10 in Å, the microscope's readout is the starting point), `--rotation` (the
scan-to-detector rotation in degrees) and `--center` (the bright-field disk's centre on the
detector in pixels, e.g. from py4DSTEM's `get_probe_size`). Tune them offline against a replayed
capture; the flags and units are identical live.

### Recording a scan

A recording lets you tune `--defocus`, `--rotation` and `--center` offline, as often as needed.
Stop the pipeline first (only one program may read the stream), then:

```bash
python3 tools/record_stream.py capture --ip <DCU_IP> --out scan01.dstream
```

and start the scan. The recorder (host Python with `pyzmq` and `cbor2`) stops by itself at the
end of the scan and reports whether every frame arrived (`COMPLETE`). It warns when nothing
accepts connections at the address (`WARNING: cannot reach …`). Replay it into the pipeline
with `tools/replay_to_zmq.py`; keep the microscope settings of each recording next to it.

## Running without the container

`--ingest python` (the default wherever Holoscan isn't importable) replaces the compiled ingest
with `PythonIngest`. It decodes the same DECTRIS Stream2 messages on the CPU (`codec.py`: cbor2,
and bslz4 via `dectris-compression` when installed, else numpy + `lz4`), batches them, and drives
the same sink. The reconstruction runs on `--device auto` (cuda, then Apple's `mps`, then cpu).

- **Sources.** A ZMQ endpoint (`--endpoint`, `--n-connections`). The ingest keeps draining every
  connection for 200 ms after `end`, so the replay race below does not lose frames here, and a
  `start` arriving early is kept for the next round. Or a file, `--source capture.dstream` or
  `--source frames.npy` ((Ny, Nx, H, W) or (N, H, W)), with `--source-rate HZ` to replay it at a
  chosen frame rate.
- **Partial batches** are always processed: the last one of a scan, and any batch that has waited
  50 ms, so slow sources still update the live view smoothly.
- **Rate** (Apple M4 Pro): decode ~80k messages/s with `dectris-compression` (~6k/s with the numpy
  fallback). End to end from a `.dstream`, a 256×256 scan of 96×96 frames with 1249 BF pixels
  reconstructs at ~75 kHz on `mps`.
- `scripts/run_local_demo.sh` runs feeder → `run.py --ingest python` → viewer with no Docker.

`pip install -e ".[local,viz,replay]"` installs what this needs. Never install the `local` extra
into the Holoscan container: it would replace the image's CUDA build of torch.

## Watching in a browser

`python -m realtime_stem.streaming.web tcp://<host>:5601` subscribes to the live stream and
serves a page at http://127.0.0.1:8050.

Or let `run.py --viewer-port 8050` serve the same page from inside the pipeline: one command
instead of two, and the page then also tunes the reconstruction (below).

### Tuning the reconstruction from the page (`run.py --viewer-port`)

The command line's values stay the target, met by turning the microscope's knobs. The
page's Reconstruction section fine-tunes what the knobs can't easily reach: the rotation above
all, which would otherwise need a scan rotation.

- **What it tunes:** C10, the rotation, and C12 as Cartesian components a = C12 cos 2φ12,
  b = C12 sin 2φ12 (like stigmators; the polar values are shown). It can also switch the
  `--accumulate` mode of a repeated scan.
  - Each has a slider about its command-line value and a number field.
  - "reset" goes back to the command line's values.
  - "copy flags" gives the `run.py` flags of the values now in use.
- **How a change is applied:** between batches (a lock shared with the ingest), and into the
  run's arguments, so later scans and the saved results use it. A snapshot is published at
  once, even with no frames arriving.
  - `--accumulate latest` re-deposits every frame it holds with the new shifts: the image is
    at once exactly what the new values give. On an M4 Pro that takes ~0.2 s for a 256² scan at
    ×4.
  - The other modes keep no frames, so their image restarts and fills in within a pass.
  - A new mode starts a new image.
- **After the run:** the viewer stays up until Ctrl-C. Tuned results are saved again on exit,
  and the tuned flags are printed.
  - So `run.py --source scan.dstream … --accumulate latest --viewer-port 8050` tunes a
    recording offline.
- **Under the hood:** the algorithm declares what is tunable (`Algorithm.live_params`,
  `retune`, `tuned_flags`) and the runner stays algorithm-agnostic (`apps/control.py`). For
  parallax, `ParallaxAccumulator.set_geometry` re-shifts. The bright-field pixels (centre,
  semiangle, angular sampling) can't change while running.

- **Only the screen crosses the wire.** The page asks for the region it shows, at the size it
  shows it (`/view`).
  - The server averages blocks of pixels down to about the screen's resolution, and sends native
    pixels once zoomed in.
  - So a 4096² snapshot (a 512² scan at ×8) costs the browser about what a 512² one does.
  - On the Mac, a full-field view of a 4096² channel takes ~50 ms to prepare.
- **Instant contrast.** The browser applies contrast, gamma, the colour map and invert on the GPU
  (WebGL2), so the sliders don't round-trip.
- **Controls:**
  - **The image:** one reconstruction panel, the parallax image by default. Wheel to zoom, drag
    to pan, double-click or `f` to fit, `1` for native pixels. A scale bar comes from the
    snapshot's `sampling_A`.
  - **Filters,** applied to the image and shown in its Fourier transform too:
    - the phase flip, sign(sin χ(q)): the pipeline's `phase_flipped` channel, computed from the
      same parallax sum;
    - Butterworth high- and low-pass (display only), set by their cutoff period d_c in Å: the
      high-pass removes spacings coarser than d_c, the low-pass those finer: H = 1/(1 +
      (q/q_c)^2n) and its complement, ½ at q_c = 1/d_c, the filters of quantem and of the
      pipeline's own `--lowpass` / `--highpass`. Order 12 by default, 2–16 in the "order" menu.
      The sliders are logarithmic, the high-pass from 2 Å up to the field of view, the low-pass
      from 0.3 to 50 Å. The Fourier transform marks each cutoff with a circle.
  - **The Fourier transform panel** (optional) shows the visible square at native pixels, up to
    2048², averaged to the panel's size.
    - The server computes it, as a real FFT in float32: about 35 ms at 2048² on an M4 Pro.
    - The page asks for it once the view has stopped moving.
    - The page applies the filters itself (log P + 2 log₁₀|H(q)|), so moving a slider redraws it
      at once.
    - It shows q and the d-spacing under the cursor, and has a scale bar in Å⁻¹.
    - A dashed outline on the image marks the square it transforms.
    - The phase flip changes only signs, so it leaves the power spectrum as it is.
    - While the image is still filling in (a scan's first pass, every pass of `reset`, an image
      restarted mid-run by a mode switch or a retune), it uses
      only the finished rows, from the snapshot's `complete_rows`. So does the frame's
      histogram. Otherwise the blank part and the edge of the scan would dominate both.
  - **Contrast:** limits from the whole frame's histogram, the visible view, or manual.
    - Manual starts from the limits shown; dragging a line on the histogram switches to manual.
    - Gamma, colour map and invert.
  - **Status:** the pass / progress line, pause (space), and a history slider over the kept
    snapshots (`--history-mb`, 512 MB by default).
  - **Export:** one PNG of the view (the image and, if open, its Fourier transform, with scale
    bars and a caption), and `.npy` of the shown channel at full resolution.
  - **Record:** a video, 15 frames/s by default (5–30), through the browser's `MediaRecorder`.
    - **page** (the default): the whole page with its controls, e.g. a supplementary video of
      the tuning. The browser asks once per recording to share this tab, and records the page as
      shown, at its own pixel size (up to 4K).
      - Screen capture needs a secure page: `http://127.0.0.1` or `localhost`, also through
        `ssh -L`. Elsewhere only "view" is offered.
      - Ending the share from the browser's bar also stops the recording.
    - **view:** the same composition as the PNG, redrawn at the chosen rate.
      - The frame size is fixed when recording starts. Opening the Fourier transform later
        letterboxes the composition into that size.
      - Keep the tab visible: browsers slow the timers of a hidden tab to about once a second.
    - **Format:** H.264 in `.mkv` in Chrome and Edge, VP9 `.webm` in Firefox, `.mp4` in Safari.
      The page writes the duration into the `.mkv` / `.webm`, which the browser leaves out.
- **Links:** URL parameters preset the controls, e.g.
  `?flip=1&fft=1&hp=50&lp=1.5&order=12&cmap=inferno&view=1to1` (`hp`, `lp`: cutoff periods in Å)
  (`ch=<channel>` picks another channel, e.g. `coverage`). `?snapshot[=seq]` shows one snapshot
  without the live stream.
- **Another computer:** keep the viewer next to the pipeline and forward its port
  (`ssh -L 8050:127.0.0.1:8050 <server>`): only screen-sized images travel. Or serve it with
  `--host 0.0.0.0`.
- **No new dependencies:** Python's standard library, numpy and pyzmq. The page loads nothing from
  the internet, since microscope computers are often offline.
- **Publisher cost:** the stream itself still carries full float32 snapshots. At 4096² × 2
  channels that is 134 MB per snapshot; on the Mac, `snapshot()` + `pack` take ~55 ms. Raise
  `--live-every` if the ingest falls behind.

## Watching in a notebook

`realtime_stem.streaming.notebook.LiveView` is a stream subscriber for Jupyter
(`pip install -e ".[notebook]"`); `examples/live_view.ipynb` is a worked example on a local replay.

- **Non-blocking.** `view.start()` receives on the kernel's asyncio loop (`zmq.asyncio`), so
  cells keep running while the scan fills in. `await view.wait_final()` waits for the scan's final
  frame; in `--continuous` mode the view keeps going with the next scan.
- **Backends.** With `%matplotlib widget` (ipympl) the figure redraws in place. With any other
  backend the output cell is re-rendered as a PNG on each update.
- **History.** The view keeps only the last `history` frames (64 by default, 0 = none), so a
  long session doesn't grow memory:
  - `view.snapshot_at(0.5)` returns the kept frame nearest half-scanned
    (`scan_pass=k` restricts it to one pass of a continuous scan);
  - `view.save_gif(path)` writes the fill-in as an animation;
  - `view.latest` is the newest `(stack, header)`.
- **Same rendering.** The desktop viewer and the notebook draw with the same `PanelFigure`
  (`streaming/display.py`): one panel per channel, robust contrast, progress / C10 / rotation in
  the title.
- **Subscribe first.** A PUB/SUB subscriber only sees frames sent after it connects. For a
  notebook on another machine, bind the publisher to all interfaces:
  `--live-endpoint tcp://0.0.0.0:5601`.

The publisher's socket lingers up to 1 s on close, so the final snapshot (sent just before the
runner exits) reaches connected subscribers rather than being dropped with the queue.

## Live channels (readout)

The hot path only deposits shifted bright-field values into one buffer, plus two small side
accumulators: the visits per scan position and each BF pixel's running sum (an
O(B·M) add per batch, 2–7% of `process()`). Everything linear and shift-invariant over the scan is
applied to the accumulated sum when a snapshot is taken (`processing/parallax/readout.py`), so it
costs nothing per frame.

**The frame mean.** By default each frame's bright-field values are added in as they are.
`--subtract-frame-mean` subtracts each frame's mean over its bright-field pixels first, as the
reference implementation does. That mean is the bright-field intensity at the scan position, so
subtracting it removes the amplitude contrast (mass-thickness, diffraction contrast) over regions
larger than about the probe, together with frame-to-frame changes in beam intensity. On a recorded
ARINA scan (256×256, largest shift 6 px) it removed nearly all of the image's power below 6 cycles
per field of view and 40% between 12 and 24, and lowered the image's standard deviation by a
quarter; detail finer than that is unchanged. Use it when only the phase contrast is wanted, or
when the beam intensity drifts during the scan.

| channel / flag | what it is |
|---|---|
| `parallax` | the sum, with each BF pixel's running scan mean removed (`--no-dc-correct` to skip) and the optional filters below |
| `phase_flipped` | the same, times `sign(sin χ(q))` (`--no-phase-flip` to drop the channel) |
| `coverage` (`--coverage`) | deposits per output pixel |
| `--normalize` | divide by coverage, tapered below 1% of its peak (quantem's `mean · support`): a partial scan reads at full brightness and the phase flip rings less at the scan front |
| `--lowpass`, `--highpass`, `--butterworth-order`, `--band-limit` | Butterworth filters (1/Å); the band limit clamps the low-pass at 2α/λ |
| `--suppress-nyquist {auto,on,off}` | zero the Nyquist row/column (auto: when `--upsampling` > 1) |

Why each is exact:

- **Phase flip.** The streaming reference scatters a grouped real-space kernel per frame
  (`parallax_phase_flip_accumulate_cpu`). Its kernel is the same for every BF pixel, so the
  operator is a convolution of the parallax sum, and by linearity filtering the partial sum gives
  the streamed phase-flipped image at that moment, mid-scan included (pinned to float precision in
  `tests/test_readout.py`). It costs one FFT pair per snapshot (2–4 ms on Apple M4 Pro `mps` at
  1024², ~25 ms on its CPU) instead of a U×M scatter per frame, and the grouped kernel would need
  ~12.6 GB at 1024². The reference also wraps the *flat* index `% (Ny·Nx)`, which carries a column
  overflow into the next row (a ~120% error on the apoferritin data); the readout wraps each axis.
- **Per-BF scan-mean removal** (quantem's `_preprocess`). Subtracting each BF pixel's mean from
  all its deposits is a convolution of the comb of position visits with the means placed at their
  shifts, so it is applied at readout. Together with `--subtract-frame-mean` this is
  double-centering. On a complete, unupsampled wrap it only removes a constant (every output pixel
  receives each BF pixel once); the amplitude contrast stays. With the frame mean subtracted, on
  apoferritin it changes a half scan by 22% (25–37% next to the scan front, where the static BF pattern piles up) and a complete
  upsampling-2 scan by 10% (removing 42% of its Nyquist power).
- **Coverage** is the comb convolved with the shift histogram.

The live cadence is `--live-every` seconds (0.2 ≈ 23 rows of a 1024² scan at 120 kHz) or, with
`--live-every-rows N`, every N scan rows. `result()` stays the raw sum (`parallax_recon.npy`).

## Detector orientation

The geometry describes the detector by the scan/detector rotation plus an optional column flip
(LiberTEM-parallax's convention). A transposed or mirrored detector is one of 8 layouts; name it
with `--detector-layout` (raw axis 0, raw axis 1) and the matching rotation is added to
`--rotation` and the flip set:

| `--detector-layout` | raw frame | rotation added | column flip |
|---|---|---|---|
| `rows,cols` | canonical | 0 | no |
| `rows,-cols` | `c[:, ::-1]` | 0 | yes |
| `-rows,cols` | `c[::-1, :]` | +180° | yes |
| `-rows,-cols` | `c[::-1, ::-1]` | +180° | no |
| `cols,rows` | `c.T` | +90° | yes |
| `cols,-rows` | `c.T[:, ::-1]` | +90° | no |
| `-cols,rows` | `c.T[::-1, :]` | −90° | no |
| `-cols,-rows` | `c.T[::-1, ::-1]` | −90° | yes |

`--flip-cols` sets the flip alone. `--center` is always measured on the raw frames; without it,
`--detector-layout` maps the canonical frame center onto the raw frame. That is why no even-grid
parity correction is needed (LiberTEM-parallax's `n_rot` block, which also fires for any
|rotation| ≥ 45°): a reversed even axis moves the center by one pixel, and the center carries it.
`tests/test_orientation.py` reproduces the canonical reconstruction exactly for all 8 layouts,
including non-square and odd detectors and the apoferritin data.

## Continuous scanning (live tuning)

The tuning workflow runs as follows:

1. Set the scan (e.g. 256×256), the *target* aberrations (`--defocus`, `--rotation`, `--c12`,
   `--phi12`) and the readout (cadence, `--lowpass`; both the parallax and the phase-flipped
   channel are always computed).
2. Start an acquisition that scans continuously.
3. The operator turns focus, stigmators and scan rotation on the microscope until the image is
   best. At that point the microscope matches the targets.

That needs `--accumulate latest`:

```bash
python -m realtime_stem.apps.run --accumulate latest --scan 256 256 --binned \
    --energy <eV> --semiangle <mrad> --angular-sampling <mrad/px> --scan-sampling <Å> \
    --defocus=<Å> --rotation <degrees> --center <row> <col> --lowpass 0.1 --live-every-rows 32 \
    --endpoint tcp://<DCU_IP>:31001 --n-connections 3 --live-endpoint tcp://0.0.0.0:5601
```

- **What the image holds.** Each scan position keeps only its newest frame. When a position is
  scanned again, its new deposits minus its stored ones are added at the same output pixels (the
  shifts don't change). So the image is always exactly the reconstruction of the last full pass:
  mid-pass, the current pass down to the scan cursor and the previous pass below it. A knob change
  is fully reflected one pass later (0.55 s for 256² at 120 kHz, 8.7 s for 1024²). The newest
  bright-field values are stored on the device: positions × M × 2 bytes for uint16 frames (the
  run prints the size), e.g. 160 MB for 256² and 2.6 GB for 1024² at M ≈ 1250. The cost is one
  extra gather per batch, ~1.2–1.5× the plain sum (`scripts/bench_accumulator.py`).
- **How the detector numbers frames.** For one long series, image ids run past the scan and are
  mapped back onto it (`id % positions`). For one series per pass, add `--continuous`: the same
  reconstruction is kept across series, and each series continues the pass count. Either way the
  live metadata carries `pass` and `progress` (the scan cursor), shown in the viewer title as
  "pass 3 at 47% (latest)". The final frame is published only when the run ends.
- **`--accumulate all`** keeps every frame of every pass (averaging once the microscope is
  settled); the default `scan` is one reconstruction per series, as before.
- **`--accumulate reset | swap`** start a fresh image with each pass (pass = `id // positions`)
  and keep the last completed pass as the saved image. Neither needs `latest`'s store: one more
  image instead (positions × bright-field pixels × 2 bytes → Ny·U × Nx·U × 4 bytes).
  - `reset` shows the current pass alone, filling in.
  - `swap` shows the current pass on the output rows it has completed and the last completed
    pass elsewhere. A row is complete once every scan row whose shifted deposits land on it is
    done, the torus wrap included. So the image is never partial, and the seam is no wider than
    the parallax shifts.
  - `--pass-average β` blends each completed pass into the shown one
    (front ← (1 − β)·front + β·pass): a running average for movies of a settled area.
  - Frames of a pass that arrive after the next one started (stragglers across connections) go
    into the completed pass (`swap`) or are dropped (`reset`); the snapshot metadata counts them
    (`late_frames`).
- **Movies: `--save-passes`.**
  - With any repeated-scan mode, each completed pass's channels are written to
    `<out-dir>/passes/parallax_<channel>_passNNNN.npy` (float32, or `--save-passes-dtype
    float16`), off the hot path.
  - With `latest` the image at a pass boundary is exactly that pass.
  - A pass completes when the next one starts, or when its series ends on its last position.
- **Trying it on a recording.** `python -m tools.repeat_dstream scan.dstream loop.dstream
  --passes 8` loops a recorded scan into one series of 8 passes (`--series per-pass` for a series
  each, `--drift DY DX` to move the specimen between passes). Play it with `run.py --source
  loop.dstream …`, or replay it over ZeroMQ. `scripts/compare_accumulate.py loop.dstream <flags>`
  shows each mode's live image part-way through each pass, against the true pass images:
  - On the binned Gd₂O₃ scan looped 4× with a (3, 4) px drift per pass, halfway through a pass
    `latest` and `swap` correlate 0.51 / 0.49 with the pass in progress and 0.55 / 0.56 with the
    one before.
  - `reset` correlates 0.69 with the pass in progress, with half the image empty.
  - `all` blurs the passes together.
  - At each pass's end, every mode but `all` equals that pass.
- **A focus walk on a recording** (for demos of live tuning).
  `python -m tools.refocus_dstream scan.dstream walk.dstream --flags scan.flags
  --defocus-schedule -500 -450 -400 -350 -300 -316 -316 -316` writes pass k as the recording would
  look at microscope defocus C10_k.
  - Under the parallax model a defocus change ΔC10 only moves each bright-field pixel's image by
    ΔC10·α (rotated, flipped as the geometry says), so each pixel's image is moved by that, in
    whole scan pixels.
  - Reconstruct with the recording's own flags (the target) and the passes sharpen as the
    schedule reaches it.
  - Checked with the calibration: a pass written at −500 Å from a −316 Å recording measures
    −501 Å.
- **Batch size.** With `latest`, a batch must not hold the same position twice, so
  `--batch-size` ≤ scan positions (enforced).
- **Cadence.** "Rows per update" is `--live-every-rows N`; the alternative is `--live-every`
  seconds. Each snapshot costs a few ms on the GPU.
- **Notebook history.** The notebook view keeps only its last `history` frames (default 64,
  `LiveView(endpoint, history=N)`, 0 = none), so a long tuning session doesn't grow memory.

**What the operator should look at.** This was simulated with `tools/simulate_4dstem.py
--passes 8 --defocus -2000 --defocus-schedule -4000 -3500 -3000 -2500 -2000 -1500 -2000 -2000`:
a 128² scan whose defocus walks towards the −2000 Å target, overshoots, and comes back.

- **Far from the target**, each bright-field image lands somewhere else, so the particles
  ghost into doubled shells. That is obvious by eye.
- **Within ~±500 Å** (a few depths of field; λ/α² ≈ 200 Å at 10 mrad and 300 keV) the change is
  subtle by eye, and a display low-pass hides it further. Judge fine detail with little or no
  `--lowpass`.
- **A scalar figure of merit is unambiguous.** It is the parallax variance loss: the spread of the
  shifted bright-field images at each output pixel (quantem's `residual_loss`). Its complement,
  "BF consistency" (1 − loss):

  | microscope C10 (Å) | −4000 | −3500 | −3000 | −2500 | −2000 | −1500 | −2000 | −2000 |
  |---|---|---|---|---|---|---|---|---|
  | BF consistency | 0.0009 | 0.0009 | 0.0012 | 0.0022 | 0.0038 | 0.0020 | 0.0039 | 0.0038 |

  It peaks at the target from both sides; the overshoot drops back to the −2500 Å level. It isn't
  in the live stream yet; it would cost one more scatter per batch.

## Commissioning on the microscope

The automated suite covers the mechanics on simulated data, on cpu/mps, and on CUDA where available:

- `tests/test_continuous.py` checks the cases that matter for tuning:
  - `latest` equals the newest frame per position, for whole and partial passes, any batching,
    one long series or one series per pass, and upsampling 1 and 2;
  - no drift over 40 passes;
  - `all` equals the sum of every pass;
  - the runner keeps one image across series and publishes the final frame only at the end;
  - on a simulated focus walk the live image converges to, then equals, the target
    reconstruction.
- `tests/test_simulate.py` pins the simulator's conventions.

What only the real system can show, in order:

1. **On the GPU machine, before the session.** Run `pytest tests/` (the CUDA variants run there).
   Then replay a multi-pass capture of the session geometry through the Holoscan ingest at full
   rate:
   - make it with `python -m tools.simulate_4dstem --scan 256 256 --passes 10
     --defocus-schedule … --format dstream --out tuning256`;
   - reconstruct with `run.py $(cat tuning256.flags)` and the viewer;
   - feed it with `tools/replay_to_zmq.py --fast`.

   Check that the sink rate is ≥ 120 kHz with `--accumulate latest`, all frames arrive, the live
   cadence holds, and GPU memory stays flat. Run `scripts/bench_accumulator.py` for the per-batch
   cost.
2. **How the DCU numbers a continuous scan.** Arm it exactly as the session will (SerialEM /
   DCU settings) and run with `--accumulate latest` and the viewer. Check:
   - no `image_id >= scan positions` errors;
   - the pass counter steps and the cursor sweeps 0→100% once per scan, in step with the beam;
   - the sink's frame count matches what the DCU reports (no drops);
   - whether ids run on in one series, or restart per series (then add `--continuous`).
3. **Frames map onto raster positions in every pass.** Scan a static, featureful area for
   many passes and compare the pass images (save them, or take `view.snapshot_at(1.0,
   scan_pass=k)` in a notebook). Real drift moves the image smoothly by sub-pixel amounts. A jump
   or roll by whole pixels or rows from pass to pass means the scan generator sends frames that
   aren't positions (fly-back or settling triggers), which `id % positions` cannot absorb.
4. **Orientation and center.**
   - Record one scan at a known defocus with
     `python3 tools/record_stream.py capture --ip <DCU_IP> --out s1.dstream` (with nothing else
     listening).
   - Measure it with `python -m tools.calibrate_parallax s1.dstream --scan … --energy …
     --semiangle … --angular-sampling … --scan-sampling … --rotation-prior <the rotation used live>
     --out s1_cal`.
     - It gives the disk centre on the raw frames and the flagged pixels, with whether they fall in
       the disk.
     - It measures the shift of each small group of bright-field pixels and fits the parallax
       geometry: column flip (the sign of the fit's determinant), rotation, C10 and C12. The 8
       layouts are one flip or none plus a rotation.
     - It checks the fit with the variance loss and writes `s1_cal.flags` for `run.py`, with a
       diagnostic plot.
     - It refines C10 on the lattice. The shifts are measured at low frequencies, where a support
       film dominates. A range of ×4 reconstructions then picks the C10 at which the lattice spots
       beyond the scan's Nyquist frequency stand out most, and `--defocus` gets that value.
       Without a lattice it keeps the fit; `--no-refine` skips the step.
     - The shifts can't tell (C10, θ) from (−C10, θ + 180°). `--rotation-prior` picks the twin, so
       the sign of C10 rests on knowing the rotation within ±90°.
     - A departure from a quadratic aberration surface (C30, C21, C23, or radial distortion of the
       pattern) is reported but not used. Parallax keeps χ(k ± q) to first order in q, and only for
       quadratic χ is the dropped term the same for every bright-field pixel.
   - Or replay it (`run.py --source s1.dstream …`) and compare the 8 layouts and a rotation sweep
     by eye.
5. **Knob sign and scale.** Set the target C10 to the microscope's readout. Sweep the focus knob
   through it in known steps; the best image should sit at the readout. An offset or mirror tells
   you the defocus sign convention (C10 = −Δf in abTEM's convention) or a center/rotation error.
   Repeat for stigmation (target C12 = 0) and scan rotation.
6. **Latency.** Step the focus and time until the live image settles: it should be one pass plus
   one live interval.
7. **Long run.** Tune continuously for 30–60 minutes at full rate. Check that the rate, GPU
   memory and the stream stay steady.

## Simulated data

`tools/simulate_4dstem.py` makes parallax datasets of any size with known ground truth (torch, on
cuda/mps/cpu; a 256×256 scan of 96×96 frames takes ~40 s on an M4 Pro):

```bash
python -m tools.simulate_4dstem --scan 256 256 --out out/sim256 \
    [--defocus -2000 --rotation -15 --semiangle 10 --angular-sampling 0.5 --scan-step 2 \
     --dose 2000 --detector-layout rows,cols --c12 ... --phi12 ...]
```

It is a single-slice, pure-phase simulation. Shells and blobs on a weak background form a
periodic phase object; the probe is `IFFT(A exp(-iχ))` (abTEM's convention, C10 < 0 =
underfocus). The far-field intensity is integrated into detector pixels on a 3–5× finer k-grid,
rotated into the detector frame, re-laid per `--detector-layout` and Poisson-sampled.

**Repeated scans** model a tuning session: `--passes N` scans the same area N times in one
series (image ids run on past the scan, as from a continuously scanning detector), and
`--defocus-schedule` / `--c12-schedule` set the microscope's *actual* C10 / C12 on each pass while
`--defocus` / `--c12` stay the algorithm's target. The `.flags` then include `--accumulate latest`.

It writes `<out>.npy` and/or `<out>.dstream` (real Stream2 framing, bslz4), `<out>_phase.npy`
(the ground truth on the scan grid), `<out>.json` and `<out>.flags`. The `.flags` file holds the
exact `run.py` geometry flags, including the effective scan step (positions sit on the simulation
grid).

`tests/test_simulate.py` pins that the simulator and the reconstruction share conventions. The
parallax criterion is quantem's variance loss: the spread across the shifted BF images at each
output pixel. It is smallest at the simulated C10 and rotation, not at the opposite sign or ±10°.

## Benchmark

`scripts/bench_accumulator.py` times `process()` on device-resident batches, the share spent on
the readout side accumulators, and one `snapshot()`, with device synchronization. On an Apple M4
Pro, 2048-frame batches of 96×96 frames, 1024² scan:

| device | M (BF pixels) | `--accumulate all` | `--accumulate latest` | side accumulators | snapshot |
|---|---|---|---|---|---|
| mps | 1249 | ~970 kHz | ~635 kHz | 2% | ~2–4 ms |
| mps | 49 | ~5 MHz | ~5–9 MHz | 7–10% | ~2–4 ms |
| cpu | 1249 | ~360 kHz | ~300 kHz | 2% | ~25 ms |

Even a laptop GPU accumulates far faster than 120 kHz: the end-to-end rate is set by the
ingest, not the parallax arithmetic (see "Batch size and rate" and "Scheduler" under Known
issues). For the end-to-end rate on a given machine, replay a recording with
`scripts/check_install.sh`.

## Known issues and limitations

- **A wrong DCU address looks like waiting for the scan.** ZeroMQ connects lazily and retries
  forever, so the pipeline and the recorder cannot fail on a wrong address. Both print
  `WARNING: cannot reach <ip>:<port>` at start-up and every 30 s until it answers.
  `DCU=<DCU_IP> scripts/check_install.sh` tells a wrong address (no answer to ping) from a
  closed stream port.
- **Which GPU.** `--gpus "device=N"` gives the container exactly that GPU, numbered as in
  `nvidia-smi`. With `--gpus all` on a machine with several GPUs the pipeline takes the first in
  CUDA's order, which can differ from nvidia-smi's; it then prints a warning. At start-up the run
  prints the GPU it uses (`[run] GPU: <name>, PCI bus <bus>`, the bus as in nvidia-smi's Bus-Id),
  and a note when other programs already hold much of its memory.

- **The run reports whether the scan is complete.** After every scan: `[sink] COMPLETE: all N
  scan positions received`, or `WARNING: INCOMPLETE SCAN: … (k missing, d frames for positions
  that already had one)`. The result is saved either way. Causes of an incomplete scan: a second
  reader of the DCU stream, a scan started too soon after the previous one, frames left over from
  a crashed run (the log then says the series' start message was not seen), or detector-side
  drops.
- **Scans in quick succession.** Between scans the pipeline is rebuilt (about a second). The
  connection to the DCU stays open meanwhile, so the next scan's frames wait in its receive queue
  (up to 50,000 messages per connection) and in the DCU's buffer; none are lost when scans follow
  each other directly (tested at 120 kHz with no gap). Frames of the next series are never mixed
  into the current one: the receiver tells them apart by `series_id`.
- **The batch size is capped at compile time** (`kMaxBatchSize` in
  `operators/cpp/common/constants.h`, 2048). Larger `--batch-size` values are clamped, with a
  warning in the log. Any smaller size works, but small batches are slower (see the next point).
- **Batch size and rate.** Replaying a 1024×1024 scan of binned 96×96 beam-on frames on an RTX 6000
  Ada (one connection): batch 512 → 69 kHz, 1024 → 114 kHz, 2048 → 160 kHz, all frames received.
  The default is therefore 2048 (the cap); GPU memory is reserved for the cap whatever the batch.
  These numbers are for the 4-thread `MultiThreadScheduler`: the limit at small batches is its
  polling (idle workers sleep up to 5 ms), not the GPU.
- **Scheduler.** Since 2.0.2 the default is Holoscan's single-threaded scheduler
  (`--scheduler-threads 0`): it runs the three operators one after another without sleeping, while
  ZeroMQ keeps receiving in its own I/O thread. On the test machine (IGX Orin, RTX 6000 Ada) it
  was faster in every test (1024²
  binned: full speed 166–168 kHz against 149–151 with 4 threads; paced at 120 kHz 120.5 kHz
  against 113.5, i.e. 4 threads end a scan about 0.5 s late; busy machine 117–120 against 96–114)
  and uses less CPU. `--scheduler-threads 4` gives the `MultiThreadScheduler` of 1.0.0–2.0.1.
- **Connections and network.** The DCU paces each TCP connection to about 60k beam-on binned
  frames per second, so 120 kHz needs 3 connections (the default since 2.0.1; two connections
  cap near 112 kHz, and the DCU then drops frames once its buffer of about 65k frames is full).
  The link from the DCU's data interface must be 10 Gbit/s or faster: 120 kHz of binned beam-on
  frames is about 370 MB/s (3 Gbit/s), so a 1 Gbit/s link caps near 40 kHz. Unbinned frames have
  four times as many pixels but run at most at 30 kHz (the DCU's shortest unbinned frame time is
  33.3 µs), so their data rate is of the same order. `DCU=tcp://<DCU_IP>:31001
  scripts/check_install.sh` prints the link speed.
- **Busy machine.** With 8 of the test machine's 12 cores kept busy by other programs, every scan
  was still complete. With the 4-thread scheduler the pipeline fell up to 2 s behind by the end of a
  1024² scan at a paced 120 kHz (the backlog waits in the receive queues, room for 150k frames,
  then in the DCU's buffer, which is finite); the single-threaded default kept up (117–120 kHz).
  On a shared server, also keep the container on cores of its own (`docker run --cpuset-cpus`),
  on the CPU socket next to the GPU.
- **uint32 streams are handled in code but have not been run live** through this sink. The DCU
  sends uint32 when the frame time is long (a few kHz); `--dtype uint32` configures the ops and the
  engine, and the engine's uint32 path is unit-tested, but the only live validation is uint16.
- **DECTRIS Stream2 only.** The ingest understands DECTRIS CBOR messages with bslz4 payloads (the
  DCU's stream compression must be `bslz4`). A different detector needs a different ingest; the
  processing stage plugs in behind the `Accumulator` contract and needs no change.
- **Start the pipeline before the scan.** While no receiver is connected, the DCU drops the
  frames it streams (a pipeline started during a scan receives only the rest of it, and the run
  reports the scan as incomplete). The DCU counts
  them: `GET http://<DCU_IP>/stream/api/1.8.0/status/dropped`. In `--continuous` mode the
  connection stays open between scans.
- **Continuous scanning assumes one frame per scan position.** `--accumulate latest | all |
  reset | swap` map image id → position as `id % positions`. A scan generator that adds fly-back or settling
  triggers between lines or passes breaks that mapping (commissioning test 3 detects it).
- **`--accumulate latest` costs memory and time.** It keeps every position's newest frame on the
  device (positions × bright-field pixels × 2 bytes: 160 MB at 256², 2.6 GB at 1024²) and costs
  about 1.2–1.5× the plain sum. Check the rate on your GPU with a replay
  (`scripts/check_install.sh`) before relying on it at full detector rate.
- **One consumer per DCU stream.** The DCU's PUSH socket load-balances across every connected PULL
  peer, so a second listener (or a crashed one's leftover connection) silently takes a share of
  the frames. The run's completeness report shows it as missing positions.
- **Frame order.** Each frame is placed by its `image_id` (raster index), so the order in which
  frames arrive does not matter, but the scan must deliver exactly one frame per position in raster
  order of `image_id`. Extra frames (e.g. flyback) shift the mapping; the `start` check warns when
  the detector announces more images than `--scan` has positions.

## Adding an algorithm

The pipeline depends only on the two contracts in `realtime_stem/processing/base.py`, so a new
method (e.g. strain mapping) is self-contained:

1. Implement an `Accumulator`: `process(frames, ids)` folds a batch into the reconstruction-so-far;
   `result()` returns it; expose `detector_shape` / `scan_shape`. Optionally add
   `snapshot() -> (stack, channel_names, meta)` to stream several named images per live frame.
2. Implement an `Algorithm`: declare its CLI flags, build the accumulator from them, save results.
3. Register it in `realtime_stem/processing/registry.py`.

`run.py --algorithm <name>` and the ingest graph need no changes. `processing/parallax/` is the
reference implementation to copy from.

## Layout

```
realtime_stem/
  ingest/holoscan_graph.py     ZmqRxOp → DecompressGpuOp → GenericSinkOp (drives any Accumulator)
  ingest/python_ingest.py      the pure-Python ingest: ZMQ or .dstream/.npy, CPU decode
  ingest/codec.py              DECTRIS Stream2 CBOR + bslz4 decode/encode on the CPU
  ingest/sink.py               BatchSink: the hot-path bookkeeping both ingests share
  ingest/dstream.py            the .dstream capture format
  processing/base.py           the Accumulator + Algorithm contracts
  processing/registry.py       --algorithm name → Algorithm
  processing/parallax/         engine (geometry + accumulator), readout (live channels),
                               orientation (detector layouts), config, algorithm
  streaming/live_stream.py     pack/unpack + PUB/SUB (CONFLATE) transport
  streaming/display.py         PanelFigure: the live-frame rendering shared by the viewers
  streaming/notebook.py        LiveView: the Jupyter subscriber
  streaming/web/               the browser viewer: server.py + static/index.html (WebGL2)
  apps/run.py                  the runner (transport flags + continuous/rounds loop)
tools/                         .dstream CLI, stream recorders, replay feeder, npy→.dstream, 4D-STEM
                               simulator (incl. multi-pass focus walks), synthetic-capture generator,
                               geometry calibration from a recording (calibrate_parallax.py),
                               DCU setup (detector_setup.py)
scripts/                       machine check, replay demos (Docker / local), accumulator benchmark,
                               continuous-mode comparison, the standard session analysis
                               (session_report.py), CLI-reference generator
(sessions/)                    recorded sessions, one folder each in one layout: in
                               EM_Realtime_Processing, not in this folder
examples/                      viewer, notebook (live_view.ipynb), headless probe, static result plot
tests/                         parallax oracle tests, readout/orientation/codec/ingest/simulator
                               tests, streaming transport tests, runner CLI tests
operators/                     the C++/CUDA Holoscan operators (ZmqRxOp, DecompressGpuOp), MIT
docker/Dockerfile              builds the complete image: Holoscan base + operators + pyzmq
docs/GUIDE.md                  this guide
docs/CLI.md                    every command-line parameter (generated: scripts/gen_cli_reference.py)
CHANGELOG.md                   what changed between versions
```

## Tests

Pure-Python (no detector, no container). From the repo root:

```bash
pip install -e ".[local,dev,oracles]"
python -m pytest tests/
```

Device-parametrized tests run on cpu, and on `mps` / `cuda` where available; the GPU-parity test
skips cleanly without CUDA.

- The parallax tests pin the engine against a float64 transcription of the reference math and
  pin the CLI unit conversions.
- The readout tests pin the live channels against the streamed reference kernel, a batch
  double-centering oracle and brute-force coverage.
- The orientation tests check all 8 detector layouts.
- The codec tests check against bitshuffle, the DECTRIS decoder and the HDF5 bitshuffle filter
  (the `oracles` extra; skipped if absent).
- The ingest tests replay over ZMQ, including the multi-connection end race and back-to-back
  scans, and round-trip the recorder.
- The continuous-scanning tests pin `--accumulate latest` / `all` (see "Commissioning on the
  microscope"), `tests/test_pass_modes.py` `reset` / `swap` / `--pass-average` / `--save-passes`
  and the looping tool, the notebook-view tests the bounded history and pass-aware snapshots.
- `tests/test_cli_reference.py` fails if `docs/CLI.md` no longer matches the parsers, or a
  parameter has no description.
- The simulator tests pin the shared conventions through the parallax variance criterion.
- A stub-Holoscan test drives the `GenericSinkOp` wrapper without the container.

## Credits and relation to the reference

The real-time parallax / tcBF algorithm is the work of Georgios Varnavides, as are the live
phase flip, the detector orientation and the continuous-scanning modes added in 2.1.0. This
implementation follows his streaming-parallax reference and reproduces its output on the public
apoferritin dataset (Zenodo 10.5281/zenodo.18346854) to float32 precision. Implementation choices:

- The scatter wraps `% Ny` and `% Nx` independently.
- The phase flip is applied to the accumulated image at each snapshot rather than per frame; it
  is the same operator (see "Live channels").
- The one-pixel bright-field-index correction for quarter-turn rotations is not applied, because
  here the disk position is the measured `--center` (see "Detector orientation").

The per-BF scan-mean removal, the moments behind `--normalize`, and the Fourier filters follow
quantem's `DirectPtychographyMontage`.

The GPU ingest (`operators/`) is built on the Holoscan operators from
[electronmicroscopy/live4d](https://github.com/electronmicroscopy/live4d) (Denis Leshchev,
NVIDIA; MIT, see `operators/LICENSE`) with the generic-geometry, header-driven block size and
multi-connection extensions described under Prerequisites.
