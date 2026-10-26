# Notes for AI coding assistants

This file orients an assistant (Claude Code, Codex, Copilot, …) working in this folder on a
user's machine. Humans: start with `README.md`; the full description is `docs/GUIDE.md`. The
repository root has its own [`AGENTS.md`](../AGENTS.md) for the shared Holoscan 5 pipeline
(`holoscan_pipeline/`, `dectris/`); read it too before changing anything there.

## What this is

A real-time STEM reconstruction pipeline. A DECTRIS ARINA detector's control unit (the DCU)
streams frames; the pipeline receives them on an NVIDIA GPU, folds them into a parallax / tcBF
image while the scan runs, and publishes the image so far (parallax and phase-flipped) for a live
viewer or notebook. A pure-Python ingest runs the same processing on CPU or Apple `mps`, without
Docker, from a stream or a recording.

There are two Docker images and two transports (README, "Which image, which transport"):

- **Holoscan 5 image** (`docker/Dockerfile.hsdk5`, tag `realtime-stem:hsdk5`, 3.0.0): the
  repository's shared receivers, `--transport zmq` (bslz4-compressed Stream2, decompressed on the
  GPU) or `--transport rdma` (imagemux RDMA, uncompressed, written straight into a GPU ring).
- **Holoscan 4 image** (`docker/Dockerfile`, tag `realtime-stem:latest`, 2.2.0's ingest): this
  folder's own operators, ZMQ only. Kept because it is what ran live first and it takes uint32
  frames; it does not run on Blackwell GPUs (its PyTorch lacks sm_120).

The pipeline only **reads** the DCU's stream. The one exception is the helper
`tools/detector_setup.py`, which sets up and arms the DCU when the user runs it. Nothing here
drives the scan generator or the microscope.

## Where is what

Three stages, each changeable on its own:

| Stage | Code | Language | Rebuild the image after a change? |
|---|---|---|---|
| ingest, Holoscan 5 | receivers in `../holoscan_pipeline/cpp/operators/` (`ZmqRxOp`, `DecompressGpuOp`, `RdmaRxOp`) and `../dectris/` (RDMA connection); wired in `realtime_stem/ingest/hsdk5_graph.py` | C++/CUDA, Python | `../holoscan_pipeline/`, `../dectris/`: **yes**; `hsdk5_graph.py`: no |
| ingest, Holoscan 4 | `operators/` (`ZmqRxOp`, `DecompressGpuOp`), wired in `realtime_stem/ingest/holoscan_graph.py` | C++/CUDA, Python | `operators/`: **yes**; the rest: no |
| ingest, no Holoscan | `ingest/python_ingest.py` + `codec.py` (CPU decode) | Python | no |
| processing: the reconstruction | `realtime_stem/processing/` (`base.py` contracts, `registry.py`, `parallax/`) | Python + PyTorch | no |
| stream-out + viewing | `realtime_stem/streaming/` (`live_stream.py`, `display.py`, `notebook.py`, `web/`: the browser viewer), `examples/viewer.py`, `examples/live_view.ipynb` | Python (+ the viewer's page: JS/WebGL2) | no |

All three ingests hand batches to the same `ingest/sink.py` (`BatchSink`), so the processing and
the viewer do not know which ingest or transport is running.

- `realtime_stem/apps/run.py`: the program that is run. It owns the transport flags (`--transport`,
  endpoint, `--rdma-…`, batch, scheduler, connections, live preview, `--continuous`) and never
  names an algorithm. `holoscan_generation()` tells the images apart (5: `live4d` importable;
  4: `dectris_holoscan_ops`). `--viewer-port` serves the browser viewer in process, with live
  control of the reconstruction (`apps/control.py`; the algorithm declares what is tunable:
  `Algorithm.live_params` / `retune`). Changes are applied under `ingest/sink.ACC_LOCK`, between
  batches.
- `realtime_stem/ingest/hsdk5_graph.py`: `Hsdk5Session` holds one connection and one running
  Holoscan graph for the whole run; each scan is a `round(acc)`. The next series' START waits
  until a round has handed over its accumulator, so a series never lands in the previous scan's
  image and nothing reconnects between scans. `GenericSinkOp` feeds `BatchSink` and releases each
  batch to the receiver (`_PublicationWindow`).
- `realtime_stem/processing/parallax/`: `config.py` (the physics flags and their units),
  `engine.py` (`preprocess_geometry`: bright-field pixels and their shifts, NumPy;
  `ParallaxAccumulator`: the scatter-add, `--accumulate` modes), `readout.py` (the live channels:
  phase flip, scan-mean removal, filters), `orientation.py` (the 8 detector layouts),
  `algorithm.py` (CLI wiring and saving results).
- `tools/`: the `.dstream` recording format, a stream recorder (`record_stream.py`, host Python
  with pyzmq + cbor2; stop the pipeline while it records), the ZMQ replay feeder
  (`replay_to_zmq.py`), the RDMA stand-in sender (`rdma_sender.py`: imagemux protocol on port
  41003, Holoscan 5 image only), a 4D-STEM simulator (`simulate_4dstem.py`), the geometry
  calibration from a recording (`calibrate_parallax.py`: centre, flip, rotation, C10/C12 as
  `run.py` flags; read only, offline), recording converters/generators (`make_dstream.py`,
  `npy_to_dstream.py`, `repeat_dstream.py`, `refocus_dstream.py`), `check_gpu_copies.py` (a
  hardware check) and `detector_setup.py` (the DCU's ZMQ settings over its REST API: `check`,
  `apply`, `arm`, `disarm`, `scan`; standard library only).
- `scripts/check_install.sh`: the step-by-step machine check (DCU reachability and link speed,
  GPU, image, GPU copies, replays; `IMAGE=` selects either image; ZMQ only).
  `scripts/run_replay_demo.sh`: the no-detector demo (Holoscan 4 image).
- `sessions/<date>_<specimen>/` (in [EM_Realtime_Processing](https://github.com/kucukogluberk/EM_Realtime_Processing/tree/v2.2.0/sessions), not in this folder):
  recorded sessions, one layout each (`sessions/README.md`). Take a new session's acquisition
  values from that session, never from another one.
- `docker/Dockerfile.hsdk5` (build from the **repository root**: it copies `holoscan_pipeline/`
  and `dectris/`) and `docker/Dockerfile` (build from this folder).
- `tests/`: unit tests (parallax math against a float64 reference, the stream transport, the
  runner's CLI, the C frame parser, the recording tools).

## How it runs

The pipeline runs inside a Docker image; this folder is bind-mounted into the container at
`/work` (`-v "$PWD":/work`). So **a Python edit takes effect at the next `docker run`**; only a
change to compiled code needs `docker build` again (a few minutes): `../holoscan_pipeline/` or
`../dectris/` for the Holoscan 5 image, `operators/` for the Holoscan 4 image. The viewer and the
ZMQ replay feeder can run on the host (`pip install -e ".[viz,replay]"`). The live commands are
in `README.md`; `run.py --help` and `docs/CLI.md` list every flag. Without Docker:
`pip install -e ".[local,viz,replay]"`, then `run.py --ingest python` (see
`scripts/run_local_demo.sh`).

Run containers with `--user $(id -u):$(id -g)`, or the files they write belong to root. RDMA
also needs `--device /dev/infiniband --ulimit memlock=-1:-1` and `--network host`.

## Setting up on a new machine

Ask the user before running anything that waits for frames. Do not guess these, and do not take
them from examples, older notes or another machine:

1. **Which image and transport.** Holoscan 5 with ZMQ or RDMA, or Holoscan 4 (ZMQ). Check the
   driver (`nvidia-smi`: CUDA 13 for the Holoscan 5 image) and the GPU generation (Blackwell needs
   the Holoscan 5 image).
2. **The DCU's addresses.** ZMQ: the stream address (IP, and the port if not 31001), which can
   differ from the address of the DCU's web page. RDMA: the DCU's RDMA address (port 31003), this
   machine's address on the same RDMA network, and the verbs device (`ibv_devinfo`, e.g.
   `mlx5_0`). Also the DCU's REST address for `detector_setup.py`.
3. **Which GPU** to use, as `nvidia-smi` numbers them. Show the user the `nvidia-smi` list
   (name, memory in use) and let them choose, also when there is only one. With RDMA, show
   `nvidia-smi topo -m` too: a GPU on the RDMA card's NUMA node is best.
4. **The image tag** they built (`docker images realtime-stem`).
5. **The acquisition values**, which have no defaults: `--energy` (eV), `--semiangle` (mrad),
   `--angular-sampling` (mrad/px of the mode running), `--scan-sampling` (Å), `--scan`,
   `--defocus` (Å), `--rotation` (degrees). Ask for each; never fill one in with a typical
   value. `--center` (disk centre, px on a raw frame) is optional: the frame center by default.
   The disk radius is not an input (semiangle ÷ angular sampling; the run prints it).
   Keep them in a file (`site.flags`) and pass `$(cat site.flags)`.

Then run `GPU=<N> DCU=<DCU_IP> IMAGE=<tag> scripts/check_install.sh` (ZMQ checks; for RDMA it
still checks the GPU, the image and a ZMQ replay). The pipeline prints the GPU it uses
(`[run] GPU: …`); compare it with the user's choice.

On a **shared machine**, give every container you start a name of its own (`--name`), check
`docker ps` and `nvidia-smi` before starting, and never stop, remove or reuse the name of a
container or image you did not start: it can be someone's live acquisition.

Do not wait on a silent command. If the pipeline or the recorder prints `cannot reach` or
`no RDMA sender answers`, or nothing arrives within a minute of the user starting a scan, stop and
check the addresses, the port, the GPU and `docker ps` (a leftover container) with the user.

## The detector's settings

The DCU's compression must match the transport. The pipeline cannot change it.

| | ZMQ | RDMA |
|---|---|---|
| stream | `stream/mode` `enabled`, `stream/format` `cbor` | `bstream/mode` `cbor-rdma`, `bstream/rdma_batch_size` (e.g. 64), `bstream/rdma_device` (the DCU's RDMA port; must not be empty); `stream/mode` `disabled` |
| compression | `detector/compression` `bslz4` | `detector/compression` `none` |
| both | binning `2x2` for `--binned`, trigger mode `exts`, `ntrigger` = Ny×Nx, `nimages` = 1 | same |

`tools/detector_setup.py` checks and sets the ZMQ column exactly:

```bash
python -m tools.detector_setup check  --ip <DCU_IP> --scan 256 256 --binned   # read only
python -m tools.detector_setup apply  --ip <DCU_IP> --scan 256 256 --binned --frame-time 8.334e-6
python -m tools.detector_setup arm    --ip <DCU_IP>     # after the pipeline has printed "ready"
python -m tools.detector_setup disarm --ip <DCU_IP>     # after the scan: ends the series
python -m tools.detector_setup scan   --ip <DCU_IP> --scan 256 256 --binned   # arm, scan, disarm
```

For RDMA it does not know the `bstream` settings, and its `check` reports compression `none` as a
difference; set the RDMA column by hand (README, "Before a session"). `arm`, `disarm` and the
trigger settings work the same for both.

- `check` only reads, so you may run it to diagnose. It ends with the matching `run.py` flags
  (`--scan`, `--binned`, `--dtype`).
- `apply`, `arm`, `disarm`, `scan` and any REST `PUT` change the detector: run them only when the
  user asks for that specific run, or let the user run them. A DCU can be shared (another
  person's RDMA or ZMQ setup); agree with its other users before changing its configuration.
- `--ip` is the DCU's REST address; the stream and RDMA addresses can be other interfaces.
- Order for every scan: pipeline running and `[holoscan] ready …` printed → arm → scan → disarm.
  The DCU sends the series' START at arm and its END at disarm.
- The scan generator is site-specific and not part of this repository. `start_scan()` in the
  helper is an empty slot (pseudo-code only) for the site's own code; until it is filled in,
  `scan` asks the user to start the scan by hand. Never start a scan or move the beam from here.

## Checking a change

1. **Unit tests.** Locally (no GPU needed): `pip install -e ".[local,dev,oracles]"` and
   `python -m pytest tests`. Inside the Holoscan 5 image (the CUDA and Holoscan 5 tests run there;
   the extra packages are for the oracle and notebook tests):

   ```bash
   docker run --rm --gpus "device=<N>" -e HOME=/tmp -v "$PWD":/work -w /work \
     --entrypoint bash realtime-stem:hsdk5 -c "pip install -q bitshuffle h5py hdf5plugin \
     matplotlib ipympl ipywidgets && python3 -m pytest -q -p no:cacheprovider tests"
   ```

   The same with `realtime-stem:latest` tests the Holoscan 4 image. The shared pipeline's own
   tests are under `../holoscan_pipeline/tests` and `../dectris/tests` (see the root `AGENTS.md`).
2. **The whole pipeline without a detector**, per transport (README, "Testing the Holoscan 5
   image without a detector"): ZMQ with `tools/replay_to_zmq.py`, RDMA with `tools/rdma_sender.py`
   (needs an RDMA card; sender and pipeline on the same card work). A full-size replay of a
   recorded scan: `RECORDING=<file.dstream> SCAN="1024 1024" BINNED=1 FLAGS="$(cat site.flags)"
   IMAGE=<tag> scripts/check_install.sh`. Expect `[sink] COMPLETE: all N scan positions received`
   and a `TOTAL … rate` above 120 kHz.
3. **A change to the ingest or the reconstruction:** replay the same recording before and after,
   or through both transports, or through both images, and compare the saved
   `out/parallax_recon.npy` files. The reconstruction does not depend on the transport:
   `parallax_recon.npy` from ZMQ, RDMA and the Holoscan 4 image is bit-identical on the same
   frames (`parallax_corrected.npy` differs by less than 1e-6, relative, between the images: cuFFT 12 vs 13).
4. **Several scans in one run:** `tools/repeat_dstream.py --series per-pass` makes a recording
   with one series per pass (each with its own `series_id`, as the DCU numbers its arms);
   `run.py --continuous --rounds N` must print `COMPLETE` N times.

Keep a working version to return to (`git stash` / a branch, or the image tag of a release) before
changing anything used live.

## Rules that the code relies on

- **One reader per DCU stream.** ZMQ: the DCU deals frames round-robin to every connected reader;
  a second reader (another tool, a forgotten container) silently takes a share. RDMA: the DCU's
  sender serves one receiver at a time. Stop the other reader first.
- **Start the pipeline before the scan**, and arm only after `[holoscan] ready …`. The DCU drops
  frames while no reader is connected.
- **Recordings and simulators must send a full START** (`image_size_x`, `image_size_y`,
  `image_dtype`, `number_of_images`, `series_unique_id`) and END, as a DCU does: the Holoscan 5
  receivers reject a minimal START as "Unknown Stream2 message". Use
  `codec.encode_start_message` / `encode_end_message`. Each series needs its own `series_id`.
- The acquisition values (`--energy`, `--semiangle`, `--angular-sampling`, `--scan-sampling`,
  `--scan`, `--defocus`, `--rotation`) are required: `Algorithm.check_args` stops the run and
  lists the missing ones. Do not add defaults for them. `--center` defaults to the frame center.
- The per-frame mean is kept unless `--subtract-frame-mean` is given (it removes the amplitude
  contrast). Do not make the subtraction the default again.
- `--scan NY NX` must equal the scan; frames are placed by their `image_id` (raster index), so
  arrival order does not matter but extra frames (e.g. flyback) do.
- The frame size and type must match the detector mode: `--binned` (96×96) or `--detector`, and
  `--dtype uint16|uint32`. A mismatch stops the run with an error. **The Holoscan 5 image takes
  uint16 only**; the DCU sends uint32 for long dwells (seen at 250 µs; 120 kHz down to about
  30 kHz gave uint16).
- **RDMA batches:** `RdmaRxOp`'s tensors keep the full batch capacity; only the first
  `len(image_ids)` frames are valid. Slice before use (`GenericSinkOp` does).
- **ZMQ connections:** `--n-connections 3` (default) is needed for 120 kHz; the DCU paces each
  connection (about 60k binned frames/s). With more than one, the shared `ZmqRxOp` tolerates
  messages reordered across connections (END before the last images, the next series before this
  END); keep that when changing it.
- **Batch size:** the Holoscan 5 receivers take at most 1000 frames per batch (`MAX_BATCH`;
  larger values are clamped with a message). The Holoscan 4 operators are capped at 2048
  (`kMaxBatchSize`, `operators/cpp/common/constants.h`), the tested default there.
- `--scheduler-threads 0` (single-threaded, default) is the tested, fastest setting (Holoscan 4).
- **The DCU's RDMA stream state** (`bstream/api/1.8.0/status/state`, read only) must be `ready`
  before arming. In `error` the DCU drops every message (`status/dropped` counts them), START
  included, while the pipeline sits at "waiting for a scan" and shows no error of its own. An
  empty `bstream/rdma_device` or an earlier failed series leaves it there; re-arming does not
  clear it. The fix is `bstream/command/initialize`, then setting `mode`, `rdma_batch_size` and
  `rdma_device` again (README, "If something goes wrong"). Diagnose with GETs; let the user run
  the reset.
- `--rdma-ring-memory gpu` (default) needs GPUDirect RDMA on the host (`nvidia_peermem`);
  `host_mapped` works without it. Both were run live.
- `--accumulate latest | all | reset | swap` map image id → position as `id % positions` (pass =
  `id // positions`), so they need exactly one frame per scan position; `latest` also needs
  `--batch-size` ≤ the scan positions, and `swap`'s live composite assumes raster order.
- A change to a command-line flag needs `python scripts/gen_cli_reference.py` (a test checks
  that `docs/CLI.md` is current).
- After every scan the runner prints `COMPLETE` or `WARNING: INCOMPLETE SCAN …`; that line is the
  test of a live run.

## Adapting to a machine

- Which GPU: `--gpus "device=N"` (the pipeline uses one GPU). With `--gpus all` on a machine
  with several GPUs the pipeline takes the first in CUDA's order, which can differ from
  nvidia-smi's numbering. GPU type missing from the build ("no kernel image is available"):
  `docker build --build-arg CUDA_ARCHITECTURES="…"` (Holoscan 5 default `86;89;90;120`,
  Holoscan 4 default `80;86;89;90`).
- Rate below the detector's on a shared or two-socket server: `docker run --cpuset-cpus …` on the
  socket next to the GPU (`nvidia-smi topo -m`).
- Podman: `--device nvidia.com/gpu=all` instead of `--gpus all`. SELinux enforcing: add `:z` to
  the `-v` mounts. Rootless Docker: `--network host` is not the host network, so a viewer outside
  the container cannot connect.
- Building behind a proxy: configure the Docker daemon's proxy, and pass
  `--build-arg HTTP_PROXY=… --build-arg HTTPS_PROXY=…`.
- Viewer on another computer: `ssh -L 8050:127.0.0.1:8050 <server>` and open
  http://127.0.0.1:8050 there (only screen-sized images travel). The matplotlib viewer works the
  same way through `ssh -L 5601:127.0.0.1:5601`.

Diagnosis: `README.md`, "If something goes wrong"; `docs/GUIDE.md`, "Known issues and
limitations"; and `scripts/check_install.sh`.

## Repository conventions

- This is a public repository. Keep personal notes, internal hostnames beyond the documented
  examples, and credentials out of it.
- Do not commit, push, publish or deploy without the user's explicit OK.
- Keep this folder's changes separate from changes to the shared pipeline (`holoscan_pipeline/`,
  `dectris/`): a fix there goes to `main` on its own, so the other users of those receivers get it.
