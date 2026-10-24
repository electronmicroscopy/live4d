# Notes for AI coding assistants

This file orients an assistant (Claude Code, Codex, Copilot, …) working in this repository on a
user's machine. Humans: start with `README.md`; the full description is `docs/GUIDE.md`.

## What this is

A real-time STEM reconstruction pipeline. A DECTRIS ARINA detector's control unit (the DCU)
streams compressed frames over ZeroMQ; the pipeline receives them, decompresses them on an NVIDIA
GPU, folds them into a parallax / tcBF image while the scan runs, and publishes the image so far
(parallax and phase-flipped) for a live viewer or notebook. A pure-Python ingest runs the same
processing on CPU or Apple `mps`, without Docker, from a stream or a recording. The pipeline only
**reads** the DCU's stream. The one exception is the helper `tools/detector_setup.py`, which sets
up and arms the DCU when the user runs it. Nothing here drives the scan generator or the
microscope.

## Where is what

Three stages, each changeable on its own:

| Stage | Code | Language | Rebuild the image after a change? |
|---|---|---|---|
| ingest: receive + decompress | `operators/` (`ZmqRxOp`, `DecompressGpuOp`), wired in `realtime_stem/ingest/holoscan_graph.py`; or `ingest/python_ingest.py` + `codec.py` (CPU); both feed `ingest/sink.py` | C++/CUDA (operators), Python | operators: **yes**; the rest: no |
| processing: the reconstruction | `realtime_stem/processing/` (`base.py` contracts, `registry.py`, `parallax/`) | Python + PyTorch | no |
| stream-out + viewing | `realtime_stem/streaming/` (`live_stream.py`, `display.py`, `notebook.py`), `examples/viewer.py`, `examples/live_view.ipynb` | Python | no |

- `realtime_stem/apps/run.py`: the program that is run. It owns the transport flags (endpoint,
  batch, scheduler, connections, live preview, `--continuous`) and never names an algorithm.
- `realtime_stem/processing/parallax/`: `config.py` (the physics flags and their units),
  `engine.py` (`preprocess_geometry`: bright-field pixels and their shifts, NumPy;
  `ParallaxAccumulator`: the scatter-add, `--accumulate` modes), `readout.py` (the live channels:
  phase flip, scan-mean removal, filters), `orientation.py` (the 8 detector layouts),
  `algorithm.py` (CLI wiring and saving results).
- `tools/`: the `.dstream` recording format, a stream recorder (`record_stream.py`, host Python
  with pyzmq + cbor2; stop the pipeline while it records), a replay feeder (`replay_to_zmq.py`),
  a 4D-STEM simulator (`simulate_4dstem.py`), recording converters/generators,
  `check_gpu_copies.py` (a hardware check) and `detector_setup.py` (the DCU's settings over its
  REST API: `check`, `apply`, `arm`, `disarm`, `scan`; standard library only).
- `scripts/check_install.sh`: the step-by-step machine check (DCU reachability and link speed,
  GPU, image, GPU copies, replays). `scripts/run_replay_demo.sh`: the no-detector demo.
- `docker/Dockerfile`: builds the image (NVIDIA's pinned Holoscan base + the compiled operators).
- `tests/`: unit tests (parallax math against a float64 reference, the stream transport, the
  runner's CLI, the C frame parser).

## How it runs

The pipeline runs inside the Docker image; this folder is bind-mounted into the container at
`/work` (`-v "$PWD":/work`). So **a Python edit takes effect at the next `docker run`**; only a
change under `operators/` needs `docker build` again (a few minutes). The viewer and the replay
feeder run on the host (`pip install -e ".[viz,replay]"`). The live command is in `README.md`;
`run.py --help` and `docs/CLI.md` list every flag. Without Docker:
`pip install -e ".[local,viz,replay]"`, then `run.py --ingest python` (see `scripts/run_local_demo.sh`).

## Setting up on a new machine

Ask the user before running anything that waits for frames. Do not guess these, and do not take
them from examples, older notes or another machine:

1. **The DCU's stream address** (IP, and the port if not 31001). It is the address of the DCU's
   data interface on the network to this machine, which can differ from the one used for the
   DCU's web page.
2. **Which GPU** to use, as `nvidia-smi` numbers them. Show the user the `nvidia-smi` list
   (name, memory in use) and let them choose, also when there is only one.
3. **The image tag** they built (`docker images realtime-stem`).
4. **The acquisition values**, which have no defaults: `--energy` (eV), `--semiangle` (mrad),
   `--angular-sampling` (mrad/px of the mode running), `--scan-sampling` (Å), `--scan`,
   `--defocus` (Å), `--rotation` (degrees). Ask for each; never fill one in with a typical
   value. `--center` (disk centre, px on a raw frame) is optional: the frame center by default. The disk radius is not an input (semiangle ÷
   angular sampling; the run prints it).
   Keep them in a file (`site.flags`) and pass `$(cat site.flags)`.

Then run `GPU=<N> DCU=<DCU_IP> IMAGE=<tag> scripts/check_install.sh`. It checks the DCU first;
every later command uses the values it prints at the end (`--gpus "device=N"`,
`--endpoint tcp://<DCU_IP>:31001`). The pipeline prints the GPU it uses (`[run] GPU: …`);
compare it with the user's choice.

Do not wait on a silent command. If the pipeline or the recorder prints `cannot reach`, or
nothing arrives within a minute of the user starting a scan, stop and check the address, the
port, the GPU and `docker ps` (a leftover container) with the user.

## The detector's settings (`tools/detector_setup.py`)

The pipeline needs the DCU set up as follows: stream `enabled`, format `cbor`, compression
`bslz4`, binning `2x2` for `--binned`, trigger mode `exts`, `ntrigger` = Ny×Nx, `nimages` = 1.
The helper checks and sets exactly these:

```bash
python -m tools.detector_setup check  --ip <DCU_IP> --scan 256 256 --binned   # read only
python -m tools.detector_setup apply  --ip <DCU_IP> --scan 256 256 --binned --frame-time 8.334e-6
python -m tools.detector_setup arm    --ip <DCU_IP>     # after the pipeline has started
python -m tools.detector_setup disarm --ip <DCU_IP>     # after the scan: ends the series
python -m tools.detector_setup scan   --ip <DCU_IP> --scan 256 256 --binned   # arm, scan, disarm
```

- `check` only reads, so you may run it to diagnose. It ends with the matching `run.py` flags
  (`--scan`, `--binned`, `--dtype`).
- `apply`, `arm`, `disarm` and `scan` change the detector: run them only when the user asks for
  that specific run, or let the user run them.
- `--ip` is the DCU's REST address; the stream address (`run.py --endpoint`) can be another
  interface of the DCU.
- Order for every scan: pipeline running → arm → scan → disarm. The DCU sends the series' `start`
  message at arm and its `end` at disarm.
- The scan generator is site-specific and not part of this repository. `start_scan()` in the
  helper is an empty slot (pseudo-code only) for the site's own code; until it is filled in,
  `scan` asks the user to start the scan by hand. Never start a scan or move the beam from here.

## Checking a change

1. Unit tests. Locally (no GPU needed): `pip install -e ".[local,dev,oracles]"` and
   `python -m pytest tests`. Inside the image (the CUDA tests run there):
   `docker run --rm --gpus all -v "$PWD":/work -w /work --entrypoint bash realtime-stem:latest -c "python3 -m pip install -q pytest && python3 -m pytest -q -p no:cacheprovider tests"`
2. The whole pipeline without a detector: `scripts/run_replay_demo.sh`, or a full-size replay of a
   recorded scan: `RECORDING=<file.dstream> SCAN="1024 1024" BINNED=1 FLAGS="$(cat site.flags)"
   scripts/check_install.sh` (FLAGS: the recording's acquisition values).
   Expect `[sink] COMPLETE: all N scan positions received`.
3. A change to the reconstruction: replay the same recording before and after, and compare the
   saved `out/parallax_recon.npy` files.

Keep a working version to return to (`git stash` / a branch, or the image tag of a release) before
changing anything used live.

## Rules that the code relies on

- **One reader per DCU stream.** The DCU deals frames round-robin to every connected reader; a
  second reader (another tool, a forgotten container) silently takes a share. Stop it first.
- **Start the pipeline before the scan.** The DCU drops frames while no reader is connected.
- The acquisition values (`--energy`, `--semiangle`, `--angular-sampling`, `--scan-sampling`,
  `--scan`, `--defocus`, `--rotation`) are required: `Algorithm.check_args` stops the run and
  lists the missing ones. Do not add defaults for them. `--center` defaults to the frame center.
- The per-frame mean is kept unless `--subtract-frame-mean` is given (it removes the amplitude
  contrast). Do not make the subtraction the default again.
- `--scan NY NX` must equal the scan; frames are placed by their `image_id` (raster index), so
  arrival order does not matter but extra frames (e.g. flyback) do.
- The frame size and type must match the detector mode: `--binned` (96×96) or `--detector`, and
  `--dtype uint16|uint32`. A mismatch stops the run with an error.
- `--n-connections 3` (default) is needed for 120 kHz: the DCU paces each connection.
- `--batch-size` is capped at 2048 at compile time (`kMaxBatchSize`,
  `operators/cpp/common/constants.h`); 2048 is the tested default.
- `--scheduler-threads 0` (single-threaded, default) is the tested, fastest setting.
- `--accumulate latest | all` map image id → position as `id % positions`, so they need exactly
  one frame per scan position; `latest` also needs `--batch-size` ≤ the scan positions.
- A change to a command-line flag needs `python scripts/gen_cli_reference.py` (a test checks
  that `docs/CLI.md` is current).
- The DCU's stream compression must be `bslz4`.
- After every scan the runner prints `COMPLETE` or `WARNING: INCOMPLETE SCAN …`; that line is the
  test of a live run.

## Adapting to a machine

- Which GPU: `--gpus "device=N"` (the pipeline uses one GPU). With `--gpus all` on a machine
  with several GPUs the pipeline takes the first in CUDA's order, which can differ from
  nvidia-smi's numbering. GPU type missing from the build
  ("no kernel image is available"): `docker build --build-arg CUDA_ARCHITECTURES="…"`.
- Rate below the detector's on a shared or two-socket server: `docker run --cpuset-cpus …` on the
  socket next to the GPU (`nvidia-smi topo -m`).
- Podman: `--device nvidia.com/gpu=all` instead of `--gpus all`. SELinux enforcing: add `:z` to
  the `-v` mounts. Rootless Docker: `--network host` is not the host network, so a viewer outside
  the container cannot connect.
- Building behind a proxy: configure the Docker daemon's proxy, and pass
  `--build-arg HTTP_PROXY=… --build-arg HTTPS_PROXY=…`.
- Files the container writes belong to root; delete them with sudo or from a container.
- Viewer on another computer: `ssh -L 5601:127.0.0.1:5601 <server>`, then
  `python examples/viewer.py tcp://127.0.0.1:5601` there.

Diagnosis: `docs/GUIDE.md`, "Known issues and limitations", and `scripts/check_install.sh`.
