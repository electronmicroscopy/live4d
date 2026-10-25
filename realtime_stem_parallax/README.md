# realtime-stem

> **About this folder.** realtime-stem 2.2.0: real-time parallax / tilt-corrected bright-field
> (tcBF) reconstruction for 4D-STEM on a DECTRIS ARINA, with a browser viewer that can fine-tune
> the reconstruction while the scan runs. It is the version after the first live sessions in
> October 2026; its history, and the recorded sessions with their analysis, are in
> [kucukogluberk/EM_Realtime_Processing](https://github.com/kucukogluberk/EM_Realtime_Processing)
> (tag `v2.2.0`). It is self-contained: it builds its own Docker image (Holoscan 4.4, CUDA 12)
> and does not use the rest of this repository. Its GPU ingest (`operators/`) is an earlier fork
> of this repository's Holoscan operators, kept as it was used live; the current operators are in
> [`holoscan_pipeline/`](../holoscan_pipeline/). Bringing the two together is discussed in
> [#9](https://github.com/electronmicroscopy/live4d/issues/9). Run every command below from this folder.

This program makes a parallax (tcBF) image **while the scan is running**. The ARINA detector
sends its frames over the network, an NVIDIA GPU unpacks them, and every frame is added to the
image as soon as it arrives. You watch the image fill in, in a browser. It keeps up with the ARINA
at full speed (120,000 frames per second, 2×2 binned; measured on an NVIDIA RTX 6000 Ada).

![The browser viewer: the Reconstruction section's sliders, the parallax image phase-flipped and filtered, and its Fourier transform](docs/images/web_viewer.jpg)

In the browser you can:

- turn on the **phase flip**, high-pass and low-pass filters, and the **Fourier transform** of
  what is on screen;
- **fine-tune the defocus, the rotation and the astigmatism** (C10, rotation, C12) while the scan
  runs. The values you started with remain the target for the microscope's knobs;
- save a PNG, the full image as `.npy`, or **record a video** of the page.

The same processing runs on a laptop without a detector or an NVIDIA GPU, on recorded or
simulated scans (see "Try it without a detector").

More detail: [`docs/GUIDE.md`](docs/GUIDE.md) (the full guide),
[`docs/WORKFLOWS.md`](docs/WORKFLOWS.md) (from a recording to a live demo: calibration, offline
reconstruction, looping a scan, a focus-walk demo) and [`docs/CLI.md`](docs/CLI.md) (every
option).

## What is where

```
realtime_stem/        the Python package
  apps/run.py           the program you run (with --viewer-port it also serves the viewer)
  apps/control.py       live tuning from the viewer
  ingest/               receives the detector frames
  processing/parallax/  the parallax / tcBF reconstruction
  streaming/            publishes the live image; web/ is the browser viewer
operators/            the C++/CUDA frame receiver and GPU decompressor
docker/Dockerfile     builds the Docker image
examples/             matplotlib viewer, notebook, sample recording
tools/                detector setup, stream recorder, replay, simulator, geometry calibration,
                      looping and refocusing a recording
scripts/              installation check, demos, the standard session analysis
tests/                unit tests
```

Recorded sessions and their analysis (Gd₂O₃ nanoparticles, an Au grating) are in
[EM_Realtime_Processing/sessions](https://github.com/kucukogluberk/EM_Realtime_Processing/tree/v2.2.0/sessions).

## Try it without a detector

On a laptop (CPU, Apple silicon or an NVIDIA GPU), without Docker. This simulates a 256×256 scan
(about a minute), reconstructs it and serves the viewer:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[local,viz,replay]"
python -m tools.simulate_4dstem --scan 256 256 --format dstream --out out/sim256
python -m realtime_stem.apps.run --ingest python --source out/sim256.dstream $(cat out/sim256.flags) \
    --accumulate latest --viewer-port 8050 --out-dir out
```

Open http://127.0.0.1:8050. The viewer stays up after the replay: change C10 or the rotation in
the Reconstruction section and the image is re-rendered from the frames it holds. Ctrl-C saves the
tuned image to `out/` and prints the tuned flags.

`scripts/run_replay_demo.sh` does the same in Docker on the GPU. [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md)
streams a recording in a loop, as a microscope scanning the same area would, and walks it through
focus.

## What you need

- **A Linux PC with an NVIDIA GPU** and a network connection to the ARINA's detector control
  unit (the DCU). We call it the *processing PC*.
- **An NVIDIA driver for CUDA 12 or newer.** Run `nvidia-smi`: it must show `CUDA Version: 12.x`
  or higher.
- **Docker** and the **NVIDIA Container Toolkit**, so that Docker can use the GPU.
- **Python 3.10 or newer**, for the detector commands (and the optional standalone viewers).
- **A web browser**, on the processing PC or on another computer (see "Watching from another
  computer").
- **Disk space:** the NVIDIA base image is large. Keep several tens of GB free.
- The image is compiled for A100, RTX 30xx / A6000, RTX 40xx / RTX 6000 Ada / L4 / L40 and
  H100 / H200 GPUs; it was tested on an RTX 6000 Ada. Older GPUs (V100, T4) need an extra build
  option: see
  [`docs/GUIDE.md`](docs/GUIDE.md), "Host requirements".

## Install

Do this once, on the processing PC.

**1. Check that Docker can see the GPU.**

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

It must print your GPU. If it fails, install or fix the NVIDIA Container Toolkit first.

**2. Get the code and go into this folder.**

```bash
git clone https://github.com/electronmicroscopy/live4d.git
cd live4d/realtime_stem_parallax
```

**3. Build the Docker image.** The first build downloads NVIDIA's base image (large; this can take
a while, depending on your network) and then compiles the GPU code (a few minutes).

```bash
docker build -t realtime-stem:latest -f docker/Dockerfile .
```

**4. Install the host tools** (the detector commands, and the optional standalone viewers) in a
Python virtual environment. (On recent Ubuntu, `pip install` outside a virtual environment is
refused.)

```bash
sudo apt install python3-venv python3-tk   # Ubuntu: virtual environments, the matplotlib window
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[viz,replay]"
```

In every new terminal, run `source .venv/bin/activate` again before using the detector commands.

**5. Find two things.**

- **Your GPU number:** run `nvidia-smi -L` and pick one (`GPU 0`, `GPU 1`, …).
- **The DCU's stream address:** the IP address of the DCU on the network to this PC. It can be
  different from the address of the DCU's web page; ask whoever set up the detector.

**6. Run the installation check.**

```bash
GPU=<N> DCU=<DCU_IP> scripts/check_install.sh
```

Every step prints `PASS`, `FAIL` or `SKIP`. At the end it prints the `--gpus` and `--endpoint`
values to use in the run command below.

## Before a session: set up the detector

The DCU must send its frames as `cbor` with `bslz4` compression, and one frame per scan position.
Run these in the virtual environment (`source .venv/bin/activate`). Check the settings (this only
reads them):

```bash
python -m tools.detector_setup check --ip <DCU_IP> --scan 256 256 --binned
```

If something is wrong, set it. `--frame-time` is the scan's dwell time per position in seconds
(8.334e-6 = 120 kHz); use the same value as on the scan generator.

```bash
python -m tools.detector_setup apply --ip <DCU_IP> --scan 256 256 --binned --frame-time 8.334e-6
```

You can also do this on the DCU's web page instead.

## Run an experiment

### The setup

- **The processing PC** runs this program, in two terminals and a browser:
  1. the pipeline (it receives the frames, builds the image and serves the viewer);
  2. the detector commands (arm, disarm);
  3. the viewer, a page in the browser.
- **The scan generator** (often on its own PC, with its own software) moves the beam. It sends
  one trigger to the detector at every scan position, and the detector takes one frame for
  each trigger.

### Step 1: write your values into a file

The program needs these values from the microscope. None of them has a default.

| option | what it is | unit |
|---|---|---|
| `--scan NY NX` | the scan size, in positions: the same as on the scan generator | positions |
| `--binned` | give it if the detector runs 2×2 binned (96×96 frames); leave it out for 192×192 | |
| `--energy` | the beam energy (200 kV = `200000`) | eV |
| `--semiangle` | the probe's convergence semi-angle | mrad |
| `--angular-sampling` | the angle per detector pixel, for the mode you use (a binned pixel is twice as large as an unbinned one) | mrad per pixel |
| `--scan-sampling` | the scan step: the field of view divided by the number of positions | Å |
| `--defocus` | the defocus; negative means underfocus (−30 nm = `-300`) | Å |
| `--rotation` | the rotation between the scan and the detector | degrees |

Two more options are optional:

- `--detector-layout`: whether the detector image is mirrored or turned relative to the scan.
  The default is `rows,cols`. If no defocus value gives a correct image, try `rows,-cols`
  (mirrored); all 8 options are in [`docs/CLI.md`](docs/CLI.md).
- `--center ROW COL`: the centre of the bright-field disk on the detector, in pixels. The default
  is the middle of the frame.

Put them on one line in a file, for example `site.flags`. These are example values; use yours:

```
--binned --scan 256 256 --energy 200000 --semiangle 30 --angular-sampling 1.066 --scan-sampling 1.98 --defocus=-300 --rotation=3.75
```

Write negative numbers with `=` (`--defocus=-300`), as above.

### Step 2: start the pipeline (terminal 1)

Always start it **before** the scan: the DCU throws frames away while nobody is listening.

```bash
docker run --rm --network host --gpus "device=<N>" --ulimit stack=33554432:33554432 \
  -v "$PWD":/work -v "$PWD/out":/out -w /work --entrypoint python3 \
  realtime-stem:latest -m realtime_stem.apps.run \
    --endpoint tcp://<DCU_IP>:31001 $(cat site.flags) \
    --viewer-port 8050 --continuous --out-dir /out
```

It prints the GPU it uses and the viewer's address, and then waits. If it prints
`WARNING: cannot reach …`, the DCU address is wrong or the network is down; no frames can arrive
until that is fixed.

### Step 3: open the viewer

Open http://127.0.0.1:8050 in a browser on the processing PC.

- **Image:** wheel to zoom, drag to pan, double-click to fit. The Fourier transform of what is on
  screen can be shown next to it.
- **Filters:** the phase flip, and a high-pass and low-pass set by their cutoff period in Å
  (Butterworth, order 12 by default). Their cutoffs are drawn as circles on the Fourier
  transform.
- **Contrast:** drag the lines on the histogram, or let it follow the image.
- **Reconstruction:** C10, the rotation and C12 (see "Setting the right values"), and the
  `--accumulate` mode.
- **Save:** a PNG of the view, the full image as `.npy`, or a video of the page (● record).

Instead of `--viewer-port`, the pipeline can publish the image with
`--live-endpoint tcp://127.0.0.1:5601`, for the standalone browser viewer
(`python -m realtime_stem.streaming.web tcp://127.0.0.1:5601`), the matplotlib window
(`python examples/viewer.py tcp://127.0.0.1:5601`) or the notebook
([`examples/live_view.ipynb`](examples/live_view.ipynb)).

### Step 4: arm, scan, disarm (terminal 2 and the scan generator)

In terminal 2, first run `source .venv/bin/activate`.

1. Arm the detector:
   `python -m tools.detector_setup arm --ip <DCU_IP>`
2. Start the scan on the scan generator. The image fills in in the viewer.
3. When the scan has finished, disarm the detector:
   `python -m tools.detector_setup disarm --ip <DCU_IP>`

(You can also arm and disarm from the DCU's web page or your own software.)

After the scan, terminal 1 prints `COMPLETE: all N scan positions received`. If it prints
`WARNING: INCOMPLETE SCAN`, some frames were lost (see "If something goes wrong").

For the next scan, repeat step 4. The pipeline keeps running (`--continuous`). The images are
saved in the `out/` folder after every scan. Stop the pipeline with Ctrl-C.

### Scanning again and again

If the microscope scans the same area over and over, add `--accumulate latest` to the pipeline
command. The image then keeps updating: every position shows its newest frame. Two other modes
start every scan afresh: `swap` shows the last complete scan while the next one fills in, and
`reset` shows only the scan in progress. The viewer can switch between them. The detector must
expect all those frames: either arm it for every scan, or set its number of triggers to
(scan positions × number of scans).

### Setting the right values

The reconstruction is only correct when `--defocus`, `--rotation` and `--center` match the
microscope: the program uses them to work out how far to shift each detector pixel's image before
adding it. With wrong values the shifts are wrong, and so is the image.

- **The values in `site.flags` are the target.** Turn the microscope's knobs (focus, stigmators)
  until the image is sharp.
- **Fine-tune in the viewer.** The Reconstruction section has a slider for C10, the rotation and
  C12 (as a and b, like stigmators). This helps above all with the rotation, which the microscope
  can only change with a scan rotation.
  - With `--accumulate latest` the image is re-rendered at once from the frames it holds. The
    other modes start a new image.
  - "copy flags" gives the values in use, to paste into `site.flags`. Ctrl-C saves the tuned
    images and prints the same flags.
- **Measure them from a recorded scan** with `tools/calibrate_parallax.py`: the disk centre, the
  mirror, the rotation, C10 and C12. See [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md).

### Watching from another computer

On the other computer, open a tunnel to the processing PC, then open http://127.0.0.1:8050 there:

```bash
ssh -L 8050:127.0.0.1:8050 <user>@<processing-PC>
```

Only screen-sized images travel, so large images stay smooth. Recording the page works through
the tunnel too.

## If something goes wrong

- **Nothing arrives:**
  - check the DCU address (`WARNING: cannot reach …`);
  - check that the detector is armed;
  - check that no other pipeline is still running (`docker ps`).
- **`INCOMPLETE SCAN`:** only one program may read the detector's stream at a time. A second one
  (another tool, a forgotten container) silently takes part of the frames. Stop it.
- **The run stops with a frame-size error:** `--binned` does not match the detector's binning.
- **The DCU's compression** must be `bslz4` (its default). `detector_setup check` shows it.
- **Files in `out/` belong to root** (Docker writes them). Delete them with `sudo`.
- More in [`docs/GUIDE.md`](docs/GUIDE.md), "Known issues and limitations".

## For AI coding assistants

[`AGENTS.md`](AGENTS.md) explains this folder to an assistant (Claude Code, Codex, Copilot, …),
including how to set it up on a new machine.

## Credits

The real-time (streaming) parallax implementation is the work of Georgios Varnavides
(Varnavides et al., *Microsc. Microanal.* 32(2), ozaf139). The GPU ingest in
`operators/` builds on [electronmicroscopy/live4d](https://github.com/electronmicroscopy/live4d)
(Denis Leshchev, NVIDIA; MIT license, see `operators/LICENSE`).
