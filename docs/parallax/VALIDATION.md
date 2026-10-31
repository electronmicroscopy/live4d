# Validation records

## Live OBF, 2026-10-31

Live optimum bright field (OBF) beside parallax, the pacing that keeps it from stalling the
parallax branch, and the viewer changes, on two TU Delft hosts, jupiter and saturn. Each has two
NVIDIA A40 GPUs (46 GB), driver 580 with CUDA 13.0, two AMD EPYC 9454 processors (192 threads) and
1.5 TB of memory. Neither has Docker, so the `parallax` target was built with Docker on a Mac
(`--platform linux/amd64`, `CUDA_ARCHITECTURES=86`), saved with `docker save` and run as an
Apptainer 1.5.4 image (`apptainer exec --nv`). The final image (`sha256:73f1dca0`) was built from
this branch's code, which has changed since only in `--obf-share`'s help text, the documentation
and the benchmark helpers. Neither host has a verbs device, so only ZMQ was run and RDMA is
unverified. The source was DECTRIS daqsim `daqsim-and-rdma-26.10.22.48491-g15b27e55d8`, also
under Apptainer, on the receiver's host.

### Correctness

- The suite in the image on one A40 (`holoscan_pipeline/tests`) gives 421 passed, 47 skipped and
  1 failed. The failure, `test_codec.py::test_sample_capture_matches_hdf5_filter`, imports h5py
  into a process that already loaded the native writer's HDF5 1.10, and it passes alone. The
  skips need matplotlib, quantem, bitshuffle, the apoferritin capture or an RDMA device.
- This covers `test_stream.py`'s live graph cases, among them OBF configured on, OBF chosen over
  HTTP during a series, and the service-mode tick applying a retune and finishing OBF after END.
- On macOS the CPU suites (`tests/parallax`, `test_pipeline_settings.py`) pass but for
  `test_telemetry.py::test_resource_sampler_handles_unsupported_gpu`, which reads `/proc`.

### Real frames through daqsim

daqsim's test images are zeros, which compress 144 to 1, cost almost nothing to decode and make
meaningless images. So daqsim also replayed real frames. Its filewriter wrote a series at the
chosen frame time (2×2, BSLZ4), and `tests/benchmarks/dstream_into_daqsim.py` replaced each
frame's chunk with one of a `.dstream` recording, without recompressing, since a Stream2 BSLZ4
blob is a bitshuffle HDF5 chunk. A sample of frames read back through the HDF5 filter matched
live4d's own decoding pixel for pixel. The recordings are the 2026-10-20 Gd₂O₃ binned scan (ARINA,
256² positions, 30 mrad, about 2.25 kB a frame) and focus walks made from it with
`live4d.parallax.tools.refocus_dstream`. The parallax settings are that scan's calibration
(EM_Realtime_Processing `sessions/2026-10-20_Gd2O3/calibration/binned.flags`), with `accumulate:
latest`.

### Stress at 120,000 fps: target not met

3,600,000 frames of 96×96 at 2×2 binning in 30 s, BSLZ4, one ZMQ connection, batch 512 and queue
depth 4, after a 128-frame warm-up series, on jupiter without the viewer. daqsim raises a frame
time of 1/120000 s to its minimum at 2×2, 8.334 µs (119,990 fps). These runs predate the pacing
changes below.

| Source | Branches | Processed | Dropped by daqsim | Delivered |
| --- | --- | ---: | ---: | ---: |
| zeros | BF/ADF, parallax `scan`, 2000×1800 | 3,434,222 | 165,778 | 112.4 kfps |
| zeros | BF/ADF, parallax `latest` ×2 | 3,323,983 | 276,017 | 109.0 kfps |
| Gd₂O₃ walk | BF/ADF, parallax `latest` ×2, OBF off | 1,723,739 | 1,876,261 | 55.7 kfps |
| Gd₂O₃ walk | BF/ADF, parallax `latest` ×2, OBF on | 1,864,140 | 1,735,860 | 60.2 kfps |
| Gd₂O₃ walk | BF/ADF only | 1,772,495 | 1,827,505 | 57.2 kfps |

With real frames BF/ADF alone delivers what parallax and OBF do, so the limit lies upstream of the
processing. Host RAM and GPU memory stayed flat in every run.

### One ZMQ connection carries about 55 thousand real frames a second

A counter that only receives (`tests/benchmarks/zmq_probe.py`) took daqsim's stream during the
same replays.

| Host | Replay | Connections | Received | Dropped by daqsim |
| --- | --- | ---: | ---: | ---: |
| jupiter | Gd₂O₃ walk at 119,990 fps | 1 | 54.9 k/s | 1,891,566 of 3,600,000 |
| jupiter | Gd₂O₃ walk at 119,990 fps | 3 | 116.9 k/s in all | 42,352 of 3,600,000 |
| jupiter | zeros at 119,990 fps | 1 | 116.3 k/s | 43,744 of 3,600,000 |
| saturn | Gd₂O₃ walk at 100,000 fps for 30 s | 1 | 57.5 k/s | 1,207,774 |

One connection carries about 140 MB/s on both hosts, which is 55 to 57 thousand of these frames a
second. Three connections reach about 117 thousand messages a second, daqsim's own ceiling here.
The same imagemux runs on a DCU. The 2026-10-20 session needed three connections because the DCU
paced each one (issue #9), as its recording's interleaved image ids show. So 120 kHz of real
frames from an ARINA over ZMQ needs a receiver with several connections, or RDMA.

### What Live4D sustains on one socket

`tests/benchmarks/paced_send.py` sends a recording over one PUSH socket at a set rate. PUSH waits
for the receiver rather than dropping, so a rate below the one asked for means the receiver fell
behind. Six series of the 524,288-frame focus walk, on saturn, with OBF on from the start, its
weights precomputed, and the viewer's server publishing snapshots but no browser:

| Upsampling | Rate | Received | Missing | Delivered | OBF |
| --- | ---: | ---: | ---: | ---: | --- |
| ×2 | 100 kHz | 3,145,728 | 0 | 100.0 kfps | live from the 2nd series |
| ×4 | 100 kHz | 3,145,728 | 0 | 100.0 kfps | live from the 3rd series |
| ×4 | 120 kHz | 3,145,728 | 0 | 119.9 to 120.4 kfps | live from the 3rd series, its share scaled to 0.625 |

Unpaced (`replay_to_zmq --fast`) at ×2 with OBF on, all 524,288 frames arrived without a gap at
112 kfps, the sender's own limit. OBF then yields the whole time, since every batch waits.

### With the viewer open

A scripted demo drives the viewer in a browser through the Chrome DevTools protocol and records
it with the page's own recorder. It zooms, filters, tunes C10, switches OBF's weights on and
chooses OBF while daqsim replays a focus walk followed by the in-focus scan. Before the pacing
changes, choosing OBF took the branch from 50.0 to about 40 kfps at ×2 and 50 kHz, and daqsim
dropped 959,063 of 6,291,456 frames. At ×4 and 33.3 kHz it dropped 522,880 of 4,194,304. Timing
the parallax callbacks found the causes.

- A deposit's batched FFTs saw a new batch size at almost every step, and each new size made
  cuFFT build a plan in the callback, about 80 ms.
- Pieces of the weights' build ran 0.3 to 0.9 s of NumPy between yields.
- The service-mode tick stepped OBF on top of the batches during the scan.
- The in-process viewer's histograms took about 6 ms a channel in the publishing callback.
- A queued request's reply described the state before it, so the page took the OBF just chosen
  for a channel switched off and went back to parallax.

After the changes (see the [changelog](CHANGELOG.md)):

| Upsampling | Frame time | Host | Frames | Dropped | Branch | OBF live after choosing it |
| --- | --- | --- | ---: | ---: | ---: | --- |
| ×2 | 20 µs | saturn | 5,058,847 | 0 | 50.0 kHz | 9 s, and 4 s the second time |
| ×4 | 20 µs | saturn | 5,336,710 | 0 | 50.0 kHz | 8 s |
| ×4 | 30 µs | jupiter | 3,524,792 | 0 | 33.3 kHz | 8 s |

Those seconds are OBF covering the 65,536 frames held while the scan rewrites them at the frame
rate. Parallax is not slowed meanwhile. The ×4 run at 50 kHz is
[live4d_obf_demo_x4_50kHz.mp4](images/live4d_obf_demo_x4_50kHz.mp4).

### Caveats

- The first OBF build in a run creates cuFFT plans and stalls the branch once, for up to about
  0.8 s. daqsim's buffer absorbed it in every run.
- On jupiter, while another user's job held the host's load near 96, a run delivered 30 to 40
  kfps from its first batch, before OBF was on. Timing results need a quiet host.
- Not run: RDMA, a real DCU, and browser-attached runs above 50 kHz.

### Reproducing

The image, on a host with Docker or built elsewhere and moved, as here:

```bash
docker buildx build --platform linux/amd64 -f docker/Dockerfile.holoscan --target parallax \
  --build-arg CUDA_ARCHITECTURES=86 -t live4d:parallax --load .
docker save live4d:parallax | ssh host 'cat > /tmp/live4d.tar'
ssh host 'apptainer build live4d.sif docker-archive:///tmp/live4d.tar'
```

Under Apptainer, `apptainer exec --nv --cleanenv --no-home live4d.sif python3 -P -m live4d
--config pipeline.yaml` runs the application. The suite runs as the
[pipeline guide](../holoscan_basic_ingest.md) says, with the repository bound at `/work`.
daqsim's image converts with `apptainer build daqsim.sif docker-archive://daqsim_x86_64.tar` and
runs unprivileged with writable directories bound at `/daqsim` and `/opt/dectris` and its HTTP
moved off port 80 (`DCU_BASE_SYSTEM_HTTP_ADDRESS=127.0.0.1:8080`). On the host network its
Stream2 output listens on port 31001, and an RDMA device is needed only for its `cbor-rdma` mode.

## Integration, 2026-10-27

The integration incorporates `origin/main` at `a83f583` (verified with a fresh fetch).
The parallax engine, readout, and orientation files are unchanged apart from imports;
the migrated tests retain their independent float64 numerical references.

### Correctness and packaging

- CPU configuration, numerical, replay, continuous accumulation, pass saving, tuning,
  viewer, telemetry, CBOR parser, and protocol checks: **329 passed, 32 skipped**.
  The skips require optional comparison data/oracles, unavailable devices, or a display.
- Final GPU suite: **61 passed, 14 skipped** (optional RDMA/codec paths).
  Nine combined CUDA integration cases passed for branch combinations, slow consumers,
  empty/partial batches, and consecutive 96×96 / 192×192 series. An additional real
  HTTP tuning test passed: the viewer request queues a change, a runtime callback
  applies it, and the final latest-mode image and shifts match a fresh reconstruction. Interruption saves partial results without fabricating END;
  an injected compute failure saves partial results and closes the viewer.
- The baseline GPU/protocol run passed 90 checks with 29 optional skips. Five existing
  `dectris/tests/test_imagemux.py::ProbeTest` cases fail because `tools.probe_imagemux`
  is absent from current main. A focused rerun confirms the missing helper; changing
  Python search paths does not resolve it. They are excluded from the final CPU run.
- Both shared Docker targets build native bindings and generated schemas from a fresh
  build directory. Imports and browser assets were checked. The standalone wheel
  imports offline replay without importing Holoscan and includes the browser assets.
- The combined graph has five partitions and four CUDA compute streams. Its only
  diagnostic is `OPERATOR_SERIALIZATION_DOMAIN_SPLIT_ADVISORY` (413, `python.gil`);
  no compile errors were reported.

Run the CPU tests from the repository root in an environment with the parallax/dev,
notebook, and visualization dependencies. Use the built package for live tests:

```bash
PYTHONPATH=holoscan_pipeline/python:dectris/src python -m pytest -q \
  holoscan_pipeline/tests/parallax holoscan_pipeline/tests/test_pipeline_settings.py \
  dectris/tests/test_messages.py dectris/tests/test_imagemux.py -k 'not ProbeTest'

docker run --rm --gpus all -v "$PWD:/work:ro" --entrypoint python3 \
  live4d-integration:parallax -B -m pytest -q -p no:cacheprovider \
  /work/holoscan_pipeline/tests/test_stream.py
```

### Stress result: target not met

Host: RTX A3000 Laptop GPU (6 GiB), driver 615.71.09, 16 logical CPUs. Source:
`daqsim:latest`, version `daqsim-and-rdma-26.10.22.48491-g15b27e55d8`.
The simulator supports ZMQ and exposes RDMA configuration; this host has no
`/dev/infiniband`, so RDMA was not run. The local simulator archive is outside the
repository; the test used an existing image and stored its own state under ignored
`daqsim/integration/`. An existing `daqsim-hdf5` container was left untouched.

The receiver and simulator ran on the same host through one ZMQ connection.
The measured series requested **3,600,000 uint16 frames at 120,000 fps**, 96×96 pixels,
2×2 binning, BSLZ4. The trigger call took 30.089 s. A separate 128-frame warm-up
series ran first on the same graph/connection and had no transport drops. Only the
second acquisition contributes to the measured counts and timing. The simulator
sent zero-valued frames; these are not representative of real diffraction data.

Shared configuration: device 0, batch size 512, queue depth 4, scan 2000 columns ×
1800 rows, BF/ADF enabled, parallax enabled, writer and ANS disabled. Explicit
parallax settings for this measurement: energy 200000 eV, semiangle 30 mrad,
angular sampling 1.066 mrad/pixel, scan sampling 1.98 Å, C10 −300 Å, rotation 3.75°,
upsampling 1, accumulation `scan`, frame mean retained. This selects 2477 BF pixels.
These are measurement settings, not acquisition defaults.

| Measurement | Warmed combined run |
| --- | ---: |
| Source frames requested / announced | 3,600,000 |
| BF/ADF frames completed | 2,277,814 |
| Parallax frames completed | 2,277,814 |
| Missing positions / simulator dropped frames | 1,322,186 |
| Duplicate positions | 0 |
| BF/ADF delivered interval / delivered rate | 30.138 s / 75,564 fps |
| Parallax branch submission rate | 75.6 kfps |
| Trigger to both branches finished and graph stopped | 30.771 s |
| Completed combined rate over that wall interval | 74,025 fps |
| Observed pipeline container RAM | 1.465–1.466 GiB |
| Observed total GPU memory (including desktop) | 3077–3079 MiB |
| Observed GPU utilization | 22–27% |

The parallax report correctly says `INCOMPLETE SCAN`; source drops equal missing
positions. The simulator resets its dropped counter on each arm; these are per-series
counts. Both branches finished the delivered series, results were saved, and
cooperative shutdown completed. Host/GPU memory remained bounded in the samples.
The last parallax progress lower bound at 30.0 s was 2,252,800 frames; the final
2,277,814 were complete by 30.771 s. Internal publication windows permit at most
three outstanding decoded and three compressed publications. Source-side queue
occupancy and exact end-to-end latency were not instrumented, so this does not
establish a maximum backlog or a pipeline bottleneck. The final completion occurred
about 0.68 s after the trigger call returned.

Two diagnostic runs also missed the target. An initial combined run overlapping CPU
tests delivered 1,195,181 frames at 27.2 kfps and dropped 2,404,819; it is not a fair
performance comparison. A BF/ADF-only run delivered 1,393,911 at 34.1 kfps and dropped
2,206,089, with about 910–912 MiB container RAM. The runs differed in warm-up and
CPU activity and thread settings (the cold combined run used `OMP_NUM_THREADS=2`;
the BF/ADF-only and warmed combined runs left it unset), so they do not measure the incremental cost of parallax. None supports
a claim of sustained 120,000 fps on this host.

The [machine-readable result](validation.json) retains completion counts, resource
samples, and diagnostics. Measurement helpers are in
`holoscan_pipeline/tests/benchmarks/combined.py` and `simulator.py`. The first wraps
the production `PipelineApp` and stops after both branches finish two series; the
second drives only the isolated test simulator and samples Docker/GPU resources.
They are validation fixtures, not another application graph.

### Simulator setup used

The isolated simulator was launched from the repository root:

```bash
docker run --rm --name live4d-integration-sim \
  -p 127.0.0.1:18081:80 -p 127.0.0.1:35555:31001 \
  -v "$PWD/daqsim/integration:/daqsim" daqsim:latest
```

Only that simulator was initialized/configured using SIMPLON API 1.8.0: initialize,
filewriter mode disabled, stream mode enabled, stream format cbor, compression bslz4,
binning mode 2x2, trigger mode ints, ntrigger 1, nimages 3,600,000, count_time 7e−6,
and frame_time 1/120000. Geometry readback was 96×96; dropped counts were read after
each acquisition. The measurement helper arms a 128-frame warm-up, waits for its
saved result, sets nimages to 3,600,000, arms again, triggers, and waits for both ENDs.
Use the measurement settings above in `/tmp/live4d-stress.yaml`, then run:

```bash
docker run --rm --gpus all --network host --name live4d-integration-isolated-stress \
  --entrypoint python3 \
  -v /tmp/live4d-stress.yaml:/etc/live4d.yaml:ro \
  -v "$PWD/holoscan_pipeline/tests/benchmarks/combined.py:/benchmark.py:ro" \
  live4d-integration:parallax /benchmark.py > /tmp/live4d-isolated-stress.log 2>&1 &
python holoscan_pipeline/tests/benchmarks/simulator.py > /tmp/live4d-isolated-stress-monitor.log
```

### Unverified optional paths

RDMA hardware smoke/stress tests, MPS execution, real detector acquisition, and real
`quantem.gpu` ANS encode/decode were unavailable. ANS staging participates in the
combined branch tests through an instrumented sink; this verifies consumer accounting
and frame preservation without claiming codec performance. The HDF5 metadata and
writer integration tests use the incorporated main implementation.
