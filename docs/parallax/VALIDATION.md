# Integration validation — 2026-10-27

The integration incorporates `origin/main` at `a83f583` (verified with a fresh fetch).
The parallax engine, readout, and orientation files are unchanged apart from imports;
the migrated tests retain their independent float64 numerical references.

## Correctness and packaging

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

## Stress result: target not met

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

## Simulator setup used

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

## Unverified optional paths

RDMA hardware smoke/stress tests, MPS execution, real detector acquisition, and real
`quantem.gpu` ANS encode/decode were unavailable. ANS staging participates in the
combined branch tests through an instrumented sink; this verifies consumer accounting
and frame preservation without claiming codec performance. The HDF5 metadata and
writer integration tests use the incorporated main implementation.
