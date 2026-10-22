# Holoscan 5 Basic Ingest

The Python-composed graph uses native receive and GPU decode operators:

```text
ZmqRxOp -> DecompressGpuOp -> StatsSinkOp
```

The build targets **Holoscan SDK 5.0.0, CUDA 13, and Python 3.12**. SDK 4 binaries
and Python bindings cannot be reused. The legacy `pipeline/` remains a separate
implementation.

## Build and run

```bash
docker build -f docker/Dockerfile.holoscan -t dectris-holoscan-basic:hsdk5 .
docker run --rm --gpus all --network host --ulimit stack=33554432 \
  dectris-holoscan-basic:hsdk5 --endpoint tcp://127.0.0.1:31001 --batch-size 1000
```

The default CUDA architectures are `86;89` (RTX A6000 and RTX 6000 Ada). Override
with `--build-arg 'CUDA_ARCHITECTURES=86;89'` for another target. nvCOMP 5 for
CUDA 13 is installed explicitly through `libnvcomp5-dev-cuda-13`; SDK 5 does not
bundle it. CMake uses its imported target. PyTorch is optional: the default
stats sink reads tensor metadata without copying pixels.

On hosts where Docker has stale DNS but the host resolver works, build with:

```bash
docker build --network host --secret id=build_resolv,src=/etc/resolv.conf \
  -f docker/Dockerfile.holoscan -t dectris-holoscan-basic:hsdk5 .
```

Compose uses the same Dockerfile:

```bash
docker compose -f docker/compose.holoscan.yaml up --build pipeline
```

The default configuration in `holoscan_pipeline/config/basic_ingest.yaml` uses
`tcp://127.0.0.1:5555`, batch size 1000, CUDA device 0, and queue depth 4. Input
must be `96x96` `uint16` BSLZ4. Override the endpoint, batch size, device, and queue
depth with `--endpoint`, `--batch-size`, `--device`, and `--queue-depth`. The old
`--scheduler-threads` option is removed: SDK 5 compiles execution policy into a
plan. `--plan /path/plan.json` saves that plan for inspection.

## SDK 5 implementation

* `Graph -> compile -> RunSession` replaces `Application` and scheduler objects.
  The linear ingest graph runs in one explicit partition with bounded FIFO edges.
* Native types register with `HOLOSCAN_REGISTER_OPERATOR`; Python loads the
  provider library and uses reflected operator proxies instead of SDK 4 pybind trampolines.
* FlatBuffers schemas carry compressed payloads, offsets, sizes, frame IDs,
  batch IDs, and decoded tensors. Generated Python schema facades replace
  `TensorMap` and dictionary-style access.
* Compressed output uses runtime-owned pinned host tensors; decoded output uses
  runtime-owned device tensors. Loan/write/commit/emit tracks ownership and CUDA
  readiness, replacing the implicit three-slot circular-pool lifetime contract.
* RX polls ZMQ nonblocking every 100 microseconds and retains partial batches.
  Each activation receives at most 1002 messages. It yields when no data is ready,
  allowing cooperative shutdown while the detector is idle.
* An ordered end marker crosses both FIFO edges. The sink requests completion
  after consuming the final data, including partial and empty acquisitions. There
  is no fixed sleep to drain the graph. Empty markers have no image IDs; their
  placeholder tensor is not an image. SIGINT and SIGTERM request cooperative stop.
* CUDA work uses the stream borrowed from `ExecutionContext`; publication commits
  its readiness. The existing decompressor still allocates scratch per batch and
  synchronizes for header/status readback. Those costs are follow-up optimization
  work, separate from this API migration.

SDK references: [C++ migration](https://docs.nvidia.com/holoscan/sdk-user-guide/introduction/migrating-from-4x),
[Python migration](https://docs.nvidia.com/holoscan/sdk-user-guide/introduction/migrating-python-from-4x),
[tensor ownership](https://docs.nvidia.com/holoscan/sdk-user-guide/getting-started/tensors-101).

## SmB6 DAQ simulator validation

The local ignored `daqsim/` directory contains the simulator, replay scripts,
SmB6 preview/full-data links, and CPU reference samples used previously. Start
one receiver at a time: multiple ZMQ PULL receivers divide the stream.

First start the validation receiver (terminal 1):

```bash
docker run --rm --gpus all --network host --ulimit stack=33554432 \
  -e HOLOSCAN_LOG_LEVEL=WARN -v "$PWD/daqsim:/validation" \
  --entrypoint python3 dectris-holoscan-basic:hsdk5 \
  -m dectris_holoscan.validate \
  --reference /validation/holoscan-reference.npz --expected-frames 1000000 \
  --report /validation/holoscan5-verification.json --plan /validation/holoscan5-plan.json
```

After `[holoscan] ready`, trigger the paced replay (terminal 2):

```bash
docker start stanford-daqsim  # only if it is stopped
bash daqsim/replay.sh 30k
```

The validator checks all frame IDs for order/gaps and compares sampled decoded
GPU pixels exactly with CPU reference images. It exits nonzero for missing frames,
unvisited references, pixel mismatches, or runtime failures. It uses an explicit
CuPy/DLPack host copy only in the correctness sink. Reports and execution plans remain
in the ignored local simulator folder.

Use the **30,000 fps** replay for lossless simulator validation. The simulator's
observed publishing limit is below 40,000 fps; the original 100,000 fps timing
causes simulator drops. DAQ replay throughput is therefore not a pipeline-capacity
measurement. Do not compare it directly with a backpressured synthetic sender.

## GPU regression checks

After building the image, run on a Docker/GPU host with local SmB6 preview and
reference files present:

```bash
python3 -m pytest -q holoscan_pipeline/tests/test_stream.py
```

Host test dependencies: `pytest`, `numpy`, `h5py`, `pyzmq`, and `cbor2`. Tests use
private ZMQ ports and real compressed SmB6 chunks, covering empty streams, one
frame, exact batch boundaries, a partial final batch, pixel/ID correctness, and
SIGTERM while idle. Override data/image locations with `SMB6_TEST_MASTER`,
`SMB6_TEST_REFERENCE`, and `HOLOSCAN_TEST_IMAGE`.

## Recorded SDK 5 validation (2026-10-21)

RTX A6000, driver 595.99.02; image `dectris-holoscan-basic:hsdk5`, built from
`nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13` with nvCOMP 5.3.0.16.

| Check | Result |
| --- | --- |
| Paced SmB6 DAQ replay | 1,000,000 frames, 1,000 data batches |
| Image IDs | 0–999,999, no gaps |
| GPU pixels against CPU reference | All 10 sampled frames exact |
| Observed replay throughput | 30,005 fps (simulator-paced) |
| End-of-stream termination | `REQUESTED_COMPLETED` from the final sink |
| GPU regression suite | 5 passed |

Local evidence: `daqsim/holoscan5-verification.json`,
`daqsim/holoscan5-validated.log`, and `daqsim/holoscan5-plan.json`.
