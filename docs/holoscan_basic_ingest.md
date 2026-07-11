# Holoscan Basic Ingest

This path builds the first Python-composed Holoscan pipeline for the DECTRIS Stream2 ingest path:

```
ZmqRxOp -> DecompressGpuOp -> StatsSinkOp
```

`ZmqRxOp` is a native C++ Holoscan operator that owns both ZMQ receive and fast CBOR extraction.
It emits compressed bslz4 batches with pinned host payload storage, offsets, sizes, and image IDs.
`DecompressGpuOp` is a native C++/CUDA Holoscan operator that performs nvCOMP LZ4 decompression and
CUDA bit-unshuffle, then emits a `holoscan.TensorMap`. The Python sink converts the `frames` tensor
to a CUDA `torch.Tensor` with DLPack.

## Docker Build

Build the deployable Holoscan pipeline image:

```bash
docker build -f docker/Dockerfile.holoscan -t dectris-holoscan-basic .
```

The image builds the native Holoscan extension during `docker build`; there is no host-side or
manual in-container CMake step. The default CUDA architecture list is `86;89` for RTX A6000 and
RTX 6000 Ada validation hosts. Override it for another GPU target:

```bash
docker build -f docker/Dockerfile.holoscan \
  --build-arg 'CUDA_ARCHITECTURES=86;89' \
  -t dectris-holoscan-basic .
```

## Docker Run

Start the pipeline and wait for a compatible DECTRIS Stream2 ZMQ source:

```bash
docker run --rm -it --gpus all --network host --ulimit stack=33554432 dectris-holoscan-basic
```

Override endpoint or batch size through container arguments:

```bash
docker run --rm -it --gpus all --network host --ulimit stack=33554432 dectris-holoscan-basic \
  --endpoint tcp://127.0.0.1:5555 \
  --batch-size 128
```

The same deployment can be run with Docker Compose:

```bash
docker compose -f docker/compose.holoscan.yaml up --build pipeline
```

The default config expects binned ARINA-compatible frames:

- endpoint: `tcp://127.0.0.1:5555`
- detector shape: `96x96`
- dtype: `uint16`
- compression: `bslz4`
- batch size: `1000`

## Validation

Use any compatible DECTRIS Stream2 simulator or data source that publishes `start`, `image`, and
`end` CBOR messages over ZMQ with bslz4-compressed `96x96` `uint16` frames.

Expected sink output looks like:

```text
[stats] batch=1 batches=1 frames=128 fps=... last_id=127 shape=(128, 96, 96) dtype=torch.uint16 device=cuda:0
```

If validation on x86 is delayed by simulator compression or pre-serialization startup, record the
observed blocker and investigate that simulator path in parallel with the core pipeline work.

## Current Limitations

This is the basic ingest/decompress path and intentionally keeps the legacy Python pipeline in place
while Holoscan is validated. Current assumptions:

- detector frames are `96x96` `uint16` bslz4 payloads
- nvCOMP is inherited from the Holoscan base image
- compressed payload batches point into a circular pinned host pool owned by `ZmqRxOp`

The pinned pool has three slots. The current pipeline relies on downstream operators consuming a
batch before `ZmqRxOp` reuses that slot. This held for the validation runs, but production work
should make the contract explicit with slot ownership, reference-counted buffers, or scheduler
backpressure.

## Follow-ups

- Add focused correctness tests against known frames.
- Generalize detector shape, dtype, and compression settings beyond the initial binned ARINA path.
- Replace the implicit pinned-pool producer/consumer contract with explicit ownership or backpressure.
- Consider preallocating nvCOMP per-batch workspace once the target batch/shape set is finalized.
