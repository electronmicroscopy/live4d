"""Ingest stage: receive detector frames and hand decoded batches to an `Accumulator`.

Two interchangeable ingests drive the same `sink.BatchSink` (batch → `accumulator.process`,
throughput, throttled live snapshots):

* `holoscan_graph` — full detector rate. Frames arrive over ZeroMQ from the DECTRIS detector
  control unit and are bslz4-decompressed on the GPU by two compiled Holoscan C++ operators —
  `ZmqRxOp` and `DecompressGpuOp` — from `operators/`, compiled into the image by
  `docker/Dockerfile`. `GenericSinkOp` receives each decoded batch as a zero-copy CUDA tensor.
* `python_ingest` — no Holoscan, no NVIDIA GPU: the same messages over ZeroMQ, or a `.dstream` /
  `.npy` file, decoded on the CPU (`codec`), for laptops, CI and offline replays (tens of kHz).

Nothing here is parallax-specific.
"""
