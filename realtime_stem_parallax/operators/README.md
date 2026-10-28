# operators — the C++/CUDA Holoscan ingest

The two compiled Holoscan operators that receive and decompress the detector stream. `docker/Dockerfile`
builds them into the image; the Python side imports them as `dectris_holoscan_ops`
(`realtime_stem/ingest/holoscan_graph.py`).

- **`ZmqRxOp`** (`cpp/operators/zmq_rx_operator.cu`): connects to the DCU over ZeroMQ (PULL,
  `n_connections` sockets), parses each DECTRIS Stream2 CBOR message (`cpp/common/fast_cbor_parse.c`),
  validates frame geometry and dtype against the stream, and packs the still-compressed frames of a
  batch into pinned host memory.
- **`DecompressGpuOp`** (`cpp/operators/decompress_gpu_operator.cu`): copies a batch to the GPU,
  decompresses all bslz4 blocks in one nvcomp LZ4 call, undoes the bitshuffle
  (`cpp/common/bitshuffle_kernel_gpu.cu`), and emits the decoded frames as a GPU tensor.
- `cpp/bindings/dectris_holoscan_ops.cpp`: the pybind11 module that exposes both to Python.
- `cpp/common/constants.h`: defaults and the batch ceiling `kMaxBatchSize` (2048). Buffers are
  preallocated for the ceiling, so a larger `--batch-size` needs this raised and the image rebuilt.
- `python/dectris_holoscan/` and `config/`: a minimal standalone ingest app from the upstream
  project. The build copies it into the image; `realtime_stem` does not use it.

## Origin and license

From [electronmicroscopy/live4d](https://github.com/electronmicroscopy/live4d) (`holoscan_pipeline/`;
Denis Leshchev, NVIDIA), MIT license, see `LICENSE`. Extended here with:

1. runtime `detector_width` / `detector_height` / `dtype` parameters, validated against the stream;
2. the bslz4 block size read from each frame's header (needed for uint32 streams);
3. acceptance of CBOR self-describe-tagged (55799) messages, as sent by real ARINA DCUs;
4. `n_connections` (the DCU paces each TCP connection; 120 kHz needs 3);
5. a batch ceiling of 2048 with a warning when a request is clamped, and a 200 ms end-of-stream
   drain instead of 2 s.
