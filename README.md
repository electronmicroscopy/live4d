# Live 4D-STEM with DECTRIS and Holoscan

The `live4d` application receives DECTRIS 96×96 (2×2 binned) or 192×192 (unbinned, 1×1) uint16
images over ZMQ or imagemux RDMA and processes both routes through one Holoscan SDK 5
pipeline. ZMQ input uses GPU BSLZ4 decompression. Uncompressed RDMA input is gathered
directly from a registered GPU or mapped-host ring into runtime-owned GPU tensors.
Shared operators place images in a raster scan and build bright-field and annular
dark-field images.

See the [pipeline guide](docs/holoscan_basic_ingest.md) for the single container, application,
configuration, batching/ownership contracts, and validation commands.

The application selects image dimensions from each series' START metadata and supports
a configured raster scan and BF/ADF integration. It prints per-series JSON summaries;
it does not yet provide a live image viewer, save diffraction datasets, or export BF/ADF
maps from the CLI.

Service mode can route consecutive acquisitions across an ordered `devices` list for
both transports.
One GPU and multiple GPUs use the same processing and connection management code.

## Documentation

- [Build, configure, run, and troubleshoot Live4D](docs/holoscan_basic_ingest.md)
- [Reusable imagemux RDMA package and protocol tests](dectris/README.md)
- [Stream2 wire-format reference](pipeline/docs/stream_format.md)
- [Earlier Python experiments and migration notes](pipeline/README.md)

The container workflow targets a Linux GPU host. RDMA additionally requires an active
verbs/RoCE interface; the ZMQ route can run without an RDMA device. Start with the pipeline
guide to select a transport and configure the sender endpoint before acquisition.

| Directory | Purpose |
| --- | --- |
| `holoscan_pipeline/` | The `live4d` application, numerical algorithms, typed ingest operators, and tests |
| `docker/` | Unified HSDK 5/CUDA 13 container and compose service |
| `dectris/` | Reusable imagemux protocol, handshake, native verbs backend, and ring ownership |
| `pipeline/` | Earlier Python/PyTorch experiments; separate dependencies and configuration from Live4D |
| `simple_listener/` | Historical Stream2 decoding example with a hard-coded endpoint and older CBOR callback |
