# Live 4D-STEM with DECTRIS and Holoscan

The `live4d` application receives DECTRIS 96×96 uint16 images over ZMQ or imagemux RDMA
and processes both routes through one Holoscan SDK 5 pipeline. ZMQ input uses GPU BSLZ4
decompression. Uncompressed RDMA input is gathered directly from a registered GPU or
mapped-host ring into runtime-owned GPU tensors. Shared operators place images in a raster
scan and build bright-field and annular dark-field images.

See the [pipeline guide](docs/holoscan_basic_ingest.md) for the single container, application,
configuration, batching/ownership contracts, and validation commands.

| Directory | Purpose |
| --- | --- |
| `holoscan_pipeline/` | The `live4d` application, numerical algorithms, typed ingest operators, and tests |
| `docker/` | Unified HSDK 5/CUDA 13 container and compose service |
| `dectris/` | Reusable imagemux protocol, handshake, native verbs backend, and ring ownership |
| `pipeline/` | Earlier Python processing experiments and Stream2 format notes |
| `simple_listener/` | Small Stream2 protocol reference |
