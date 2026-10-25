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
a configured raster scan and BF/ADF integration. It prints per-series JSON summaries and
can optionally save the diffraction images as NXmx HDF5 with GPU bitshuffle/zstd compression;
it does not yet provide a live image viewer or export BF/ADF maps from the CLI.

## HDF5 writer and the Python GIL

The HDF5 writer (`H5WriterOp`, see [HDF5 output](docs/holoscan_basic_ingest.md#hdf5-output))
is a native C++ operator, so it does not add load to the Python GIL:

- Its `compute()` runs on Holoscan runtime threads, and all HDF5 calls run on a plain C++
  writer thread; neither takes the GIL.
- Its only contact with Python is one call per series at END, when the detector checks the
  shared finite-mode completion latch.
- Compiled graph diagnostics are the same with the writer on and off. The only one is the
  existing GIL serialization advisory on the Python `detector` operator.
- The GIL-bound operators are the Python `scan` and `detector`; the writer does not change their
  load. In 120,000 fps stress runs with 96×96 images, the detector delivered 120,001–120,003 fps
  with the writer enabled.

The writer does compete with the rest of the pipeline for other resources:

- **CPU:** the writer thread and HDF5 chunk writes.
- **GPU time:** bitshuffle/zstd compression runs on the same GPU as the BF/ADF integration.
- **Memory and disk bandwidth:** about 143 MB/s of compressed output at 120,000 fps with the
  synthetic stress frames (compression ratio 15.4); real data compresses differently.

Those runs were paced at 120,000 fps. They show that the pipeline sustains that rate with the
writer enabled, not how much headroom the writer costs.

## Documentation

- [Build, configure, run, and troubleshoot Live4D](docs/holoscan_basic_ingest.md)
- [START header_appendix keys](docs/header_appendix.md)
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
| `realtime_stem_parallax/` | Live parallax / tcBF reconstruction with a browser viewer and live tuning; self-contained (its own Holoscan 4.4 image and ingest) |
| `pipeline/` | Earlier Python/PyTorch experiments; separate dependencies and configuration from Live4D |
| `simple_listener/` | Historical Stream2 decoding example with a hard-coded endpoint and older CBOR callback |
