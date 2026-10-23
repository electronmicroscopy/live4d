# Earlier Python processing experiments

This directory retains the Python/PyTorch ZMQ decoding, virtual bright-field, and
Matplotlib visualization experiments that preceded the unified Live4D application.
For the current ZMQ/RDMA application, use the
[Holoscan pipeline guide](../docs/holoscan_basic_ingest.md).

| Earlier path | Current Live4D counterpart |
| --- | --- |
| `ingest/` CPU/threaded or PyTorch GPU decode | Native ZMQ receive and GPU decode operators in `holoscan_pipeline/cpp/operators/` |
| `virtual_detectors/main.py` and `config.py` | `live4d/app.py`, `live4d/config.py`, and `holoscan_pipeline/config/` YAML |
| `virtual_detectors/vbf.py` | CuPy BF/ADF functions in `live4d.algorithms.virtual_detectors` |
| `visualize/` Matplotlib viewer | No current Live4D CLI viewer; visualization remains separate work |

The old `requirements.txt` includes PyTorch, CUDA 12 nvCOMP, and `dectris-compression`.
It is not the dependency list for the HSDK 5/CUDA 13 container. Old presets, masks,
geometry, and CLI flags are not automatically portable to Live4D, which currently
accepts only 96×96 uint16 frames. Keep any experiment environment separate from the
unified application environment.

The [Stream2 notes](docs/stream_format.md) remain useful as a wire-format reference.
The sibling `simple_listener/` is also historical: it embeds a sender address and uses
the older `(decoder, tag)` CBOR callback convention. For callback compatibility handling,
consult `ingest/dectris_decode.py` rather than copying that listener unchanged.

Retain these experiments while they serve as decoding references or comparison tools.
Before removing them, migrate any required viewer, calibration-mask, or CPU-reference
workflow and its validation. New Live4D algorithms belong under
`holoscan_pipeline/python/live4d/algorithms/`.
