# DECTRIS Stream2 ZMQ format and decoding reference

These notes describe the CBOR image envelope and compression framing used by the
repository's ZMQ decoders. The current application already implements GPU BSLZ4 decoding;
use the [Live4D guide](../../docs/holoscan_basic_ingest.md) for running it and for supported
geometry. The broader tag support in the earlier Python decoder is not a promise that
Live4D accepts every dtype, channel layout, or compression mode.

Imagemux RDMA uses a separate framing and credit protocol, described in the
[transport package guide](../../dectris/README.md#protocol-and-flow-control).

## Transport

- The receiver connects a **ZMQ `PULL`**
  socket to the DCU (`tcp://<dcu_ip>:<zmq_port>`); the DCU is the `PUSH` side.
- Each receive yields one CBOR-encoded message. In the earlier CPU decoder,
  `cbor2.loads(message, tag_hook=...)` parses it **and** decompresses the image through
  tag `56500`. Live4D instead extracts compressed bytes in C++ and decompresses on the GPU.
- Messages carry a `type` field (`image`, plus series-level start/end). **Frame messages
  carry a message/image id** -- use it for scan-position mapping rather than relying on
  arrival order (important once decode is parallelized and can reorder).

### CBOR tag decoders (in the listener)

- Tag `40` / `1040` -- multi-dim array (row-major / column-major).
- Tags `64`-`87` -- typed arrays (endian + dtype variants).
- Tag `56500` -- **DECTRIS compression**: value is `(algorithm, elem_size, encoded)`, decoded
  by calling `dectris.compression.decompress(encoded, algorithm, elem_size=elem_size)`.

### Message envelope

An acquisition consists of START, zero or more IMAGE messages, and END. The current
receiver requires matching ordered boundaries and rejects images outside a series.

- **`start`** -- `type='start'` + series metadata: `series_id`, `series_unique_id`,
  `image_size_x` / `image_size_y` (detector shape), `image_dtype`, `number_of_images`, `frame_time`,
  `count_time`, `channels` (e.g. `('threshold_1',)`), `saturation_value`, ...
- **`image`** -- `type='image'`, **`image_id`** (0-based, contiguous -> scan position),
  `series_id`, `real_time` / `start_time` / `stop_time`, and **`data`**.
  - `data` is a **per-channel mapping**: `{'threshold_1': <frame>}`.
  - the frame is nested tags:
    `CBORTag(40, ((H,W), CBORTag(69, CBORTag(56500, (algo, elem_size, bytes)))))`
    = multi-dim array (40) > typed array `<u2`/uint16 (69) > compression (56500, `bslz4`).
- **`end`** -- `type='end'`.

Live4D requires START fields `series_id`, `image_size_x`, `image_size_y`, and
`image_dtype='uint16'`. Both dimensions must be 96 for 2×2-binned images or 192 for
unbinned (1×1) images. START selects the dimensions for that series, and each IMAGE
must match them. `number_of_images` is announced metadata, not the scan shape;
configure scan width/height separately. START metadata is retained with a 16 KiB bound.
The [stream tests](../../holoscan_pipeline/tests/test_stream.py) construct executable
examples of the START/image envelopes, including an optional outer CBOR tag `55799`.

### Python decoder compatibility

The compatibility hook in [dectris_decode.py](../ingest/dectris_decode.py) locates a
`CBORTag` among its arguments rather than assuming `(decoder, tag)` or
`(tag, immutable_flag)`. Decoded mappings can be immutable; accept
`collections.abc.Mapping` or the required mapping methods rather than assuming `dict`.
These Python details do not describe Live4D's native fast CBOR parser.

## Historical CPU library inspection

The original notes inspected `dectris-compression` v0.3.1. That version's reported
algorithm set and GIL behavior are historical observations, not a check of whichever
version is installed today. The earlier CPU decoder calls
`decompress(buffer, algorithm, elem_size=...) -> bytes`; the unified container does not
depend on this library.

### Algorithms described by those notes

| Name     | Meaning                  | `elem_size` |
|----------|--------------------------|-------------|
| `lz4`    | plain LZ4 block stream   | not needed  |
| `bslz4`  | **bitshuffle + LZ4**     | required    |

Live4D's ZMQ route requires **`bslz4`**; plain `lz4` is not a supported Live4D input mode.

### Compression container layout

The local [framing parser](../ingest/bslz4.py) describes the following layout (BSLZ4 adds
an unshuffle step after LZ4 decompression):

```
[ 12-byte header ]
    uint64 big-endian   uncompressed size (bytes)
    uint32 big-endian   block size (bytes)
[ block 0 ]
    uint32 big-endian   compressed size of this block
    <compressed LZ4 bytes>
[ block 1 ]
    ...
```

- Number of full blocks = `uncompressed_size // block_size`; a final short block holds the
  remainder.
- Per block: LZ4-decompress, then (for `bslz4`) **bit-unshuffle** within the block using
  `elem_size`.

For the 96×96 uint16 test messages, the uncompressed size is 18,432 bytes, with an
8,192-byte block size: two full blocks and a 2,048-byte remainder. The 192×192 test
messages contain 73,728 uncompressed bytes in nine full 8,192-byte blocks. Live4D's
GPU operators support these two image sizes; the Python framing parser also serves
as a reference for the earlier experiments.

### Historical GIL observation

The v0.3.1 binding inspection reported a GIL release around decompression:

```c
Py_BEGIN_ALLOW_THREADS
n = compression_decompress_buffer(algorithm, ...);
Py_END_ALLOW_THREADS
```

This motivated the earlier threaded CPU decoder. It does not establish decoder scaling
or identify the bottleneck for another library version, detector, or machine. Measure
those separately if reviving the experiment. Keep receive ownership on one thread.

## Implemented GPU decoding

The former GPU feasibility proposal is implemented in Live4D:

- `ZmqRxOp` extracts and batches compressed image payloads into pinned storage.
- `DecompressGpuOp` parses block metadata on the GPU, calls nvCOMP batched LZ4
  decompression, and runs the CUDA bit-unshuffle kernel.
- Runtime-owned GPU tensors feed the shared scan and BF/ADF processing operators.

Use the [measurement guidance](../../docs/holoscan_basic_ingest.md#interpreting-results)
to distinguish source rate, batch submission rate, completed results, and NIC traffic.
The format alone does not establish an achievable frame rate.

## Source pointers

- [Native Stream2 parser](../../holoscan_pipeline/cpp/common/fast_cbor_parse.c)
- [GPU decoder](../../holoscan_pipeline/cpp/operators/decompress_gpu_operator.cu)
- [CUDA bit-unshuffle kernel](../../holoscan_pipeline/cpp/common/bitshuffle_kernel_gpu.cu)
- [START validation and raster geometry](../../dectris/src/dectris/messages.py)
