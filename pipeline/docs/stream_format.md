# DECTRIS Stream Decoding & Decompression

Reference notes on how frames arrive from the DECTRIS DCU and how they are decoded
and decompressed. Relevant to any real-time pipeline that consumes the stream.

Established by inspecting the installed `dectris-compression` package and its
public source (`https://github.com/dectris/compression`).

## Transport

- The stream listener (the live source; e.g. `ingest/pipeline.py`) connects a **ZMQ `PULL`**
  socket to the DCU (`tcp://<dcu_ip>:<zmq_port>`); the DCU is the `PUSH` side.
- Each `socket.recv()` returns one CBOR-encoded message. `cbor2.loads(message, tag_hook=...)`
  parses it **and** decompresses any image payload via the tag hook (tag `56500`).
- Messages carry a `type` field (`image`, plus series-level start/end). **Frame messages
  carry a message/image id** -- use it for scan-position mapping rather than relying on
  arrival order (important once decode is parallelized and can reorder).

### CBOR tag decoders (in the listener)

- Tag `40` / `1040` -- multi-dim array (row-major / column-major).
- Tags `64`-`87` -- typed arrays (endian + dtype variants).
- Tag `56500` -- **DECTRIS compression**: value is `(algorithm, elem_size, encoded)`, decoded
  by calling `dectris.compression.decompress(encoded, algorithm, elem_size=elem_size)`.

### Message schema (real QUADRO capture)

From a 32x32 QUADRO capture (1 `start` + 1024 `image` + 1 `end`):

- **`start`** -- `type='start'` + series metadata: `series_id`, `series_unique_id`,
  `image_size_x` / `image_size_y` (detector shape), `number_of_images`, `frame_time`,
  `count_time`, `channels` (e.g. `('threshold_1',)`), `saturation_value`, ...
- **`image`** -- `type='image'`, **`image_id`** (0-based, contiguous -> scan position),
  `series_id`, `real_time` / `start_time` / `stop_time`, and **`data`**.
  - `data` is a **per-channel mapping**: `{'threshold_1': <frame>}`.
  - the frame is nested tags:
    `CBORTag(40, ((H,W), CBORTag(69, CBORTag(56500, (algo, elem_size, bytes)))))`
    = multi-dim array (40) > typed array `<u2`/uint16 (69) > compression (56500, `bslz4`).
- **`end`** -- `type='end'`.

### Two cbor2 decoding notes (both handled in `dectris_decode.py` / `pipeline.py`)

1. **`tag_hook` signature changed in cbor2 >=6.** It calls `tag_hook(tag, immutable_flag)` --
   the `CBORTag` is the **first** argument (older cbor2 used `(decoder, tag)`). `tag_hook`
   locates the `CBORTag` among its args instead of assuming a position.
2. **cbor2 returns an immutable `frozendict`, not `dict`.** `isinstance(x, dict)` is `False`
   for decoded messages and for `data`. Use duck typing (`hasattr(x, 'get')` / `.values`) or
   `collections.abc.Mapping`.

## The compression library: `dectris-compression`

- Installed as a compiled CPython C-extension (no Python source); v0.3.1 here.
- A **raw CPython C-API** binding (`python/dectris/compression.c`) over a C core
  (`src/compression.c`), bundling **LZ4** and **bitshuffle** as submodules. LZ4-only build
  (no zstd in this version).
- Public API used by the pipeline: `decompress(buffer, algorithm, elem_size=...) -> bytes`.

### Supported algorithms (exactly two)

| Name     | Meaning                  | `elem_size` |
|----------|--------------------------|-------------|
| `lz4`    | plain LZ4 block stream   | not needed  |
| `bslz4`  | **bitshuffle + LZ4**     | required    |

ARINA frames use **`bslz4`**.

### `bslz4` / `lz4` container layout (`src/compression.c`)

Simple, regular, and the same framing for both (bslz4 just adds an unshuffle step):

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

- Number of full blocks = `uncompressed_size / block_size`; a final short block holds the
  remainder.
- Per block: LZ4-decompress, then (for `bslz4`) **bit-unshuffle** within the block using
  `elem_size` (temp buffer sized `block_size * 2`).

## GIL behavior -- decode threads scale

The binding releases the GIL around the decompress call:

```c
Py_BEGIN_ALLOW_THREADS
n = compression_decompress_buffer(algorithm, ...);
Py_END_ALLOW_THREADS
```

**Implication:** a *thread* pool of decoders gives real CPU parallelism -- no need for a
process pool (which would otherwise be forced by the GIL but would cost IPC to move frames).
The remaining ceiling on a Python pipeline is per-frame **Python dispatch overhead**, not the
decode work itself. ZMQ sockets are not thread-safe, so exactly one thread may own `recv()`;
that thread should do nothing but `recv()` and hand raw bytes to decode workers.

## GPU decompression is feasible (the path to full frame rate)

ARINA targets ~**30 kHz @ 192x192** (and ~**120 kHz @ 96x96 binned**) -- rates at which any
per-frame Python work is infeasible and CPU decompression becomes a bottleneck. The `bslz4`
format maps cleanly onto a GPU decode path:

- The container is **N independent LZ4 blocks** -> maps directly onto **nvCOMP's batched
  low-level LZ4 decompress** (arrays of input/output pointers + sizes).
- **Bitshuffle unshuffle** is an embarrassingly-parallel bit-transpose -> a small CUDA kernel.
- Flow: ship the *compressed* bytes to the GPU (small), parse block offsets, nvCOMP
  batched-LZ4, then the unshuffle kernel -> frames live on the GPU and **the CPU never touches
  a pixel**. A thin glue layer parses the DECTRIS block framing; no algorithmic blocker.

This is the natural home for an NVIDIA **Holoscan** operator and is what makes the full-rate
target reachable.

## Source pointers

- `dectris/compression` (GitHub): `python/dectris/compression.c` (binding + GIL release),
  `src/compression.c` (container parsing + block loop), `third_party/{lz4,bitshuffle}`.
- Local install: `dectris/compression.cpython-312-*.so` (LZ4 symbols statically linked).
