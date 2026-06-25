"""
DECTRIS lz4 / bslz4 container framing parser (the "data in" boundary needs this on the CPU).

The GPU decode path (`ingest/gpu_decode.py`) feeds nvCOMP a batch of independent LZ4 blocks described
by arrays of (input pointer, input size, output size). nvCOMP cannot parse the DECTRIS container
framing -- that is a few microseconds of CPU work on the small *compressed* bytes (the CPU never
touches a pixel). This module is that parser.

Container layout (see `docs/stream_format.md`, both `lz4` and `bslz4`):

    [ 12-byte header ]
        uint64 big-endian   uncompressed size (bytes)
        uint32 big-endian   block size (bytes)
    [ block i ]  for i in 0..N-1
        uint32 big-endian   compressed size of this block
        <compressed LZ4 bytes>

  - N = ceil(uncompressed_size / block_size); only the last block may be short (the remainder).
  - Per block: LZ4-decompress to its uncompressed size, then (bslz4 only) bit-unshuffle by elem_size.
    The framing is identical for lz4 and bslz4 -- only the post-LZ4 unshuffle differs.
"""

from __future__ import annotations

from typing import NamedTuple


class Bslz4Block(NamedTuple):
    data_offset: int      # byte offset of the LZ4 bytes within `encoded` (past the 4-byte prefix)
    comp_size: int        # number of compressed LZ4 bytes in this block
    uncomp_size: int      # bytes this block decompresses to (== block_size, except the last)


class Bslz4Layout(NamedTuple):
    uncompressed_size: int        # total decompressed bytes
    block_size: int               # bytes per full block
    blocks: list[Bslz4Block]


def parse_bslz4_blocks(encoded: bytes) -> Bslz4Layout:
    """Parse the DECTRIS lz4/bslz4 container framing without decompressing anything.

    Returns the header sizes + one `Bslz4Block` per LZ4 block. Raises `ValueError` on any framing
    inconsistency (truncated header/block, sizes that don't add up) so a malformed blob fails loudly
    rather than silently feeding garbage to the GPU.
    """
    if len(encoded) < 12:
        raise ValueError(f"bslz4 blob too short for a 12-byte header: {len(encoded)} bytes")

    uncompressed_size = int.from_bytes(encoded[0:8], "big")
    block_size = int.from_bytes(encoded[8:12], "big")
    if block_size == 0:
        raise ValueError("bslz4 header reports block_size == 0")

    blocks: list[Bslz4Block] = []
    pos = 12
    remaining = uncompressed_size
    n = len(encoded)
    while remaining > 0:
        this_uncomp = min(block_size, remaining)
        if pos + 4 > n:
            raise ValueError(f"truncated block header at byte {pos} (blob is {n} bytes)")
        comp_size = int.from_bytes(encoded[pos:pos + 4], "big")
        pos += 4
        if comp_size == 0 or pos + comp_size > n:
            raise ValueError(f"block at {pos} claims {comp_size} compressed bytes; only "
                             f"{n - pos} remain")
        blocks.append(Bslz4Block(data_offset=pos, comp_size=comp_size, uncomp_size=this_uncomp))
        pos += comp_size
        remaining -= this_uncomp

    if pos != n:
        raise ValueError(f"{n - pos} trailing bytes after the last block (framing mismatch)")
    return Bslz4Layout(uncompressed_size=uncompressed_size, block_size=block_size, blocks=blocks)
