"""The `.dstream` capture-file format: a raw DECTRIS ZMQ stream recorded verbatim to disk.

Layout: 8-byte magic `DSTREAM1`, then a sequence of length-prefixed records
`[uint32 little-endian length][raw message bytes]`. The first record is the `start` message and
the last is `end`; the image records in between are the raw CBOR messages the DCU sent (their
bslz4 pixel payload is never touched here). Streaming on both write (constant memory) and read
(`iter_records`), so a multi-GB capture is no problem.

The pure-Python ingest reads captures directly with `iter_records` (`--source capture.dstream`);
`tools/dstream.py` adds the `info` CLI, `tools/replay_to_zmq.py` replays them over ZMQ.
"""

from __future__ import annotations

# Streaming-container magic. Bump the trailing digit if the record layout ever changes.
MAGIC = b"DSTREAM1"


def write_magic(fh) -> None:
    fh.write(MAGIC)


def write_record(fh, raw: bytes) -> None:
    fh.write(len(raw).to_bytes(4, "little"))
    fh.write(raw)


def is_dstream(path) -> bool:
    """True if `path` is a `.dstream` capture (cheap magic-byte sniff)."""
    with open(path, "rb") as fh:
        return fh.read(len(MAGIC)) == MAGIC


def iter_records(path):
    """Yield each raw message from a `.dstream`, streaming with bounded memory.

    A short read (truncated tail from an interrupted capture) ends iteration cleanly rather than
    raising — the records read so far are still valid.
    """
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: not a DSTREAM capture (bad magic).")
        while True:
            header = fh.read(4)
            if len(header) < 4:
                return
            n = int.from_bytes(header, "little")
            payload = fh.read(n)
            if len(payload) < n:
                return  # truncated final record
            yield payload
