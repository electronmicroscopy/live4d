"""The `.dstream` capture-file format: a raw DECTRIS ZMQ stream recorded verbatim to disk.

Layout: 8-byte magic `DSTREAM1`, then a sequence of length-prefixed records
`[uint32 little-endian length][raw message bytes]`. The first record is the `start` message and
the last is `end`; the image records in between are the raw CBOR messages the DCU sent (their
bslz4 pixel payload is never touched here). Streaming on both write (constant memory) and read
(`iter_records`), so a multi-GB capture is no problem.

This is deliberately small and dependency-light: the byte-level helpers need nothing, and only the
envelope peek needs `cbor2`. `tools/replay_to_zmq.py` reads captures with it; `tools/make_dstream.py`
writes them.

    python -m tools.dstream info examples/data/sample.dstream
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → `import realtime_stem`

# The record format itself lives in the package (the pure-Python ingest reads captures directly);
# re-exported here so `tools.*` and existing imports keep working.
from realtime_stem.ingest.dstream import (  # noqa: E402,F401
    MAGIC,
    is_dstream,
    iter_records,
    write_magic,
    write_record,
)


def scan_index(path):
    """Return `(n_records, first_raw, last_raw)`, seeking past the payloads in between so this
    stays cheap even on a multi-GB file."""
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: not a DSTREAM capture (bad magic).")
        n = 0
        first = None
        last_pos = last_len = None
        while True:
            header = fh.read(4)
            if len(header) < 4:
                break
            length = int.from_bytes(header, "little")
            pos = fh.tell()
            if n == 0:
                first = fh.read(length)
            else:
                last_pos, last_len = pos, length
                fh.seek(length, 1)  # skip the payload; we only need first + last
            n += 1
        if last_pos is not None:
            fh.seek(last_pos)
            last = fh.read(last_len)
        else:
            last = first  # only one record
        return n, first, last


def envelope_type(raw: bytes):
    """The message 'type' ('start'/'image'/'end') without decompressing the image payload.

    Decoding with no tag_hook leaves the bslz4 blob (CBOR tag 56500) as an inert `CBORTag`, so the
    small envelope fields are read for free. Requires `cbor2`.
    """
    import cbor2
    try:
        msg = cbor2.loads(raw)
    except Exception:
        return None
    return msg.get("type") if hasattr(msg, "get") else None


def info(path) -> None:
    """Print a one-line summary of a capture."""
    path = Path(path)
    size_mb = path.stat().st_size / 1e6
    n, first, last = scan_index(path)
    print(f"{path.name}: DSTREAM, {n:,} records (~{max(n - 2, 0):,} frames), "
          f"first={envelope_type(first)!r}, last={envelope_type(last)!r}, {size_mb:,.1f} MB")


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python -m tools.dstream", description="Inspect a .dstream capture.")
    sub = p.add_subparsers(dest="cmd", required=True)
    ic = sub.add_parser("info", help="print a summary of a capture file")
    ic.add_argument("file", help="The .dstream capture.")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if args.cmd == "info":
        info(args.file)


if __name__ == "__main__":
    main()
