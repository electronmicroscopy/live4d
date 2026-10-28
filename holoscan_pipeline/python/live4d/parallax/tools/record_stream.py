"""Record one scan of a DECTRIS detector stream to a .dstream file, message for message.

Connects to the DCU's stream (ZeroMQ PULL) and writes every raw message unchanged: the `start`
message, every image message, the `end` message. Nothing is decoded, so the file can later be
replayed through the pipeline exactly as the detector sent it.

    python3 record_stream.py capture --ip <DCU_IP> --out scan01.dstream
    python3 record_stream.py info scan01.dstream

Start it BEFORE the scan. It stops by itself after the scan's `end` message (or on Ctrl-C).
Only one program may read the stream at a time: stop the pipeline first.
Needs: python3 with pyzmq and cbor2 (pip install pyzmq cbor2).

File format: 8 bytes `DSTREAM1`, then records [uint32 little-endian length][raw message].
"""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import cbor2
import zmq

MAGIC = b"DSTREAM1"


def write_record(fh, raw: bytes) -> None:
    fh.write(len(raw).to_bytes(4, "little"))
    fh.write(raw)


def envelope_type(raw: bytes):
    """The message 'type', without decompressing the image payload."""
    try:
        msg = cbor2.loads(raw)
    except Exception:
        return None
    return msg.get("type") if hasattr(msg, "get") else None


def reachable(ip: str, port: int, timeout: float = 2.0) -> str | None:
    """None when something accepts TCP connections at ip:port, else the reason. A plain TCP
    connection without the ZeroMQ handshake, so the DCU never counts it as a reader."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return None
    except OSError as exc:
        return str(exc)


def capture(ip: str, port: int, out_path, max_frames: int | None = None) -> None:
    """Record one ordered series through a single connection."""
    from live4d.parallax.tools.record_dstream import record
    count = record(f"tcp://{ip}:{port}", Path(out_path), max_frames)
    print(f"[record] {count} frames -> {out_path}", flush=True)


def info(path) -> None:
    """One-line summary: number of records, first and last message type, size."""
    path = Path(path)
    n = 0
    first = last = None
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise SystemExit(f"{path}: not a .dstream file")
        while True:
            header = fh.read(4)
            if len(header) < 4:
                break
            length = int.from_bytes(header, "little")
            if n == 0:
                first = fh.read(length)
            else:
                pos = fh.tell()
                fh.seek(length, 1)
                last = (pos, length)
            n += 1
        if last is not None:
            fh.seek(last[0])
            last = fh.read(last[1])
    print(f"{path.name}: {n:,} records (~{max(n - 2, 0):,} frames), first={envelope_type(first)!r}, "
          f"last={envelope_type(last if last is not None else first)!r}, "
          f"{path.stat().st_size / 1e6:,.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description="Record / inspect a DECTRIS stream (.dstream).")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="record one scan from the DCU")
    c.add_argument("--ip", required=True, help="the DCU's data IP address")
    c.add_argument("--port", type=int, default=31001, help="the DCU's stream port")
    c.add_argument("--out", required=True, help="output file, e.g. scan01_1024_bin2.dstream")
    c.add_argument("--max-frames", type=int, default=None,
                   help="stop after this many frames (default: at the end of the scan)")
    i = sub.add_parser("info", help="summary of a .dstream file")
    i.add_argument("file")
    args = p.parse_args()
    if args.cmd == "capture":
        capture(args.ip, args.port, args.out, args.max_frames)
    else:
        info(args.file)


if __name__ == "__main__":
    main()
