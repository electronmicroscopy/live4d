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


def capture(ip: str, port: int, out_path, max_frames: int | None = None,
            n_connections: int = 3, drain_quiet_ms: int = 1000) -> None:
    """Record one series. The DCU paces each connection (about 60k binned frames/s), so 120 kHz
    needs 3. `start` and `end` each arrive on one connection; images read before `start` are kept
    and written after it, and after `end` the other connections are drained."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context.instance()
    socks = []
    poller = zmq.Poller()
    for _ in range(n_connections):
        sock = ctx.socket(zmq.PULL)
        sock.set_hwm(0)  # unbounded receive queue: a short stall here never drops frames
        sock.connect(f"tcp://{ip}:{port}")
        poller.register(sock, zmq.POLLIN)
        socks.append(sock)

    print(f"[record] tcp://{ip}:{port} x{n_connections} -> {out_path}")
    print("[record] waiting for 'start' ... start the scan now (Ctrl-C to stop).")
    unreachable = reachable(ip, port)
    if unreachable:
        print(f"[record] WARNING: cannot reach {ip}:{port} ({unreachable}). Nothing can be "
              f"recorded until this is fixed: check the DCU address and port, and that its stream "
              f"interface is enabled. Still trying; Ctrl-C stops.")
    last_check = time.perf_counter()

    n_msgs = n_img = nbytes = 0
    started = False
    detector = None
    expected = None
    end_raw = None
    pre_start: list[bytes] = []
    t0 = None
    last_msg = time.perf_counter()

    with open(out_path, "wb") as fh:
        fh.write(MAGIC)

        def emit(raw: bytes, mtype) -> None:
            nonlocal n_msgs, n_img, nbytes
            write_record(fh, raw)
            n_msgs += 1
            nbytes += len(raw)
            if mtype == "image":
                n_img += 1
                if n_img % 100_000 == 0:
                    rate = n_img / max(time.perf_counter() - t0, 1e-9)
                    print(f"[record] {n_img:>9,} frames  {rate:>9,.0f}/s  {nbytes / 1e6:>8,.0f} MB")

        try:
            done = False
            while not done:
                events = dict(poller.poll(timeout=500))
                if not events:
                    if unreachable and time.perf_counter() - last_check >= 30:
                        last_check = time.perf_counter()
                        unreachable = reachable(ip, port)
                        print(f"[record] {ip}:{port} is reachable now." if not unreachable else
                              f"[record] WARNING: still cannot reach {ip}:{port}.")
                    if end_raw is not None and (time.perf_counter() - last_msg) * 1000 >= drain_quiet_ms:
                        break
                    continue
                for sock in events:
                    for _ in range(4096):
                        try:
                            raw = sock.recv(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        last_msg = time.perf_counter()
                        mtype = envelope_type(raw)

                        if mtype == "start":
                            if started:
                                continue
                            started = True
                            t0 = time.perf_counter()
                            m = cbor2.loads(raw)
                            detector = (m.get("image_size_y"), m.get("image_size_x"))
                            expected = m.get("number_of_images")
                            print(f"[record] start: detector {detector}, number_of_images {expected}")
                            emit(raw, mtype)
                            for buffered in pre_start:
                                emit(buffered, "image")
                            pre_start.clear()
                        elif mtype == "end":
                            end_raw = raw
                            print("[record] end of series; reading what is still queued ...")
                        elif not started:
                            if mtype == "image":
                                pre_start.append(raw)
                                if len(pre_start) > 100_000:
                                    print("[record] 100k+ images without 'start' (started during "
                                          "a scan?); discarding them.")
                                    pre_start.clear()
                        else:
                            emit(raw, mtype)

                        if max_frames is not None and n_img >= max_frames:
                            print(f"[record] reached --max-frames {max_frames}; stopping.")
                            done = True
                            break
                        if end_raw is not None and expected is not None and n_img >= expected:
                            done = True
                            break
                    if done:
                        break
        except KeyboardInterrupt:
            print("\n[record] interrupted, closing the file.")
        finally:
            if started:
                if end_raw is not None:
                    write_record(fh, end_raw)
                    n_msgs += 1
                    nbytes += len(end_raw)
                else:
                    write_record(fh, cbor2.dumps({"type": "end"}))
                    print("[record] no 'end' received; added a minimal one so the file replays.")

    for sock in socks:
        sock.close(linger=0)
    print(f"[record] wrote {n_msgs:,} messages ({n_img:,} frames), {nbytes / 1e6:,.1f} MB -> {out_path}")
    if expected is not None and max_frames is None:
        if n_img < expected:
            print(f"[record] WARNING: incomplete, {n_img:,} of {expected:,} frames. Check the DCU's "
                  f"dropped counter and that nothing else reads the stream.")
        else:
            print(f"[record] COMPLETE: {n_img:,} of {expected:,} frames.")


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
    c.add_argument("--n-connections", type=int, default=3,
                   help="connections to the DCU (3 carries 120 kHz binned)")
    i = sub.add_parser("info", help="summary of a .dstream file")
    i.add_argument("file")
    args = p.parse_args()
    if args.cmd == "capture":
        capture(args.ip, args.port, args.out, args.max_frames, args.n_connections)
    else:
        info(args.file)


if __name__ == "__main__":
    main()
