"""A stand-in for the DCU's RDMA sender (imagemux): replay a scan over RDMA to `run.py --transport rdma`.

    # terminal 1: the sender, on this machine's RDMA address (port 41003, not the DCU's 31003)
    python3 -m live4d.parallax.tools.rdma_sender --source capture.dstream --listen 10.10.10.1 --rate 120000
    # terminal 2: the pipeline, exactly as at the microscope
    python3 -m live4d.parallax.offline --transport rdma --rdma-remote 10.10.10.1:41003 \
        --rdma-local 10.10.10.1 ...

It speaks imagemux's protocol: a TCP handshake (Hello -> Accept -> Ready), then each frame
written into the receiver's ring with RDMA WRITE (header, then data with the immediate), START and
END around the images, `--rdma-batch` uncompressed images per write, and the receiver's credit
word for flow control. The traffic stays inside the network card when both ends use the same port.

`--source`: a `.dstream` recording (its bslz4 frames are decoded once and cached next to it as
`<name>.frames.npy` + `<name>.control.cbor`; its own START and END messages are sent, so
`header_appendix` and the other metadata are the recorded ones) or a `.npy` of uint16 frames,
(Ny, Nx, H, W) or (N, H, W).

One receiver connection gets `--series N` series, one after another (the pipeline keeps one
connection for a whole session, as with the DCU), `--gap` seconds apart; then the sender waits for
the receiver to hang up and exits. `--cut-after N` closes the connection after N images of the
first series, without END (a sender that goes away mid-scan).

Runs in the Holoscan 5 image with `--network host --device /dev/infiniband --ulimit memlock=-1:-1`.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path


import numpy as np   # noqa: E402

PORT = 41003   # not the DCU's 31003, so a stand-in never looks like the real sender


def load_source(path: Path, limit: int = 0) -> tuple[np.ndarray, bytes | None, bytes | None]:
    """(frames (N, H, W) uint16 in RAM, START bytes or None, END bytes or None)."""
    import cbor2
    if path.suffix == ".npy":
        frames = np.load(path)
        frames = frames.reshape(-1, *frames.shape[-2:])
        start = end = None
    else:
        cache, control = path.with_suffix(".frames.npy"), path.with_suffix(".control.cbor")
        if not cache.exists():
            _decode_dstream(path, cache, control)
        frames = np.load(cache, mmap_mode="r")
        start, end = cbor2.loads(control.read_bytes())
    if frames.dtype != np.uint16:
        raise SystemExit(f"{path}: {frames.dtype} frames; the RDMA path carries uint16")
    if limit:
        frames = frames[:limit]
    # A copy in RAM: RDMA cannot register a read-only file mapping.
    return np.array(frames, order="C"), start, end


def _decode_dstream(path: Path, cache: Path, control: Path) -> None:
    """Decode a recording's frames once into `cache` (placed by image id), and keep its START and
    END messages in `control`."""
    import cbor2
    from live4d.parallax.ingest.codec import decode_message
    from live4d.parallax.ingest.dstream import iter_records
    t0 = time.perf_counter()
    start = end = None
    out = None
    n = 0
    for raw in iter_records(path):
        mtype, image_id, frame = decode_message(raw)
        if mtype == "start":
            start = raw
            meta = cbor2.loads(raw)
            meta = getattr(meta, "value", meta)
            count = int(meta["number_of_images"])
            shape = (int(meta["image_size_y"]), int(meta["image_size_x"]))
            out = np.lib.format.open_memmap(cache.with_suffix(".tmp.npy"), mode="w+",
                                            dtype=np.uint16, shape=(count, *shape))
        elif mtype == "image":
            out[image_id] = frame
            n += 1
            if n % 100_000 == 0:
                print(f"[sender] decoding {path.name}: {n:,} frames", flush=True)
        elif mtype == "end":
            end = raw
            break
    out.flush()
    del out
    cache.with_suffix(".tmp.npy").rename(cache)
    control.write_bytes(cbor2.dumps([start, end]))
    print(f"[sender] decoded {n:,} frames into {cache} in {time.perf_counter() - t0:.0f} s",
          flush=True)


def _series_message(raw: bytes | None, kind: str, series_id: int, frames: np.ndarray) -> bytes:
    """The recorded START/END with this series' id, or a minimal one for a .npy source."""
    import cbor2
    if raw is None:
        fields = {"type": kind, "series_id": series_id}
        if kind == "start":
            fields.update(image_size_x=int(frames.shape[2]), image_size_y=int(frames.shape[1]),
                          image_dtype="uint16", number_of_images=int(frames.shape[0]))
        return cbor2.dumps(cbor2.CBORTag(55799, fields))
    msg = cbor2.loads(raw)
    tagged = isinstance(msg, cbor2.CBORTag)
    body = dict(msg.value if tagged else msg)   # decoded maps can be read-only
    body["series_id"] = series_id
    if kind == "start":
        body["number_of_images"] = int(frames.shape[0])
    return cbor2.dumps(cbor2.CBORTag(msg.tag, body) if tagged else body)


class Sender:
    """imagemux's sending side for one RDMA port: the frames stay registered across series."""

    def __init__(self, listen: str, frames: np.ndarray, rdma_batch: int, device: str | None):
        from dectris import imxrdma
        self.imxrdma = imxrdma
        self.ctx = imxrdma.Context(local_addr=listen, device=device)
        self.frames = frames
        self.frame_bytes = frames[0].nbytes
        self.rdma_batch = rdma_batch
        self.frames_mr = self.ctx.register(frames.ctypes.data, frames.nbytes)
        n_records = -(-len(frames) // rdma_batch) + 3
        # One header per record; a ring wrap adds a header, at most one per record.
        self.headers = np.zeros((2 * n_records, 128), dtype=np.uint8)
        self.headers_mr = self.ctx.register(self.headers.ctypes.data, self.headers.nbytes)
        self.control = np.zeros(2 * 65536, dtype=np.uint8)
        self.control_mr = self.ctx.register(self.control.ctypes.data, self.control.nbytes)
        self.credit = np.zeros(1, dtype=np.uint64)
        self.credit_mr = self.ctx.register(self.credit.ctypes.data, 8, remote_write=True)
        self.imms = np.empty(256, dtype=np.uint32)

    def credit_position(self) -> int:
        import ctypes
        return ctypes.c_uint64.from_address(self.credit.ctypes.data).value

    def handshake(self, conn: socket.socket):
        from dectris.imagemux import (ACCEPT_LEN, HELLO_LEN, Accept, Hello, QpEndpoint,
                                      _read_exact, decode_ready)
        hello = Hello.decode(_read_exact(conn, HELLO_LEN))
        self.credit[0] = 0
        qp = self.ctx.create_qp(recv_depth=1, send_depth=256, signal_every=1)
        e = qp.endpoint
        conn.sendall(Accept(QpEndpoint(e.qp_num, e.psn, e.lid, e.mtu, e.gid),
                            self.credit_mr.address, self.credit_mr.rkey).encode())
        r = hello.endpoint
        qp.connect(self.imxrdma.Endpoint(r.qp_num, r.psn, r.lid, r.mtu, r.gid))
        decode_ready(_read_exact(conn, 8))
        # The ring position and the frame sequence run on across the series of a connection.
        self.state = dict(position=0, sequence=0, header=0)
        return qp, hello

    def send_series(self, qp, hello, series_id: int, start: bytes, end: bytes, rate: float,
                    cut_after: int = 0, timeout: float = 30.0) -> tuple[int, float]:
        """Write one series into the ring. Returns (images written, seconds)."""
        from dectris.imagemux import (Compression, DataType, FrameHeader, FrameKind, ImageHeader,
                                      frame_span)
        ring_addr, ring_len, rkey = hello.ring_addr, hello.ring_len, hello.ring_rkey
        state = self.state
        n, h, w = self.frames.shape

        def poll():
            _, _, failed = qp.poll(self.imms)
            if failed:
                raise RuntimeError(f"RDMA write failed: {qp.errors}")

        def wait_for(cond, what):
            deadline = time.monotonic() + timeout
            while not cond():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"the receiver {what}")
                poll()

        def write(mr, offset, length, remote, imm=None):
            wait_for(lambda: qp.write(mr, offset, length, remote, rkey, imm=imm),
                     "left the send queue full")

        def record(kind, data_mr=None, data_offset=0, data_len=0, image=None):
            offset = state["position"] % ring_len
            span = frame_span(data_len)
            if kind != FrameKind.WRAP and offset + span > ring_len:
                record(FrameKind.WRAP)
                offset = 0
            end_pos = state["position"] + (ring_len - offset if kind == FrameKind.WRAP else span)
            wait_for(lambda: end_pos - self.credit_position() <= ring_len,
                     "returned no ring credit")
            hdr = (FrameHeader(kind, state["sequence"], state["position"], data_len, series_id,
                               image=image) if image is not None else
                   FrameHeader(kind, state["sequence"], state["position"], data_len, series_id))
            slot = state["header"] % len(self.headers)
            self.headers[slot] = np.frombuffer(hdr.encode(), dtype=np.uint8)
            imm = state["sequence"] & 0xFFFFFFFF
            if data_len:
                write(self.headers_mr, slot * 128, 128, ring_addr + offset)
                write(data_mr, data_offset, data_len, ring_addr + offset + 128, imm)
            else:
                write(self.headers_mr, slot * 128, 128, ring_addr + offset, imm)
            state["header"] += 1
            state["sequence"] += 1
            state["position"] = end_pos

        self.control[:len(start)] = np.frombuffer(start, dtype=np.uint8)
        self.control[65536:65536 + len(end)] = np.frombuffer(end, dtype=np.uint8)
        record(FrameKind.START, self.control_mr, 0, len(start))
        t0 = time.perf_counter()
        sent = 0
        for first in range(0, n, self.rdma_batch):
            count = min(self.rdma_batch, n - first)
            if cut_after and sent + count > cut_after:
                return sent, time.perf_counter() - t0
            if rate > 0:
                while time.perf_counter() - t0 < sent / rate:
                    pass
            image = ImageHeader(image_id=first, batch_id=first // self.rdma_batch,
                                batch_size=count, dim_x=w, dim_y=h, dtype=DataType.UINT16,
                                compression=Compression.NONE,
                                uncompressed_size=count * self.frame_bytes, channel=0,
                                channel_count=1)
            record(FrameKind.IMAGE, self.frames_mr, first * self.frame_bytes,
                   count * self.frame_bytes, image)
            sent += count
        elapsed = time.perf_counter() - t0
        record(FrameKind.END, self.control_mr, 65536, len(end))
        wait_for(lambda: qp.stats["writes_outstanding"] == 0, "did not complete the writes")
        return sent, elapsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", type=Path, required=True,
                   help="A .dstream recording or a .npy of uint16 frames, in scan order.")
    p.add_argument("--limit", type=int, default=0, metavar="N",
                   help="Send only the first N frames of the source (0 = all).")
    p.add_argument("--listen", required=True, metavar="IP",
                   help="This machine's address on an active RDMA port; the pipeline's "
                        "--rdma-remote is IP:PORT.")
    p.add_argument("--port", type=int, default=PORT, help="TCP handshake port.")
    p.add_argument("--rdma-device", default=None, help="Verbs device (default: from --listen).")
    p.add_argument("--rdma-batch", type=int, default=64,
                   help="Images per RDMA write (the DCU's bstream rdma_batch_size).")
    p.add_argument("--rate", type=float, default=0.0, metavar="HZ",
                   help="Send at this many frames/s, like a scan at that dwell (0 = as fast as "
                        "possible).")
    p.add_argument("--series", type=int, default=1, metavar="N",
                   help="Send N series over the connection (one per scan), then wait for the "
                        "receiver to hang up and exit.")
    p.add_argument("--gap", type=float, default=1.0, metavar="S",
                   help="Seconds between two series (the time between two scans).")
    p.add_argument("--first-series-id", type=int, default=1)
    p.add_argument("--cut-after", type=int, default=0, metavar="N",
                   help="Close the connection after N images of the first series, without "
                        "END.")
    args = p.parse_args()

    frames, start, end = load_source(args.source, args.limit)
    print(f"[sender] {len(frames):,} frames {frames.shape[1]}x{frames.shape[2]} uint16 "
          f"({frames.nbytes / 2**30:.2f} GiB) from {args.source.name}", flush=True)
    sender = Sender(args.listen, frames, args.rdma_batch, args.rdma_device)
    with socket.create_server((args.listen, args.port)) as server:
        print(f"[sender] listening on {args.listen}:{args.port} "
              f"({args.series} series, rate {args.rate or 'max'} Hz)", flush=True)
        conn, peer = server.accept()
        with conn:
            qp, hello = sender.handshake(conn)
            try:
                for k in range(args.series):
                    series_id = args.first_series_id + k
                    if k:
                        time.sleep(args.gap)
                    sent, elapsed = sender.send_series(
                        qp, hello, series_id, _series_message(start, "start", series_id, frames),
                        _series_message(end, "end", series_id, frames), args.rate,
                        args.cut_after)
                    print(f"[sender] series {series_id}: {sent:,} frames in {elapsed:.3f} s "
                          f"({sent / max(elapsed, 1e-9) / 1e3:,.1f} kHz) to {peer[0]}"
                          + (" (cut, no END)" if args.cut_after else ""), flush=True)
                    if args.cut_after:
                        return
                conn.settimeout(None)   # stay connected, as the DCU does, until the receiver hangs up
                try:
                    while conn.recv(4096):
                        pass
                except OSError:
                    pass
                print("[sender] the receiver hung up", flush=True)
            finally:
                qp.close()


if __name__ == "__main__":
    main()
