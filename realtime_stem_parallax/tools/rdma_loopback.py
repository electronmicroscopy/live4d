"""Send a scan through the RDMA ingest on this machine alone: a loopback test without a DCU.

    python3 -m tools.rdma_loopback --frames out/sim256.npy --rdma-local 10.10.10.1 \
        $(cat out/sim256.flags) --out-dir out/rdma

A sender in this process writes the frames into the receiver's ring exactly as the DCU's imagemux
does (uncompressed, `--rdma-batch` images per RDMA write, START and END around them), over a
second queue pair on the same RDMA port. The receiver is the real one (`RdmaRxOp`, then
`rdma_graph.GenericSinkOp` and the reconstruction), so this checks the full RDMA path, its
correctness against the same frames through `--ingest python --source`, and its rate.

Runs in the Holoscan 5 image (`docker/Dockerfile.hsdk5`) with `--device /dev/infiniband
--ulimit memlock=-1:-1`. `--rdma-local` is this machine's address on an active RDMA port (the
traffic stays inside the network card). The algorithm flags are those of `apps/run.py`.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np   # noqa: E402

from realtime_stem.apps.run import build_parser   # noqa: E402
from realtime_stem.processing import registry   # noqa: E402


class LoopbackSender:
    """A second queue pair that writes imagemux frames into the receiver's ring and follows its
    credit word, as the DCU does. `records` are built up front into one registered buffer."""

    def __init__(self, ctx, ring, ring_bytes: int):
        from dectris import imxrdma
        self.imxrdma = imxrdma
        self.ring = ring
        self.ring_bytes = ring_bytes
        self.credit = np.zeros(1, dtype=np.uint64)
        self.credit_mr = ctx.register(self.credit.ctypes.data, 8, remote_write=True)
        self.qp = ctx.create_qp(recv_depth=1, send_depth=256, signal_every=1)
        self.ctx = ctx
        self.wires: list[tuple[int, int, int]] = []   # (source offset, length, ring position)
        self.chunks: list[bytes | memoryview] = []
        self.position = self.sequence = self.size = 0
        self.source = self.source_mr = None

    def add(self, kind, data=b"", *, series_id=1, image=None) -> None:
        from dectris.imagemux import FrameHeader, FrameKind, frame_span
        offset = self.position % self.ring_bytes
        span = frame_span(len(data))
        if kind != FrameKind.WRAP and offset + span > self.ring_bytes:
            self.add(FrameKind.WRAP)
            offset = 0
        header = (FrameHeader(kind, self.sequence, self.position, len(data), series_id, image=image)
                  if image is not None else
                  FrameHeader(kind, self.sequence, self.position, len(data), series_id))
        wire_len = 128 + len(data)
        self.wires.append((self.size, wire_len, self.position))
        self.chunks += [header.encode(), data]
        self.size += wire_len
        self.position += self.ring_bytes - offset if kind == FrameKind.WRAP else span
        self.sequence += 1

    def build(self) -> None:
        self.source = np.empty(self.size, dtype=np.uint8)
        at = 0
        for chunk in self.chunks:
            n = len(chunk)
            self.source[at:at + n] = np.frombuffer(chunk, dtype=np.uint8)
            at += n
        self.chunks = []
        self.source_mr = self.ctx.register(self.source.ctypes.data, self.source.nbytes)

    def credit_position(self) -> int:
        import ctypes
        return ctypes.c_uint64.from_address(self.credit.ctypes.data).value

    def send(self, frames_per_record: list[int], rate: float = 0.0, timeout: float = 30.0) -> float:
        """Write every record; with `rate` > 0 frames/s, pace IMAGE records to it. Returns the
        seconds from the first to the last write."""
        imms = np.empty(256, dtype=np.uint32)

        def poll():
            _, _, failed = self.qp.poll(imms)
            if failed:
                raise RuntimeError(f"sender completion failed: {self.qp.errors}")

        t0 = time.perf_counter()
        sent = 0
        for (src, length, pos), n_frames in zip(self.wires, frames_per_record):
            deadline = time.monotonic() + timeout
            end = pos + length
            while end - self.credit_position() > self.ring_bytes:
                if time.monotonic() > deadline:
                    raise TimeoutError("the receiver returned no ring credit")
                poll()
            if rate > 0 and n_frames:
                while time.perf_counter() - t0 < sent / rate:
                    pass
            while not self.qp.write(self.source_mr, src, length,
                                    self.ring.addr + pos % self.ring_bytes, self.ring.rkey,
                                    imm=self.sequence_of(pos) & 0xFFFFFFFF):
                if time.monotonic() > deadline:
                    raise TimeoutError("the send queue stayed full")
                poll()
            sent += n_frames
        elapsed = time.perf_counter() - t0
        deadline = time.monotonic() + timeout
        while self.qp.stats["writes_outstanding"]:
            if time.monotonic() > deadline:
                raise TimeoutError("writes did not complete")
            poll()
        return elapsed

    def sequence_of(self, pos: int) -> int:
        return self._seq_by_pos[pos]

    def index(self) -> None:
        self._seq_by_pos = {pos: i for i, (_, _, pos) in enumerate(self.wires)}

    def close(self) -> None:
        for name in ("qp", "source_mr", "credit_mr"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.close()
                setattr(self, name, None)


def main() -> None:
    import cbor2
    import torch
    from dectris import VerbsConn
    from dectris.imagemux import Compression, DataType, FrameKind, ImageHeader
    from realtime_stem.ingest.rdma_graph import MAX_BATCH, RdmaIngestApp

    argv = sys.argv[1:]
    algorithm = registry.get("parallax")
    parser = build_parser(algorithm)
    g = parser.add_argument_group("loopback sender")
    g.add_argument("--frames", type=Path, required=True,
                   help="A .npy of uint16 frames, (Ny, Nx, H, W) or (N, H, W) in scan order.")
    g.add_argument("--rdma-batch", type=int, default=64,
                   help="Images per RDMA write (the DCU's bstream rdma_batch_size).")
    g.add_argument("--send-rate", type=float, default=0.0, metavar="HZ",
                   help="Pace the sender to this many frames/s (0 = as fast as possible).")
    args = parser.parse_args(algorithm.preprocess_argv(argv))
    if not args.rdma_local:
        parser.error("--rdma-local IP (this machine's address on an active RDMA port) is needed")
    args.batch_size = min(args.batch_size, MAX_BATCH)

    frames = np.load(args.frames)
    if frames.dtype != np.uint16:
        raise SystemExit(f"{args.frames}: {frames.dtype} frames; the RDMA path carries uint16")
    frames = np.ascontiguousarray(frames.reshape(-1, *frames.shape[-2:]))
    n, h, w = frames.shape
    acc = algorithm.build(args)
    if (h, w) != tuple(acc.detector_shape):
        raise SystemExit(f"frames are {h}x{w}, the reconstruction expects {acc.detector_shape}")

    ring_bytes = args.rdma_ring_mib << 20
    verbs = VerbsConn(args.rdma_local, torch.cuda.current_device(), ring_bytes,
                      args.rdma_device or None, ring_memory=args.rdma_ring_memory)
    sender = LoopbackSender(verbs.ctx, verbs.ring, ring_bytes)
    try:
        verbs.connect(SimpleNamespace(endpoint=sender.qp.endpoint))
        sender.qp.connect(verbs.qp.endpoint)

        start = cbor2.dumps(cbor2.CBORTag(55799, {
            "type": "start", "series_id": 1, "image_size_x": w, "image_size_y": h,
            "image_dtype": "uint16", "number_of_images": n}))
        per_record = [0]
        sender.add(FrameKind.START, start)
        for first in range(0, n, args.rdma_batch):
            chunk = frames[first:first + args.rdma_batch]
            payload = memoryview(chunk).cast("B")
            image = ImageHeader(image_id=first, batch_id=first // args.rdma_batch,
                                batch_size=len(chunk), dim_x=w, dim_y=h, dtype=DataType.UINT16,
                                compression=Compression.NONE, uncompressed_size=len(payload),
                                channel=0, channel_count=1)
            sender.add(FrameKind.IMAGE, payload, image=image)
            per_record.append(len(chunk))
        sender.add(FrameKind.END, cbor2.dumps({"type": "end", "series_id": 1}))
        per_record.append(0)
        sender.build()
        sender.index()
        counts = _frames_per_wire(sender, per_record)   # WRAP records carry no frames
        print(f"[loopback] {n} frames {h}x{w} uint16 in {len(sender.wires)} RDMA writes "
              f"({sender.size / 2**30:.2f} GiB), ring {args.rdma_ring_memory} "
              f"{args.rdma_ring_mib} MiB", flush=True)

        app = RdmaIngestApp(acc, remote="loopback:0", local=args.rdma_local,
                            batch_size=args.batch_size, ring_memory=args.rdma_ring_memory)
        result = {}

        def send():
            if not app.ready.wait(60):
                result["error"] = "the receiver never became ready"
                return
            try:
                result["elapsed"] = sender.send(counts, rate=args.send_rate)
            except BaseException as exc:   # reported after the run
                result["error"] = repr(exc)

        thread = threading.Thread(target=send, name="loopback-sender", daemon=True)
        thread.start()
        credit = SimpleNamespace(addr=sender.credit_mr.address, rkey=sender.credit_mr.rkey)
        app.run_on(verbs.qp, verbs.ring, credit, verbs.ring_device_address)
        thread.join(timeout=10)
        if "error" in result:
            raise SystemExit(f"[loopback] sender: {result['error']}")
        print(f"[loopback] sender wrote {n} frames in {result['elapsed']:.3f} s "
              f"({n / result['elapsed'] / 1e3:,.1f} kHz)", flush=True)
        print(app.sink.summary(), flush=True)
        print(app.sink.completeness()[1], flush=True)
        algorithm.save(acc, args.out_dir)
    finally:
        sender.close()
        verbs.close()


def _frames_per_wire(sender: LoopbackSender, per_record: list[int]) -> list[int]:
    """Frame counts per written record, with 0 for the WRAP records the ring inserted."""
    from dectris.imagemux import FrameHeader, FrameKind
    out, it = [], iter(per_record)
    for src, _, _ in sender.wires:
        kind = FrameHeader.decode(sender.source[src:src + 128].tobytes()).kind
        out.append(0 if kind == FrameKind.WRAP else next(it))
    return out


if __name__ == "__main__":
    main()
