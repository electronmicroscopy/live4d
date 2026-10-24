"""The Holoscan ingest graph: ZmqRxOp → DecompressGpuOp → GenericSinkOp.

`ZmqRxOp` and `DecompressGpuOp` are this repository's C++/CUDA operators (`operators/`), compiled
into the image by `docker/Dockerfile` (imported as `dectris_holoscan_ops`): they receive raw ARINA messages over ZeroMQ and
bslz4-decompress them on the GPU, GIL-free, at full detector rate. The decompressor emits each
batch as CUDA tensors via DLPack — `frames` (B, H, W) uint16/uint32 and `image_ids` (B,) — so
`GenericSinkOp` is a zero-copy handoff into any `Accumulator`.

The graph is algorithm-agnostic: `GenericSinkOp` and `IngestApp` are built around the `Accumulator`
contract, so the same ingest drives parallax today and any future method unchanged.

Runs inside the container (this package is bind-mounted; the image supplies the compiled ops).
See `apps/run.py` for the command line and the README for the docker invocation.
"""

from __future__ import annotations

import torch
from holoscan.core import Application, Operator, OperatorSpec

import dectris_holoscan_ops
from dectris_holoscan_ops import DecompressGpuOp, ZmqRxOp

from realtime_stem.ingest.sink import BatchSink
from realtime_stem.processing.base import Accumulator

#: Version of the compiled operators in this image (None: built before 2.0, without the
#: end-of-scan and integrity fixes).
OPERATORS_VERSION: str | None = getattr(dectris_holoscan_ops, "__version__", None)


def _tensor_from_dlpack(tensor):
    """Convert a Holoscan Tensor-like object to a torch.Tensor without a copy."""
    if hasattr(tensor, "__dlpack__"):
        return torch.from_dlpack(tensor)
    if hasattr(tensor, "to_dlpack"):
        return torch.from_dlpack(tensor.to_dlpack())
    raise TypeError(f"Object does not expose DLPack: {type(tensor)!r}")


def _get_tensor(message, name: str):
    if isinstance(message, dict):
        return message[name]
    if hasattr(message, "get"):
        value = message.get(name)
        if value is not None:
            return value
    return message[name]


class GenericSinkOp(Operator):
    """Feed decompressed batches into any `Accumulator` via the shared `BatchSink` (throughput
    prints and throttled live snapshots live there, identical for the pure-Python ingest)."""

    def __init__(self, fragment, *args, acc: Accumulator, print_every: int = 50,
                 live_publisher=None, live_every: float = 0.2, live_every_rows: int = 0,
                 name: str = "sink"):
        super().__init__(fragment, *args, name=name)
        self.acc = acc
        self.core = BatchSink(acc, print_every=print_every, live_publisher=live_publisher,
                              live_every=live_every, live_every_rows=live_every_rows)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("input")

    def compute(self, op_input, op_output, context) -> None:
        message = op_input.receive("input")
        frames = _tensor_from_dlpack(_get_tensor(message, "frames"))
        # ids stay on the device: a .cpu().numpy() here would cost a full D2H round trip per batch
        # (the engine would re-upload them) and cut throughput. BatchSink's id guard is the
        # cross-stream fence — see the comment there.
        ids = _tensor_from_dlpack(_get_tensor(message, "image_ids")).to(torch.long)
        self.core.feed(frames, ids)

    def summary(self) -> str:
        return self.core.summary()

    def completeness(self) -> tuple[bool, str]:
        return self.core.completeness()


class IngestApp(Application):
    """ZmqRxOp → DecompressGpuOp → GenericSinkOp, configured from the accumulator's geometry."""

    def __init__(self, endpoint: str, batch_size: int, scheduler_threads: int,
                 acc: Accumulator, dtype: str = "uint16", n_connections: int = 3,
                 live_publisher=None, live_every: float = 0.2, live_every_rows: int = 0):
        super().__init__()
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.scheduler_threads = scheduler_threads
        self.acc = acc
        self.dtype = dtype
        self.n_connections = n_connections
        self.live_publisher = live_publisher
        self.live_every = live_every
        self.live_every_rows = live_every_rows
        self.sink: GenericSinkOp | None = None

    def compose(self) -> None:
        if self.scheduler_threads > 0:
            try:
                from holoscan.schedulers import MultiThreadScheduler
                # The grace period stops the scheduler from mistaking the instant between two
                # operators for a deadlock and ending the run early (seen on a busy machine).
                self.scheduler(MultiThreadScheduler(
                    self, worker_thread_number=self.scheduler_threads,
                    stop_on_deadlock_timeout=500))
            except Exception as exc:
                print(f"[holoscan] using default scheduler: {exc}", flush=True)

        # The generic C++ ops take detector geometry + dtype and validate them against the stream.
        # n_connections is a recent parameter (the DCU paces each TCP connection, so full rate
        # needs 3); if the installed ops predate it, construct without it. Ops without the
        # geometry parameters at all (the public live4d "basic" build: fixed 96x96 uint16) are
        # refused with a clear message rather than a bare TypeError.
        hy, hx = self.acc.detector_shape
        op_kwargs = {"detector_width": int(hx), "detector_height": int(hy),
                     "dtype": self.dtype}
        n_images = int(self.acc.scan_shape[0] * self.acc.scan_shape[1])
        try:
            try:
                rx = ZmqRxOp(self, endpoint=self.endpoint, batch_size=self.batch_size,
                             name="zmq_rx", n_connections=int(self.n_connections),
                             expected_images=n_images, **op_kwargs)
            except TypeError:
                # Operators before 2.0 have no expected_images (the start-message check).
                rx = ZmqRxOp(self, endpoint=self.endpoint, batch_size=self.batch_size,
                             name="zmq_rx", n_connections=int(self.n_connections), **op_kwargs)
            decompress = DecompressGpuOp(self, name="decompress_gpu", **op_kwargs)
        except TypeError:
            try:
                rx = ZmqRxOp(self, endpoint=self.endpoint, batch_size=self.batch_size,
                             name="zmq_rx", **op_kwargs)
                decompress = DecompressGpuOp(self, name="decompress_gpu", **op_kwargs)
            except TypeError as exc:
                raise RuntimeError(
                    "the installed dectris_holoscan_ops do not accept detector_width/"
                    "detector_height/dtype: this image carries the fixed 96x96 uint16 'basic' "
                    "operators (e.g. the public live4d build), not the generic ones this "
                    "pipeline needs. Build the image from this repository's docker/Dockerfile "
                    "(README, Prerequisites)."
                ) from exc
            if self.n_connections != 1:
                print("[holoscan] installed ops predate n_connections — using a single "
                      "connection (the DCU caps one connection below full rate; rebuild the "
                      "image from docker/Dockerfile for multi-connection)", flush=True)
        self.sink = GenericSinkOp(self, acc=self.acc, live_publisher=self.live_publisher,
                                  live_every=self.live_every,
                                  live_every_rows=self.live_every_rows, name="sink")

        self.add_flow(rx, decompress, {("output", "input")})
        self.add_flow(decompress, self.sink, {("output", "input")})
