"""The Holoscan SDK 5 ingest: the DCU's frames → GenericSinkOp → any Accumulator, over either transport.

    zmq:  ZmqRxOp (bslz4-compressed Stream2, N connections) → DecompressGpuOp → GenericSinkOp
    rdma: RdmaRxOp (imagemux RDMA, uncompressed, into GPU or mapped host memory) → GenericSinkOp

The receivers and the GPU decompressor are the repository's shared operators (`holoscan_pipeline/`,
compiled into the image built by `docker/Dockerfile.hsdk5`). This module adds only the sink: it
hands each batch to the same `BatchSink` as the other ingests, so the reconstruction, the live
stream and the viewer are unchanged.

The DCU side:
  zmq:  `stream/config/mode` = `enabled`, `stream/config/format` = `cbor`,
        `detector/config/compression` = `bslz4` (the receiver decodes bslz4 only).
  rdma: `bstream/config/mode` = `cbor-rdma`, `detector/config/compression` = `none` (the
        receiver refuses compressed frames), `bstream/config/rdma_batch_size` images per write.

One `Hsdk5Session` = one connection and one running graph for the whole run (as the shared
`live4d` app's service mode). Each scan is a `round(acc)`: the next series' START waits until a
round has handed over its accumulator, so a series never lands in the previous scan's
reconstruction, and nothing is reconnected between two scans. Runs inside the Holoscan 5 container
only (holoscan, live4d and dectris come from the image).
"""

from __future__ import annotations

import socket
import threading

import numpy as np
import torch
import holoscan as hs

from live4d.operators import DecompressGpuOp, RdmaRxOp, ZmqRxOp, _PublicationWindow
from live4d.schemas.decoded_batch import DecodedBatch

from realtime_stem.ingest.sink import BatchSink
from realtime_stem.processing.base import Accumulator

#: The shared receivers' largest batch (holoscan_pipeline/python/live4d/config.py).
MAX_BATCH = 1000

#: What a receiver's protocol error usually means on the DCU side, per transport.
_HINTS = {
    "Invalid BSLZ4 image": "the frames are not bslz4-compressed uint16 images of the size the "
                           "START announced: for --transport zmq set the DCU's "
                           "detector/config/compression to bslz4 (and keep the dwell short "
                           "enough for uint16 frames).",
    "Invalid RDMA stream": "the RDMA frames are compressed or not uint16: for --transport rdma "
                           "set the DCU's detector/config/compression to none (and keep the "
                           "dwell short enough for uint16 frames).",
    "Invalid START image geometry": "the detector sends a frame size or type the receiver does "
                                    "not take: 96x96 (binning 2x2) or 192x192, uint16.",
}


def _torch_stream(runtime_stream) -> torch.cuda.ExternalStream:
    """The runtime's CUDA stream for this callback, as a torch stream: our GPU work queued on it
    is ordered before the runtime reuses the batch's memory (HSDK 5's tensor ownership)."""
    if hasattr(runtime_stream, "__cuda_stream__"):
        handle = runtime_stream.__cuda_stream__()[1]
    else:
        handle = getattr(runtime_stream, "ptr", runtime_stream)
    return torch.cuda.ExternalStream(int(handle))


class GenericSinkOp(hs.Operator):
    """Feed each `DecodedBatch` into the current round's `BatchSink`, on the runtime's CUDA stream."""
    inp = hs.Input("input", DecodedBatch)

    def __init__(self, *, session: "Hsdk5Session"):
        super().__init__()
        self.session = session
        self.core: BatchSink | None = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, ctx):
        batch = self.inp.receive().data
        try:
            with torch.cuda.stream(_torch_stream(ctx.stream())):
                self._handle(batch)
        finally:
            self.session.release(int(batch.batch_id))

    def _handle(self, batch) -> None:
        s = self.session
        if batch.pipeline_ready:
            torch.from_dlpack(batch.frames)          # exercise the tensor handoff before a scan
            torch.cuda.current_stream().synchronize()
            print(f"[holoscan] ready ({s.transport.upper()} receiver connected)", flush=True)
            s.ready.set()
            return
        if batch.start_of_series:
            import cbor2
            self.core = s.take_round()               # waits for run.py to arm the next scan
            if self.core is None:                    # stopping
                return
            announced = cbor2.loads(bytes(batch.start_metadata)).get("number_of_images")
            print(f"[{s.transport}] series {int(batch.series_id)} started ({announced} images "
                  f"announced)", flush=True)
            positions = self.core.acc.scan_shape[0] * self.core.acc.scan_shape[1]
            if announced and announced != positions and not getattr(self.core.acc, "wrap_ids",
                                                                    False):
                print(f"[{s.transport}] WARNING: the detector will send {announced} images, but "
                      f"the scan has {positions} positions: check --scan (or the DCU's "
                      f"ntrigger).", flush=True)
        if self.core is None:
            if len(batch.image_ids):
                raise ValueError("images arrived before their series' START")
            return
        count = len(batch.image_ids)
        if count:
            frames = torch.from_dlpack(batch.frames)
            if tuple(frames.shape[1:]) != s.detector_shape:
                raise ValueError(
                    f"the DCU sends {tuple(frames.shape[1:])} frames but the reconstruction "
                    f"expects {s.detector_shape}: check --binned / --detector against the "
                    f"DCU's binning_mode.")
            # Only the first `count` images are valid (RDMA tensors keep the full batch capacity).
            frames = frames[:count]
            ids = torch.as_tensor(np.asarray(batch.image_ids, dtype=np.int64)).to(
                frames.device, non_blocking=True)
            self.core.feed(frames, ids)
        if batch.end_of_series:
            if int(batch.series_images) != self.core.frames:
                print(f"[{s.transport}] WARNING: the DCU reports {int(batch.series_images)} "
                      f"images in series {int(batch.series_id)}, {self.core.frames} reached the "
                      f"sink", flush=True)
            torch.cuda.current_stream().synchronize()   # the scan's GPU work is done
            self.core = None
            s.series_ended.set()


class Hsdk5Round:
    """One scan on the session: same surface as the Holoscan 4 `IngestApp` for `apps/run.py`,
    `run()` and `sink` (`summary()`, `completeness()`)."""

    def __init__(self, session: "Hsdk5Session", sink: BatchSink, should_stop):
        self.session = session
        self.sink = sink
        self.should_stop = should_stop

    def run(self) -> None:
        """Returns after the series' END or a stop; raises when the connection or graph fails."""
        s = self.session
        s.arm(self.sink)
        while not s.series_ended.wait(0.05):
            if self.should_stop():
                return
            if s.finished.is_set():
                s.raise_failure()


class Hsdk5Session:
    """One connection to the DCU and one running graph for the whole run."""

    def __init__(self, *, transport: str, detector_shape, batch_size: int,
                 endpoint: str = "", n_connections: int = 3,
                 remote: str = "", local: str = "", rdma_device: str = "",
                 ring_memory: str = "gpu", ring_mib: int = 64, max_batch: int = 512,
                 timeout: float = 10.0, queue_depth: int = 4):
        if transport not in ("zmq", "rdma"):
            raise ValueError(f"transport must be zmq or rdma, not {transport!r}")
        self.transport = transport
        self.detector_shape = tuple(int(n) for n in detector_shape)
        self.batch_size = min(int(batch_size), MAX_BATCH)
        self.endpoint, self.n_connections = endpoint, int(n_connections)
        self.remote, self.local = remote, local
        self.rdma_device = rdma_device
        self.ring_memory = ring_memory
        self.ring_mib = int(ring_mib)
        self.max_batch = int(max_batch)
        self.timeout = float(timeout)
        self.queue_depth = int(queue_depth)
        self.ready = threading.Event()
        self.series_ended = threading.Event()
        self.finished = threading.Event()
        self.sender_gone = threading.Event()   # RDMA: the DCU hung up
        self.failure: BaseException | None = None
        self._armed: BatchSink | None = None
        self._cond = threading.Condition()
        self._stopping = threading.Event()
        self._thread = self.verbs = self.link = self.closing = None
        self._windows = []
        self.release = None

    # ── one scan ──────────────────────────────────────────────────────────────────────────────
    def round(self, acc: Accumulator, *, should_stop=None, live_publisher=None,
              live_every: float = 0.2, live_every_rows: int = 0) -> Hsdk5Round:
        if tuple(acc.detector_shape) != self.detector_shape:
            raise ValueError(f"accumulator detector {acc.detector_shape} != {self.detector_shape}")
        sink = BatchSink(acc, live_publisher=live_publisher, live_every=live_every,
                         live_every_rows=live_every_rows)
        return Hsdk5Round(self, sink, should_stop or (lambda: False))

    def arm(self, sink: BatchSink) -> None:
        with self._cond:
            self.series_ended.clear()
            self._armed = sink
            self._cond.notify_all()

    def take_round(self) -> BatchSink | None:
        """Called at a START: the armed scan's sink (waits for it), or None when stopping."""
        with self._cond:
            while self._armed is None and not self._stopping.is_set():
                self._cond.wait(0.05)
            sink, self._armed = self._armed, None
            return sink

    # ── the connection and the graph ──────────────────────────────────────────────────────────
    def open(self) -> None:
        """Connect (RDMA: the handshake with the DCU), start the graph, wait until it is ready."""
        if self.transport == "rdma":
            self._connect_rdma()
        self._plan = self._compile()
        self._thread = threading.Thread(target=self._run, name="hsdk5-graph", daemon=True)
        self._thread.start()
        while not self.ready.wait(0.05):
            if self.finished.is_set():
                self.raise_failure()

    def _connect_rdma(self) -> None:
        from dectris import ImagemuxCodec, LocalInfo, TcpConnector, VerbsConn, watch_liveness
        host, _, port = self.remote.rpartition(":")
        self.verbs = VerbsConn(self.local, torch.cuda.current_device(), self.ring_mib << 20,
                               self.rdma_device or None, ring_memory=self.ring_memory)
        verbs = self.verbs
        try:
            self.link = TcpConnector(host, ImagemuxCodec(),
                                     lambda: LocalInfo(verbs.ring, verbs.endpoint),
                                     verbs.connect, port=int(port), timeout_s=self.timeout).connect()
        except OSError as exc:
            self.close()
            raise ConnectionError(
                f"no RDMA sender answers at {self.remote} ({exc}). Check --rdma-remote (the "
                f"DCU's RDMA address, port 31003), that the DCU's bstream mode is cbor-rdma, "
                f"and --rdma-local (this machine's address on the same RDMA network).") from None
        self.closing, self.sender_gone = watch_liveness(self.link)

    def _compile(self):
        decoded_window = _PublicationWindow(self.queue_depth - 1)
        # The windows must outlive the running graph.
        self._windows.append(decoded_window)
        self.release = decoded_window.release
        graph = hs.Graph(f"realtime-stem-{self.transport}", default_clock=hs.RealtimeClock)
        device = torch.cuda.current_device()
        if self.transport == "zmq":
            compressed_window = _PublicationWindow(self.queue_depth - 1)
            self._windows.append(compressed_window)
            rx = ZmqRxOp(graph, "zmq_rx", endpoint=self.endpoint, batch_size=self.batch_size,
                         n_connections=self.n_connections,
                         publication_window=compressed_window.handle)
            source = DecompressGpuOp(graph, "decompress_gpu",
                                     publication_window=decoded_window.handle,
                                     input_window=compressed_window.handle)
            graph.add_flow(rx.output, source.input, queue_depth=self.queue_depth)
            graph.partition("receiver").add(rx)
            graph.partition("decode").add(source)
            source_name = "decompress_gpu"
        else:
            source = RdmaRxOp(graph, "rdma_rx", qp_handle=self.link.conn.native_handle,
                              ring_address=self.verbs.ring.addr,
                              ring_device_address=self.verbs.ring_device_address,
                              ring_bytes=self.verbs.ring.length, ring_memory=self.ring_memory,
                              credit_address=self.link.peer.credit.addr,
                              credit_rkey=self.link.peer.credit.rkey,
                              batch_size=self.batch_size, max_batch=self.max_batch,
                              device=device, publication_window=decoded_window.handle)
            graph.partition("receiver").add(source)
            source_name = "rdma_rx"
        session = self

        class BoundSink(GenericSinkOp):
            def __init__(self):
                super().__init__(session=session)

        sink = graph.op(BoundSink, "sink")
        # Ports are addressed by their names ("input"), as in live4d.app.
        graph.add_flow(source.output, sink.input, queue_depth=self.queue_depth)
        graph.partition("sink").add(sink)
        deployment = hs.execution.DeploymentBinding()
        deployment.bind_tensor_output_device(operator_path=source_name, output_port="output",
                                             device=hs.execution.DeviceId(device))
        plan = hs.compile(graph, hs.CompileOptions(deployment=deployment))
        if not plan.ok:
            raise RuntimeError(plan.summary())
        return plan

    def _run(self) -> None:
        try:
            with hs.run_async(self._plan) as session:
                while not session.wait(timeout=0.05):
                    if self._stopping.is_set() or self.sender_gone.is_set():
                        session.request_stop()
                if self.sender_gone.is_set() and not self._stopping.is_set():
                    self.failure = ConnectionError("the DCU closed the RDMA connection")
                elif session.termination not in (hs.RunTermination.REQUESTED_COMPLETED,
                                                 hs.RunTermination.COMPLETED,
                                                 hs.RunTermination.STOPPED):
                    summary = session.summary()
                    hints = [hint for key, hint in _HINTS.items() if key in summary]
                    self.failure = RuntimeError(f"Holoscan run failed: {summary}"
                                                + "".join(f"\n  → {h}" for h in hints))
        except BaseException as exc:   # reported by the waiting round
            self.failure = exc
        finally:
            self.finished.set()

    def raise_failure(self) -> None:
        raise self.failure or RuntimeError(f"the {self.transport} receiver stopped")

    def close(self) -> None:
        """Stop the graph, then hang up and release the ring (in that order)."""
        self._stopping.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.link is not None:
            self.closing.set()
            # Link.close is the socket's close(); shut it down first so the DCU sees the hang-up
            # at once (a plain close waits for the liveness thread's pending recv).
            sock = getattr(self.link.close, "__self__", None)
            if isinstance(sock, socket.socket):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self.link.close()
            self.link = None
        if self.verbs is not None:
            self.verbs.close()
            self.verbs = None
