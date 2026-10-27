"""One connection to the detector and one running Holoscan graph, handing every batch to a consumer.

    rdma: RdmaRxOp (imagemux RDMA, uncompressed, into GPU memory) -> SinkOp
    zmq:  ZmqRxOp (bslz4 Stream2) -> DecompressGpuOp -> SinkOp

The receivers are live4d's (`live4d.operators`). `SinkOp` calls the consumer on the runtime's CUDA
stream, inside the graph:

    consumer.on_start(start: dict, series_id: int)     # the parsed START, once per series
    consumer.process(frames, ids)                      # torch (n, H, W) on the GPU, NumPy int64 ids
    consumer.end(series_images: int)                   # the series' END

`frames` are lent for the call only. A consumer must not raise for a faulty series (it would stop
the graph and every later series); it drops that series instead.
"""

from __future__ import annotations

import socket
import threading

import cbor2
import holoscan as hs
import numpy as np
import torch

from live4d.operators import DecompressGpuOp, RdmaRxOp, ZmqRxOp, _PublicationWindow
from live4d.schemas.decoded_batch import DecodedBatch

#: The receivers' largest batch (live4d.config).
MAX_BATCH = 1000


def _torch_stream(runtime_stream) -> torch.cuda.ExternalStream:
    """The runtime's CUDA stream for this callback, as a torch stream: work queued on it is ordered
    before the runtime reuses the batch's memory."""
    if hasattr(runtime_stream, "__cuda_stream__"):
        handle = runtime_stream.__cuda_stream__()[1]
    else:
        handle = getattr(runtime_stream, "ptr", runtime_stream)
    return torch.cuda.ExternalStream(int(handle))


class SinkOp(hs.Operator):
    """Hand each `DecodedBatch` to the session's consumer, then release it to the receiver."""
    inp = hs.Input("input", DecodedBatch)

    def __init__(self, *, session: "ReceiverSession"):
        super().__init__()
        self.session = session

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
            s.ready.set()
            return
        consumer = s.consumer
        if batch.start_of_series:
            consumer.on_start(cbor2.loads(bytes(batch.start_metadata)), int(batch.series_id))
        count = len(batch.image_ids)
        if count:
            # Only the first `count` images are valid (RDMA tensors keep the full batch capacity).
            frames = torch.from_dlpack(batch.frames)[:count]
            consumer.process(frames, np.asarray(batch.image_ids, dtype=np.int64))
        if batch.end_of_series:
            torch.cuda.current_stream().synchronize()
            consumer.end(int(batch.series_images))


class ReceiverSession:
    """One connection to the detector and one running graph for the whole session."""

    def __init__(self, consumer, *, transport: str = "rdma", remote: str = "", local: str = "",
                 rdma_device: str = "", endpoint: str = "", n_connections: int = 3,
                 batch_size: int = 1000, ring_memory: str = "gpu", ring_mib: int = 256,
                 max_batch: int = 512, timeout: float = 10.0, queue_depth: int = 4):
        if transport not in ("zmq", "rdma"):
            raise ValueError(f"transport must be zmq or rdma, not {transport!r}")
        if transport == "rdma" and not (remote and local):
            raise ValueError("rdma needs remote (the detector's HOST:PORT) and local (this machine's address)")
        self.consumer = consumer
        self.transport = transport
        self.remote, self.local, self.rdma_device = remote, local, rdma_device
        self.endpoint, self.n_connections = endpoint, int(n_connections)
        self.batch_size = min(int(batch_size), MAX_BATCH)
        self.ring_memory, self.ring_mib = ring_memory, int(ring_mib)
        self.max_batch, self.timeout, self.queue_depth = int(max_batch), float(timeout), int(queue_depth)
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.sender_gone = threading.Event()
        self.failure: BaseException | None = None
        self._stopping = threading.Event()
        self._thread = self.verbs = self.link = self.closing = None
        self._windows = []
        self.release = None

    def open(self) -> None:
        """Connect (RDMA: the imagemux handshake), start the graph, wait until it is ready."""
        if self.transport == "rdma":
            self._connect_rdma()
        self._plan = self._compile()
        self._thread = threading.Thread(target=self._run, name="live4d-live-graph", daemon=True)
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
            self.link = TcpConnector(host, ImagemuxCodec(), lambda: LocalInfo(verbs.ring, verbs.endpoint),
                                     verbs.connect, port=int(port), timeout_s=self.timeout).connect()
        except OSError as exc:
            self.close()
            raise ConnectionError(f"no RDMA sender answers at {self.remote} ({exc}): check the address, "
                                  "that the detector's bstream mode is cbor-rdma, and the local address.") from None
        self.closing, self.sender_gone = watch_liveness(self.link)

    def _compile(self):
        decoded_window = _PublicationWindow(self.queue_depth - 1)
        self._windows.append(decoded_window)        # the windows must outlive the running graph
        self.release = decoded_window.release
        graph = hs.Graph(f"live4d-live-{self.transport}", default_clock=hs.RealtimeClock)
        device = torch.cuda.current_device()
        if self.transport == "zmq":
            compressed_window = _PublicationWindow(self.queue_depth - 1)
            self._windows.append(compressed_window)
            rx = ZmqRxOp(graph, "zmq_rx", endpoint=self.endpoint, batch_size=self.batch_size,
                         n_connections=self.n_connections, publication_window=compressed_window.handle)
            source = DecompressGpuOp(graph, "decompress_gpu", publication_window=decoded_window.handle,
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

        class BoundSink(SinkOp):
            def __init__(self):
                super().__init__(session=session)

        sink = graph.op(BoundSink, "sink")
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
                    self.failure = ConnectionError("the detector closed the RDMA connection")
                elif session.termination not in (hs.RunTermination.REQUESTED_COMPLETED,
                                                 hs.RunTermination.COMPLETED, hs.RunTermination.STOPPED):
                    self.failure = RuntimeError(f"Holoscan run failed: {session.summary()}")
        except BaseException as exc:   # reported by raise_failure
            self.failure = exc
        finally:
            self.finished.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and not self.finished.is_set()

    def raise_failure(self) -> None:
        raise self.failure or RuntimeError(f"the {self.transport} receiver stopped")

    def close(self) -> None:
        """Stop the graph, then hang up and release the ring (in that order)."""
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.link is not None:
            self.closing.set()
            # shut the socket down first so the sender sees the hang-up at once
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
