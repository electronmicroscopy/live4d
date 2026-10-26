"""The Holoscan SDK 5 RDMA ingest: DCU imagemux RDMA (uncompressed) → GenericSinkOp → any Accumulator.

The DCU writes each frame, uncompressed, straight into a registered ring (GPU memory, or mapped
pinned host memory); the repository's shared `RdmaRxOp` (`holoscan_pipeline/`, compiled into the
image built by `docker/Dockerfile.hsdk5`) gathers them into GPU batches. This module adds only
the sink: it hands each batch to the same `BatchSink` as the ZMQ ingest, so the reconstruction,
the live stream and the viewer are unchanged.

The DCU side: `bstream/config/mode` = `cbor-rdma`, `detector/config/compression` = `none` (the
receiver refuses compressed frames), `bstream/config/rdma_batch_size` images per RDMA write.
`tools/detector_setup.py --transport rdma` sets them.

One `RdmaIngestApp` = one series, like the ZMQ `IngestApp`: it connects (TCP handshake on the
DCU's port 31003, then RDMA), runs until the series' END (or a stop), and disconnects.
Runs inside the Holoscan 5 container only (holoscan, live4d and dectris come from the image).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import torch
import holoscan as hs

from live4d.operators import RdmaRxOp, _PublicationWindow
from live4d.schemas.decoded_batch import DecodedBatch

from realtime_stem.ingest.sink import BatchSink
from realtime_stem.processing.base import Accumulator

#: The shared receiver's largest batch (holoscan_pipeline/python/live4d/config.py).
MAX_BATCH = 1000


def _torch_stream(runtime_stream) -> torch.cuda.ExternalStream:
    """The runtime's CUDA stream for this callback, as a torch stream: our GPU work queued on it
    is ordered before the runtime reuses the batch's memory (HSDK 5's tensor ownership)."""
    if hasattr(runtime_stream, "__cuda_stream__"):
        handle = runtime_stream.__cuda_stream__()[1]
    else:
        handle = getattr(runtime_stream, "ptr", runtime_stream)
    return torch.cuda.ExternalStream(int(handle))


class GenericSinkOp(hs.Operator):
    """Feed each `DecodedBatch` into the shared `BatchSink`, on the runtime's CUDA stream."""
    inp = hs.Input("input", DecodedBatch)

    def __init__(self, *, core: BatchSink, detector_shape, ready: threading.Event, release,
                 ended: threading.Event):
        super().__init__()
        self.core = core
        self.detector_shape = tuple(int(n) for n in detector_shape)
        self.ready = ready
        self.release = release
        self.ended = ended
        self.series_id = None
        self.announced = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, ctx):
        batch = self.inp.receive().data
        try:
            with torch.cuda.stream(_torch_stream(ctx.stream())):
                self._handle(batch, ctx)
        finally:
            self.release(int(batch.batch_id))

    def _handle(self, batch, ctx) -> None:
        if batch.pipeline_ready:
            torch.from_dlpack(batch.frames)          # exercise the tensor handoff before a scan
            torch.cuda.current_stream().synchronize()
            print("[holoscan] ready (RDMA receiver connected)", flush=True)
            self.ready.set()
            return
        if batch.start_of_series:
            import cbor2
            self.series_id = int(batch.series_id)
            self.announced = cbor2.loads(bytes(batch.start_metadata)).get("number_of_images")
            print(f"[rdma] series {self.series_id} started ({self.announced} images announced)",
                  flush=True)
        count = len(batch.image_ids)
        if count:
            frames = torch.from_dlpack(batch.frames)
            if tuple(frames.shape[1:]) != self.detector_shape:
                raise ValueError(
                    f"the DCU sends {tuple(frames.shape[1:])} frames but the reconstruction "
                    f"expects {self.detector_shape}: check --binned / --detector against the "
                    f"DCU's binning_mode.")
            # Only the first `count` images are valid (RDMA tensors keep the full batch capacity).
            frames = frames[:count]
            ids = torch.as_tensor(np.asarray(batch.image_ids, dtype=np.int64)).to(
                frames.device, non_blocking=True)
            self.core.feed(frames, ids)
        if batch.end_of_series:
            if int(batch.series_images) != self.core.frames:
                print(f"[rdma] WARNING: the DCU reports {int(batch.series_images)} images in "
                      f"series {int(batch.series_id)}, {self.core.frames} reached the sink",
                      flush=True)
            self.ended.set()
            ctx.request_completion()


class RdmaIngestApp:
    """One series over RDMA: connect, receive until the series' END (or a stop), disconnect.

    Same surface as the ZMQ `IngestApp` for `apps/run.py`: `run()` and `sink` (`summary()`,
    `completeness()`)."""

    def __init__(self, acc: Accumulator, *, remote: str, local: str, batch_size: int,
                 rdma_device: str = "", ring_memory: str = "gpu", ring_mib: int = 64,
                 max_batch: int = 512, timeout: float = 10.0, queue_depth: int = 4,
                 live_publisher=None, live_every: float = 0.2, live_every_rows: int = 0,
                 should_stop=None):
        self.acc = acc
        self.remote, self.local = remote, local
        self.batch_size = min(int(batch_size), MAX_BATCH)
        self.rdma_device = rdma_device
        self.ring_memory = ring_memory
        self.ring_mib = int(ring_mib)
        self.max_batch = int(max_batch)
        self.timeout = float(timeout)
        self.queue_depth = int(queue_depth)
        self.should_stop = should_stop or (lambda: False)
        self.core = BatchSink(acc, live_publisher=live_publisher, live_every=live_every,
                              live_every_rows=live_every_rows)
        # run.py reads `sink` for the end-of-scan report (BatchSink has summary/completeness).
        self.sink = self.core
        self.ready = threading.Event()
        self.ended = threading.Event()

    def _compile(self, conn, ring, credit, ring_device_address):
        window = _PublicationWindow(self.queue_depth - 1)
        graph = hs.Graph("realtime-stem-rdma", default_clock=hs.RealtimeClock)
        rx = RdmaRxOp(graph, "rdma_rx", qp_handle=conn.native_handle,
                      ring_address=ring.addr, ring_device_address=ring_device_address,
                      ring_bytes=ring.length, ring_memory=self.ring_memory,
                      credit_address=credit.addr, credit_rkey=credit.rkey,
                      batch_size=self.batch_size, max_batch=self.max_batch,
                      device=torch.cuda.current_device(), publication_window=window.handle)
        core, shape, ready, ended = self.core, self.acc.detector_shape, self.ready, self.ended
        release = window.release

        class BoundSink(GenericSinkOp):
            def __init__(self):
                super().__init__(core=core, detector_shape=shape, ready=ready, release=release,
                                 ended=ended)

        sink = graph.op(BoundSink, "sink")
        # Ports are addressed by their names ("input"), as in live4d.app.
        graph.add_flow(rx.output, sink.input, queue_depth=self.queue_depth)
        graph.partition("receiver").add(rx)
        graph.partition("sink").add(sink)
        deployment = hs.execution.DeploymentBinding()
        deployment.bind_tensor_output_device(
            operator_path="rdma_rx", output_port="output",
            device=hs.execution.DeviceId(torch.cuda.current_device()))
        plan = hs.compile(graph, hs.CompileOptions(deployment=deployment))
        if not plan.ok:
            raise RuntimeError(plan.summary())
        return plan, window

    def run(self) -> None:
        from dectris import ImagemuxCodec, LocalInfo, TcpConnector, VerbsConn, watch_liveness
        host, _, port = self.remote.rpartition(":")
        verbs = VerbsConn(self.local, torch.cuda.current_device(), self.ring_mib << 20,
                          self.rdma_device or None, ring_memory=self.ring_memory)
        try:
            link = TcpConnector(host, ImagemuxCodec(), lambda: LocalInfo(verbs.ring, verbs.endpoint),
                                verbs.connect, port=int(port), timeout_s=self.timeout).connect()
            closing, sender_gone = watch_liveness(link)
            try:
                self.run_on(link.conn, verbs.ring, link.peer.credit, verbs.ring_device_address,
                            sender_gone)
            finally:
                closing.set()
                link.close()
        finally:
            verbs.close()

    def run_on(self, conn, ring, credit, ring_device_address, sender_gone=None) -> None:
        """Receive one series on an already connected queue pair (`run`, or a loopback test:
        `tools/rdma_loopback.py`). Returns after the series' END, a stop, or the sender leaving."""
        sender_gone = sender_gone or threading.Event()
        # The connection, the ring and the window must outlive the running graph.
        plan, window = self._compile(conn, ring, credit, ring_device_address)
        with hs.run_async(plan) as session:
            while not session.wait(timeout=0.05):
                if self.should_stop() or sender_gone.is_set():
                    if sender_gone.is_set():
                        print("[rdma] the DCU closed the connection", flush=True)
                    session.request_stop()
            if session.termination not in (hs.RunTermination.REQUESTED_COMPLETED,
                                           hs.RunTermination.COMPLETED,
                                           hs.RunTermination.STOPPED):
                raise RuntimeError(f"Holoscan run failed: {session.summary()}")
        del window
