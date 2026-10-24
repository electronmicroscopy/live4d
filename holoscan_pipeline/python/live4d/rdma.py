"""Acquisition routing around fixed-device RDMA/processing branches.

One registered ring and one result slot per GPU; only one live sender connection.
The detector controller must wait for each receiver_ready notification before ARM.
"""
import json
import threading
import time

import cupy as cp
import numpy as np

from dectris import imxrdma
from dectris.connection import DRAIN_S, RegisteredRing, RECV_DEPTH, SEND_DEPTH, connect_qp
from dectris.imagemux import ImagemuxCodec, LocalInfo, QpEndpoint, TcpConnector
from .rdma_session import RdmaSession


class ResultSlot:
    """A bounded asynchronous CPU result, independent of runtime stream lifetime."""

    def __init__(self, device):
        self.device = device
        self.available = threading.Event()
        self.available.set()
        self.lock = threading.Lock()
        self.pending = None
        with cp.cuda.Device(device):
            self.host = cp.cuda.alloc_pinned_memory(16)

    def submit(self, totals, finish):
        with self.lock:
            if self.pending is not None or self.available.is_set():
                raise RuntimeError("RDMA result slot was not reserved for this acquisition")
            stream = cp.cuda.get_current_stream()
            event = cp.cuda.Event(block=False, disable_timing=True)
            try:
                cp.cuda.runtime.memcpyAsync(self.host.ptr, totals.data.ptr, 16,
                                           cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
                event.record(stream)
            except BaseException:
                # Error path only: keep both copy buffers alive if event recording failed.
                stream.synchronize()
                raise
            # finish retains the map and END metadata. Never retain the runtime stream.
            self.pending = (event, totals, finish)

    def poll(self):
        with self.lock:
            if self.pending is None:
                return False
            event, totals, finish = self.pending
            with cp.cuda.Device(self.device):
                if not event.done:
                    return False
                finish(np.frombuffer(self.host, dtype=np.float64, count=2).tolist())
            self.pending = None
            self.available.set()
            return True

    def close(self):
        # Called only after runtime join, including failure/stop with GPU work pending.
        with self.lock, cp.cuda.Device(self.device):
            if self.pending is not None:
                self.pending[0].synchronize()
                self.pending = None


class RdmaReceiver:
    """Rotate whole acquisitions across a nonempty device list, including length one.

    tick() runs on the application's control thread. Native operators exclusively
    poll attached QPs; detach acknowledgment precedes closing a QP. Ring and mailbox
    destruction is permitted only after the Holoscan runtime has joined.
    """

    def __init__(self, config, receiver_ready=None):
        self.config, self.receiver_ready = config, receiver_ready
        self.context = None
        self.rings, self.sessions, self.results = [], [], []
        self.link = None
        self.active = self.acquisition = self.generation = self.completed = 0
        self.detaching = False
        self.connect_deadline = None
        self.next_attempt = 0.0
        self.last_ready = None
        self.dead_since = None
        rdma = config.rdma
        try:
            self.context = imxrdma.Context(local_addr=rdma["local_address"],
                                          device=rdma["device"] or None)
            for device in config.gpu_devices:
                self.rings.append(RegisteredRing(self.context, device, rdma["ring_mib"] << 20,
                                                 ring_memory=rdma["ring_memory"]))
                self.sessions.append(RdmaSession())
                self.results.append(ResultSlot(device))
        except BaseException:
            self.close()
            raise

    def _close_link(self):
        if self.link is not None:
            self.link.close()
            self.link.conn.close()
            self.link = None
        self.last_ready = None

    def _connect(self):
        rdma = self.config.rdma
        host, _, port = rdma["remote_address"].rpartition(":")
        ring = self.rings[self.active]
        qp = self.context.create_qp(recv_depth=RECV_DEPTH, send_depth=SEND_DEPTH, signal_every=1)
        endpoint = qp.endpoint
        local = LocalInfo(ring.region, QpEndpoint(endpoint.qp_num, endpoint.psn, endpoint.lid,
                                                endpoint.mtu, endpoint.gid))
        try:
            link = TcpConnector(host, ImagemuxCodec(), lambda: local,
                lambda peer: connect_qp(qp, peer), port=int(port),
                timeout_s=min(1.0, rdma["timeout"])).connect()
        except BaseException:
            qp.close()
            raise
        self.link = link
        self.dead_since = None
        self.generation = self.sessions[self.active].attach(link.conn, link.peer.credit)
        self.results[self.active].available.clear()

    def tick(self):
        for slot in self.results:
            if slot.poll():
                self.completed += 1
        if self.config.execute["mode"] == "finite" and self.completed:
            return
        control = self.sessions[self.active]
        if self.link is None:
            if not self.results[self.active].available.is_set():
                return
            now = time.monotonic()
            if now < self.next_attempt:
                return
            if self.connect_deadline is None:
                self.connect_deadline = now + self.config.rdma["timeout"]
            try:
                self._connect()
            except (ConnectionError, OSError):
                if time.monotonic() >= self.connect_deadline:
                    raise
                self.next_attempt = time.monotonic() + 0.01
            return
        status = control.snapshot()
        if self.detaching:
            if status["detached"] == self.generation:
                control.release_detached(self.generation)
                self._close_link()
                self.active = (self.active + 1) % len(self.rings)
                self.acquisition += 1
                self.detaching = False
                self.connect_deadline = None
            return
        if status["drained"] == self.generation:
            self.last_ready = None
            if self.config.execute["mode"] == "service":
                control.detach(self.generation)
                self.detaching = True
            return
        if not self.link.liveness.alive:
            self.dead_since = self.dead_since or time.monotonic()
            if time.monotonic() - self.dead_since >= DRAIN_S:
                raise ConnectionError(self.link.liveness.reason)
            return
        if status["attached"] == self.generation and self.last_ready is None:
            self.last_ready = dict(acquisition=self.acquisition, device=self.rings[self.active].gpu,
                                   generation=self.generation)
            print("[rdma-ready] " + json.dumps(self.last_ready), flush=True)
            if self.receiver_ready is not None:
                self.receiver_ready(dict(self.last_ready))

    def close(self):
        self._close_link()
        for result in self.results:
            result.close()
        for session in self.sessions:
            session.close()
        for ring in reversed(self.rings):
            ring.close()
        self.context = None
