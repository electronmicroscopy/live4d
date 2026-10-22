"""Connection and run helpers of a receiver app: the queue pair and GPU ring (VerbsConn), stopping
when imagemux goes away (watch_liveness), and the scheduler (run). examples/pipeline/pipeline.py
puts them together with its own operators.

The RDMA verbs come from imxrdma (native/, plain libibverbs).
"""

import threading
import time

import cupy as cp
from holoscan.core import Application
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler

from . import imxrdma
from .imagemux import Link, Peer, QpEndpoint, RemoteRegion

DRAIN_S = 1.0  # time to process what is still in the ring when imagemux closes the handshake socket
RECV_DEPTH = 1024  # RECEIVEs kept posted; must exceed the sender's in-flight frame limit
SEND_DEPTH = 64  # credit writes in flight


class VerbsConn:
    """Our RC queue pair on the RoCE interface of ``local_addr`` and the ring in the memory of
    GPU ``gpu`` (GPUDirect RDMA, needs nvidia_peermem), registered for imagemux to write into.
    The queue pair parameters are exchanged over imagemux's handshake, not rdma_cm."""

    def __init__(self, local_addr: str, gpu: int, ring_bytes: int, device: str | None = None):
        self.ctx = imxrdma.Context(local_addr=local_addr, device=device)
        with cp.cuda.Device(gpu):
            # Straight from cudaMalloc, not the pool: one allocation, aligned for GPUDirect.
            self.ring_mem = cp.cuda.Memory(ring_bytes)
            cp.cuda.runtime.memset(self.ring_mem.ptr, 0, ring_bytes)
        self.ring_mr = self.ctx.register(self.ring_mem.ptr, ring_bytes, remote_write=True)
        self.qp = self.ctx.create_qp(recv_depth=RECV_DEPTH, send_depth=SEND_DEPTH)
        e = self.qp.endpoint
        self.endpoint = QpEndpoint(e.qp_num, e.psn, e.lid, e.mtu, e.gid)
        self.ring = RemoteRegion(self.ring_mr.address, self.ring_mr.length, self.ring_mr.rkey)

    def connect(self, peer: Peer):
        """Connect the queue pair to imagemux's endpoint, then post RECV_DEPTH zero-length
        RECEIVEs. Called before Ready is sent, so frames may arrive as soon as it returns."""
        e = peer.endpoint
        self.qp.connect(imxrdma.Endpoint(e.qp_num, e.psn, e.lid, e.mtu, e.gid))
        posted = self.qp.post_recvs(RECV_DEPTH)
        if posted != RECV_DEPTH:
            raise RuntimeError(f"posted {posted}/{RECV_DEPTH} RECEIVEs")
        return self.qp

    def close(self):
        """Queue pair first, then the ring's registration, then the ring."""
        self.qp.close()
        self.ring_mr.close()
        self.ring_mem = None


def watch_liveness(app: Application, link: Link) -> threading.Event:
    """Stop the app when imagemux's handshake connection drops, after a moment to drain the ring
    (a series can be over in milliseconds). Set the returned event before closing the link
    ourselves, which would otherwise look like imagemux going away."""
    closing = threading.Event()

    def stop():
        link.liveness.wait_dead()
        if closing.is_set():
            return
        print(f"sender gone: {link.liveness.reason}", flush=True)
        time.sleep(DRAIN_S)
        app.stop_execution()
    threading.Thread(target=stop, name="imagemux-stop", daemon=True).start()
    return closing


def run(app, seconds, threads=1):
    """Run for `seconds` (0 = until Ctrl+C). threads > 1 lets operators run concurrently."""
    limit = {"max_duration_ms": int(seconds * 1000)} if seconds > 0 else {}
    if threads > 1:
        app.scheduler(EventBasedScheduler(app, name="sched", worker_thread_number=threads,
                                          **limit))
    else:
        app.scheduler(GreedyScheduler(app, name="sched", **limit))
    app.run()
