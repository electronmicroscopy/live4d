"""Own RDMA connection resources and report sender disconnection.

The RDMA verbs come from imxrdma (native/, plain libibverbs).
"""

import ctypes
import threading

import cupy as cp

from . import imxrdma
from .dispatch import mapped_device_pointer
from .imagemux import FRAME_ALIGN, MAX_RINGS, Link, Peer, QpEndpoint, RemoteRegion

DRAIN_S = 1.0  # time to process what is still in the ring when imagemux closes the handshake socket
RECV_DEPTH = 1024  # RECEIVEs kept posted; must exceed the sender's in-flight frame limit
SEND_DEPTH = 64  # credit writes in flight


class VerbsConn:
    """An RC queue pair and rings registered for imagemux's writes.

    ``gpu`` selects the GPU that processes the images. ``ring_memory='gpu'`` allocates the
    rings with cudaMalloc and requires GPUDirect RDMA support. ``'host_mapped'`` registers
    pinned host memory instead; ``ring_device_address`` is its CUDA alias. ``rings`` equal
    rings of ``ring_bytes`` each (protocol 3) share one registration, ring i at offset
    ``i * ring_bytes``. ``ring`` describes the whole registered region and ``rings`` each
    ring advertised to imagemux; with one ring both are the same.
    """

    def __init__(self, local_addr: str, gpu: int, ring_bytes: int, device: str | None = None,
                 *, ring_memory: str = "gpu", rings: int = 1):
        if ring_memory not in ("gpu", "host_mapped"):
            raise ValueError("ring_memory must be 'gpu' or 'host_mapped'")
        if not 1 <= rings <= MAX_RINGS or ring_bytes % FRAME_ALIGN:
            raise ValueError(f"need 1 to {MAX_RINGS} rings of a multiple of {FRAME_ALIGN} bytes")
        self.gpu, self.ring_memory = gpu, ring_memory
        self.ring_bytes, self.ring_count = ring_bytes, rings
        ring_bytes *= rings
        self.ctx = self.ring_mem = self.ring_mr = self.qp = None
        try:
            self.ctx = imxrdma.Context(local_addr=local_addr, device=device)
            with cp.cuda.Device(gpu):
                if ring_memory == "host_mapped":
                    self.ring_mem = cp.cuda.PinnedMemory(ring_bytes, cp.cuda.runtime.hostAllocMapped)
                    ctypes.memset(self.ring_mem.ptr, 0, ring_bytes)
                    self.ring_device_address = mapped_device_pointer(self.ring_mem.ptr, gpu)
                else:
                    self.ring_mem = cp.cuda.Memory(ring_bytes)
                    cp.cuda.runtime.memset(self.ring_mem.ptr, 0, ring_bytes)
                    self.ring_device_address = self.ring_mem.ptr
            self.ring_mr = self.ctx.register(self.ring_mem.ptr, ring_bytes, remote_write=True)
            self.qp = self.ctx.create_qp(recv_depth=RECV_DEPTH, send_depth=SEND_DEPTH)
            e = self.qp.endpoint
            self.endpoint = QpEndpoint(e.qp_num, e.psn, e.lid, e.mtu, e.gid)
            self.ring = RemoteRegion(self.ring_mr.address, self.ring_mr.length, self.ring_mr.rkey)
            self.rings = tuple(RemoteRegion(self.ring.addr + i * self.ring_bytes, self.ring_bytes,
                                            self.ring.rkey) for i in range(rings))
        except Exception:
            self.close()
            raise

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
        """Stop NIC writes, finish GPU reads, then deregister and release the ring."""
        if self.qp is not None:
            self.qp.close()
            self.qp = None
        if self.ring_mem is not None and self.ring_memory == "host_mapped":
            with cp.cuda.Device(self.gpu):
                cp.cuda.runtime.deviceSynchronize()
        if self.ring_mr is not None:
            self.ring_mr.close()
            self.ring_mr = None
        self.ring_mem = None
        self.ctx = None


def watch_liveness(link: Link) -> tuple[threading.Event, threading.Event]:
    """Return events for local closure and sender disconnection after the drain interval.

    Set the first event before closing the link locally. The application's session loop
    watches the second event and requests cooperative shutdown when it is set.
    """
    closing = threading.Event()
    sender_gone = threading.Event()

    def stop():
        link.liveness.wait_dead()
        if closing.is_set():
            return
        print(f"sender gone: {link.liveness.reason}", flush=True)
        if not closing.wait(DRAIN_S):
            sender_gone.set()

    threading.Thread(target=stop, name="imagemux-stop", daemon=True).start()
    return closing, sender_gone
