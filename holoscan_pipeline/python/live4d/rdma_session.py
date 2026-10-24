"""Process-local attach/drain/detach control for a persistent native RDMA source.

Keep this owner, the registered ring, and the QueuePair alive through the run.
Only the native source may poll an attached QP. Controlled QPs must be created
with signal_every=1 so the final credit write has a completion to drain.
"""
import ctypes

from .operators import _library


class _Snapshot(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "generation", "phase", "attached", "drained", "detached", "series_id", "batch_id",
        "position", "safe_credit", "sent_credit", "writes_outstanding", "starts", "stops",
        "attached_ns", "drained_ns", "detached_ns")]


_library.dectris_rdma_session_create.restype = ctypes.c_void_p
_library.dectris_rdma_session_create.argtypes = []
_library.dectris_rdma_session_destroy.argtypes = [ctypes.c_void_p]
_library.dectris_rdma_session_destroy.restype = None
_library.dectris_rdma_session_attach.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                               ctypes.c_uint64, ctypes.c_uint32]
_library.dectris_rdma_session_attach.restype = ctypes.c_uint64
_library.dectris_rdma_session_detach.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
_library.dectris_rdma_session_detach.restype = ctypes.c_int
_library.dectris_rdma_session_snapshot.argtypes = [ctypes.c_void_p]
_library.dectris_rdma_session_snapshot.restype = _Snapshot


class RdmaSession:
    """Single-slot command mailbox; acknowledgments are tagged by connection generation."""

    def __init__(self):
        self.handle = _library.dectris_rdma_session_create()
        if not self.handle:
            raise MemoryError("RDMA session allocation failed")
        self._owner = None

    def snapshot(self):
        if not self.handle:
            raise RuntimeError("RDMA session is closed")
        value = _library.dectris_rdma_session_snapshot(self.handle)
        result = {name: int(getattr(value, name)) for name, _ in value._fields_}
        result["state"] = ("idle", "attaching", "attached", "drained", "detaching",
                           "detached", "stopped")[result["phase"]]
        return result

    def attach(self, qp, credit):
        if self._owner is not None:
            raise RuntimeError("Previous QP has not been released after detach")
        generation = _library.dectris_rdma_session_attach(
            self.handle, qp.native_handle, credit.addr, credit.rkey)
        if not generation:
            raise RuntimeError("RDMA session cannot attach in its current state")
        self._owner = qp
        return int(generation)

    def detach(self, generation):
        if not _library.dectris_rdma_session_detach(self.handle, generation):
            raise RuntimeError("RDMA detach requires the current generation to be drained")

    def release_detached(self, generation):
        status = self.snapshot()
        if status["state"] != "detached" or status["detached"] != generation:
            raise RuntimeError("RDMA detach has not been acknowledged")
        self._owner = None

    def close(self):
        if not self.handle:
            return
        state = self.snapshot()
        if state["state"] not in ("idle", "detached", "stopped") or (
                state["starts"] and not state["stops"]):
            raise RuntimeError("Stop/join the HSDK run before closing its session mailbox")
        _library.dectris_rdma_session_destroy(self.handle)
        self.handle = None
        self._owner = None
