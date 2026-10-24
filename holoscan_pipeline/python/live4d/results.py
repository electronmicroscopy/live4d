"""Bounded CUDA result completion shared by all transport controllers."""
import threading

import cupy as cp
import numpy as np


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
                raise RuntimeError("GPU result slot was not reserved for this acquisition")
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
