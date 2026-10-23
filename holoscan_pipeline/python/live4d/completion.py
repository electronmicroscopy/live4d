"""Bound asynchronous submissions without borrowing a callback's stream after it returns."""
from collections import deque
from dataclasses import dataclass
import threading
import time

import cupy as cp


@dataclass
class _Completion:
    batch_id: int
    event: object
    owners: tuple
    callback: object
    acknowledge: bool


class CompletionTracker:
    """Poll callback-recorded events in order, independently of subsequent callbacks.

    Tensor retirement belongs to the SDK. This queue retains algorithm scratch and
    reports completion for admission and metrics; it never submits work on a saved
    runtime stream. The application must close it before destroying native windows.
    """

    def __init__(self, capacity, device=0, release=None):
        if capacity < 1:
            raise ValueError("Completion capacity must be positive")
        self.capacity = capacity
        self.device = device
        self.release = release
        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._pending = deque()
        self._tail = None
        self._thread = None
        self._closing = False
        self._error = None

    def start(self):
        with self._condition:
            if self._thread is not None or self._closing:
                raise RuntimeError("Completion tracker has already started")
            self._thread = threading.Thread(target=self._poll, name="gpu-completion")
            self._thread.start()

    def check(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Asynchronous GPU completion failed") from self._error

    def wait_for_previous(self, stream):
        """Order mutable algorithm state on the current callback's stream."""
        self.check()
        with self._condition:
            previous = self._tail
        if previous is not None:
            stream.wait_event(previous)

    def track(self, batch_id, stream, *, owners=(), on_complete=None, acknowledge=True):
        """Record completion before leaving the callback, retaining scratch as needed."""
        event = cp.cuda.Event(disable_timing=True)
        event.record(stream)
        item = _Completion(batch_id, event, tuple(owners), on_complete, acknowledge)
        try:
            with self._condition:
                self.check()
                if self._thread is None or self._closing:
                    raise RuntimeError("Completion tracker is not running")
                if len(self._pending) >= self.capacity:
                    raise RuntimeError("Asynchronous completion capacity exhausted")
                self._pending.append(item)
                self._tail = event
                self._condition.notify_all()
        except BaseException:
            # Only error cleanup blocks: rejected submissions still need their owners.
            event.synchronize()
            raise

    def drain(self):
        """Wait at a host-readback or teardown boundary, never for routine pacing."""
        with self._condition:
            while self._pending and self._error is None:
                self._condition.wait(timeout=0.05)
            self.check()

    def close(self):
        """Drain outstanding work before releasing owners, events, or native resources."""
        with self._close_lock:
            self._finish()

    def _finish(self):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join()
        # On an event-query or completion-callback failure, the worker leaves
        # outstanding owners here for a deliberate teardown wait.
        try:
            with cp.cuda.Device(self.device):
                for item in self._pending:
                    item.event.synchronize()
        finally:
            with self._condition:
                self._pending.clear()
                self._tail = None
        self.check()

    def _poll(self):
        try:
            with cp.cuda.Device(self.device):
                while True:
                    with self._condition:
                        if not self._pending:
                            if self._closing:
                                return
                            self._condition.wait(timeout=0.001)
                            continue
                        item = self._pending[0]
                    if not item.event.done:
                        with self._condition:
                            self._condition.wait(timeout=0.001)
                        continue
                    completed_at = time.perf_counter()
                    # Finish callbacks before making drain() observe an empty queue.
                    with self._condition:
                        if item.callback is not None:
                            item.callback(completed_at)
                        if item.acknowledge and self.release is not None:
                            self.release(item.batch_id)
                        self._pending.popleft()
                        self._condition.notify_all()
                    # Owners are released only after the event is complete.
                    del item
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
