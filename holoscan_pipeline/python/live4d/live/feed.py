"""The hand-off queue between a push-style receiver (Holoscan's observer) and the encoding thread."""

import cupy as cp


class BatchFeed:
    """Hand decoded batches from a push-style receiver (Holoscan, RDMA) to a consumer thread.

    A receiver that calls back with each batch (rather than being iterated) cannot run the
    reconstruction in its callback without stalling its own pipeline. It calls
    ``start_series(start)``, ``put(first_image_id, patterns)`` and ``end_series()`` instead; the
    consumer iterates ``series()``. ``put`` copies the batch on
    the GPU by default, because a receiver's buffer is usually borrowed only for the callback and
    ``LiveEncoder`` corrects flagged pixels in place. The queue is bounded (``max_batches``),
    so a consumer that falls behind blocks the receiver instead of growing GPU memory without limit.
    """

    _END = object()

    def __init__(self, *, max_batches: int = 64, copy: bool = True):
        import queue

        self.items = queue.Queue(maxsize=max_batches)
        self.copy = copy
        self.closed = False
        self._error = None
        self.pending = None

    def start_series(self, start: dict) -> None:
        """Begin one acquisition with its START metadata (detector shape, pixel mask, series id)."""
        self._put(("start", dict(start)))

    def put(self, first_image_id: int, patterns) -> None:
        """Queue one batch of decoded patterns ``(n, det_row, det_col)`` on the GPU."""
        batch = cp.array(patterns, copy=True) if self.copy else cp.asarray(patterns)
        if self.copy:
            # The copy must finish before the receiver reuses its buffer after this call returns.
            cp.cuda.get_current_stream().synchronize()
        self._put(("batch", (int(first_image_id), batch)))

    def end_series(self) -> None:
        """End the current acquisition; a short one is reported incomplete by ``LiveEncoder``."""
        self._put(("end", None))

    def close(self) -> None:
        """No more acquisitions: the consumer's ``series()`` ends after what is queued.

        Never blocks: a consumer that stopped (an error in the reconstruction) leaves the queue full, and
        the caller must get back to read that error rather than wait for room forever.
        """
        self.closed = True

    # Full producers and empty consumers periodically check for cancellation.
    _POLL_S = 0.2

    def _put(self, item):
        """Wait for room without trapping a producer after close or consumer failure."""
        import queue

        while not self.closed:
            try:
                self.items.put(item, timeout=self._POLL_S)
            except queue.Full:
                continue
            if self._error is None:
                return
            break
        if self._error is not None:
            # A put may have gained room while abort drained the full queue.
            self._abort(self._error)
            raise self._error
        raise RuntimeError("This batch feed is closed; start another feed.")

    def _abort(self, error):
        """Wake producers with the consumer's failure and discard abandoned batches."""
        import queue

        self._error, self.closed, self.pending = error, True, None
        while True:
            try:
                self.items.get_nowait()
            except queue.Empty:
                return

    def _next(self):
        """The next queued item, or the one an acquisition handed back because it was not its own."""
        import queue

        if self.pending is not None:
            item, self.pending = self.pending, None
            return item
        while True:
            try:
                return self.items.get(timeout=self._POLL_S)
            except queue.Empty:
                if self.closed:
                    return ("close", None)

    def series(self):
        """Acquisitions in arrival order, each with ``start`` and ``batches()``."""
        while True:
            kind, value = self._next()
            if kind == "close":
                return
            if kind == "start":
                yield _FedAcquisition(value, self)


class _FedAcquisition:
    def __init__(self, start, feed):
        self.start = start
        self.feed = feed

    def batches(self):
        while True:
            kind, value = self.feed._next()
            if kind == "batch":
                yield value
            elif kind == "end":
                return
            else:
                # A new START or close before END: this acquisition was cut short; the item is the next one's.
                self.feed.pending = (kind, value)
                return
