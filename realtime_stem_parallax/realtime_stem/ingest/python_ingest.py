"""The pure-Python ingest: no Holoscan, no NVIDIA GPU.

Receives the same DECTRIS Stream2 messages as the compiled ingest — over ZeroMQ from the DCU or the
replay feeder, or straight from a file — decodes them on the CPU (`codec.py`), batches them, and
drives the shared `BatchSink`, so the processing and the live stream-out are exactly those of the
Holoscan path. The accumulator can sit on any torch device (cuda, Apple's mps, cpu).

It trades rate for portability: roughly 80k messages/s decode with `dectris-compression` installed
(one core), ~6k/s with the numpy fallback, before the accumulator. Use it for laptops, CI and
offline replays; the detector at full rate needs the Holoscan ingest.

Sources:
  * ZeroMQ (`endpoint`): `n_connections` PULL sockets. After the `end` message it keeps draining
    every socket until `end_drain_s` passes with no message, so frames still queued on another
    connection are not lost (the compiled ingest stops at the first `end`).
  * `.dstream` capture (`source`): the recorded messages, decoded in order.
  * `.npy` (`source`): a (Ny, Nx, H, W) or (N, H, W) array of decoded frames, memory-mapped; ids
    are the raster order. Any dtype the accumulator accepts.

Partial batches are always processed: the final one at the end of a scan, and any batch whose
first frame has waited `flush_after_s`, so a slow source still updates the live view smoothly.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from realtime_stem.ingest.codec import decode_message
from realtime_stem.ingest.sink import BatchSink
from realtime_stem.processing.base import Accumulator

__all__ = ["PythonIngest", "ZmqStream"]


class ZmqStream:
    """`n_connections` PULL sockets to one endpoint, polled together.

    Owned by the runner across scans, so a message that belongs to the next scan (a `start`
    arriving while the previous scan is still draining) can be pushed back rather than lost.
    """

    def __init__(self, endpoint: str, n_connections: int = 1, rcvhwm: int = 100_000):
        import zmq

        self._zmq = zmq
        self.endpoint = endpoint
        self._ctx = zmq.Context.instance()
        self._socks = []
        self._poller = zmq.Poller()
        for _ in range(max(1, int(n_connections))):
            s = self._ctx.socket(zmq.PULL)
            s.setsockopt(zmq.RCVHWM, rcvhwm)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(endpoint)
            self._socks.append(s)
            self._poller.register(s, zmq.POLLIN)
        self._pushback: list[bytes] = []

    def unread(self, messages: list[bytes]) -> None:
        """Push messages back, in order, to be returned by the next `receive`."""
        self._pushback = list(messages) + self._pushback

    def receive(self, timeout_ms: float, max_messages: int = 4096) -> list[bytes]:
        """Every message available within `timeout_ms` (up to `max_messages`), or []."""
        if self._pushback:
            out, self._pushback = self._pushback, []
            return out
        events = dict(self._poller.poll(timeout_ms))
        out = []
        for s in self._socks:
            if s not in events:
                continue
            while len(out) < max_messages:
                try:
                    out.append(s.recv(flags=self._zmq.NOBLOCK, copy=True))
                except self._zmq.Again:
                    break
        return out

    def close(self) -> None:
        for s in self._socks:
            s.close(linger=0)


class _Batcher:
    """Collect decoded frames into batches and hand them to the sink."""

    def __init__(self, sink: BatchSink, acc: Accumulator, batch_size: int, flush_after_s: float,
                 expect_dtype: str | None):
        self.sink = sink
        self.batch_size = int(batch_size)
        self.flush_after_s = float(flush_after_s)
        self.expect_dtype = None if expect_dtype is None else np.dtype(expect_dtype)
        self.detector_shape = tuple(int(d) for d in acc.detector_shape)
        self._frames = None
        self._ids = np.empty(self.batch_size, dtype=np.int64)
        self._count = 0
        self._first_t = 0.0

    def push(self, image_id: int, frame: np.ndarray) -> None:
        if self._frames is None:
            if tuple(frame.shape) != self.detector_shape:
                raise ValueError(f"stream frames are {tuple(frame.shape)} but the geometry expects "
                                 f"{self.detector_shape} — set --detector / --binned to match.")
            if self.expect_dtype is not None and frame.dtype != self.expect_dtype:
                raise ValueError(f"stream pixels are {frame.dtype} but --dtype is "
                                 f"{self.expect_dtype} — set --dtype to match the stream.")
            # A fresh buffer per batch: the accumulator may still be reading the previous one
            # (an asynchronous host-to-device copy) when we start filling the next.
            self._frames = np.empty((self.batch_size, *self.detector_shape), dtype=frame.dtype)
            self._first_t = time.perf_counter()
        self._frames[self._count] = frame
        self._ids[self._count] = image_id
        self._count += 1
        if self._count == self.batch_size:
            self.flush()

    @property
    def pending(self) -> int:
        """Frames collected but not yet processed."""
        return self._count

    def due(self) -> bool:
        return self._count > 0 and time.perf_counter() - self._first_t >= self.flush_after_s

    def flush(self) -> None:
        if self._count:
            self.sink.feed(self._frames[:self._count], self._ids[:self._count].copy())
            self._frames = None
            self._count = 0


class PythonIngest:
    """Drive an `Accumulator` from a ZMQ stream or a file; `run()` returns at the scan's end.

    Same surface as the Holoscan `IngestApp` (`run()`, `.sink.summary()`), so the runner treats
    the two alike.
    """

    def __init__(self, acc: Accumulator, *, stream: ZmqStream | None = None,
                 source: str | Path | None = None, batch_size: int = 2048,
                 dtype: str | None = "uint16", source_rate: float = 0.0,
                 flush_after_s: float = 0.05, end_drain_s: float = 0.2,
                 live_publisher=None, live_every: float = 0.2, live_every_rows: int = 0,
                 print_every: int = 50):
        if (stream is None) == (source is None):
            raise ValueError("give exactly one of `stream` (ZMQ) or `source` (a file)")
        self.acc = acc
        self.stream = stream
        self.source = None if source is None else Path(source)
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.source_rate = float(source_rate)
        self.flush_after_s = float(flush_after_s)
        self.end_drain_s = float(end_drain_s)
        self.sink = BatchSink(acc, print_every=print_every, live_publisher=live_publisher,
                              live_every=live_every, live_every_rows=live_every_rows)

    # ── sources ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        if self.stream is not None:
            self._run_zmq()
        elif self.source.suffix == ".npy":
            self._run_npy()
        else:
            self._run_dstream()

    def _pace(self, t0: float, n_done: int) -> None:
        """Sleep so that `n_done` frames take at least `n_done / source_rate` seconds."""
        if self.source_rate > 0:
            ahead = n_done / self.source_rate - (time.perf_counter() - t0)
            if ahead > 0:
                time.sleep(ahead)

    def _run_npy(self) -> None:
        data = np.load(self.source, mmap_mode="r")
        if data.ndim < 3:
            raise ValueError(f"{self.source}: expected frames as (..., H, W), e.g. (Ny, Nx, H, W) "
                             f"or (passes, Ny, Nx, H, W), got shape {data.shape}")
        data = data.reshape(-1, *data.shape[-2:])
        if tuple(data.shape[1:]) != tuple(self.acc.detector_shape):
            raise ValueError(f"{self.source}: frames are {tuple(data.shape[1:])} but the geometry "
                             f"expects {tuple(self.acc.detector_shape)} — set --detector.")
        n_pos = self.acc.scan_shape[0] * self.acc.scan_shape[1]
        if data.shape[0] > n_pos and not getattr(self.acc, "wrap_ids", False):
            raise ValueError(f"{self.source}: {data.shape[0]} frames but --scan has only "
                             f"{n_pos} positions (repeated scans: --accumulate latest / all).")
        # with pacing, keep batches short enough for the live view to fill in smoothly
        step = self.batch_size
        if self.source_rate > 0:
            step = max(1, min(step, int(self.source_rate * self.flush_after_s)))
        t0 = time.perf_counter()
        for start in range(0, data.shape[0], step):
            stop = min(start + step, data.shape[0])
            self.sink.feed(np.array(data[start:stop]), np.arange(start, stop))
            self._pace(t0, stop)

    def _run_dstream(self) -> None:
        from realtime_stem.ingest.dstream import iter_records

        batcher = _Batcher(self.sink, self.acc, self.batch_size, self.flush_after_s, self.dtype)
        t0, n = time.perf_counter(), 0
        for raw in iter_records(self.source):
            mtype, image_id, frame = decode_message(raw)
            if mtype == "image":
                batcher.push(image_id, frame)
                n += 1
                if self.source_rate > 0 and n % 64 == 0:
                    self._pace(t0, n)
                    if batcher.due():
                        batcher.flush()
            elif mtype == "end":
                break
        batcher.flush()

    def _run_zmq(self) -> None:
        batcher = _Batcher(self.sink, self.acc, self.batch_size, self.flush_after_s, self.dtype)
        ended = False
        last_msg = time.perf_counter()
        while True:
            timeout_ms = 1000 * (self.flush_after_s if batcher.pending else
                                 self.end_drain_s if ended else 0.5)
            messages = self.stream.receive(timeout_ms)
            now = time.perf_counter()
            if not messages:
                batcher.flush()                     # idle: process the partial batch
                if ended and now - last_msg >= self.end_drain_s:
                    break
                continue
            last_msg = now
            for i, raw in enumerate(messages):
                mtype, image_id, frame = decode_message(raw)
                if mtype == "image":
                    batcher.push(image_id, frame)   # incl. frames still draining after `end`
                elif mtype == "start":
                    if ended:                       # the next scan: hand it back, finish this one
                        self.stream.unread(messages[i:])
                        batcher.flush()
                        return
                    print("[ingest] series start", flush=True)
                elif mtype == "end":
                    ended = True
            if batcher.due():
                batcher.flush()
        batcher.flush()
