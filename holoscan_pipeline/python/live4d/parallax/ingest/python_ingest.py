"""The pure-Python ingest: no Holoscan, no NVIDIA GPU.

Receives the same DECTRIS Stream2 messages as the compiled ingest — over an ordered ZeroMQ connection or the
replay feeder, or straight from a file — decodes them on the CPU (`codec.py`), batches them, and
drives the shared `BatchSink`, so the processing and the live stream-out are exactly those of the
Holoscan path. The accumulator can sit on any torch device (cuda, Apple's mps, cpu).

It trades rate for portability: roughly 80k messages/s decode with `dectris-compression` installed
(one core), ~6k/s with the numpy fallback, before the accumulator. Use it for laptops, CI and
offline replays; the detector at full rate needs the Holoscan ingest.

Sources:
  * ZeroMQ (`endpoint`): one ordered PULL connection. END flushes the partial batch.
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

from live4d.parallax.ingest.codec import decode_message
from live4d.parallax.ingest.sink import BatchSink
from live4d.algorithms.base import Accumulator

__all__ = ["PythonIngest", "ZmqStream"]


class ZmqStream:
    """One ordered PULL connection, retained across acquisitions."""
    def __init__(self, endpoint: str, rcvhwm: int = 100_000):
        import zmq
        self._zmq = zmq
        self.endpoint = endpoint
        self._socket = zmq.Context.instance().socket(zmq.PULL)
        self._socket.setsockopt(zmq.RCVHWM, rcvhwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._pushback = []

    def unread(self, messages):
        self._pushback = list(messages) + self._pushback

    def receive(self, timeout_ms, max_messages=4096):
        if self._pushback:
            out, self._pushback = self._pushback[:max_messages], self._pushback[max_messages:]
            return out
        if not self._socket.poll(int(timeout_ms)):
            return []
        out = []
        while len(out) < max_messages:
            try:
                out.append(self._socket.recv(flags=self._zmq.NOBLOCK))
            except self._zmq.Again:
                break
        return out

    def close(self):
        self._socket.close(linger=0)


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
        if tuple(frame.shape) != self.detector_shape:
            raise ValueError("Frame geometry changed within acquisition")
        if self.expect_dtype is not None and frame.dtype != self.expect_dtype:
            raise ValueError("Frame dtype differs from acquisition")
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

    Same surface as the offline replay driver (`run()`, `.sink.summary()`), so the runner treats
    the two alike.
    """

    def __init__(self, acc: Accumulator, *, stream: ZmqStream | None = None,
                 source: str | Path | None = None, batch_size: int = 2048,
                 dtype: str | None = "uint16", source_rate: float = 0.0,
                 flush_after_s: float = 0.05,
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
        from live4d.parallax.ingest.dstream import iter_records

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
        while True:
            messages = self.stream.receive(1000 * (self.flush_after_s if batcher.pending else 0.5))
            if not messages:
                batcher.flush()
                continue
            for i, raw in enumerate(messages):
                mtype, image_id, frame = decode_message(raw)
                if mtype == "image":
                    batcher.push(image_id, frame)
                elif mtype == "end":
                    batcher.flush()
                    self.stream.unread(messages[i + 1:])
                    return
            if batcher.due():
                batcher.flush()
