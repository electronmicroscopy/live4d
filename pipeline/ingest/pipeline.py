"""
Streaming intake pipeline (host side).

Turns a *source of frames* into *batches handed to a callback*. Frames flow through coarse,
chunk-based hand-offs between threads:

    bytes source --> [producer] --> chunk queue --> [decoder pool] --> batch queue --> [consumer]
                     1 thread       (list[bytes])   N threads          (ids, frames)    on_batch

**Why chunks.** Passing one frame at a time across bounded queues, with a per-frame `qsize()`
lock and per-frame object churn, makes the queue orchestration -- not the decode -- the wall:
the decode pool runs far below its tight-loop ceiling once routed through per-frame machinery.
The fix is to pass *chunks* of frames between threads. Each decoder pulls a chunk of raw
messages and grinds it in a tight `decode_message` loop, writing results straight into a
preallocated `(C, H, W)` buffer (no `np.stack`, no per-frame `np.asarray`). The queues are
touched ~`chunk_size`x less often, so the GIL ping-pong mostly disappears and the decoders
settle into GIL-released decompression.

Two structural simplifications follow:
  - The **start gate lives in the producer**: it validates `start` *before* emitting any frame
    chunk, so there is no cross-decoder gate race.
  - Decoded chunks *are* the GPU batches -- the consumer hands each `(ids, frames)` straight to
    `on_batch`, doing zero per-frame work.

`BatchedLiveSource` builds on a base, `_BatchedDecodingSource`, which owns the decoder pool, the
start-validation gate, error reporting, and clean shutdown; the byte source is a socket connected
to the DCU.

A CPU and a GPU decode mode are both supported: 'cpu' runs threaded bslz4 decompress on the host;
'gpu' parses only the envelope on the host and runs the bslz4 decompress on the GPU.

Everything here stays on the host (system RAM) for the CPU path. The host->GPU transfer and any
batching into pinned memory belong to the consumer's `on_batch` (the VBF engine).
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import zmq

from ingest.dectris_decode import decode_message, decode_envelope, envelope_type

# An item flowing to the consumer is (ids, frames). `_END` is a unique sentinel object placed
# on the queue exactly once, after every batch, to signal "no more frames".
_END = object()


# --------------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------------


class PipelineStats:
    """Lightweight, method-agnostic throughput counters for one pipeline run.

    Counters are plain ints incremented under the GIL (a bare `+=` on an int is atomic
    enough for monotonic counters that are only read in the end-of-run summary), so the hot
    path stays cheap. Wall time uses `time.perf_counter`. Queue-depth samples reveal *which*
    stage is the bottleneck: the decode->batch queue sitting chronically full means the
    consumer (GPU) can't keep up; chronically empty means decode/IO can't keep up.
    """

    def __init__(self) -> None:
        self.received = 0          # raw messages pulled from the source (socket/file)
        self.decoded = 0           # 'image' frames successfully decoded + extracted
        self.batched = 0           # frames handed to on_batch
        self.batches = 0           # number of on_batch calls
        self._t0: float | None = None
        self._t1: float | None = None
        self._q_sum = 0
        self._q_samples = 0
        self._q_max = 0

    def start_clock(self) -> None:
        self._t0 = time.perf_counter()

    def stop_clock(self) -> None:
        self._t1 = time.perf_counter()

    def mark_received(self, n: int = 1) -> None:
        self.received += n

    def mark_decoded(self, n: int = 1) -> None:
        self.decoded += n

    def mark_batch(self, n: int) -> None:
        self.batched += n
        self.batches += 1

    def sample_queue(self, depth: int) -> None:
        self._q_sum += depth
        self._q_samples += 1
        if depth > self._q_max:
            self._q_max = depth

    @property
    def elapsed_s(self) -> float:
        if self._t0 is None:
            return 0.0
        end = self._t1 if self._t1 is not None else time.perf_counter()
        return end - self._t0

    def summary(self, queue_maxsize: int | None = None) -> str:
        e = self.elapsed_s or float("nan")
        avg_q = (self._q_sum / self._q_samples) if self._q_samples else 0.0
        cap = f"/{queue_maxsize}" if queue_maxsize else ""
        # The headline number is frames decoded per second.
        return (
            "=== Pipeline throughput ===\n"
            f"  elapsed:            {e:8.3f} s\n"
            f"  messages received:  {self.received:8d}  ({self.received / e:10.1f} /s)\n"
            f"  frames decoded:     {self.decoded:8d}  ({self.decoded / e:10.1f} /s)\n"
            f"  frames batched:     {self.batched:8d}  in {self.batches} batches"
            f"  ({self.batched / e:10.1f} /s)\n"
            f"  decode->GPU queue:  avg {avg_q:6.1f}{cap}, max {self._q_max}{cap}"
            "   (full -> GPU-bound; empty -> decode/IO-bound)"
        )


# --------------------------------------------------------------------------------
# Decoded-message helpers
# --------------------------------------------------------------------------------


def extract_image(msg) -> tuple[int, np.ndarray]:
    """Pull `(message_id, frame)` out of a decoded DECTRIS 'image' message.

    The scan index is under 'image_id' (0-based, raster order) and the pixel array under
    'data' (a per-channel mapping).
    """
    msg_id = int(msg["image_id"])
    frame = msg["data"]
    # `data` is a per-channel mapping, e.g. {'threshold_1': array}. Unwrap to the single
    # channel's array. Duck-typed on `.values` because cbor2 returns an immutable
    # `frozendict`, which is NOT a `dict` subclass -- an `isinstance(..., dict)` check
    # silently misses it and returns the whole mapping.
    if hasattr(frame, "values") and not isinstance(frame, np.ndarray):
        frame = next(iter(frame.values()))
    return msg_id, np.asarray(frame)


def validate_start(start_msg, detector_shape, num_positions=None) -> bool:
    """Validate a decoded 'start' message against what we preallocated for; raise on mismatch.

    We deliberately do **not** reconfigure from the stream. All buffers and precompute (weight
    map, masks, and for heavier methods the geometry/kernels) are built up front from config, off
    the hot path -- a method like parallax can't rebuild in the brief start->first-frame gap
    without falling behind. So the start message is a **gate**: catch a config/detector mismatch
    loudly *before* any frame is processed, rather than corrupting silently or crashing mid-stream.
    """
    sy, sx = start_msg.get("image_size_y"), start_msg.get("image_size_x")
    if sy is not None and sx is not None and (int(sy), int(sx)) != tuple(detector_shape):
        raise ValueError(
            f"detector shape mismatch: configured {tuple(detector_shape)}, stream sends "
            f"({int(sy)}, {int(sx)}). Fix config.detector_shape."
        )
    n = start_msg.get("number_of_images")
    if num_positions is not None and n is not None and int(n) > num_positions:
        raise ValueError(
            f"scan too small: stream will send {int(n)} images but the output buffer holds "
            f"only {num_positions} (config.scan_shape). Frames would index past the buffer."
        )
    return True


# --------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------


class _BatchedDecodingSource:
    """Shared machinery for chunk-based sources: a decoder pool that decodes whole chunks.

    A *producer* (subclass-provided) validates the start gate, then drops chunks of raw image
    bytes (`list[bytes]`) onto `self._raw_q`; the decoder pool pulls a chunk, decodes it into a
    preallocated `(n, H, W)` array plus an `(n,)` id array, and puts `(ids, frames)` on the out
    queue. The byte source is provided by the subclass; here it is the DCU socket
    (`BatchedLiveSource`).
    """

    def __init__(self, config, on_start=None, stats: "PipelineStats | None" = None,
                 decode_mode: str = "cpu"):
        self.config = config
        self.on_start = on_start
        self.start_error: Exception | None = None
        self.stats = stats
        self.detector_shape = tuple(config.detector_shape)
        self.chunk_size = int(config.chunk_size)
        self.n_decode_threads = int(config.n_decode_threads)
        if decode_mode not in ("cpu", "gpu"):
            raise ValueError(f"decode_mode must be 'cpu' or 'gpu', got {decode_mode!r}")
        self.decode_mode = decode_mode

        # Queues carry *chunks*, so a small maxsize already buys plenty of buffering and
        # bounds memory (each decoded chunk is chunk_size*H*W*dtype bytes -- sizeable).
        self._raw_q: queue.Queue = queue.Queue(maxsize=config.chunk_queue_maxsize)
        self._stop = threading.Event()                 # hard stop (gate failure / end)
        self._producing_done = threading.Event()       # producer has emitted its last chunk
        self._threads: list[threading.Thread] = []
        self._joiner: threading.Thread | None = None
        self._err_lock = threading.Lock()
        self._decode_errors = 0

    # -- subclass hooks --------------------------------------------------------------
    def _producer_threads(self) -> list[threading.Thread]:
        raise NotImplementedError

    def _cleanup(self) -> None:
        """Release any resources (e.g. a socket) after all threads have finished. Optional."""

    # -- lifecycle -------------------------------------------------------------------
    def start(self, out_queue: queue.Queue) -> None:
        producers = self._producer_threads()
        if self.decode_mode == "gpu":
            # One GPU-decode thread: the GPU serializes, and the CPU does only the cheap envelope
            # parse, so a thread pool buys nothing (and would contend for one CUDA context).
            decoders = [threading.Thread(target=self._decode_chunks_gpu, args=(out_queue,),
                                         name="gpu-decoder", daemon=True)]
        else:
            decoders = [
                threading.Thread(target=self._decode_chunks, args=(out_queue,),
                                 name=f"decoder-{i}", daemon=True)
                for i in range(self.n_decode_threads)
            ]
        self._threads = [*producers, *decoders]
        for t in self._threads:
            t.start()
        self._joiner = threading.Thread(
            target=self._join_and_signal, args=(out_queue,), name="joiner", daemon=True
        )
        self._joiner.start()

    def _validate_start(self, start_raw: bytes) -> bool:
        """Decode + validate the start message *before* any frame chunk is emitted.

        Running this on the single producer thread (not in the decoder pool) is what removes the
        gate race: no image can be decoded before the gate has passed. On failure we record the
        error and signal a hard stop, so the consumer ends cleanly with no frames processed.
        """
        if self.on_start is None:
            return True
        try:
            self.on_start(decode_message(start_raw))
            return True
        except Exception as exc:
            self.start_error = exc
            self._stop.set()
            return False

    def _put_chunk(self, chunk: list[bytes]) -> bool:
        """Put a chunk on the raw queue, staying responsive to `_stop` if the queue is full."""
        while not self._stop.is_set():
            try:
                self._raw_q.put(chunk, timeout=0.1)
                if self.stats is not None:
                    self.stats.mark_received(len(chunk))
                return True
            except queue.Full:
                continue
        return False

    def _decode_chunks(self, out_queue: queue.Queue) -> None:
        """Pull chunks, decode each in a tight loop into a preallocated buffer, emit (ids, frames)."""
        H, W = self.detector_shape
        while True:
            try:
                chunk = self._raw_q.get(timeout=0.1)
            except queue.Empty:
                # Nothing waiting: exit once the producer is done (or we've been told to stop).
                if self._producing_done.is_set() or self._stop.is_set():
                    break
                continue

            # Preallocate the id buffer; the frame buffer is allocated on the first frame, once we
            # know its dtype (the stream's typed-array tag decides uint8/uint16/uint32 -- see
            # dectris_decode). np.empty: every cell is overwritten, so no need to zero.
            n = len(chunk)
            ids = np.empty(n, dtype=np.int64)
            frames: np.ndarray | None = None
            count = 0
            for raw in chunk:
                try:
                    msg = decode_message(raw)
                    mid, frame = extract_image(msg)
                except Exception as exc:
                    self._report_decode_error(exc, None)
                    continue
                if frames is None:
                    frames = np.empty((n, *frame.shape), dtype=frame.dtype)
                frames[count] = frame   # raises loudly if a frame's dims != (H, W) -- fail fast
                ids[count] = mid
                count += 1

            if count:
                out_queue.put((ids[:count], frames[:count]))
                if self.stats is not None:
                    self.stats.mark_decoded(count)
                    self.stats.sample_queue(out_queue.qsize())  # per chunk -- no per-frame lock

    def _decode_chunks_gpu(self, out_queue: queue.Queue) -> None:
        """GPU decode path: the CPU parses only the envelope; bslz4 decompress runs on the GPU.

        One thread parses each chunk's messages with `decode_envelope` (no decompress), hands the
        still-compressed blobs to a `GpuBslz4Decoder`, and emits on-device integer frames
        `(n, H, W)` straight to the consumer (the VBF accumulator casts them to float on the GPU).
        The decoder is built here, on this thread, because it compiles a CUDA kernel + binds nvCOMP.
        """
        from ingest.gpu_decode import GpuBslz4Decoder   # lazy: torch/nvCOMP only for the GPU path
        H, W = self.detector_shape
        try:
            decoder = GpuBslz4Decoder()
        except Exception as exc:                         # no CUDA / kernel build failed -> stop cleanly
            self._report_decode_error(exc, None)
            self._stop.set()
            return
        while True:
            try:
                chunk = self._raw_q.get(timeout=0.1)
            except queue.Empty:
                if self._producing_done.is_set() or self._stop.is_set():
                    break
                continue

            ids: list[int] = []
            encoded: list[bytes] = []
            elem_size: int | None = None
            for raw in chunk:
                try:
                    env = decode_envelope(raw)
                except Exception as exc:
                    self._report_decode_error(exc, None)
                    continue
                if env is None:                          # not an image (shouldn't happen in a chunk)
                    continue
                ids.append(env.image_id)
                encoded.append(env.encoded)
                elem_size = env.elem_size
            if not encoded:
                continue
            try:
                frames = decoder.decode_int_frames(encoded, elem_size, (H, W))
            except Exception as exc:
                self._report_decode_error(exc, None)
                continue
            out_queue.put((np.asarray(ids, dtype=np.int64), frames))
            if self.stats is not None:
                self.stats.mark_decoded(len(ids))
                self.stats.sample_queue(out_queue.qsize())   # per chunk -- no per-frame lock

    def _report_decode_error(self, exc: Exception, msg) -> None:
        with self._err_lock:
            self._decode_errors += 1
            first = self._decode_errors == 1
        if first:
            print(f"[decoder] ERROR on a frame: {type(exc).__name__}: {exc}")
            print("[decoder] further errors will be counted, not printed.")

    def _join_and_signal(self, out_queue: queue.Queue) -> None:
        for t in self._threads:
            t.join()
        out_queue.put(_END)
        self._cleanup()
        if self._decode_errors:
            print(f"[decoder] total frames that failed to decode/extract: {self._decode_errors}")

    def join(self) -> None:
        if self._joiner is not None:
            self._joiner.join()


class BatchedLiveSource(_BatchedDecodingSource):
    """Receive + chunk the live DECTRIS stream, decode through the chunked pool.

    The producer peeks each message's envelope type (cheap -- no decompress) to find the single
    `start` (validate the gate), accumulate `image` messages into chunks, and flush + stop on
    `end`. The peek costs ~one CBOR envelope parse per message, comfortably above the unbinned
    target frame rate.
    """

    def __init__(self, config, on_start=None, stats: "PipelineStats | None" = None,
                 decode_mode: str = "cpu"):
        super().__init__(config, on_start, stats, decode_mode=decode_mode)
        self._ctx: zmq.Context | None = None
        self._sock: zmq.Socket | None = None

    def _producer_threads(self) -> list[threading.Thread]:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.connect(f"tcp://{self.config.dcu_ip}:{self.config.zmq_port}")
        return [threading.Thread(target=self._receive, name="receiver", daemon=True)]

    def _receive(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        started = False
        chunk: list[bytes] = []
        try:
            while not self._stop.is_set():
                if not dict(poller.poll(timeout=100)):
                    continue
                raw = self._sock.recv()
                mtype = envelope_type(raw)
                if not started:
                    if mtype != "start":
                        continue  # ignore chatter before a clean start
                    started = True
                    if not self._validate_start(raw):
                        return
                    continue
                if mtype == "image":
                    chunk.append(raw)
                    if len(chunk) >= self.chunk_size:
                        if not self._put_chunk(chunk):
                            return
                        chunk = []
                elif mtype == "end":
                    if chunk:
                        self._put_chunk(chunk)
                    return
        finally:
            self._producing_done.set()

    def _cleanup(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)


def run_pipeline_batched(
    source: "_BatchedDecodingSource",
    on_batch,
    stats: "PipelineStats | None" = None,
) -> None:
    """Drive a batched source: each decoded `(ids, frames)` chunk is one `on_batch` call.

    Runs on the calling thread and blocks until the source is exhausted. There is no per-frame
    buffering or `np.stack` here -- the decode pool already produced batched arrays, so the
    consumer does zero per-frame work.
    """
    out_q: queue.Queue = queue.Queue(maxsize=source.config.chunk_queue_maxsize)
    if stats is not None:
        stats.start_clock()
    source.start(out_q)
    while True:
        item = out_q.get()
        if item is _END:
            break
        ids, frames = item
        on_batch(frames, ids)
        if stats is not None:
            stats.mark_batch(len(ids))
    source.join()
    if stats is not None:
        stats.stop_clock()
