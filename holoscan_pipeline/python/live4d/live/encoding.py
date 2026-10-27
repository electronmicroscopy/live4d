"""ANS-encode each series on the GPU as its batches arrive: a decoded-batch consumer of the pipeline.

`AnsEncoderOp` is one more consumer of the decoded batches, beside the virtual detectors and the
HDF5 writer, in its own partition. Each activation copies the batch's images once, on the runtime's
CUDA stream, into a chunk of `CHUNK` patterns and releases the batch. A full chunk is corrected for
flagged pixels and ANS-encoded in place by quantem.gpu's `LiveEncoder`; the encoder waits for its
own result once per chunk, which blocks only this partition.

What becomes of each series is the sink's (`LiveFrames`):

    sink.begin(series_id, scan_shape, detector_shape) -> LiveEncoder, or None to skip the series
    sink.complete(encoder)                            # after END: finish, keep, save
    sink.drop(encoder, reason)                        # gap, repeat or overrun: the series is not kept
"""
from __future__ import annotations

import functools
import math

import cupy as cp
import numpy as np

#: Patterns encoded together: ANS codes each pixel over 512-scan runs, and an encode call costs about
#: the same whatever its size, so whole 4096-pattern chunks keep both the rate and the compression.
CHUNK = 4096


def series_scan(series, configured) -> tuple[int, int]:
    """(rows, cols) of one series: the START header_appendix scan, else a square raster of
    number_of_images, else the configured scan."""
    if series.scan is not None:
        return series.scan.height, series.scan.width
    if series.expected_images:
        side = math.isqrt(series.expected_images)
        if side * side == series.expected_images:
            return side, side
    return int(configured[0]), int(configured[1])


class SeriesEncoder:
    """Stage arriving batches into whole chunks and hand them to the current series' encoder.

    Holds no Holoscan state; the operator calls it on the runtime's stream.
    """

    def __init__(self, sink, configured_scan, log=print):
        self.sink = sink
        self.configured_scan = tuple(configured_scan)
        self.log = log
        self.encoder = None
        self.staging = None          # (CHUNK, H, W) owned buffer, reused across chunks and series
        self.staged = 0
        self.first = 0               # image id of staging[0]
        self.next_id = 0

    def start(self, series) -> None:
        if self.encoder is not None:               # START without END: the previous series ends short
            self.end()
        scan = series_scan(series, self.configured_scan)
        self.encoder = self.sink.begin(series.id, scan, tuple(series.shape))
        self.staged = self.first = self.next_id = 0
        if self.encoder is not None and (self.staging is None or self.staging.shape[1:] != tuple(series.shape)):
            self.staging = cp.empty((CHUNK, *series.shape), cp.uint16)

    def add(self, frames, ids) -> None:
        """Copy `frames` (n, H, W) with consecutive `ids` into the staging chunk; encode full chunks."""
        if self.encoder is None:
            return
        n = int(frames.shape[0])
        first, last = int(ids[0]), int(ids[-1])
        if first != self.next_id or last - first + 1 != n:
            return self._drop(f"images {first}..{last} do not follow image {self.next_id - 1}")
        if last >= self.encoder.total:
            return self._drop(f"image {last} is beyond the {self.encoder.scan_shape} scan")
        offset = 0
        while offset < n:
            take = min(n - offset, CHUNK - self.staged)
            self.staging[self.staged:self.staged + take] = frames[offset:offset + take]
            self.staged += take
            offset += take
            if self.staged == CHUNK:
                self._encode()
        self.next_id = last + 1

    def _encode(self) -> None:
        # LiveEncoder corrects the chunk in place and waits for its encode, so staging is free after it.
        self.encoder.add(self.first, self.staging[:self.staged])
        self.first += self.staged
        self.staged = 0

    def end(self) -> None:
        if self.encoder is None:
            return
        if self.staged:
            self._encode()
        encoder, self.encoder = self.encoder, None
        self.sink.complete(encoder)

    def _drop(self, reason: str) -> None:
        encoder, self.encoder = self.encoder, None
        self.staged = 0
        self.log(f"[ans] series dropped: {reason}")
        self.sink.drop(encoder, reason)


@functools.cache
def _operator():
    """`AnsEncoderOp`, defined on first use so `SeriesEncoder` imports without Holoscan."""
    import holoscan as hs

    from dectris.messages import parse_start

    from ..schemas.decoded_batch import DecodedBatch

    class AnsEncoderOp(hs.Operator):
        """ANS-encode each series as it arrives (one decoded-batch consumer)."""
        inp = hs.Input("input", DecodedBatch)
        device_id = hs.Param(0)
        scan_width = hs.Param(1000)
        scan_height = hs.Param(1000)

        def __init__(self, *, sink, release, consumer, finite=True, completion=None):
            super().__init__()
            self.sink, self.release, self.consumer = sink, release, consumer
            self.finite, self.completion = finite, completion

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def start(self):
            self.device = cp.cuda.Device(self.device_id)
            self.series = SeriesEncoder(self.sink, (self.scan_height, self.scan_width))

        def compute(self, ctx):
            batch = self.inp.receive().data
            runtime_stream = ctx.stream()
            try:
                with self.device, runtime_stream, cp.cuda.Stream.from_external(runtime_stream):
                    if batch.pipeline_ready:
                        cp.from_dlpack(batch.frames)          # the runtime Tensor-to-CuPy handoff
                        return
                    if batch.start_of_series:
                        self.series.start(parse_start(bytes(batch.start_metadata)))
                    count = len(batch.image_ids)
                    if count:
                        # The runtime orders input readiness and release on this activation's stream.
                        frames = cp.from_dlpack(batch.frames)[:count]
                        self.series.add(frames, np.asarray(batch.image_ids, dtype=np.int64))
                    if batch.end_of_series:
                        self.series.end()
                        if self.finite and (self.completion is None or self.completion.arrive()):
                            ctx.request_completion()
            finally:
                self.release(int(batch.batch_id), self.consumer)

    return AnsEncoderOp


def __getattr__(name):
    if name == "AnsEncoderOp":
        return _operator()
    raise AttributeError(name)
