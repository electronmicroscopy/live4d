"""Stage each series' decoded images into chunks for lossless ANS encoding on the GPU.

The encoding itself is quantem.gpu's `LiveEncoder`: flagged-pixel correction and ANS, giving the
`Dataset4dstemGPU` that `quantem.gpu.io.load` returns for the same counts. This module holds no
Holoscan state; `processing.AnsEncoderOp` calls `SeriesEncoder` on the runtime's CUDA stream.

What becomes of each series is the sink's:

    sink.begin(series_id, scan_shape, image_shape) -> LiveEncoder, or None to skip the series
    sink.complete(encoder) -> dict                     # after END; returns the series summary
    sink.drop(encoder, reason)                         # gap, repeat or overrun: not kept
"""
from __future__ import annotations

import math

import cupy as cp

#: Patterns encoded together. ANS codes each pixel over 512-scan runs and an encode call costs about
#: the same whatever its size, so whole chunks of 4096 keep both the rate and the compression.
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
    """Copy arriving batches into a reused staging chunk; encode each full chunk."""

    def __init__(self, sink, configured_scan, log=print):
        self.sink = sink
        self.configured_scan = tuple(configured_scan)
        self.log = log
        self.encoder = None
        self.staging = None          # (CHUNK, H, W), reused across chunks and series of one size
        self.staged = 0
        self.first = 0               # image id of staging[0]
        self.next_id = 0
        # Each activation may run on its own runtime stream: the copies into staging are ordered
        # before the encode, which runs on a later activation's stream, by these events.
        self.copies = []

    def start(self, series) -> None:
        if self.encoder is not None:               # START without END: the previous series ends short
            self.end()
        self.encoder = self.sink.begin(series.id, series_scan(series, self.configured_scan),
                                       tuple(series.shape))
        self.staged = self.first = self.next_id = 0
        self.copies = []
        if self.encoder is not None and (self.staging is None or self.staging.shape[1:] != tuple(series.shape)):
            self.staging = cp.empty((CHUNK, *series.shape), cp.uint16)

    def add(self, frames, ids) -> None:
        """Copy `frames` (n, H, W) with consecutive image `ids`; encode each full chunk."""
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
            copied = cp.cuda.Event(disable_timing=True)
            copied.record()
            self.copies.append(copied)
            self.staged += take
            offset += take
            if self.staged == CHUNK:
                self._encode()
        self.next_id = last + 1

    def _encode(self) -> None:
        stream = cp.cuda.get_current_stream()
        for copied in self.copies:       # GPU-side waits: no host synchronization
            stream.wait_event(copied)
        self.copies = []
        # LiveEncoder corrects the chunk in place and waits for its encode, so staging is free after it.
        self.encoder.add(self.first, self.staging[:self.staged])
        self.first += self.staged
        self.staged = 0

    def end(self) -> dict | None:
        """Encode the last partial chunk and hand the series to the sink; its summary, if kept."""
        if self.encoder is None:
            return None
        if self.staged:
            self._encode()
        encoder, self.encoder = self.encoder, None
        return self.sink.complete(encoder)

    def _drop(self, reason: str) -> None:
        encoder, self.encoder = self.encoder, None
        self.staged = 0
        self.copies = []
        self.log(f"[ans] series dropped: {reason}")
        self.sink.drop(encoder, reason)


class ResidentSeries:
    """The default sink: keep the latest `keep` encoded series on the GPU (0: all)."""

    def __init__(self, device: int = 0, keep: int = 1, pixel_mask=None):
        self.device, self.keep, self.pixel_mask = int(device), int(keep), pixel_mask
        self.datasets, self.dropped = [], []

    def begin(self, series_id, scan_shape, image_shape):
        from quantem.gpu.streaming import LiveEncoder

        mask = self.pixel_mask if self.pixel_mask is not None and tuple(self.pixel_mask.shape) == image_shape else None
        return LiveEncoder(scan_shape, image_shape, pixel_mask=mask, device=f"cuda:{self.device}",
                           metadata={"series": f"series_{series_id}"})

    def complete(self, encoder) -> dict:
        from quantem.gpu.streaming import IncompleteAcquisition

        summary = dict(series=encoder.metadata["series"], images=encoder.received, total=encoder.total,
                       encoded_bytes=int(encoder.source.nbytes),
                       raw_bytes=encoder.received * math.prod(encoder.detector_shape) * 2)
        try:
            dataset = encoder.finish()
        except IncompleteAcquisition as error:
            self.drop(encoder, str(error))
            return dict(summary, kept=False)
        self.datasets.append(dataset)
        while self.keep and len(self.datasets) > self.keep:
            self.datasets.pop(0).close()
        return dict(summary, kept=True)

    def drop(self, encoder, reason: str) -> None:
        self.dropped.append(dict(series=encoder.metadata.get("series"), reason=reason))
        encoder.release()
