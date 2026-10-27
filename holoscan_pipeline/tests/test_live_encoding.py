"""live4d.live.encoding: SeriesEncoder stages batches into chunks for quantem.gpu's LiveEncoder.

Runs on a CUDA device without Holoscan: batches go straight to SeriesEncoder, as AnsEncoderOp hands
them over, and every kept series is decoded and compared with the counts that were sent.
"""
import cupy as cp
import numpy as np
import pytest

from dectris.messages import ScanShape, Series
from live4d.live.encoding import CHUNK, SeriesEncoder, series_scan

pytest.importorskip("quantem.gpu.streaming")


@pytest.mark.parametrize("series, expected", [
    (Series(1, expected_images=512 * 512), (512, 512)),
    (Series(1, expected_images=3072, scan=ScanShape(64, 48)), (48, 64)),   # header_appendix wins
    (Series(1, expected_images=3072), (100, 200)),                         # not square: configured
    (Series(1), (100, 200)),
])
def test_series_scan(series, expected):
    assert series_scan(series, (100, 200)) == expected


class Sink:
    """What LiveFrames does with each series, without the page."""

    def __init__(self):
        self.kept, self.dropped = [], []

    def begin(self, series_id, scan_shape, detector_shape):
        from quantem.gpu.streaming import LiveEncoder
        return LiveEncoder(scan_shape, detector_shape, metadata={"series": series_id})

    def complete(self, encoder):
        self.kept.append(encoder.finish())

    def drop(self, encoder, reason):
        encoder.release()
        self.dropped.append(reason)


def counts(n, shape=(96, 96), seed=0):
    patterns = np.random.default_rng(seed).poisson(3.0, size=(n, *shape)).astype(np.uint16)
    patterns[:, 0, 0] = np.arange(n) % 60000          # each pattern's position is checked
    return patterns


def send(encoder, patterns, batch):
    for first in range(0, len(patterns), batch):
        part = patterns[first:first + batch]
        encoder.add(cp.asarray(part), np.arange(first, first + len(part)))


def decoded(dataset):
    rows, cols, *detector = dataset.shape
    return cp.asnumpy(dataset.data.decode_scan_range_device(0, rows * cols)).reshape(-1, *detector)


@pytest.mark.parametrize("side, batch", [(64, 64), (96, 1000), (128, 512)])
def test_series_is_bit_exact_across_chunk_boundaries(side, batch):
    sink = Sink()
    encoder = SeriesEncoder(sink, (side, side), log=lambda message: None)
    patterns = counts(side * side)
    encoder.start(Series(7, expected_images=side * side, shape=(96, 96)))
    send(encoder, patterns, batch)
    encoder.end()
    assert not sink.dropped
    assert np.array_equal(decoded(sink.kept[0]), patterns)


def test_gap_drops_the_series_and_the_next_one_is_kept():
    sink = Sink()
    encoder = SeriesEncoder(sink, (64, 64), log=lambda message: None)
    patterns = counts(64 * 64)
    encoder.start(Series(1, expected_images=4096, shape=(96, 96)))
    encoder.add(cp.asarray(patterns[:100]), np.arange(100))
    encoder.add(cp.asarray(patterns[200:300]), np.arange(200, 300))       # gap
    encoder.add(cp.asarray(patterns[300:400]), np.arange(300, 400))       # ignored
    encoder.end()
    encoder.start(Series(2, expected_images=4096, shape=(96, 96)))
    send(encoder, patterns, 512)
    encoder.end()
    assert len(sink.dropped) == 1 and "do not follow" in sink.dropped[0]
    assert len(sink.kept) == 1 and np.array_equal(decoded(sink.kept[0]), patterns)


def test_short_series_is_not_kept():
    sink = Sink()
    encoder = SeriesEncoder(sink, (64, 64), log=lambda message: None)
    sink.complete = lambda live: sink.dropped.append("incomplete") if not live.complete else sink.kept.append(live)
    encoder.start(Series(3, expected_images=4096, shape=(96, 96)))
    send(encoder, counts(CHUNK // 2), 256)
    encoder.end()
    assert sink.dropped == ["incomplete"] and not sink.kept


def test_binning_change_between_series():
    sink = Sink()
    encoder = SeriesEncoder(sink, (32, 32), log=lambda message: None)
    for series, shape in ((1, (96, 96)), (2, (192, 192)), (3, (96, 96))):
        patterns = counts(32 * 32, shape, seed=series)
        encoder.start(Series(series, expected_images=1024, shape=shape))
        send(encoder, patterns, 100)
        encoder.end()
        assert np.array_equal(decoded(sink.kept[-1]), patterns)
