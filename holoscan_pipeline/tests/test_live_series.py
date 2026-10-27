"""live4d.live SeriesFeed and the scan raster from START: no Holoscan, no detector (CPU tensors, a recording feed)."""

import sys
import types

import numpy as np
import pytest
import torch

# graph.py imports Holoscan and live4d; these tests need neither.
for name in ("holoscan", "live4d.operators", "live4d.schemas.decoded_batch"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["live4d.operators"].__dict__.update(DecompressGpuOp=None, RdmaRxOp=None, ZmqRxOp=None,
                                               _PublicationWindow=None)
sys.modules["live4d.schemas.decoded_batch"].DecodedBatch = None
sys.modules["holoscan"].Operator = object
sys.modules["holoscan"].Input = lambda *args: None

from live4d.live import receiver  # noqa: E402
from live4d.live.receiver import SeriesFeed, scan_shape_from_start  # noqa: E402


@pytest.mark.parametrize("start, expected", [
    ({"number_of_images": 512 * 512}, (512, 512)),
    ({"number_of_images": 0}, (64, 32)),                                               # configured
    ({"number_of_images": 3072}, None),                                                 # not square
    ({"number_of_images": 3072, "header_appendix": {"scan_width": 64, "scan_height": 48}}, (48, 64)),
    ({"number_of_images": 3000, "header_appendix": {"scan_width": 64, "scan_height": 48}}, None),
    ({"number_of_images": 64, "header_appendix": {"scan_width": 64}}, None),            # pair only
    ({"number_of_images": 1, "header_appendix": {"scan_width": True, "scan_height": 1}}, None),
    ({"number_of_images": 4096, "header_appendix": "not a map"}, (64, 64)),
])
def test_scan_shape_from_start(start, expected):
    assert scan_shape_from_start(start, (64, 32)) == expected


class Feed:
    def __init__(self):
        self.events = []

    def start_series(self, start):
        self.events.append(("start", start["series_id"], start["scan_shape"]))

    def put(self, first, patterns):
        self.events.append(("batch", first, len(patterns)))

    def end_series(self):
        self.events.append(("end",))


@pytest.fixture
def series(monkeypatch):
    monkeypatch.setattr(receiver.torch.cuda, "current_stream", lambda: types.SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(receiver.cp, "from_dlpack", lambda frames: frames)
    feed = Feed()
    return feed, SeriesFeed(feed, (512, 512), log=lambda message: None)


def batch(first, n):
    return torch.zeros((n, 96, 96), dtype=torch.uint16), np.arange(first, first + n)


def test_series_reaches_the_feed(series):
    feed, consumer = series
    consumer.on_start({"series_id": 7, "number_of_images": 16, "image_size_y": 96, "image_size_x": 96}, 7)
    consumer.process(*batch(0, 10))
    consumer.process(*batch(10, 6))
    consumer.end(16)
    assert feed.events == [("start", "series_7", (4, 4)), ("batch", 0, 10), ("batch", 10, 6), ("end",)]


def test_gap_drops_the_series_and_the_next_one_arrives(series):
    feed, consumer = series
    consumer.on_start({"series_id": 1, "number_of_images": 16}, 1)
    consumer.process(*batch(0, 4))
    consumer.process(*batch(8, 4))          # gap: the series ends short, the rest is ignored
    consumer.process(*batch(12, 4))
    consumer.end(16)
    consumer.on_start({"series_id": 2, "number_of_images": 4}, 2)
    consumer.process(*batch(0, 4))
    consumer.end(4)
    assert feed.events == [("start", "series_1", (4, 4)), ("batch", 0, 4), ("end",),
                           ("start", "series_2", (2, 2)), ("batch", 0, 4), ("end",)]
    assert [item["series"] for item in consumer.dropped] == ["series_1"]


def test_unplaceable_series_never_reaches_the_feed(series):
    feed, consumer = series
    consumer.on_start({"series_id": 3, "number_of_images": 3072}, 3)
    consumer.process(*batch(0, 64))
    consumer.end(3072)
    assert feed.events == []
    assert consumer.dropped[0]["series"] == "series_3"


def test_start_without_end_ends_the_previous_series(series):
    feed, consumer = series
    consumer.on_start({"series_id": 4, "number_of_images": 16}, 4)
    consumer.process(*batch(0, 4))
    consumer.on_start({"series_id": 5, "number_of_images": 16}, 5)
    assert feed.events[-1] == ("end",)
