"""live4d.algorithms.series_store: the encoder's sink owns each finished series; leases keep one open.

Runs on a CUDA device without Holoscan: series are built with quantem.gpu's LiveEncoder, as AnsEncoderOp
builds them, and handed to the store at END.
"""
import cupy as cp
import numpy as np
import pytest

pytest.importorskip("quantem.gpu.streaming")
from live4d.algorithms.series_store import SeriesGone, SeriesStore


def finish(store, series_id, scan=(2, 2), shape=(8, 8)):
    encoder = store.begin(series_id, scan, shape)
    frames = cp.full((scan[0] * scan[1], *shape), series_id, cp.uint16)
    encoder.add(0, frames)
    return store.complete(encoder)


def test_keeps_the_latest_and_closes_the_rest():
    store = SeriesStore(keep=2)
    for n in (1, 2, 3):
        assert finish(store, n)["kept"]
    assert store.names == ["series_2", "series_3"]


def test_a_leased_series_outlives_keep_and_is_closed_on_release():
    store = SeriesStore(keep=1)
    finish(store, 1)
    with store.lease("series_1") as dataset:
        finish(store, 2)
        finish(store, 3)
        assert store.names == ["series_1", "series_3"]          # 2 retired, 1 held
        assert dataset.shape == (2, 2, 8, 8)
        assert float(dataset.mean(dim=(2, 3)).mean()) == 1.0     # still readable
    assert store.names == ["series_3"]


def test_leases_nest_and_unknown_series_are_refused():
    store = SeriesStore(keep=1)
    finish(store, 1)
    store.acquire("series_1"); store.acquire("series_1")
    finish(store, 2)
    store.release("series_1")
    assert "series_1" in store.names                            # one lease still open
    store.release("series_1")
    assert store.names == ["series_2"]
    with pytest.raises(SeriesGone):
        store.acquire("series_1")
    with pytest.raises(RuntimeError, match="not leased"):
        store.release("series_2")


def test_at_most_max_leased_series_are_held():
    store = SeriesStore(keep=0, max_leased=2)
    for n in (1, 2, 3):
        finish(store, n)
    store.acquire("series_1"); store.acquire("series_2")
    with pytest.raises(RuntimeError, match="max_leased"):
        store.acquire("series_3")
    store.acquire("series_2")                                   # one more lease on a held series is fine
