"""Drive the Holoscan sink wrapper without Holoscan.

`holoscan_graph.py` only runs inside the container, so here `holoscan.core` and the compiled
`dectris_holoscan_ops` are replaced by stubs in `sys.modules`, and `GenericSinkOp.compute` is fed
messages shaped like `DecompressGpuOp`'s output (a dict of DLPack-capable tensors). This pins the
wrapper's hand-off to `BatchSink`: frames and ids reach the accumulator unchanged, the id guard
fires, live snapshots are published, and `summary()` reports.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_stem.processing.parallax.engine import ParallaxAccumulator, preprocess_geometry


@pytest.fixture
def holoscan_graph(monkeypatch):
    core = types.ModuleType("holoscan.core")

    class Operator:
        def __init__(self, fragment, *args, name=""):
            self.name = name

    class Application:
        def __init__(self, *args, **kwargs):
            pass

    core.Operator, core.Application, core.OperatorSpec = Operator, Application, object
    holoscan = types.ModuleType("holoscan")
    holoscan.core = core
    ops = types.ModuleType("dectris_holoscan_ops")
    ops.DecompressGpuOp = ops.ZmqRxOp = object
    monkeypatch.setitem(sys.modules, "holoscan", holoscan)
    monkeypatch.setitem(sys.modules, "holoscan.core", core)
    monkeypatch.setitem(sys.modules, "dectris_holoscan_ops", ops)
    monkeypatch.delitem(sys.modules, "realtime_stem.ingest.holoscan_graph", raising=False)
    return importlib.import_module("realtime_stem.ingest.holoscan_graph")


class _Input:
    def __init__(self, message):
        self.message = message

    def receive(self, name):
        assert name == "input"
        return self.message


class _Publisher:
    def __init__(self):
        self.sent = []

    def publish(self, arr, *, final=False, channels=None, meta=None):
        self.sent.append((arr.shape, channels, meta))


def _geometry():
    return preprocess_geometry(det_shape=(32, 32), scan_shape=(8, 8), scan_sampling=(20.0, 20.0),
                               energy=300e3, semiangle_cutoff=4.0,
                               angular_sampling=(0.307617, 0.307617),
                               aberration_coefs={"C10": -1.5e4})


def test_compute_feeds_accumulator(holoscan_graph):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 60000, size=(64, 32, 32), dtype=np.uint16)
    ids = np.arange(64)[::-1].copy()

    acc = ParallaxAccumulator(_geometry(), device="cpu")
    pub = _Publisher()
    op = holoscan_graph.GenericSinkOp(None, acc=acc, live_publisher=pub, live_every_rows=4)
    for sl in (slice(0, 32), slice(32, 64)):    # two batches, as DecompressGpuOp emits them
        message = {"frames": torch.from_numpy(frames[sl]),                    # torch.uint16
                   "image_ids": torch.from_numpy(ids[sl].astype(np.uint64))}
        op.compute(_Input(message), None, None)

    ref = ParallaxAccumulator(_geometry(), device="cpu")
    ref.process(frames, ids)
    scale = np.abs(ref.result()).max()
    np.testing.assert_allclose(acc.result(), ref.result(), atol=1e-5 * scale)
    assert len(pub.sent) == 2                   # every 4 rows of 8 = every 32 frames
    shape, channels, meta = pub.sent[-1]
    assert channels == ["parallax", "phase_flipped"] and shape == (2, 8, 8)
    assert meta["frames"] == 64
    assert "TOTAL frames=64 batches=2" in op.summary()


def test_compute_guards_scan_shape(holoscan_graph):
    acc = ParallaxAccumulator(_geometry(), device="cpu")
    op = holoscan_graph.GenericSinkOp(None, acc=acc)
    message = {"frames": torch.zeros((2, 32, 32), dtype=torch.uint16),
               "image_ids": torch.tensor([3, 64])}        # 64 ≥ 8·8 positions
    with pytest.raises(ValueError, match="--scan"):
        op.compute(_Input(message), None, None)


def test_compute_accepts_repeated_scans(holoscan_graph):
    """--accumulate latest: ids past the scan (continuous scanning) wrap instead of failing."""
    acc = ParallaxAccumulator(_geometry(), device="cpu", accumulate="latest", wrap_ids=True)
    op = holoscan_graph.GenericSinkOp(None, acc=acc)
    frames = np.random.default_rng(1).integers(0, 60000, size=(2, 32, 32), dtype=np.uint16)
    for i, image_id in enumerate((3, 64 + 3)):                  # position 3 on passes 1 and 2
        message = {"frames": torch.from_numpy(frames[i:i + 1]),
                   "image_ids": torch.tensor([image_id])}
        op.compute(_Input(message), None, None)
    ref = ParallaxAccumulator(_geometry(), device="cpu")
    ref.process(frames[1:], np.array([3]))                      # only the newer frame remains
    np.testing.assert_allclose(acc.result(), ref.result(), atol=1e-5 * np.abs(ref.result()).max())
