"""The shared live-frame renderer and the Jupyter view, without a notebook.

`LiveView.run(until_final=True)` is driven on a plain asyncio loop against a real PUB socket (the
transport the pipeline uses); the figure uses the Agg backend, so nothing is displayed.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from realtime_stem.streaming.display import PanelFigure, meta_line  # noqa: E402
from realtime_stem.streaming.live_stream import LivePublisher  # noqa: E402
from realtime_stem.streaming.notebook import LiveView  # noqa: E402


def _stack(value, coverage):
    arr = np.full((2, 16, 16), value, dtype=np.float32)
    arr[:, 0, 0] += 1.0
    meta = {"frames": int(coverage * 256), "coverage": coverage, "C10": -2000.0,
            "rotation_deg": -15.0}
    return arr, meta


def test_panel_figure_layouts():
    from matplotlib.figure import Figure

    panels = PanelFigure(Figure())
    arr, meta = _stack(1.0, 0.5)
    panels.update(arr, {"channels": ["parallax", "phase_flipped"], "meta": meta})
    assert [ax.get_title() for ax in panels.axes] == ["parallax", "phase_flipped"]
    panels.update(arr[0], {"final": True})                     # a plain 2D frame → one panel
    assert len(panels.axes) == 1
    assert "50% scanned" in meta_line({"meta": meta}) and "final" in meta_line({"final": True})


def test_live_view_receives_until_final(tmp_path):
    endpoint = "tcp://127.0.0.1:5781"
    pub = LivePublisher(endpoint)

    def feed():
        time.sleep(0.5)                                         # the view subscribes first
        for i, cov in enumerate((0.25, 0.5, 0.75)):
            arr, meta = _stack(float(i), cov)
            pub.publish(arr, channels=["parallax", "phase_flipped"], meta=meta)
            time.sleep(0.1)
        arr, meta = _stack(9.0, 1.0)
        pub.publish(arr, final=True, channels=["parallax", "phase_flipped"], meta=meta)
        pub.close()

    view = LiveView(endpoint)
    t = threading.Thread(target=feed, daemon=True)
    t.start()
    asyncio.run(asyncio.wait_for(view.run(until_final=True), timeout=10))
    t.join(5)

    assert view.n_updates == 4 and view.latest[1]["final"]
    assert view._final.is_set()
    arr, header = view.snapshot_at(0.5)
    assert header["meta"]["coverage"] == 0.5 and arr[0, 1, 1] == 1.0
    view.save_gif(tmp_path / "live.gif", fps=4)
    from PIL import Image
    assert Image.open(tmp_path / "live.gif").n_frames == 4


def test_history_is_bounded_and_picks_by_pass():
    """Continuous scanning: the view keeps only the last `history` frames; `snapshot_at` can pick
    a cursor position within a given scan pass; history=0 keeps none."""
    view = LiveView("tcp://127.0.0.1:5782", history=5)
    for k in range(3):                                          # three passes, 4 frames each
        for j, prog in enumerate((0.25, 0.5, 0.75, 1.0)):
            arr = np.full((2, 8, 8), 10 * k + j, dtype=np.float32)
            meta = {"pass": k, "progress": prog, "accumulate": "latest", "frames": 0}
            view._on_frame(arr, {"channels": ["parallax", "phase_flipped"], "meta": meta})
    assert len(view.history) == 5 and view.n_updates == 12
    assert view.snapshot_at(0.5)[0][0, 0, 0] == 21              # newest kept frame near 50%
    assert view.snapshot_at(1.0, scan_pass=1)[0][0, 0, 0] == 13
    with pytest.raises(RuntimeError):
        view.snapshot_at(0.5, scan_pass=0)                     # pass 0 has left the history
    assert "pass 3 at 100% (latest)" in meta_line(view.latest[1])

    none = LiveView("tcp://127.0.0.1:5783", history=0)
    none._on_frame(np.zeros((8, 8), np.float32), {"meta": {"frames": 1}})
    assert len(none.history) == 0 and none.latest is not None
