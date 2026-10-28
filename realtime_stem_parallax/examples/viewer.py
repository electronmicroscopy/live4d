"""Live viewer: subscribe to a `run.py --live-endpoint` PUB and redraw as the scan fills in.

The reconstruction "wipes in" during the scan because the sink publishes a throttled snapshot of
the reconstruction-so-far (~5 Hz). This viewer keeps only the newest frame (CONFLATE), so it never
lags behind the pipeline. Each named channel of a snapshot (for parallax: the corrected image and
the phase-flipped one, optionally coverage) gets its own panel.

    python examples/viewer.py tcp://127.0.0.1:5601
    python examples/viewer.py tcp://127.0.0.1:5601 --high-pass 25   # subtract a slow background

One example consumer of the stream — the pipeline doesn't depend on it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root on path

from realtime_stem.streaming.display import PanelFigure, meta_line   # noqa: E402


def _log(msg: str) -> None:
    print(f"[viewer {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _View:
    """A desktop window around the shared `PanelFigure` renderer."""

    def __init__(self, label: str, hp_sigma: float):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.label = label
        self.fig = plt.figure(figsize=(7.5, 7.5))
        try:
            self.fig.canvas.manager.set_window_title(f"live view — {label}")
        except Exception:
            pass
        self.panels = PanelFigure(self.fig, high_pass_sigma=hp_sigma, label=label)
        self.n = 0
        plt.ion()
        plt.show(block=False)
        self.fig.suptitle(f"waiting for {label} …")
        self.fig.canvas.draw()

    @property
    def axes(self):
        return self.panels.axes

    def alive(self) -> bool:
        return self.plt.fignum_exists(self.fig.number)

    def pump(self, seconds: float) -> None:
        self.fig.canvas.flush_events()
        self.fig.canvas.start_event_loop(seconds)

    def render(self, raw, header: dict) -> None:
        self.n += 1
        ranges = self.panels.update(raw, header,
                                    title=f"update #{self.n} at {time.strftime('%H:%M:%S')}")
        self.fig.canvas.draw()
        shape = raw.shape[-2:]
        _log(f"update #{self.n}: {len(ranges)}×{shape} "
             f"{' '.join(f'[{lo:.3g}, {hi:.3g}]' for lo, hi in ranges)} {meta_line(header)}")


def run(endpoint: str, label: str, hp_sigma: float) -> None:
    from realtime_stem.streaming.live_stream import LiveSubscriber   # lazy: needs pyzmq
    view = _View(label, hp_sigma)
    sub = LiveSubscriber(endpoint)
    _log(f"subscribed to {endpoint}")
    while view.alive():
        try:
            msg = sub.recv_latest(timeout_ms=100)
            if msg is not None:
                view.render(*msg)
            view.pump(0.01)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            _log(f"cycle error ({type(exc).__name__}: {exc}); continuing")
            time.sleep(0.2)
    sub.close()
    _log("window closed — exiting")


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python examples/viewer.py", description=__doc__)
    p.add_argument("endpoint", help="tcp://host:port of the run.py --live-endpoint PUB.")
    p.add_argument("--high-pass", type=float, default=0.0, metavar="SIGMA_PX",
                   help="Gaussian high-pass sigma in pixels (0 = off).")
    p.add_argument("--label", default=None, help="Window/title label.")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    run(args.endpoint, args.label or args.endpoint, args.high_pass)


if __name__ == "__main__":
    main()
