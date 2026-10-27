"""Live control of a running reconstruction, for the browser viewer that `run.py --viewer-port`
serves: the algorithm's tunable parameters (for parallax C10, the rotation and C12) and the
accumulate mode.

The command line's values are the starting point and stay the target: the user still turns the
microscope's knobs to meet them. The control is for fine-tuning what the knobs can't easily
reach, the rotation above all (a scan rotation otherwise).

A change is applied between batches (the sinks' `ACC_LOCK`), and kept in `args`, so the next
rounds of a continuous run and the saved results use it.
- **Same mode:** the algorithm re-shifts the running accumulator (`Algorithm.retune`). With
  `--accumulate latest` that is exact at once, from the frames it holds; the other modes restart
  their image.
- **Another mode:** a new accumulator is built from `args` and handed to the sink.

A snapshot is published straight after, so the viewer shows the result without waiting for
frames.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import Callable

from live4d.parallax.ingest.sink import ACC_LOCK, publish_snapshot
from live4d.algorithms.base import Accumulator, Algorithm

MODES = ("latest", "swap", "reset", "all")       # for repeated scans; one scan has no choice


class LiveControl:
    def __init__(self, algorithm: Algorithm, args: argparse.Namespace, *, publisher=None,
                 on_new_accumulator: Callable[[Accumulator], None] | None = None,
                 execution_context=nullcontext):
        self.algorithm, self.args = algorithm, args
        self.execution_context = execution_context
        self.publisher = publisher
        self.on_new_accumulator = on_new_accumulator   # the runner's hooks, e.g. --save-passes
        self.acc: Accumulator | None = None            # the running accumulator
        self._sink: Callable[[], object] = lambda: None   # → the running BatchSink, if any yet
        self.initial = {p["name"]: p["value"] for p in algorithm.live_params(args)}
        self.initial_mode = args.accumulate
        self.modes = list(MODES) if args.accumulate in MODES else []
        self.note = ""
        self.revision = 0                              # changes applied so far

    def attach(self, acc: Accumulator, sink: Callable[[], object] = lambda: None) -> None:
        """The runner's accumulator for this round, and how to find the sink driving it (the
        Holoscan graph makes its sink only when it starts)."""
        with ACC_LOCK, self.execution_context():
            self.acc, self._sink = acc, sink

    def state(self) -> dict:
        flags = self.algorithm.tuned_flags(self.args)
        if self.modes:
            flags += ["--accumulate", self.args.accumulate]
        return {"params": self.algorithm.live_params(self.args), "initial": self.initial,
                "mode": self.args.accumulate, "modes": self.modes,
                "initial_mode": self.initial_mode, "flags": " ".join(flags), "note": self.note}

    def update(self, changes: dict) -> dict:
        """Apply `changes` ({name: value}, optionally "mode", or {"reset": true} for the command
        line's values) and return the new state. Raises ValueError for a change it can't make."""
        changes = dict(changes)
        if changes.pop("reset", False):
            changes = dict(self.initial)
            if self.modes:
                changes["mode"] = self.initial_mode
        mode = changes.pop("mode", None)
        notes = []
        with ACC_LOCK, self.execution_context():
            if mode is not None and mode != self.args.accumulate:
                notes.append(self._switch(mode))
            if changes:
                notes.append(self.algorithm.retune(self.acc, self.args, changes))
            self.note = "; ".join(n for n in notes if n)
            self.revision += 1
            sink = self._sink()
            if sink is not None:
                sink.publish_now()
            elif self.publisher is not None and self.acc is not None:
                publish_snapshot(self.publisher, self.acc)
        return self.state()

    def _switch(self, mode: str) -> str:
        if mode not in self.modes:
            raise ValueError(f"mode {mode!r}: choose one of {', '.join(self.modes) or 'none'}")
        if mode == "latest" and self.acc is not None:
            n_pos = self.acc.scan_shape[0] * self.acc.scan_shape[1]
            if getattr(self.args, "batch_size", 0) > n_pos:
                raise ValueError(f"latest needs --batch-size <= the scan's {n_pos} positions")
        self.args.accumulate = mode
        if self.acc is None:
            return f"{mode} from the first frame"
        new = self.algorithm.build(self.args)
        if self.on_new_accumulator is not None:
            self.on_new_accumulator(new)
        sink = self._sink()
        if sink is not None:
            sink.replace_accumulator(new)
        self.acc = new
        return f"{mode}: a new image from the next frames"


class QueuedLiveControl:
    """HTTP threads enqueue bounded requests; only active operator callbacks touch CUDA state."""
    def __init__(self, control, capacity=16):
        import queue
        self.control = control
        self.pending = queue.Queue(maxsize=capacity)
        self.closed = False

    def state(self):
        with ACC_LOCK:
            state = self.control.state()
            if not self.pending.empty():
                state["note"] = "Tuning queued for the next parallax batch"
            return state

    def update(self, changes):
        import queue
        with ACC_LOCK:
            if self.closed:
                raise ValueError("Parallax acquisition has stopped")
            try:
                self.pending.put_nowait(dict(changes))
            except queue.Full:
                raise ValueError("Parallax tuning queue is full") from None
            return self.state()

    def apply_pending(self):
        import queue
        while True:
            try:
                changes = self.pending.get_nowait()
            except queue.Empty:
                return
            try:
                self.control.update(changes)
            except (ValueError, TypeError) as exc:
                self.control.note = str(exc)
