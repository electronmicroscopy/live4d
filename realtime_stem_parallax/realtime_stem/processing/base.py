"""The two contracts every processing algorithm implements.

Keeping these tiny is the whole point of the modular design: the ingest graph and the runner
depend only on `Accumulator` / `Algorithm`, never on a specific method, so a new algorithm
(e.g. strain mapping) drops in by implementing them and registering itself — no changes to
ingest or `apps/run.py`.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Accumulator(Protocol):
    """The hot-path contract the ingest sink drives, once per decoded batch.

    An accumulator holds a *valid reconstruction-so-far*: `process` folds one batch in, and
    `result()` is cheap enough to call mid-scan (it drives the live preview). `frames` is
    (B, H, W) integer/float — a host ndarray or a device torch tensor (a Holoscan DLPack handoff);
    `ids` is the (B,) raster scan index of each frame (a host array-like or a device tensor).
    """

    detector_shape: tuple[int, int]     # (H, W) — the ingest graph configures the ops from this
    scan_shape: tuple[int, int]         # (Ny, Nx) — used to guard image ids and shape the output

    def process(self, frames, ids) -> None:
        """Fold one batch of frames (placed by scan id) into the reconstruction-so-far."""
        ...

    def result(self) -> np.ndarray:
        """Return the reconstruction-so-far as a host array (safe to call at any time)."""
        ...

    # Optional hook (not part of the runtime-checked protocol): an accumulator may also define
    #
    #     def snapshot(self) -> tuple[np.ndarray, list[str], dict]:
    #         """(C, Ny, Nx) float32 stack of named live channels, their names, and metadata."""
    #
    # The sink publishes that instead of `result()` when present (`ingest.sink.publish_snapshot`),
    # so a method can stream several derived images (parallax: corrected + phase-flipped) and
    # display metadata without the ingest or the runner knowing what they are.
    #
    # For repeated scans (`--accumulate latest|all`) an accumulator may also expose
    # `wrap_ids: bool` (ids past the scan wrap onto it, so the sink doesn't reject them) and
    # `start_series()` (called by the runner when a new detector series starts).


class Algorithm(ABC):
    """Wires one method into the runner: its CLI flags, how to build its accumulator, how to save.

    A single stateless instance per method is registered in `processing.registry`. `apps/run.py`
    owns the transport flags (endpoint, batching, live preview, …); an `Algorithm` owns everything
    that shapes the reconstruction geometry and prints its own physics summary in `build`.
    """

    #: the `--algorithm` name this instance is registered under.
    name: str = ""

    @abstractmethod
    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        """Attach this algorithm's flags (geometry, physics, tuning knobs) to the runner's parser."""

    def preprocess_argv(self, argv: list[str]) -> list[str]:
        """Optionally rewrite argv before argparse sees it (default: identity).

        Parallax uses this to make scientific-notation negatives like `--defocus -1.5e4` parse.
        """
        return argv

    def check_args(self, args: argparse.Namespace) -> list[str]:
        """Problems with the parsed args that must stop the run before it starts, e.g. a missing
        acquisition value with no safe default (default: none)."""
        return []

    @abstractmethod
    def build(self, args: argparse.Namespace) -> Accumulator:
        """Build a fresh accumulator from parsed args, and print a one-line physics summary.

        Called once per scan (continuous mode rebuilds it), so building must be cheap.
        """

    @abstractmethod
    def save(self, acc: Accumulator, out_dir: Path, tag: str = "") -> None:
        """Write this scan's results (e.g. reconstruction .npy) under `out_dir`.

        `tag` is non-empty in continuous mode (e.g. "scan003"): then also keep a copy of this
        scan's result under a name containing it, so a session does not overwrite its scans.
        """

    def save_pass(self, acc: Accumulator, out_dir: Path, k: int, dtype: str = "float32",
                  executor=None) -> None:
        """Write the image of completed pass `k` of a repeated scan (`--save-passes`), called
        from the hot path when the pass completes: take the arrays now, and write them through
        `executor` (a single-thread pool) when one is given. Optional (default: unsupported)."""
        raise NotImplementedError(f"{self.name} does not save passes")

    # ── live tuning (optional): the parameters a viewer may change while the pipeline runs ──

    def live_params(self, args: argparse.Namespace) -> list[dict]:
        """The reconstruction parameters a viewer may tune while running, each a dict with
        `name`, `label`, `unit`, `value` (now), `step` and `span` (the slider's half range about
        its starting value). Default: none."""
        return []

    def retune(self, acc: Accumulator, args: argparse.Namespace, values: dict) -> str:
        """Apply new values of `live_params` (name → value): to `args`, so later rounds and the
        saved results use them, and to the running `acc`. Called between batches. Returns what
        happened, for the viewer."""
        raise NotImplementedError(f"{self.name} has no live parameters")

    def tuned_flags(self, args: argparse.Namespace) -> list[str]:
        """The command-line flags that reproduce the current `live_params` values."""
        return []
