"""The ingest-agnostic sink: feed decoded batches into an `Accumulator`, report throughput and
completeness, and publish throttled live snapshots.

Both ingests drive it — the Holoscan graph (`GenericSinkOp` wraps one per stream) and the pure-Python
ingest (`python_ingest.PythonIngest`) — so the hot-path bookkeeping and the live stream-out behave
identically whichever one delivered the frames. No Holoscan or GPU dependency.
"""

from __future__ import annotations

import threading
import time

from realtime_stem.processing.base import Accumulator

# Held while a batch is folded in and published, and by a live control (`apps/control.py`) while
# it changes the reconstruction, so a change never lands in the middle of a batch. One pipeline
# per process, so one lock.
ACC_LOCK = threading.RLock()


def publish_snapshot(publisher, acc: Accumulator, *, final: bool = False) -> None:
    """Publish the accumulator's live channels (its `snapshot()` hook) or, without one, `result()`."""
    if hasattr(acc, "snapshot"):
        stack, names, meta = acc.snapshot()
        publisher.publish(stack, final=final, channels=names, meta=meta)
    else:
        publisher.publish(acc.result(), final=final)


class BatchSink:
    """Fold batches into `acc`; print throughput every `print_every` batches; optionally publish a
    snapshot every `live_every` seconds, or every `live_every_rows` scan rows when that is > 0."""

    def __init__(self, acc: Accumulator, *, print_every: int = 50, live_publisher=None,
                 live_every: float = 0.2, live_every_rows: int = 0):
        self.print_every = print_every
        self.live_publisher = live_publisher
        self._live_every_rows = int(live_every_rows)
        self.replace_accumulator(acc)
        self._throttle = None
        if live_publisher is not None and self._rows_frames <= 0:
            from realtime_stem.streaming.live_stream import Throttle   # lazy: no zmq otherwise
            self._throttle = Throttle(live_every)
        self._start = None
        self._last = None          # when the last batch was folded in
        self._first_frames = 0     # frames of the first batch (they arrived before _start)
        self._batches = 0
        self._frames = 0
        self._last_pub_frames = 0
        # Which scan positions have received a frame (on the ids' device; one tiny scatter per
        # batch). The end-of-scan report counts missing positions and duplicate frames from it.
        self._seen = None

    def replace_accumulator(self, acc: Accumulator) -> None:
        """Fold the following batches into `acc` (a live control switching modes)."""
        self.acc = acc
        self._n_positions = acc.scan_shape[0] * acc.scan_shape[1]
        self._wrap_ids = bool(getattr(acc, "wrap_ids", False))   # repeated scans: ids wrap
        self._rows_frames = self._live_every_rows * int(acc.scan_shape[1])

    def publish_now(self) -> None:
        """Publish a snapshot now, e.g. after a live change, without waiting for frames."""
        if self.live_publisher is not None:
            with ACC_LOCK:
                publish_snapshot(self.live_publisher, self.acc)

    @property
    def frames(self) -> int:
        return self._frames

    def feed(self, frames, ids) -> None:
        """Fold one batch: `frames` (B, H, W), `ids` (B,) raster scan indices (host or device)."""
        if self._start is None:
            self._start = time.perf_counter()

        # Real ids from the stream (not assumed-sequential). Guard against a scan-shape
        # mismatch before they hit a GPU indexed op (device asserts are opaque). int() of a
        # device max is a scalar sync — far cheaper than shipping the whole array to host.
        #
        # That scalar sync is also the Holoscan pipeline's only cross-stream fence: the
        # accumulator's torch work runs on torch's stream while DecompressGpuOp writes its 3-slot
        # output pool on its own CUDA stream, and this sync waits for the PREVIOUS batch's
        # scatter before the pool slot can be recycled under it. If you ever remove the guard,
        # replace it with an explicit torch.cuda.synchronize()/event.
        n = int(frames.shape[0])
        max_id = int(ids.max()) if n else -1
        if max_id >= self._n_positions and not self._wrap_ids:
            raise ValueError(
                f"image_id {max_id} >= scan positions {self._n_positions} — --scan doesn't match "
                f"the acquisition, or the scan repeats (use --accumulate latest, all, reset or swap)."
            )
        with ACC_LOCK:
            self.acc.process(frames, ids)
        self._mark_seen(ids)

        if self._batches == 0:
            self._first_frames = n
        self._batches += 1
        self._frames += n
        self._last = time.perf_counter()
        if self._batches % self.print_every == 0:
            elapsed = max(time.perf_counter() - self._start, 1e-9)
            print(f"[sink] batches={self._batches} frames={self._frames} "
                  f"fps={self._frames / elapsed:,.1f}", flush=True)

        # Live preview: a snapshot of the reconstruction-so-far (D2H sync gated by the throttle).
        if self.live_publisher is None:
            return
        if self._rows_frames > 0:
            ready = self._frames - self._last_pub_frames >= self._rows_frames
        else:
            ready = self._throttle.ready()
        if ready:
            self._last_pub_frames = self._frames
            with ACC_LOCK:
                publish_snapshot(self.live_publisher, self.acc)

    def _mark_seen(self, ids) -> None:
        import torch
        ids = torch.as_tensor(ids)
        if self._seen is None:
            self._seen = torch.zeros(self._n_positions, dtype=torch.bool, device=ids.device)
        if self._wrap_ids:
            ids = ids % self._n_positions
        self._seen[ids.to(device=self._seen.device, dtype=torch.long)] = True

    def completeness(self) -> tuple[bool, str]:
        """(complete, one-line report): did every scan position receive exactly one frame?

        With repeated scans (`wrap_ids`) a position is expected more than once; the report then
        gives the positions visited and the number of passes instead.
        """
        covered = int(self._seen.sum()) if self._seen is not None else 0
        missing = self._n_positions - covered
        if self._wrap_ids:
            passes = self._frames / self._n_positions
            status = "COMPLETE" if missing == 0 else "WARNING: INCOMPLETE SCAN"
            return missing == 0, (f"[sink] {status}: {covered} of {self._n_positions} scan "
                                  f"positions received ({passes:.2f} passes)")
        duplicates = self._frames - covered
        if missing == 0 and duplicates == 0:
            return True, f"[sink] COMPLETE: all {self._n_positions} scan positions received"
        parts = [f"{missing} missing"] if missing else []
        if duplicates:
            parts.append(f"{duplicates} frames for positions that already had one")
        return False, (f"[sink] WARNING: INCOMPLETE SCAN: {covered} of {self._n_positions} scan "
                       f"positions received ({', '.join(parts)}).")

    def summary(self) -> str:
        """Frames, batches, and the rate from the first to the last batch: the time spent waiting
        for the series' end afterwards does not count, nor do the first batch's frames (they
        arrived before the clock started)."""
        if self._start is None:
            return "[sink] no batches received"
        elapsed = max(self._last - self._start, 1e-9)
        rate = (f"{(self._frames - self._first_frames) / elapsed / 1e3:,.1f} kHz"
                if self._batches > 1 else "n/a (one batch)")
        return (f"[sink] TOTAL frames={self._frames} batches={self._batches} "
                f"elapsed={elapsed:.2f}s rate={rate}")
