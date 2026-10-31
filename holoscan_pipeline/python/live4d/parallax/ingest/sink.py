"""The ingest-agnostic sink: feed decoded batches into an `Accumulator`, report throughput and
completeness, and publish throttled live snapshots.

Both ingests drive it — the Holoscan graph (`ParallaxOp` wraps one per stream) and the pure-Python
ingest (`python_ingest.PythonIngest`) — so the hot-path bookkeeping and the live stream-out behave
identically whichever one delivered the frames. No Holoscan or GPU dependency.
"""

from __future__ import annotations

import threading
import time

from live4d.algorithms.base import Accumulator

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


#: OBF's device time per batch at least and at most (`BatchSink.obf_share`)
OBF_MIN_S, OBF_MAX_S = 0.0005, 0.05
#: OBF's budget is saved up over batches into steps of at least this long: a step's own cost
#: (gathering the positions, launching the deposit) is then a small part of it, and the branch
#: still stalls for far less than it takes to fill the receiver's publication window
OBF_STEP_S = 0.008
#: the branch's load (its batches' time over their intervals, per LOAD_WINDOW_S) above which OBF's
#: share halves, and below which it grows back (`BatchSink.obf_scale`)
LOAD_HIGH, LOAD_LOW, LOAD_WINDOW_S = 0.8, 0.6, 0.5


class BatchSink:
    """Fold batches into `acc`; print throughput every `print_every` batches; optionally publish a
    snapshot every `live_every` seconds, or every `live_every_rows` scan rows when that is > 0.
    An accumulator followed by OBF (`acc.obf`) gets a step after each batch: `obf_share` of the
    interval between batches, within [OBF_MIN_S, OBF_MAX_S], so that OBF's part of the branch
    follows the frame rate. The live branch gives the source's interval (`interval_s`, from the
    batches' publish times) and how long the batch waited (`queued_s`): a batch that waited longer
    than the interval means the branch is behind, and OBF then waits. Without them (offline) the
    interval is the time since the previous batch arrived here.

    OBF also yields to the rest of the branch. It skips the batch that publishes a snapshot, and
    its share is scaled by `obf_scale`, which halves while the branch's batches take more than
    LOAD_HIGH of their intervals (live only) and grows back below LOAD_LOW. A stall of the branch
    fills the receiver's publication window and then the source's buffers within tens of ms, so the
    branch must keep up batch by batch, not only on average. The shares are saved up into steps of
    OBF_STEP_S, so that at a high frame rate OBF steps every few batches, efficiently."""

    def __init__(self, acc: Accumulator, *, print_every: int = 50, live_publisher=None,
                 live_every: float = 0.2, live_every_rows: int = 0, obf_share: float = 0.4):
        self.print_every = print_every
        self.live_publisher = live_publisher
        self.obf_share = float(obf_share)
        self.obf_scale = 1.0       # OBF's share scaled down while the branch is busy
        self._obf_credit = 0.0     # OBF's budget saved up towards a step of OBF_STEP_S, s
        self._busy = self._span = 0.0   # this load window's batch time and intervals, s
        self._fed = None           # when the previous batch arrived here (offline: the interval)
        self._live_every_rows = int(live_every_rows)
        self.replace_accumulator(acc)
        self.telemetry = getattr(live_publisher, "telemetry", None)
        if self.telemetry is not None:
            self.telemetry.begin(self._n_positions)
        self._throttle = None
        if live_publisher is not None and self._rows_frames <= 0:
            from live4d.parallax.streaming.live_stream import Throttle   # lazy: no zmq otherwise
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

    def feed(self, frames, ids, *, interval_s: float | None = None,
             queued_s: float | None = None) -> None:
        """Fold one batch: `frames` (B, H, W), `ids` (B,) raster scan indices (host or device);
        `interval_s`, `queued_s`: the source's batch interval and this batch's wait, when known."""
        entered = time.perf_counter()
        if self._start is None:
            self._start = entered

        # Real ids from the stream (not assumed-sequential). Guard against a scan-shape
        # mismatch before they hit a GPU indexed op (device asserts are opaque). int() of a
        # device max is a scalar sync — far cheaper than shipping the whole array to host.
        #
        # Live operators execute all tensor work on the runtime CUDA stream.
        n = int(frames.shape[0])
        max_id = int(ids.max()) if n else -1
        if max_id >= self._n_positions and not self._wrap_ids:
            raise ValueError(
                f"image_id {max_id} >= scan positions {self._n_positions} — --scan doesn't match "
                f"the acquisition, or the scan repeats (use --accumulate latest, all, reset or swap)."
            )
        arrived, gap = time.perf_counter(), interval_s
        if gap is None and self._fed is not None:
            gap = arrived - self._fed
        self._fed = arrived
        with ACC_LOCK:
            self.acc.process(frames, ids)
        self._mark_seen(ids)

        if self._batches == 0:
            self._first_frames = n
        self._batches += 1
        self._frames += n
        publishing = self._snapshot_due()
        obf = getattr(self.acc, "obf", None)
        if obf is not None:
            obf.share_scale = self.obf_scale
            self._obf_credit = min(self._obf_credit + self._obf_budget(gap, queued_s, publishing),
                                   OBF_MAX_S)
            if self._obf_credit >= OBF_STEP_S:
                with ACC_LOCK:
                    obf.step(self._obf_credit)
                self._obf_credit = 0.0
        self._last = time.perf_counter()
        if self.telemetry is not None:
            self.telemetry.record(n, entered, self._last)
        if self._batches % self.print_every == 0:
            elapsed = max(time.perf_counter() - self._start, 1e-9)
            print(f"[sink] parallax branch batches={self._batches} frames={self._frames} "
                  f"fps={self._frames / elapsed:,.1f}", flush=True)

        # Live preview: a snapshot of the reconstruction-so-far (D2H sync gated by the throttle).
        if publishing:
            self._last_pub_frames = self._frames
            with ACC_LOCK:
                publish_snapshot(self.live_publisher, self.acc)
        self._pace_load(time.perf_counter() - entered, interval_s)

    def _snapshot_due(self) -> bool:
        """Whether this batch publishes a snapshot (every `live_every` s, or `live_every_rows`)."""
        if self.live_publisher is None:
            return False
        if self._rows_frames > 0:
            return self._frames - self._last_pub_frames >= self._rows_frames
        return self._throttle.ready()

    def _obf_budget(self, gap, queued_s, publishing: bool) -> float:
        """OBF's device time for this batch, s: `obf_share` of the interval times `obf_scale`,
        within [OBF_MIN_S, OBF_MAX_S]; none while the branch is behind (the batch waited longer
        than the interval), when this batch also publishes a snapshot, or at a zero scale."""
        behind = queued_s is not None and gap is not None and queued_s > gap
        if behind or publishing or self.obf_scale <= 0:
            return 0.0
        budget = self.obf_scale * self.obf_share * (OBF_MAX_S if gap is None else gap)
        return min(max(budget, OBF_MIN_S), OBF_MAX_S)

    def _pace_load(self, busy_s: float, interval_s: float | None) -> None:
        """Track the branch's load, its batches' time over their source intervals (live only),
        and scale OBF's share: halved (then zero below 1/16) above LOAD_HIGH, grown back by a
        quarter below LOAD_LOW, once per LOAD_WINDOW_S of intervals."""
        if interval_s is None or interval_s <= 0:
            return
        self._busy += busy_s
        self._span += interval_s
        if self._span < LOAD_WINDOW_S:
            return
        load = self._busy / self._span
        if load > LOAD_HIGH:
            self.obf_scale = self.obf_scale / 2 if self.obf_scale > 1 / 16 else 0.0
        elif load < LOAD_LOW and self.obf_scale < 1:
            self.obf_scale = min(1.0, max(self.obf_scale, 1 / 16) * 1.25)
        self._busy = self._span = 0.0

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
        if self.telemetry is not None:
            self.telemetry.finish(covered, self._wrap_ids)
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
        return (f"[sink] TOTAL parallax branch frames={self._frames} batches={self._batches} "
                f"elapsed={elapsed:.2f}s rate={rate}")
