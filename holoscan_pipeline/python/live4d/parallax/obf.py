"""`ObfTracker`: OBF as a live channel of the parallax accumulator, advanced a step at a time.

OBF costs about a hundred times more per frame than parallax (`scripts/bench_obf.py`), so it runs
only while it is shown, and never in `process()`. Its owner advances it with `step(budget_s)`:
the sink after each batch (`BatchSink.feed`), and between batches the live branch's tick and the
offline replay's idle loop (`ParallaxRuntime.idle`). A step does a bounded amount of device work on
the caller's current stream, so in the live graph all of it runs inside the parallax operator's
callbacks, on its runtime stream, and nothing runs beside them. Its weights (the Gram) depend on
the geometry and not on the data; they are built a unit at a time (`ObfSetup.build_steps`), and
can be kept built for the current geometry while OBF is off (`precompute`), so that showing OBF
needs no more than the frames' own work.

When OBF is switched on, or after a retune, its numerator starts empty and covers the scan from the
scan line outwards:

- forward with the scan: each new frame is deposited as it arrives (the accumulator hands the
  batch's ids to `mark`, on the host);
- backward through the frames the `latest` store already holds, newest first, with the budget
  left;
- a covered position the scan visits again is updated, its new values minus those deposited before
  (kept in a shadow of the store), so the numerator is always exactly the sum over the frames it
  covers, whatever order they came in.

Until every position held is covered the channel is a composite (`compose`): OBF on the output rows
it covers, and elsewhere the phase-flipped parallax image of the same frames, scaled to OBF's
contrast. So choosing OBF shows an image at once, and OBF wipes in from the scan line. Slower than
the frames, OBF lags: positions replaced again before it reaches them are deposited once, at
their newest frame, and the ingest never waits beyond the budget.

- The budget is device time. On CUDA each step is bracketed by timing events: the next step does
  nothing until they have completed (at most one step's work is queued), and sizes its work from
  the time the last one took (`_Pace`). On the CPU, for development, a step is timed on the
  host. OBF does not run on Apple's `mps`.
- The normalization V = I/m − 1 needs each bright-field pixel's mean m. It is taken over the frames
  held when the coverage starts and frozen. A new m restarts the coverage: while fewer than
  `MIN_MEAN_POSITIONS` are held, each time they double, and when m drifts by more than
  `renorm_drift` rms. The final image (`finalize`) is re-swept with the exact means.

States: `off`, `building` (the weights), `filling` (the coverage growing), `live` (every position
held covered), `error`.
"""

from __future__ import annotations

import time
import traceback
from collections import OrderedDict

import numpy as np
import torch

from live4d.algorithms.obf.core import ObfSetup, _raw_float, geometry_key, normalize
from live4d.algorithms.parallax.readout import fourier_filters

__all__ = ["ObfTracker"]

_SETUPS: "OrderedDict[tuple, ObfSetup]" = OrderedDict()
SETUP_CACHE_SIZE = 3            # setups kept (each holds a Gram and the operands, ~0.2 GB binned)
#: positions the frozen bright-field means are taken over at least (while fewer are held, they are
#: renewed each time the positions held double). Means from the first 8 rows of the binned Gd2O3
#: scan, 1.6% rms off the scan's, change its OBF image by 1.1% (correlation 0.99995)
MIN_MEAN_POSITIONS = 2048
#: scan rows beyond the probe's radius that an output row still waits for before it shows OBF
_REACH_MARGIN = 2
#: seconds a failed step waits before OBF tries again
_RETRY_S = 5.0


def _cache_get(key) -> ObfSetup | None:
    if key in _SETUPS:
        _SETUPS.move_to_end(key)
        return _SETUPS[key]
    return None


def _cache_put(key, setup: ObfSetup) -> None:
    """Keep the last `SETUP_CACHE_SIZE` setups, so a retune and back finds the Gram again."""
    _SETUPS[key] = setup
    while len(_SETUPS) > SETUP_CACHE_SIZE:
        _SETUPS.popitem(last=False)


def _prefix(ok: np.ndarray) -> int:
    """How many leading entries of `ok` are true."""
    return int(np.argmin(ok)) if not ok.all() else int(ok.size)


def _longest_run(rows: np.ndarray) -> list[int] | None:
    """[r0, r1) of the longest run of true rows, or None for none."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], rows.astype(np.int8), [0]))))
    if edges.size == 0:
        return None
    starts, ends = edges[::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return [int(starts[k]), int(ends[k])]


class _Pace:
    """How much work fits a step's budget: units per step, scaled by the time the last step took
    (at most halved or doubled per step, at least `floor`)."""

    def __init__(self, floor: int):
        self.floor = max(1, int(floor))
        self.allow = float(self.floor)
        self.per_s = None                            # units per second, lately

    def update(self, units: int, seconds: float, budget_s: float) -> None:
        if units <= 0 or seconds <= 0:
            return
        rate = units / seconds
        self.per_s = rate if self.per_s is None else 0.7 * self.per_s + 0.3 * rate
        self.allow = min(max(rate * budget_s, self.allow / 2, self.floor), 2 * self.allow)

    @property
    def units(self) -> int:
        return max(self.floor, int(self.allow))


class ObfTracker:
    """Follow a `latest` `ParallaxAccumulator` with OBF. Attaches itself (`acc.obf`); its owner
    calls `step` and, at the end of a finite acquisition, `finalize`. `enabled`: reconstruct OBF
    (the channel); `precompute`: keep the setup, the Gram above all, built for the current
    geometry also while OBF is off. `builder(geometry, device=, dtype=, fft_sizes=)` returns a
    generator like `ObfSetup.build_steps`."""

    def __init__(self, acc, *, enabled: bool = True, precompute: bool = False,
                 live_every: float = 0.2, renorm_every: float = 10.0, renorm_drift: float = 0.02,
                 fft_sizes: bool = False, builder=None):
        if acc.accumulate != "latest":
            raise ValueError("OBF follows the frames of accumulate: latest")
        if acc.device.type == "mps":
            raise ValueError("OBF runs on cuda, or on the cpu for development, not on mps")
        self.acc = acc
        self.device = acc.device
        self.n = int(acc.n_positions)
        self.scan_shape = tuple(int(v) for v in acc.scan_shape)
        self.live_every = float(live_every)
        self.renorm_every = float(renorm_every)      # s between checks of the bright-field means
        self.renorm_drift = float(renorm_drift)      # their rms relative change that renews them
        self.fft_sizes = bool(fft_sizes)
        self._builder = builder or ObfSetup.build_steps
        self._closed = False
        self._enabled = bool(enabled)
        self.precompute = bool(precompute)
        self._geometry = acc.geometry
        self.state = "building" if enabled else "off"
        self.note = ""
        self._retry_at = 0.0
        # the published image, the output rows it covers (None: all), and when
        self.image_np: np.ndarray | None = None
        self.done_np: np.ndarray | None = None
        self._image_time = 0.0
        self._scale = None                           # the stand-in's scale to OBF's contrast
        self._last_publish = 0.0
        self.sweep_s = None                          # the last re-sweep (finalize), s
        self.coverage = 0.0                          # the share of the positions held covered
        # what the store holds and what changed in it, on the host (from the batches' ids)
        self._held = np.zeros(self.n, dtype=bool)
        self._dirty = np.zeros(self.n, dtype=bool)   # written since the coverage started
        self._max_id = -1
        if acc._store is not None:                   # attached to a store already filling
            self._held[:] = (acc._visits > 0).cpu().numpy()
            self._max_id = int(acc._max_id)
        # the weights being built: a `build_steps` generator, its geometry key and start
        self._building = None
        self._building_key = None
        self._build_t0 = 0.0
        self._setup: ObfSetup | None = None
        self._buffer = None
        self._shadow = None                          # (positions, M) raw values deposited
        self._seen = np.zeros(self.n, dtype=bool)    # positions covered
        self._lo = self._hi = 0                      # the covered stretch, [lo, hi) in raster order
        self._cursor = 0                             # round-robin start of the updates
        self._kind = None
        self._mean = None
        self._mean_count = 0
        self._mean_time = 0.0
        self._held_count = 0                         # positions held at the last readout
        self._exact = False                          # the means are those of the shadow as it is
        self._changed = False                        # deposits since the last image
        self._image_s = 0.0                          # the last readout, s
        self._base = None                            # the readout's Butterworth / Nyquist filter
        self._deposits = _Pace(1)                    # positions per step (floor: one sub-batch)
        self._builds = _Pace(1)                      # build units per step
        self._events = None                          # (start, end, positions, build units, budget, host s)
        self.share_scale = 1.0                       # the sink's: OBF's share scaled down while busy
        acc.obf = self

    # ── host-only requests (the live control's, applied in a callback) ──

    def mark(self, ids: np.ndarray) -> None:
        """The absolute image ids of a batch the store took (host, before the wrap)."""
        if ids.size == 0:
            return
        pos = ids % self.n
        self._held[pos] = True
        if self._enabled:
            self._dirty[pos] = True
        self._max_id = max(self._max_id, int(ids.max()))

    def set_geometry(self, geometry) -> None:
        """A retune: the weights are rebuilt by the next steps and the coverage restarts; the
        channel shows the parallax stand-in meanwhile."""
        self._geometry = geometry

    def set_enabled(self, on: bool) -> str:
        """Switch OBF on or off. Off frees the shadow and the image at the next step."""
        on = bool(on)
        if on == self._enabled:
            return f"OBF already {'on' if on else 'off'}"
        self._enabled = on
        self._dirty[:] = False
        self._buffer = None
        return "OBF on" if on else "OBF off"

    def set_precompute(self, on: bool) -> None:
        """Keep the setup built for the current geometry while OBF is off, or not."""
        self.precompute = bool(on)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def weights_state(self) -> str:
        """`ready` when the setup (the Gram) is that of the current geometry, `building` while it
        is built or due, `off` when nothing asks for it."""
        key = geometry_key(self._geometry)
        if self._setup is not None and self._setup.key == key:
            return "ready"
        if self._building is not None or self._enabled or self.precompute:
            return "building"
        return "off"

    def image(self) -> np.ndarray | None:
        """The latest OBF image (host float32, the parallax grid; zero where not yet covered)."""
        return self.image_np if self._enabled else None

    def meta(self) -> dict:
        meta = {"obf_state": self.state, "obf_weights_state": self.weights_state(),
                "obf_coverage": round(self.coverage, 4)}
        if self.image_np is not None:
            meta["obf_age_s"] = round(time.monotonic() - self._image_time, 3)
        if self._deposits.per_s is not None:
            meta["obf_rate_khz"] = round(self._deposits.per_s / 1e3, 2)
        if self.sweep_s is not None:
            meta["obf_sweep_s"] = round(self.sweep_s, 3)
        if self.share_scale < 1:
            meta["obf_share_scale"] = round(self.share_scale, 3)
        return meta

    def compose(self, standin: np.ndarray) -> tuple[np.ndarray, list[int] | None]:
        """The channel to show: OBF on the output rows it covers, `standin` (the phase-flipped
        parallax image of the same frames) elsewhere, scaled to OBF's contrast over the rows both
        cover; and [r0, r1), the longest run of OBF rows, while it is partial (None when all OBF).
        """
        image, done = self.image_np, self.done_np
        standin = np.asarray(standin, dtype=np.float32)
        if image is None:
            if self._scale is None:
                return standin, None
            return (self._scale * (standin - standin.mean())).astype(np.float32), None
        if done is None:
            return image, None
        if done.sum() >= 4:                  # match the stand-in to OBF where both are there
            spread = float(standin[done].std())
            if spread > 0:
                self._scale = float(image[done].std()) / spread
            offset = float(standin[done].mean()) - float(image[done].mean()) / (self._scale or 1.0)
        else:
            offset = float(standin.mean())
        scaled = (self._scale or 1.0) * (standin - offset)
        return np.where(done[:, None], image, scaled).astype(np.float32), _longest_run(done)

    def close(self) -> None:
        """Stop following and free the device buffers."""
        self._closed = True
        self._building = None
        self._buffer = self._shadow = self._mean = None

    # ── the owner's: a step, and the end of an acquisition ──

    def step(self, budget_s: float) -> bool:
        """Advance OBF by about `budget_s` of device work on the current stream; True when a new
        image was published. Does nothing while the last step's work is still running (CUDA)."""
        if self._closed or time.monotonic() < self._retry_at or not self._previous_done():
            return False
        start = self._event()
        published = False
        positions = units = 0
        t0 = time.perf_counter()
        try:
            positions, units, published = self._advance(budget_s, t0 + budget_s)
        except Exception as exc:   # report, keep the pipeline running, retry after a pause
            self.state, self.note = "error", f"{type(exc).__name__}: {exc}"
            print(f"[obf] error, OBF paused: {self.note}", flush=True)
            traceback.print_exc()
            self._building, self._buffer = None, None
            self._retry_at = time.monotonic() + _RETRY_S
        if start is not None:
            end = self._event()
            self._events = (start, end, positions, units, budget_s, time.perf_counter() - t0)
        else:
            self._account(positions, units, time.perf_counter() - t0, budget_s)
        return published

    def finalize(self, timeout_s: float = 120.0) -> np.ndarray | None:
        """The image of every frame held, with exact bright-field means (the image
        `reconstruct_once` gives for the store), computed now in the caller: the end of a finite
        acquisition. None if OBF is off or nothing is held; the latest image if `timeout_s`
        passes first."""
        if not self._enabled or self._closed:
            return None
        deadline = time.perf_counter() + timeout_s
        setup = None
        while setup is None and time.perf_counter() < deadline:
            setup = self._ensure_setup(deadline, unlimited=True)
        if setup is None:
            print(f"[obf] the weights are not built after {timeout_s:.0f} s", flush=True)
            return self.image_np
        if self._buffer is None and not self._restart(setup):
            return None
        while time.perf_counter() < deadline:
            work = self._gather(setup, self.n)
            if work is None:
                break
            self._fold(setup, *work)
        if (self._held & ~self._seen).any():
            print(f"[obf] the final image is not ready after {timeout_s:.0f} s "
                  f"({100 * self.coverage:.0f}% covered); saving the last one", flush=True)
            return self.image_np
        if not self._exact:
            self._resweep(setup)
        self._publish(setup, force=True)
        return self.image_np

    # ── the work ──

    def _event(self):
        if self.device.type != "cuda":
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _previous_done(self) -> bool:
        """Whether the last step's device work has completed (and account for its time)."""
        if self._events is None:
            return True
        start, end, positions, units, budget_s, host_s = self._events
        if not end.query():
            return False
        self._events = None
        # its device time, or its host time where that was longer (the setup's NumPy blocks)
        self._account(positions, units, max(start.elapsed_time(end) / 1e3, host_s), budget_s)
        return True

    def _account(self, positions: int, units: int, seconds: float, budget_s: float) -> None:
        if positions:
            self._deposits.update(positions, seconds, budget_s)
        elif units:
            self._builds.update(units, seconds, budget_s)

    def _advance(self, budget_s: float, deadline: float) -> tuple[int, int, bool]:
        """One step's work: (positions deposited, build units, published)."""
        if not self._enabled:
            if self._shadow is not None or self.image_np is not None:
                self._reset()
                self.image_np = self.done_np = None
            self.state = "off"
            units = 0
            if self.precompute:                      # the weights: kept for the geometry
                units = self._ensure_setup(deadline, count=True)
            return 0, units, False
        units = self._ensure_setup(deadline, count=True)
        setup = self._setup
        if setup is None or setup.key != geometry_key(self._geometry):
            return 0, units, False                   # building
        if self._buffer is None and not self._restart(setup):
            return 0, units, False                   # no frames held yet
        self.state = "live" if self.coverage >= 1 else "filling"
        allowed, done = self._deposits.units, 0
        while done < allowed and time.perf_counter() < deadline:
            work = self._gather(setup, allowed - done)
            if work is None:
                break
            done += self._fold(setup, *work)
        if self._renew_due():
            self._buffer = None                      # new means: the coverage restarts
            return done, units, False
        return done, units, self._publish(setup)

    def _ensure_setup(self, deadline: float, count: bool = False, unlimited: bool = False):
        """The setup of the current geometry, advancing its build within the step; None (or with
        `count`, the build units done) while it is being built."""
        geometry = self._geometry
        key = geometry_key(geometry)
        if self._setup is not None and self._setup.key == key:
            return 0 if count else self._setup
        cache_key = (key, str(self.device), self.fft_sizes)
        setup = _cache_get(cache_key)
        units = 0
        if setup is None:
            if self._building is None or self._building_key != key:
                if self._enabled:
                    self.state = "building"
                    self.image_np = self.done_np = None  # an image of another geometry
                self._building = self._builder(geometry, device=self.device, dtype=None,
                                               fft_sizes=self.fft_sizes)
                self._building_key, self._build_t0 = key, time.perf_counter()
            allowed = None if unlimited else self._builds.units
            while allowed is None or units < allowed:
                try:
                    next(self._building)
                except StopIteration as done:
                    setup = done.value
                    break
                units += 1
                if not unlimited and time.perf_counter() >= deadline:
                    break
            if setup is None:
                return units if count else None
            self._building = None
            setup.build_s = time.perf_counter() - self._build_t0
            _cache_put(cache_key, setup)
            print(f"[obf] weights for C10 = {geometry.aberration_coefs.get('C10', 0.0):.1f} Å "
                  f"built in {setup.build_s:.1f} s ({setup.placement}, canvas "
                  f"{setup.inner_shape[0]}x{setup.inner_shape[1]})", flush=True)
        self._setup, self._buffer = setup, None
        self._ensure_base(geometry)
        return units if count else setup

    def _ensure_base(self, geometry) -> None:
        """The readout's Butterworth / Nyquist filter, the same for every geometry of a run."""
        if self._base is not None:
            return
        cfg = self.acc.readout_config
        u = int(geometry.upsampling_factor)
        base, _ = fourier_filters(
            geometry.upsampled_scan_gpts, geometry.upsampled_scan_sampling,
            geometry.wavelength, geometry.aberration_coefs, q_lowpass=cfg.q_lowpass,
            q_highpass=cfg.q_highpass, butterworth_order=cfg.butterworth_order,
            suppress_nyquist=(u > 1) if cfg.suppress_nyquist is None else cfg.suppress_nyquist)
        if base is not None:
            self._base = torch.as_tensor(base).float().to(self.device)

    def _to_device(self, index: np.ndarray) -> torch.Tensor:
        """Host indices on the device, without waiting for the stream (CUDA: from pinned memory)."""
        t = torch.from_numpy(np.ascontiguousarray(index, dtype=np.int64))
        if self.device.type == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def _reset(self) -> None:
        self._buffer = self._shadow = self._mean = None
        self._seen[:] = False
        self._mean_count, self._exact, self._changed = 0, False, False
        self.coverage = 0.0

    # ── coverage ──

    def _restart(self, setup: ObfSetup) -> bool:
        """An empty numerator, the means of the frames held, and the coverage starting at the
        scan line. False while nothing is held."""
        acc = self.acc
        store = acc._store
        held = np.flatnonzero(self._held)
        if store is None or not held.size:
            return False
        cursor = (self._max_id + 1) % self.n if self._max_id >= 0 else 0
        self._dirty[:] = False                       # from now on: the positions written since
        kind = acc._store_kind
        self._mean = self._store_mean(held, kind, setup)
        if self._shadow is None or self._shadow.shape[1] != store.shape[1] or \
                self._shadow.dtype != store.dtype:
            self._shadow = torch.zeros((self.n, store.shape[1]), dtype=store.dtype,
                                       device=self.device)
        self._buffer = setup.buffer()
        self._seen[:] = False
        self._lo = self._hi = cursor
        self._kind = kind
        self._mean_count, self._mean_time = int(held.size), time.monotonic()
        self._exact, self._changed, self.coverage = False, False, 0.0
        self.image_np = self.done_np = None
        self.state = "filling"
        return True

    def _store_mean(self, held: np.ndarray, kind: str, setup: ObfSetup) -> torch.Tensor:
        """(M,) each bright-field pixel's mean over the positions `held` of the store."""
        total = None
        for pos in np.array_split(held, max(1, -(-held.size // max(setup._batch, 4096)))):
            part = _raw_float(self.acc._store[self._to_device(pos)], kind).sum(0)
            total = part if total is None else total + part
        return total / max(int(held.size), 1)

    def _gather(self, setup: ObfSetup, limit: int):
        """Up to `limit` positions to deposit: those to cover at the scan line (written since the
        coverage started, from `hi` on), covered positions written again, and the frames held
        behind the coverage (before `lo`, newest first); their store rows read and their marks
        cleared. None when there is nothing to do."""
        acc = self.acc
        store = acc._store
        if store is None:
            return None
        marked, unseen = self._dirty, ~self._seen
        part = max(1, -(-int(limit) // 3))           # each kind a fair share, the rest to others
        front = (self._hi + np.arange(min(part, self.n))) % self.n
        front = front[:_prefix(marked[front] & unseen[front])]
        again = np.flatnonzero(marked & self._seen)
        if again.size:                               # round-robin, so none starves
            start = int(np.searchsorted(again, self._cursor))
            again = np.concatenate((again[start:], again[:start]))
            again = again[:max(part, int(limit) - front.size - part)]
        room = int(limit) - front.size - again.size
        back = (self._lo - 1 - np.arange(max(0, min(room, self.n)))) % self.n
        k_back = _prefix(self._held[back] & unseen[back])
        back = back[:k_back]
        back = back[~np.isin(back, front)]          # the last gap, reached from both sides
        fresh = np.concatenate((front, back))
        if not (fresh.size or again.size):
            return None
        chosen = np.concatenate((fresh, again))
        index = self._to_device(chosen)
        rows = store[index]
        marked[chosen] = False
        self._hi = int((self._hi + front.size) % self.n)
        self._lo = int((self._lo - k_back) % self.n)
        if again.size:
            self._cursor = int((again[-1] + 1) % self.n)
        return fresh, again, index, rows

    def _fold(self, setup: ObfSetup, fresh: np.ndarray, again: np.ndarray, index: torch.Tensor,
              rows: torch.Tensor) -> int:
        """Deposit newly covered frames whole, and covered ones written again as new − old."""
        sfm, k = self.acc.subtract_frame_mean, fresh.size
        values = normalize(_raw_float(rows, self._kind), self._mean, sfm)
        if again.size:
            old = self._shadow[index[k:]]
            values[k:] -= normalize(_raw_float(old, self._kind), self._mean, sfm)
            del old
        self._shadow[index] = rows
        setup.deposit(self._buffer, values, index)
        del values, rows
        self._seen[fresh] = True
        self._exact, self._changed = False, True
        return int(index.numel())

    def _done_rows(self) -> np.ndarray:
        """(Ny·U,) the output rows that hold only covered frames: every scan row within the
        probe's radius (the largest parallax shift) and a margin is covered."""
        ny, nx = self.scan_shape
        geometry = self._geometry
        u = int(geometry.upsampling_factor)
        shifts = np.asarray(geometry.shifts, dtype=np.float64)
        reach = int(np.ceil(np.hypot(shifts[:, 0], shifts[:, 1]).max() / u)) + _REACH_MARGIN
        missing = np.concatenate(([0], np.cumsum(~self._seen.reshape(ny, nx).all(axis=1))))
        rows = np.arange(ny)
        lo, hi = np.clip(rows - reach, 0, ny), np.clip(rows + reach + 1, 0, ny)
        near = missing[hi] - missing[lo] > 0         # a scan row within reach is not covered
        return ~near[np.arange(ny * u) // u]

    def _renew_due(self) -> bool:
        """New means: while few are held, each time they double; else when they drifted."""
        if self._mean is None:
            return False
        if self._mean_count < MIN_MEAN_POSITIONS:
            return int(self._held.sum()) >= 2 * self._mean_count
        if time.monotonic() - self._mean_time < self.renorm_every:
            return False
        self._mean_time = time.monotonic()
        mean = self._shadow_mean()
        ratio = mean / torch.where(self._mean > 0, self._mean, torch.ones_like(self._mean))
        # a common factor (the beam current) drops out of OBF for an even χ; the rest counts
        drift = float((ratio / ratio.mean() - 1).square().mean().sqrt())
        return drift > self.renorm_drift

    def _shadow_mean(self) -> torch.Tensor:
        """(M,) each bright-field pixel's mean over the positions covered."""
        held = np.flatnonzero(self._seen)
        total = None
        for pos in np.array_split(held, max(1, -(-held.size // 4096))):
            part = _raw_float(self._shadow[self._to_device(pos)], self._kind).sum(0)
            total = part if total is None else total + part
        return total / max(int(held.size), 1)

    def _resweep(self, setup: ObfSetup) -> None:
        """A fresh numerator from the shadow, with the exact means of the positions covered."""
        t0 = time.perf_counter()
        mean = self._shadow_mean()
        held = np.flatnonzero(self._seen)
        buffer = setup.buffer()
        sfm = self.acc.subtract_frame_mean
        for pos in np.array_split(held, max(1, -(-held.size // max(setup._batch, 2048)))):
            index = self._to_device(pos)
            values = normalize(_raw_float(self._shadow[index], self._kind), mean, sfm)
            setup.deposit(buffer, values, index)
            del values
        self._mean, self._mean_count, self._mean_time = mean, int(held.size), time.monotonic()
        self._buffer, self._exact, self._changed = buffer, True, True
        self.sweep_s = time.perf_counter() - t0

    def _publish(self, setup: ObfSetup, force: bool = False) -> bool:
        """A new image, when the numerator changed since the last one, at most every
        `live_every` s and so that the readouts take at most a fifth of OBF's time; or now
        (`force`). True when published."""
        if self._buffer is None or not self._seen.any():
            return False
        interval = max(self.live_every, 4 * self._image_s)
        if not force and (not self._changed or time.monotonic() - self._last_publish < interval):
            return False
        t0 = time.perf_counter()
        image = setup.image(self._buffer, self._base).float().cpu().numpy()
        self._image_s = time.perf_counter() - t0
        self._held_count = int(self._held.sum())
        covered = int((self._held & self._seen).sum())
        self.coverage = covered / max(self._held_count, 1)
        complete = covered == self._held_count
        self.image_np, self.done_np = image, None if complete else self._done_rows()
        self._image_time = self._last_publish = time.monotonic()
        self._changed = False
        self.state = "live" if complete else "filling"
        return True
