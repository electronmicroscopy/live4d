"""Live 4D-STEM scans from a DECTRIS detector, ANS-encoded on the GPU as they arrive.

live4d's receiver (`graph.ReceiverSession`) hands each batch to `SeriesFeed`, which queues it on a
`BatchFeed`; `recording.record` encodes it with quantem.gpu's `LiveEncoder` (flagged pixels
corrected, lossless ANS on the GPU). Each detector series is one scan and becomes one
`Dataset4dstemGPU`, the same object `quantem.gpu.io.load` returns for a file.

    live = LiveFrames(remote="10.0.0.2:31003", local="10.0.0.1", folder="/data/live")
    live.start()              # connects to the detector; nothing is armed or triggered
    viewer = live.viewer()    # Show4DSTEM.stream: patterns appear as they arrive
    live.datasets             # completed scans, the latest `keep` on the GPU
    live.stop()

Each completed scan is saved (``<folder>/<date>/series_297_512x512_96px.qem``, the exact ANS bytes)
before it can be released, and `LiveScans` loads saved scans by name in any other process.
"""

from __future__ import annotations

import fcntl
import math
import os
import queue
import signal
import socket
import threading
import time
from pathlib import Path

import cupy as cp
import numpy as np
import torch

from live4d.live.feed import BatchFeed
from live4d.live.graph import ReceiverSession
from live4d.live.recording import record
from live4d.live.scans import DEFAULT_FOLDER


class ReceiverBusy(RuntimeError):
    """Another receiver holds the detector stream."""


def scan_shape_from_start(start: dict, configured):
    """The (rows, cols) raster of one series, or None when its images cannot be placed.

    The START ``header_appendix`` ``scan_height`` / ``scan_width`` (as live4d reads them) come
    first; without them, a square raster of ``number_of_images``; without a count, `configured`.
    """
    count = int(start.get("number_of_images") or 0)
    appendix = start.get("header_appendix")
    if isinstance(appendix, dict) and ("scan_width" in appendix or "scan_height" in appendix):
        width, height = appendix.get("scan_width"), appendix.get("scan_height")
        if type(width) is not int or type(height) is not int or width < 1 or height < 1:
            return None
        if count and count != width * height:
            return None
        return height, width
    if not count:
        return tuple(int(side) for side in configured)
    side = math.isqrt(count)
    return (side, side) if side * side == count else None


class SeriesFeed:
    """The graph's consumer: queue each series' batches on a `BatchFeed`, one detector series per scan.

    `LiveEncoder` needs consecutive batches. A series whose images cannot be placed (gap, repeat,
    unknown raster) is dropped and listed in `dropped`; the receiver keeps running for the next one.
    """

    def __init__(self, feed: BatchFeed, scan_shape, pixel_mask=None, log=print):
        self.feed = feed
        self.scan_shape = tuple(int(side) for side in scan_shape)
        self.detector_shape = None
        # one mask, or {side: mask} for each binning (96: 2x2, 192: unbinned)
        self.pixel_mask = pixel_mask
        self.log = log
        self.name = None
        self.active = False      # the current series has reached the feed
        self.skip = False        # the current series was dropped
        self.next_id = 0
        self.series = 0          # series accepted
        self.frames = 0          # frames queued, all series
        self.dropped = []        # [{"series", "reason"}]

    def on_start(self, start: dict, series_id: int) -> None:
        if self.active:          # a START without the previous END: that series ends short
            self.feed.end_series()
        self.active, self.skip, self.next_id = False, False, 0
        self.name = f"series_{start.get('series_id', series_id)}"
        rows, cols = start.get("image_size_y"), start.get("image_size_x")
        if rows and cols:
            self.detector_shape = (int(rows), int(cols))
        scan = scan_shape_from_start(start, self.scan_shape)
        if scan is None:
            self._drop(f"{start.get('number_of_images')} images and header_appendix "
                       f"{start.get('header_appendix')!r} give no raster scan")
            return
        self.scan_shape = scan
        self.series += 1

    def _mask(self):
        mask = self.pixel_mask.get(self.detector_shape[0]) if isinstance(self.pixel_mask, dict) else self.pixel_mask
        return mask if mask is not None and tuple(mask.shape) == self.detector_shape else None

    def process(self, frames, ids) -> None:
        if self.skip or self.name is None:
            return
        n = int(frames.shape[0])
        first, last = int(ids[0]), int(ids[-1])
        if first != self.next_id or last - first + 1 != n:
            self._drop(f"images {first}..{last} ({n}) do not follow image {self.next_id - 1}")
            return
        if last >= self.scan_shape[0] * self.scan_shape[1]:
            self._drop(f"image {last} is beyond the {self.scan_shape[0]}x{self.scan_shape[1]} scan")
            return
        if not self.active:
            self.detector_shape = tuple(int(side) for side in frames.shape[1:])   # what arrived
            self.feed.start_series({"series_id": self.name, "image_size_y": self.detector_shape[0],
                                    "image_size_x": self.detector_shape[1],
                                    "scan_shape": self.scan_shape, "pixel_mask": self._mask()})
            self.active = True
        torch.cuda.current_stream().synchronize()    # the frames are written before CuPy copies them
        self.feed.put(first, cp.from_dlpack(frames.detach()))
        self.next_id = last + 1
        self.frames += n

    def end(self, series_images: int) -> None:
        if self.active:
            self.feed.end_series()
        self.active = False

    def _drop(self, reason: str) -> None:
        if self.active:          # LiveEncoder reports the short series incomplete and frees it
            self.feed.end_series()
        self.active, self.skip = False, True
        self.dropped.append({"series": self.name, "reason": reason})
        self.log(f"[live] {self.name} dropped: {reason}")


def _acquisition_id(dataset) -> str:
    return str(dataset.metadata.get("acquisition_id") or id(dataset))


class LiveFrames:
    """The detector's stream as a `Recording` of ANS scans, one detector series after another.

    `keep` completed scans stay on the GPU (0: all); when a scan of another size or binning
    starts, scans of the old size are released. A scan is released only after it is saved, while
    `autosave` names a format ("qem", "h5"); with ``autosave=()`` unsaved scans are released too.
    """

    def __init__(self, scan=(512, 512), *, remote="", local="", rdma_device="", transport="rdma",
                 endpoint="", pixel_mask=None, keep: int = 10, folder=DEFAULT_FOLDER, autosave=("qem",),
                 batch_size=1000, ring_mib=256, device="cuda:0", metadata=None, lock_path=None, who=None):
        self.keep = int(keep)
        self.folder = Path(folder).expanduser()
        self.autosave = (autosave,) if isinstance(autosave, str) else tuple(autosave or ())
        self.device = int(str(device).split(":")[-1])
        self.who = who
        self.lock_path = Path(lock_path or os.environ.get("LIVE4D_LOCK")
                              or Path.home() / ".cache" / "live4d" / "receiver.lock")
        self.feed = BatchFeed(max_batches=16)
        self.series = SeriesFeed(self.feed, scan, pixel_mask)
        self.session = ReceiverSession(self.series, transport=transport, remote=remote, local=local,
                                       rdma_device=rdma_device, endpoint=endpoint,
                                       batch_size=batch_size, ring_mib=ring_mib)
        self.recording = record(self.feed, scan, device=device, on_acquisition=self._completed,
                                on_arrival=lambda recording, acquisition: self._prune(),
                                metadata=metadata, verbose=False)
        self._lock = None
        self._stop = threading.Event()
        self._saved = {}          # (acquisition id, folder) -> written paths
        self._pinned = {}         # acquisition id -> saves in progress; never released meanwhile
        self._pin_lock = threading.Lock()
        self._saves = queue.Queue()
        self.save_error = None
        threading.Thread(target=self._save_worker, name="live4d-save", daemon=True).start()

    @property
    def receiving(self) -> bool:
        return self.session.running

    @property
    def datasets(self):
        return self.recording.datasets

    @property
    def error(self):
        return self.session.failure or self.recording.error

    # --- one receiver per detector stream --------------------------------------------------------

    def viewer_name(self) -> str:
        """Who this receiver is, for the others' notice: `who`, the page's ?who=... (Voila passes the
        URL query as QUERY_STRING), LIVE_WHO, or this host."""
        from urllib.parse import parse_qs
        who = self.who or parse_qs(os.environ.get("QUERY_STRING", "")).get("who", [""])[0]
        return (who or os.environ.get("LIVE_WHO") or socket.gethostname()).replace("\n", " ")[:60]

    def holder(self):
        """The receiver holding the detector stream now: {"pid", "since", "who"}, or None."""
        try:
            with open(self.lock_path) as probe:            # a free lock means nobody is receiving
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(probe, fcntl.LOCK_UN)
                    return None
                except BlockingIOError:
                    pass
            pid, since, *who = open(self.lock_path).read().split(" ", 2)
            os.kill(int(pid), 0)
            return {"pid": int(pid), "since": time.strftime("%H:%M:%S", time.localtime(float(since))),
                    "who": (who[0].strip() if who else "") or "a receiver"}
        except (OSError, ValueError):
            return None

    def take_over(self, timeout_s: float = 15.0) -> None:
        """Stop the other receiver holding the detector stream (it hangs up as its process ends)."""
        other = self.holder()
        if other is None or other["pid"] == os.getpid():
            return
        os.kill(other["pid"], signal.SIGTERM)
        deadline = time.monotonic() + timeout_s
        while self.holder() is not None and time.monotonic() < deadline:
            time.sleep(0.2)
        if self.holder() is not None:
            os.kill(other["pid"], signal.SIGKILL)
            time.sleep(1)

    def start(self, wait_s: float = 5.0) -> None:
        """Connect to the detector and start the graph; arm the detector only after this returns.

        One receiver per detector stream: if another holds it, raise `ReceiverBusy` (use `page()`,
        which shows who and offers Take over, or `take_over()` first).
        """
        if self.receiving:
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.lock_path, "a+")
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    self._lock.close()
                    self._lock = None
                    other = self.holder() or {"since": "?", "who": "a receiver"}
                    raise ReceiverBusy(f"Another receiver holds the detector stream: {other['who']}, since "
                                       f"{other['since']}.") from None
                time.sleep(0.5)
        self._lock.seek(0)
        self._lock.truncate()
        self._lock.write(f"{os.getpid()} {time.time()} {self.viewer_name()}")
        self._lock.flush()
        self.session.open()
        where = self.session.remote if self.session.transport == "rdma" else self.session.endpoint
        print(f"[live] listening: {self.session.transport} {where}, ANS on cuda:{self.device}", flush=True)

    def stop(self, save_timeout_s: float = 300.0) -> None:
        """Stop listening and hang up; finish pending saves; completed scans stay in `datasets`."""
        self._stop.set()
        self.session.close()
        self.feed.close()
        self.recording.close()
        self.recording.wait(timeout=30)
        self._saves.put(None)
        deadline = time.monotonic() + save_timeout_s
        while self._pinned and time.monotonic() < deadline:
            time.sleep(0.1)
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    # --- saving and retention: a scan lives on the GPU until it is saved --------------------------

    def _pin(self, datasets) -> None:
        with self._pin_lock:
            for dataset in datasets:
                key = _acquisition_id(dataset)
                self._pinned[key] = self._pinned.get(key, 0) + 1

    def _unpin(self, datasets) -> None:
        with self._pin_lock:
            for dataset in datasets:
                key = _acquisition_id(dataset)
                self._pinned[key] -= 1
                if not self._pinned[key]:
                    del self._pinned[key]

    def _completed(self, dataset) -> None:
        """Each completed scan (on the encoding thread): pin and queue its save, then apply retention."""
        if self.autosave:
            self._pin([dataset])
            self._saves.put(dataset)
        self._prune()

    def _save_worker(self) -> None:
        while True:
            dataset = self._saves.get()
            if dataset is None:
                return
            try:
                self._write(dataset, self.folder / time.strftime("%Y-%m-%d"), self.autosave)
            except Exception as exc:   # shown in the status; the scan stays pinned until stop
                self.save_error = exc
                print(f"[live] save of {dataset.metadata.get('series')} failed: {exc}", flush=True)
                continue
            self._unpin([dataset])
            self._prune()

    def _prune(self) -> None:
        """Release scans beyond `keep` and scans of an older size, except those still being saved."""
        with self.recording.lock:
            datasets = list(self.recording.datasets)
            acquisitions = self.recording.acquisitions
            newest = tuple(acquisitions[-1].shape) if acquisitions else None
            old = datasets[:-self.keep] if self.keep else []
            old += [item for item in datasets if newest and tuple(item.shape) != newest
                    and not any(item is other for other in old)]
            with self._pin_lock:
                pinned = set(self._pinned)
            for dataset in old:
                if _acquisition_id(dataset) not in pinned:
                    self.recording.discard(dataset)

    def _write(self, dataset, folder: Path, formats) -> list:
        """Write one scan in each format; a scan already written to `folder` is skipped."""
        from quantem.gpu import io

        key = (_acquisition_id(dataset), str(folder))
        if key in self._saved:
            return []
        folder.mkdir(parents=True, exist_ok=True)
        rows, cols, det, _ = dataset.shape
        stem = f"{dataset.metadata.get('series') or 'scan'}_{rows}x{cols}_{det}px"
        written = []
        with cp.cuda.Device(self.device):
            for fmt in formats:
                suffix = ".qem" if fmt == "qem" else "_master.h5"
                path, again = folder / (stem + suffix), 1
                while path.exists():                        # never overwrite a saved scan
                    again += 1
                    path = folder / f"{stem}_{again}{suffix}"
                io.save(path, dataset)
                written.append(path)
        self._saved[key] = written
        return written

    def save(self, folder=None, formats=("qem",), which="latest") -> list:
        """Write completed scans now: "qem" keeps the ANS bytes as they are (fast, small); "h5" writes
        uint16 HDF5 with bitshuffle/LZ4 (master + data files). Both are lossless. `which` is
        "latest" or "all" (the kept scans). Scans already written to that folder are skipped."""
        formats = (formats,) if isinstance(formats, str) else tuple(formats)
        if not formats or set(formats) - {"qem", "h5"}:
            raise ValueError("formats: 'qem', 'h5' or both")
        folder = Path(folder).expanduser() if folder else self.folder / time.strftime("%Y-%m-%d")
        with self.recording.lock:
            datasets = list(self.recording.datasets)
            datasets = datasets[-1:] if which == "latest" else datasets
            self._pin(datasets)
        try:
            return [path for dataset in datasets for path in self._write(dataset, folder, formats)]
        finally:
            self._unpin(datasets)
            self._prune()

    # --- the page: status, save, Show4DSTEM ------------------------------------------------------

    def viewer(self, **kwargs):
        """Show4DSTEM following the recording: arriving patterns, one entry per scan."""
        import quantem.widget as qw

        return qw.Show4DSTEM.stream(self.recording, **kwargs)

    def save_panel(self):
        """Folder, formats, Save latest / Save all kept, and auto-save of each completed scan."""
        import ipywidgets as widgets

        folder_box = widgets.Text(value=str(self.folder / time.strftime("%Y-%m-%d")), description="Folder",
                                  layout=widgets.Layout(width="420px"))
        qem = widgets.Checkbox(value="qem" in (self.autosave or ("qem",)), description="QEM (ANS)", indent=False)
        h5 = widgets.Checkbox(value="h5" in self.autosave, description="HDF5 (bitshuffle/LZ4)", indent=False)
        auto = widgets.Checkbox(value=bool(self.autosave), description="Auto-save each scan", indent=False)
        latest = widgets.Button(description="Save latest")
        everything = widgets.Button(description="Save all kept")
        result = widgets.HTML()

        def formats():
            return tuple(name for name, box in (("qem", qem), ("h5", h5)) if box.value)

        def follow(_=None):
            self.autosave = formats() if auto.value else ()

        for box in (qem, h5, auto):
            box.observe(follow, "value")

        def run(which):
            def work():
                result.value = "saving..."
                try:
                    paths = self.save(folder_box.value, formats(), which)
                    result.value = (f"saved {len(paths)} file(s): " + ", ".join(p.name for p in paths)
                                    if paths else "nothing new to save")
                except Exception as exc:
                    result.value = f"<b style='color:#c00'>save failed: {exc}</b>"
            threading.Thread(target=work, name="live4d-save-now", daemon=True).start()

        latest.on_click(lambda _: run("latest"))
        everything.on_click(lambda _: run("all"))
        return widgets.VBox([widgets.HBox([folder_box, qem, h5]),
                             widgets.HBox([latest, everything, auto]), result])

    def page(self, **viewer_kwargs):
        """The whole live page: status, Save and Show4DSTEM. If another receiver holds the detector
        stream, it says so (who, since when) with a Take over button instead of failing."""
        import ipywidgets as widgets

        box = widgets.VBox()

        def show_live():
            box.children = [self.status(), self.save_panel(), self.viewer_box(**viewer_kwargs)]

        try:
            self.start()
            show_live()
        except ReceiverBusy as busy:
            notice = widgets.HTML(f"<div style='padding:8px;border-left:4px solid #d98a00;background:#fff4e0;color:#222'>"
                                  f"<b>Not receiving.</b> {busy} Only one receiver can hold the stream. "
                                  f"<b>Take over</b> stops that receiver and starts receiving here: tell them first "
                                  f"(the operator may be scanning).</div>")
            button = widgets.Button(description="Take over", button_style="warning")
            result = widgets.HTML()

            def take(_):
                button.disabled = True
                result.value = "stopping the other receiver..."
                try:
                    self.take_over()
                    self.start()
                    show_live()
                except Exception as exc:
                    result.value = f"<b style='color:#c00'>{exc}</b>"
                    button.disabled = False

            button.on_click(take)
            box.children = [notice, widgets.HBox([button, result])]
        return box

    def viewer_box(self, refresh_s: float = 0.5, **kwargs):
        """Show4DSTEM that survives binning changes: a new detector size gets a fresh viewer
        (its virtual detectors are in detector pixels). Scan-size changes need no new viewer."""
        import ipywidgets as widgets

        box = widgets.VBox([self.viewer(**kwargs)])
        shown = {"detector": None}

        def follow():
            while not self._stop.is_set():
                with self.recording.lock:
                    acquisitions = list(self.recording.acquisitions)
                detector = tuple(acquisitions[-1].detector_shape) if acquisitions else None
                if detector is not None and shown["detector"] is None:
                    shown["detector"] = detector
                elif detector is not None and detector != shown["detector"]:
                    old = box.children[0]
                    box.children = [self.viewer(**kwargs)]
                    shown["detector"] = detector
                    try:
                        old.close()
                    except Exception:
                        pass
                time.sleep(refresh_s)

        threading.Thread(target=follow, name="live4d-viewer", daemon=True).start()
        return box

    def status_text(self) -> str:
        error = self.error
        if error is not None:
            return f"<b style='color:#c00'>Stopped: {error}</b>"
        rows, cols = self.series.scan_shape
        state = "listening" if self.receiving else "not listening"
        with self.recording.lock:
            current = self.recording.acquisitions[-1].received if self.recording.acquisitions else 0
            resident = len(self.recording.datasets)
        det = self.series.detector_shape or ("?", "?")
        where = self.session.remote if self.session.transport == "rdma" else self.session.endpoint
        dropped = f" &nbsp; dropped {len(self.series.dropped)}" if self.series.dropped else ""
        saving = (f" &nbsp; <b style='color:#c00'>save failed: {self.save_error}</b>"
                  if self.save_error is not None else "")
        return (f"<b>{state}</b> &nbsp; {self.session.transport.upper()} {where} &nbsp; scan {rows}x{cols}, "
                f"detector {det[0]}x{det[1]} &nbsp; scans {self.series.series} (kept {resident}) &nbsp; "
                f"current {100 * current / (rows * cols):.0f}% encoded{dropped}{saving}")

    def status(self, refresh_s: float = 1.0):
        """A one-line live status (listening, scans, encoded patterns, errors) that updates itself."""
        import ipywidgets as widgets

        label = widgets.HTML(self.status_text())

        def follow():
            while not self._stop.is_set():
                label.value = self.status_text()
                time.sleep(refresh_s)
            label.value = self.status_text()

        threading.Thread(target=follow, name="live4d-status", daemon=True).start()
        return label
