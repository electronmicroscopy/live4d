"""Acquisitions received from a live source, each encoded on arrival and kept on the GPU as an ANS dataset."""

import threading
import weakref

import cupy as cp

from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.streaming import IncompleteAcquisition, LiveEncoder
from live4d.live.feed import BatchFeed


def series_fields(start, index: int):
    """Series name, detector shape and pixel mask from the detector's START mapping; ``index`` names a series
    without an id."""
    shape = (int(start["image_size_y"]), int(start["image_size_x"]))
    return str(start.get("series_id", index)), shape, start.get("pixel_mask")


class Recording:
    """Acquisitions received from a live source, each kept on the GPU as a lossless ANS dataset.

    ``datasets`` grows as acquisitions complete (``io.load``-equivalent ``Dataset4dstemGPU``, named by
    ``metadata["series"]``); ``incomplete`` lists series that ended short. ``on_arrival(recording, acquisition)``, when
    set, is called from the receiving thread as each acquisition starts, before its first pattern is encoded,
    and ``on_acquisition(dataset)`` after each one completes. ``wait`` blocks until the source ends or
    ``close`` was called and raises the receiving thread's error, if any.
    """

    def __init__(self, on_acquisition=None, on_arrival=None):
        self.lock = threading.RLock()
        self.datasets, self.incomplete = [], []
        self.acquisitions = []
        self.on_acquisition, self.on_arrival = on_acquisition, on_arrival
        self.error = None
        self.stop = threading.Event()
        self.thread = None
        self._processing = set()
        # Existing viewer sessions own these counts; monitoring must not extend their lifetime.
        self._retiring = weakref.WeakValueDictionary()

    def memory_snapshot(self) -> dict:
        """Describe retained scans without decoding counts or synchronizing CUDA.

        Buffer bytes cover encoded counts, their indexes and the arriving tail,
        not allocator reserves or separate scientific sessions. Scan IDs remain
        stable when earlier scans are removed or detector series names repeat.
        ``releasing`` lists discarded scans still borrowed by live viewers;
        weak references allow them to disappear when their final viewer closes.

        Examples
        --------
        >>> recording.memory_snapshot()["scans"]
        """
        with self.lock:
            completed = {id(dataset.data) for dataset in self.datasets}
            scans = []
            for acquisition in self.acquisitions:
                with acquisition.lock:
                    released = acquisition.source.is_released
                    ready = id(acquisition.source) in completed
                    busy = acquisition.identifier in self._processing
                    state = ("Unloaded" if released else "Processing" if busy else
                             "Ready" if ready else "Receiving" if self.receiving else "Incomplete")
                    scans.append({
                        **acquisition.memory_snapshot(),
                        "state": state,
                        "can_unload": ready and not busy and not released,
                        "unload_reason": "" if ready and not busy and not released else
                            "Finish acquisition and processing before unloading this scan.",
                    })
            releasing = []
            for acquisition in list(self._retiring.values()):
                with acquisition.lock:
                    if not acquisition.source.is_released:
                        releasing.append({**acquisition.memory_snapshot(),
                                          "state": "Waiting for viewers", "can_unload": False,
                                          "unload_reason": "Close viewers still using this scan to release its buffers."})
            return {"scans": scans, "releasing": releasing, "receiving": self.receiving,
                    "error": None if self.error is None else str(self.error)}

    def unload(self, scan_id: str) -> None:
        """Discard one ready scan by its stable ID; saved files remain intact.

        Existing live queries retain their buffers until their sessions close.
        Unsaved counts are discarded. Finish scientific consumers before calling.

        Examples
        --------
        >>> scan = recording.memory_snapshot()["scans"][0]
        >>> recording.unload(scan["id"])
        """
        with self.lock:
            acquisition = next((item for item in self.acquisitions if item.identifier == scan_id), None)
            if acquisition is None:
                raise ValueError("This scan is no longer retained; refresh the scan list.")
            if scan_id in self._processing:
                raise RuntimeError("This scan is processing; wait for it to become ready before unloading.")
            dataset = next((item for item in self.datasets if item.data is acquisition.source), None)
            if dataset is None:
                raise RuntimeError("This scan is still arriving or incomplete; finish or stop acquisition first.")
            self.discard(dataset)

    @property
    def receiving(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def wait(self, timeout: float | None = None) -> "Recording":
        if self.thread is not None:
            self.thread.join(timeout)
        if self.error is not None:
            raise self.error
        return self

    def close(self) -> "Recording":
        """Stop after the acquisition in progress; received datasets stay resident."""
        self.stop.set()
        return self

    def release(self) -> None:
        """Release completed and unfinished acquisitions after receiving stops."""
        with self.lock:
            if self.receiving:
                raise RuntimeError("The recording is still receiving; call close().wait() before release().")
            for acquisition in self.acquisitions:
                acquisition.release()
            for dataset in self.datasets:
                dataset.close()
            self.datasets.clear()
            self.acquisitions.clear()

    def discard(self, dataset: Dataset4dstemGPU) -> None:
        """Remove completed data after saving and scientific consumers finish.

        Existing live detector sessions keep the counts until their last close.
        Hold ``recording.lock`` while selecting acquisitions and opening viewer
        sessions so a discarded acquisition cannot gain a new viewer. Repeated
        discards are harmless; this does not cancel an in-progress acquisition.

        Parameters
        ----------
        dataset : Dataset4dstemGPU
            A completed dataset retained by this recording.

        Examples
        --------
        >>> io.save("acquisition.qem", dataset)
        >>> recording.discard(dataset)
        """
        with self.lock:
            if not any(item is dataset for item in self.datasets):
                return
            self.datasets[:] = [item for item in self.datasets if item is not dataset]
            acquired = [item for item in self.acquisitions if item.source is dataset.data]
            self.acquisitions[:] = [item for item in self.acquisitions if item.source is not dataset.data]
            for acquisition in acquired:
                acquisition._retire()
                if not acquisition.source.is_released:
                    self._retiring[acquisition.identifier] = acquisition
            if not acquired:
                dataset.close()


def record(source, scan_shape, *, device="cuda:0", on_acquisition=None, on_arrival=None,
           metadata: dict | None = None, verbose: bool = True,
           retire_other_shapes: bool = False) -> Recording:
    """Receive acquisitions from a live source and keep each as an ANS dataset on the GPU; returns at once.

    ``source`` yields acquisitions like ``BatchFeed`` (``series()`` of objects with
    ``start`` and ``batches()``); ``scan_shape`` is configured, because a detector announces only an image
    count. Each batch is corrected for flagged pixels and ANS-encoded as it arrives (quantem.gpu ``LiveEncoder``), so
    an acquisition is complete, encoded and usable the moment its last pattern arrives; nothing else is
    computed. Each acquisition is then analysed as a dataset: ``DirectPtycho(dataset)``, or
    ``ptycho.series(datasets, fit)`` over several, MAPED and the other offline-style workflows.

    ``metadata`` supplies measured calibration and provenance for each acquisition.
    Each START message's fields are retained and take precedence over this template;
    ``series`` is always the received acquisition name. The detector pixel mask is
    stored separately and keeps its correction policy.

    ``retire_other_shapes`` releases completed scans of another scan or detector shape when a scan of
    a new shape starts (a size or binning change), so a live viewer (which compares scans of one shape) follows a size change.
    Save those scans first.
    """
    recording = Recording(on_acquisition, on_arrival)
    scan_shape = tuple(int(side) for side in scan_shape)

    def work():
        try:
            with cp.cuda.Device(int(str(device).split(":")[-1])):
                for index, acquisition in enumerate(source.series()):
                    if recording.stop.is_set():
                        break
                    name, detector_shape, pixel_mask = series_fields(acquisition.start, index)
                    acquired_metadata = dict(metadata or {})
                    acquired_metadata.update({key: value for key, value in acquisition.start.items()
                                              if key != "pixel_mask"})
                    acquired_metadata["series"] = name
                    # A receiver that reads the scan size from the detector's START passes it per series.
                    series_scan = tuple(int(side) for side in acquisition.start.get("scan_shape", scan_shape))
                    live = LiveEncoder(series_scan, detector_shape, pixel_mask=pixel_mask, device=device,
                                           metadata=acquired_metadata)
                    with recording.lock:
                        if retire_other_shapes:
                            for old in [item for item in recording.datasets
                                        if tuple(item.shape) != (*series_scan, *detector_shape)]:
                                recording.discard(old)
                        recording.acquisitions.append(live)
                    if recording.on_arrival is not None:
                        recording.on_arrival(recording, live)
                    for first, patterns in acquisition.batches():
                        live.add(first, patterns)
                    try:
                        dataset = live.finish()
                    except IncompleteAcquisition as error:
                        live.release()
                        with recording.lock:
                            recording.incomplete.append({"series": name, "reason": str(error)})
                        if verbose:
                            print(f"record: series {name} incomplete ({error}); not kept.")
                        continue
                    with recording.lock:
                        recording.datasets.append(dataset)
                        recording._processing.add(live.identifier)
                    try:
                        if recording.on_acquisition is not None:
                            recording.on_acquisition(dataset)
                    finally:
                        with recording.lock:
                            recording._processing.discard(live.identifier)
        except BaseException as error:  # noqa: BLE001 - surfaced to the caller by wait()
            recording.error = error
        finally:
            if isinstance(source, BatchFeed) and (recording.error is not None or recording.stop.is_set()):
                source._abort(recording.error or RuntimeError("The recording stopped; start another feed."))

    recording.thread = threading.Thread(target=work, name="quantem-record", daemon=True)
    recording.thread.start()
    return recording
