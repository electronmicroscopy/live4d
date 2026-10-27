"""The one owner of the encoded series on the GPU: the encoder's sink, with leases for the per-scan operators.

`processing.AnsEncoderOp` builds each series with quantem.gpu's `LiveEncoder` and hands it to the store at END.
The store keeps the finished `Dataset4dstemGPU` and is the only code that closes one. Operators downstream
(pairing, DirectPtycho, drift) pass the series name, not the buffer, and hold a lease while they read it:

    store = SeriesStore(device=2, keep=2)
    with store.lease("series_297") as dataset:   # the dataset stays open until the lease ends
        ...

A series is closed when it is older than the latest `keep` and no lease holds it. A lease that would keep
more than `max_leased` series open is refused, so the analysis never holds the GPU against the encoder.
No Holoscan imports.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from contextlib import contextmanager


class SeriesGone(KeyError):
    """The series was dropped, closed or never completed."""


class SeriesStore:
    """Encoder sink: the latest `keep` series stay resident (0: all), and any series under a lease."""

    def __init__(self, device: int = 0, keep: int = 1, pixel_mask=None, max_leased: int = 4):
        self.device, self.keep, self.pixel_mask = int(device), int(keep), pixel_mask
        self.max_leased = int(max_leased)
        self._series = OrderedDict()           # name -> Dataset4dstemGPU, oldest first
        self._leases = {}                      # name -> open leases
        self._lock = threading.Lock()
        self.dropped = []                      # [{"series", "reason"}]

    # --- the sink AnsEncoderOp calls (SeriesEncoder) -------------------------------------------------

    def begin(self, series_id, scan_shape, image_shape):
        from quantem.gpu.streaming import LiveEncoder

        mask = self.pixel_mask if self.pixel_mask is not None and tuple(self.pixel_mask.shape) == image_shape else None
        return LiveEncoder(scan_shape, image_shape, pixel_mask=mask, device=f"cuda:{self.device}",
                           metadata={"series": f"series_{series_id}"})

    def complete(self, encoder) -> dict:
        from quantem.gpu.streaming import IncompleteAcquisition

        name = encoder.metadata["series"]
        summary = dict(series=name, images=encoder.received, total=encoder.total,
                       encoded_bytes=int(encoder.source.nbytes),
                       raw_bytes=encoder.received * math.prod(encoder.detector_shape) * 2)
        try:
            dataset = encoder.finish()
        except IncompleteAcquisition as error:
            self.drop(encoder, str(error))
            return dict(summary, kept=False)
        with self._lock:
            self._series[name] = dataset
            self._retire()
        return dict(summary, kept=True)

    def drop(self, encoder, reason: str) -> None:
        with self._lock:
            self.dropped.append(dict(series=encoder.metadata.get("series"), reason=reason))
        encoder.release()

    # --- what the per-scan operators use ------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        with self._lock:
            return list(self._series)

    @property
    def datasets(self) -> list:
        with self._lock:
            return list(self._series.values())

    def acquire(self, name: str):
        """Open a lease on `name` and return its dataset; pair every acquire with one release."""
        with self._lock:
            if name not in self._series:
                raise SeriesGone(name)
            if name not in self._leases and len(self._leases) >= self.max_leased:
                raise RuntimeError(f"{len(self._leases)} series are already leased (max_leased={self.max_leased})")
            self._leases[name] = self._leases.get(name, 0) + 1
            return self._series[name]

    def release(self, name: str) -> None:
        with self._lock:
            count = self._leases.get(name, 0)
            if count <= 0:
                raise RuntimeError(f"{name} is not leased")
            if count == 1:
                del self._leases[name]
            else:
                self._leases[name] = count - 1
            self._retire()

    @contextmanager
    def lease(self, name: str):
        dataset = self.acquire(name)
        try:
            yield dataset
        finally:
            self.release(name)

    def close(self) -> None:
        """Close every series no lease holds (at the end of the run)."""
        with self._lock:
            for name in [n for n in self._series if n not in self._leases]:
                self._series.pop(name).close()

    def _retire(self) -> None:
        """Close the series beyond the latest `keep` that no lease holds (lock held)."""
        if not self.keep:
            return
        for name in list(self._series)[:-self.keep]:
            if name not in self._leases:
                self._series.pop(name).close()
