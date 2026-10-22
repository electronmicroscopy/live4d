"""Operators of the example receiver app: WorkerOp (reconstruct the images of one ring) and
AggregateOp (merge the workers' results per series). They consume what dectris.RdmaRxOp
emits (lists of Start, Batch and End) and are wired up by add_workers.
"""

import cupy as cp
import numpy as np
from holoscan.core import Operator, OperatorSpec

from dectris import IMAGE_SHAPE, Batch, End, Start


# ---------------------------------------------------------------------------------------------
# WorkerOp
#
# Reconstruction of the images of one ring, on the ring's GPU.
#
# Stand-in reconstruction: each image's integrated intensity, i.e. a virtual bright-field image.
# The sums come ready from the dispatcher (reduced straight from the ring) or are taken from the
# copied images. The series' shape is not known; the worker keeps one value per image id (a 1-D
# array, 0 for the ids it did not receive) and the aggregator puts the workers' values together.
# ---------------------------------------------------------------------------------------------

_MIN_CAPACITY = 1024


class WorkerOp(Operator):
    """Consumes the lists of ``Start``, ``Batch`` and ``End`` items of one ring, in arrival order,
    on ``device``.

    Image ``i`` of a series is value ``i`` of the array that the end of the series sends, as
    (series id, uint64 array), on ``output`` as part of a list of such pairs. A series whose end
    never arrives is dropped when the next one starts.
    """

    def __init__(self, fragment, *args, device: int, **kwargs):
        self.device = cp.cuda.Device(device)
        self.series = None
        self.values = None  # growable (capacity,) uint64; one intensity per image id
        self.size = 0  # highest image id received + 1
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def initialize(self):
        super().initialize()
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            self._warm_up()

    def _warm_up(self):
        """Compile the CuPy kernels that ``_add`` uses now: the first call of each takes up to
        seconds, during which the source is blocked and the sender drops images."""
        with self.stream:
            images = cp.ones((2, *IMAGE_SHAPE), dtype=cp.uint16)
            sums = images.sum(axis=(1, 2), dtype=cp.uint64)
            values = cp.zeros(_MIN_CAPACITY, dtype=cp.uint64)
            values[0:2] = sums
            values[cp.asarray(np.array([0, 5], dtype=np.int64))] = sums
        self.stream.synchronize()

    def _begin(self, series):
        if self.values is not None:
            print(f"worker: series {self.series.id} has no end, dropped", flush=True)
        self.series = series
        self.size = 0
        self.values = cp.zeros(max(_MIN_CAPACITY, series.expected_images or 0), dtype=cp.uint64)

    def _reserve(self, size: int):
        if size > self.values.shape[0]:
            grown = cp.zeros(max(size, 2 * self.values.shape[0]), dtype=cp.uint64)
            grown[:self.values.shape[0]] = self.values
            self.values = grown

    def _add(self, batch: Batch):
        if self.series is None or batch.series.id != self.series.id:
            self._begin(batch.series)
        self.stream.wait_event(batch.ready)  # the data is produced on another stream
        if batch.sums is not None:  # reduced straight from the ring
            sums = batch.sums
        else:
            sums = cp.asarray(batch.images).sum(axis=(1, 2), dtype=cp.uint64)
        ids = np.asarray(batch.ids, dtype=np.int64)
        first, count = int(ids[0]), len(ids)
        last = int(ids.max()) + 1
        self._reserve(last)
        if int(ids[-1]) - first == count - 1 and (count < 2 or (np.diff(ids) == 1).all()):
            # Consecutive image ids: one slice, no index array.
            self.values[first:first + count] = sums
        else:  # images were lost in between
            self.values[cp.asarray(ids)] = sums
        self.size = max(self.size, last)

    def _finish(self, end: End):
        if self.series is None or end.series.id != self.series.id:
            self._begin(end.series)  # no image of this series came to this ring
        size = max(self.size, end.images)  # lost images keep their place: they stay 0
        self._reserve(size)
        result = (end.series.id, self.values[:size])
        self.series = self.values = None
        return result

    def compute(self, op_input, op_output, context):
        completed = []
        with self.device, self.stream:
            for item in op_input.receive("input"):
                if isinstance(item, Start):
                    self._begin(item.series)
                elif isinstance(item, End):
                    completed.append(self._finish(item))
                else:
                    self._add(item)
            if completed:
                self.stream.synchronize()  # the values are read on the aggregator's stream
                op_output.emit(completed, "output")


# ---------------------------------------------------------------------------------------------
# AggregateOp
#
# Gathers the per-GPU worker values and aggregates each series' image.
# ---------------------------------------------------------------------------------------------

# Series this far behind the newest aggregated series are abandoned (a worker lost frames).
_MAX_LAG = 4


class AggregateOp(Operator):
    """Inputs ``input0..``: lists of (series id, values) from each WorkerOp, ``values`` having one
    value per image id (0 for the ids that worker did not receive).

    The values are copied to ``device``. Once all workers' values of a series are available they
    are merged by image id into one array with a value per image, inside the operator's device and
    stream context: every image went through exactly one ring, so the workers' non-zero values are
    at different ids and the merge is their sum, padded to the longest. Which GPU handled which
    image does not matter.

    The series' shape is unknown. ``series_width`` (optional) is applied only if it divides the
    number of images the series turned out to have; otherwise the image is a single row.
    """

    def __init__(self, fragment, *args, n_workers: int, device: int,
                 series_width: int | None = None, **kwargs):
        self.n_workers = n_workers
        self.device = cp.cuda.Device(device)
        self.series_width = series_width
        self.pending = {}  # series id -> {worker: values on self.device}
        self.latest = None  # id of the newest aggregated series
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        for t in range(self.n_workers):
            spec.input(f"input{t}")

    def initialize(self):
        super().initialize()
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)

    def compute(self, op_input, op_output, context):
        with self.device, self.stream:
            for t in range(self.n_workers):
                for series, img in op_input.receive(f"input{t}"):
                    local = cp.empty(img.shape, dtype=img.dtype)
                    if img.nbytes:
                        local.data.copy_from_device_async(img.data, img.nbytes, self.stream)
                    self.pending.setdefault(series, {})[t] = local
            # The sources go back to their device's memory pool once dropped.
            self.stream.synchronize()

            for series in sorted(self.pending):
                inputs = self.pending[series]
                if len(inputs) < self.n_workers:
                    continue
                del self.pending[series]
                self._aggregate(series, inputs)
                self.latest = series
            if self.latest is not None:
                for series in [s for s in self.pending if s < self.latest - _MAX_LAG]:
                    del self.pending[series]

    def _aggregate(self, series, inputs):
        total = max(int(values.shape[0]) for values in inputs.values())
        merged = cp.zeros(total, dtype=cp.uint64)
        for t in sorted(inputs):
            merged[:inputs[t].shape[0]] += inputs[t]
        width = self.series_width
        note = ""
        if not width or width <= 0 or total % width:
            if width:
                note = f" (series_width {width} does not divide {total}; one row)"
            width = max(total, 1)
        image = merged.reshape(-1, width) if total else merged.reshape(1, 0)
        print(f"aggregate series: {series} {total} images -> {image.shape[1]}x{image.shape[0]}"
              f"{note} sum={int(merged.sum())}", flush=True)
        return image


def add_workers(app, source, series_width=None):
    """One WorkerOp per ring of ``source`` (an RdmaRxOp) on the ring's GPU, all into one
    AggregateOp."""
    paths = source.paths
    aggregate = AggregateOp(app, name="aggregate", n_workers=len(paths),
                            device=paths[0].device, series_width=series_width)
    for path in paths:
        worker = WorkerOp(app, name=f"worker{path.index}", device=path.device)
        app.add_flow(source, worker, {(f"output{path.index}", "input")})
        app.add_flow(worker, aggregate, {("output", f"input{path.index}")})
