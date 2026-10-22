"""Place decoded GPU images in a raster scan and integrate virtual detectors."""
from dataclasses import dataclass
import json
import time

import cbor2
import cupy as cp
import holoscan as hs
import numpy as np

from dectris.messages import IMAGE_SHAPE, ScanShape, parse_start, scan_positions
from .schemas.decoded_batch import DecodedBatch


@dataclass
class PositionedBatch:
    batch_id: int
    positions: np.ndarray
    selected: np.ndarray
    images: object = None
    result: dict | None = None


class ScanTensorOp(hs.Operator):
    """Publish raster positions keyed to the original typed image batch."""
    inp = hs.Input("input", DecodedBatch)
    out = hs.Output("output", hs.pyobject)
    scan_width = hs.Param(1000)
    scan_height = hs.Param(1000)

    def setup(self, spec):
        spec.input(self.inp)
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def start(self):
        self.scan = ScanShape(self.scan_width, self.scan_height)
        self.series = None

    def compute(self, ctx):
        sample = self.inp.receive()
        batch = sample.data
        if batch.start_of_series:
            series = parse_start(bytes(batch.start_metadata))
            if self.series is not None or series.id != batch.series_id:
                raise ValueError("Invalid ordered series boundary")
            self.series = series
        elif not batch.pipeline_ready and (self.series is None or self.series.id != batch.series_id):
            raise ValueError("Image batch has no matching START")
        ids = np.asarray(batch.image_ids, dtype=np.uint64)
        positions, valid = scan_positions(ids, self.scan)
        if batch.end_of_series:
            self.series = None
        positioned = PositionedBatch(int(batch.batch_id), positions, np.flatnonzero(valid))
        options = hs.EmitOptions(capture_time=sample.metadata.capture_time)
        while not ctx.stop_requested:
            try:
                self.out.emit(positioned, options=options)
                return
            except hs.errors.BackpressuredError:
                time.sleep(0.0001)


class VirtualDetectorOp(hs.Operator):
    """Integrate float32 BF/ADF values while retaining each source tensor until CUDA finishes."""
    inp = hs.Input("input", DecodedBatch)
    positions = hs.Input("positions", hs.pyobject)
    device_id = hs.Param(0)
    scan_width = hs.Param(1000)
    scan_height = hs.Param(1000)
    batch_size = hs.Param(512)
    bf_radius = hs.Param(16.0)
    adf_inner = hs.Param(24.0)
    adf_outer = hs.Param(48.0)

    def __init__(self, *, ready=None, finite=True, observer=None, release=None):
        super().__init__()
        self.ready = ready
        self.finite = finite
        self.observer = observer
        self.release = release
        self.last = None

    def setup(self, spec):
        spec.input(self.inp)
        spec.input(self.positions)

    def contract(self):
        return (hs.Contract().trigger(hs.OnAligned(ports=[self.inp, self.positions],
                                                  window=hs.time.ns(0)))
                .input(self.inp, hs.Exact()).input(self.positions, hs.Exact()))

    def start(self):
        self.device = cp.cuda.Device(self.device_id)
        self.scan = ScanShape(self.scan_width, self.scan_height)
        self.series = self.image = None
        with self.device:
            y, x = cp.mgrid[:96, :96].astype(cp.float32)
            radius = cp.hypot(y - 47.5, x - 47.5)
            self.masks = cp.stack([radius <= self.bf_radius,
                                  (radius >= self.adf_inner) & (radius <= self.adf_outer)]
                                 ).reshape(2, -1).T.astype(cp.float32)
            cp.cuda.get_current_stream().synchronize()

    def _integrate(self, frames):
        return frames.reshape(frames.shape[0], -1).astype(cp.float32) @ self.masks

    def _warm_up(self):
        # Run in the READY callback's thread and runtime CUDA stream, as image batches do.
        scratch = cp.ones((self.batch_size, *IMAGE_SHAPE), dtype=cp.uint16)
        values = self._integrate(scratch)
        image = cp.zeros((2, self.scan.height, self.scan.width), dtype=cp.float32)
        ids = np.arange(min(self.batch_size, self.scan.size), dtype=np.int64)
        positions = cp.asarray(np.stack(np.divmod(ids, self.scan.width), axis=1).astype(np.int32))
        image[:, positions[:, 0], positions[:, 1]] = values[:len(ids)].T
        image.sum(axis=(1, 2), dtype=cp.float64).get()

    def compute(self, ctx):
        original = self.inp.receive()
        position_sample = self.positions.receive()
        batch = original.data
        positioned = position_sample.data
        if positioned.batch_id != batch.batch_id:
            raise ValueError(f"Scan positions do not match the image batch: "
                             f"images={batch.batch_id}@{original.metadata.capture_timestamp_ns}, "
                             f"positions={positioned.batch_id}@{position_sample.metadata.capture_timestamp_ns}")
        runtime_stream = ctx.stream()
        with self.device, runtime_stream, cp.cuda.Stream.from_external(runtime_stream) as stream:
            if batch.pipeline_ready:
                # Exercise the actual runtime Tensor-to-CuPy handoff before acquisition.
                view = cp.from_dlpack(batch.frames)
                self._warm_up()
                stream.synchronize()
                del view
                if self.ready is not None:
                    print("[holoscan] ready", flush=True)
                    self.ready.set()
                if self.release is not None:
                    self.release(int(batch.batch_id))
                return
            if batch.start_of_series:
                self.series = cbor2.loads(bytes(batch.start_metadata))
                self.image = cp.zeros((2, self.scan.height, self.scan.width), dtype=cp.float32)
                self.received = self.placed = self.gaps = self.missing = 0
                self.next_id = 0
                self.first = self.last_image = None
                self.first_batch_count = 0
                self.batches = 0
                self.batch_sizes = {}
            count = len(batch.image_ids)
            if count:
                if self.first is None:
                    self.first_batch_count = count
                ids = np.asarray(batch.image_ids, dtype=np.uint64)
                self.gaps += int(ids[0] != self.next_id) + int(np.count_nonzero(np.diff(ids) != 1))
                self.missing += max(0, int(ids[0]) - self.next_id)
                forward = ids[1:] > ids[:-1]
                self.missing += int((np.diff(ids)[forward] - 1).sum())
                self.next_id = int(ids[-1]) + 1
                self.received += count
                self.batches += 1
                self.batch_sizes[count] = self.batch_sizes.get(count, 0) + 1
                frames = cp.from_dlpack(batch.frames)[:count]
                positioned.images = frames
                if len(positioned.selected):
                    selected = positioned.selected
                    if len(selected) != count:
                        frames = frames[cp.asarray(selected)]
                    values = self._integrate(frames)
                    positions = cp.asarray(positioned.positions)
                    self.image[:, positions[:, 0], positions[:, 1]] = values.T
                    self.placed += len(selected)
                stream.synchronize()
                self.last_image = time.perf_counter()
                if self.first is None:
                    self.first = self.last_image
            if not batch.end_of_series and self.observer is not None:
                self.observer(batch, positioned)
            positioned.images = None
            if batch.end_of_series:
                # Preserve float32 processing and report a float64 aggregate of the stored values.
                totals = cp.asnumpy(self.image.sum(axis=(1, 2), dtype=cp.float64)).tolist()
                stream.synchronize()
                elapsed = (self.last_image - self.first) if self.first is not None else 0.0
                result = dict(series_id=int(batch.series_id), images=self.received,
                              announced_images=self.series.get("number_of_images"),
                              positions=self.placed, scan_positions=self.scan.size,
                              gap_events=self.gaps, missing_images=self.missing, batches=self.batches,
                              batch_sizes=self.batch_sizes, bf_sum=totals[0], adf_sum=totals[1],
                              processing_dtype="float32", totals_dtype="float64",
                              delivered_interval_s=elapsed,
                              delivered_fps=(self.received - self.first_batch_count) / elapsed
                              if elapsed and self.batches > 1 else None)
                if int(batch.series_images) != self.received:
                    raise ValueError("END image count does not match delivered images")
                self.last = (batch.series_id, self.image, result)
                positioned.result = result
                print("[series-result] " + json.dumps(result, sort_keys=True), flush=True)
                if self.observer is not None:
                    self.observer(batch, positioned)
                self.series = self.image = None
                if self.finite:
                    ctx.request_completion()
            if batch.start_of_series:
                stream.synchronize()
            if self.release is not None:
                self.release(int(batch.batch_id))
