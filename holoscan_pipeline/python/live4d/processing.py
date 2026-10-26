"""Place decoded GPU images in a raster scan and integrate virtual detectors."""
from dataclasses import dataclass
from datetime import datetime, timezone
import cbor2
import json
import time

import cupy as cp
import holoscan as hs
import numpy as np

from dectris.messages import IMAGE_SHAPE, ScanShape, parse_start, scan_positions
from .algorithms import virtual_detectors
from .schemas.decoded_batch import DecodedBatch


def _log_series(event, values):
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    print(f"[{event}] " + json.dumps(dict(values, timestamp_utc=timestamp), sort_keys=True),
          flush=True)


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
        self.series_scan = self.scan

    def compute(self, ctx):
        sample = self.inp.receive()
        batch = sample.data
        if batch.start_of_series:
            series = parse_start(bytes(batch.start_metadata))
            if self.series is not None or series.id != batch.series_id:
                raise ValueError("Invalid ordered series boundary")
            self.series = series
            # A scan announced in START overrides the configured scan for this series.
            self.series_scan = series.scan or self.scan
        elif not batch.pipeline_ready and (self.series is None or self.series.id != batch.series_id):
            raise ValueError("Image batch has no matching START")
        ids = np.asarray(batch.image_ids, dtype=np.uint64)
        positions, valid = scan_positions(ids, self.series_scan)
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
    """Integrate float32 BF/ADF values on the runtime-managed CUDA stream."""
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
        self.announced_images = None
        with self.device:
            self.masks_by_shape = {
                shape: virtual_detectors.build_masks(
                    self.bf_radius, self.adf_inner, self.adf_outer, shape=shape)
                for shape in (IMAGE_SHAPE, (192, 192))
            }
            cp.cuda.get_current_stream().synchronize()

    def _warm_up(self):
        # Run in the READY callback's thread and runtime CUDA stream, as image batches do.
        image = cp.zeros((2, self.scan.height, self.scan.width), dtype=cp.float32)
        ids = np.arange(min(self.batch_size, self.scan.size), dtype=np.int64)
        positions = cp.asarray(np.stack(np.divmod(ids, self.scan.width), axis=1).astype(np.int32))
        for shape, masks in self.masks_by_shape.items():
            scratch = cp.ones((self.batch_size, *shape), dtype=cp.uint16)
            values = virtual_detectors.integrate(scratch, masks)
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
                start_metadata = bytes(batch.start_metadata)
                self.series = parse_start(start_metadata)
                self.announced_images = cbor2.loads(start_metadata).get("number_of_images")
                if self.series.id != batch.series_id:
                    raise ValueError("Invalid ordered series boundary")
                self.series_scan = self.series.scan or self.scan
            if self.series is None or self.series.id != batch.series_id:
                raise ValueError("Image batch has no matching START")
            if tuple(batch.frames.shape[1:]) != self.series.shape:
                raise ValueError("Image geometry does not match START")
            self.masks = self.masks_by_shape[self.series.shape]
            if batch.start_of_series:
                scan = self.series_scan
                self.image = cp.zeros((2, scan.height, scan.width), dtype=cp.float32)
                self.received = self.placed = self.gaps = self.missing = 0
                self.next_id = 0
                self.first = self.last_image = None
                self.first_batch_count = 0
                self.batches = 0
                self.batch_sizes = {}
                _log_series("series-start", dict(
                    series_id=int(batch.series_id), device=self.device_id,
                    image_shape=self.series.shape, announced_images=self.announced_images,
                    scan_width=scan.width, scan_height=scan.height,
                    scan_source="header_appendix" if self.series.scan else "config",
                    stage="processing"))
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
                # The runtime orders input readiness and release on this activation's stream.
                frames = cp.from_dlpack(batch.frames)[:count]
                positioned.images = frames
                if len(positioned.selected):
                    selected = positioned.selected
                    if len(selected) != count:
                        frames = frames[cp.asarray(selected)]
                    values = virtual_detectors.integrate(frames, self.masks)
                    positions = cp.asarray(positioned.positions)
                    self.image[:, positions[:, 0], positions[:, 1]] = values.T
                    self.placed += len(selected)
                # Host timestamps measure batch submission, before GPU completion.
                self.last_image = time.perf_counter()
                if self.first is None:
                    self.first = self.last_image
            if not batch.end_of_series and self.observer is not None:
                self.observer(batch, positioned)
            positioned.images = None
            if batch.end_of_series:
                if int(batch.series_images) != self.received:
                    raise ValueError("END image count does not match delivered images")
                _log_series("series-end", dict(
                    series_id=int(batch.series_id), device=self.device_id,
                    received_images=self.received, announced_images=self.announced_images,
                    positions=self.placed, batches=self.batches, stage="processing"))
                # Preserve float32 processing and report a float64 aggregate of the stored values.
                totals = cp.asnumpy(self.image.sum(axis=(1, 2), dtype=cp.float64)).tolist()
                elapsed = (self.last_image - self.first) if self.first is not None else 0.0
                result = dict(series_id=int(batch.series_id), images=self.received,
                              announced_images=self.announced_images,
                              positions=self.placed, scan_positions=self.series_scan.size,
                              gap_events=self.gaps, missing_images=self.missing, batches=self.batches,
                              batch_sizes=self.batch_sizes, bf_sum=totals[0], adf_sum=totals[1],
                              device=self.device_id, processed_images=self.received,
                              processing_dtype="float32", totals_dtype="float64",
                              delivered_interval_s=elapsed,
                              delivered_fps=(self.received - self.first_batch_count) / elapsed
                              if elapsed and self.batches > 1 else None)
                self.last = (batch.series_id, self.image, result)
                positioned.result = result
                _log_series("series-result", result)
                if self.observer is not None:
                    self.observer(batch, positioned)
                self.series = self.image = None
                self.announced_images = None
                if self.finite:
                    ctx.request_completion()
            if self.release is not None:
                self.release(int(batch.batch_id))
