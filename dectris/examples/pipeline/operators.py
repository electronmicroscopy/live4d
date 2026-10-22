"""Operators of the example receiver app: VirtualDetectorOp (virtual detector images of a 4D-STEM
scan), which consumes what dectris.ScanTensorOp emits, wired up after dectris.RdmaRxOp by
add_virtual_detectors.
"""

import cupy as cp
import numpy as np
from holoscan.core import ConditionType, Operator, OperatorSpec

from dectris import IMAGE_SHAPE, End, ScanShape, Segment, Start, ScanTensorOp, scan_positions


# ---------------------------------------------------------------------------------------------
# VirtualDetectorOp
#
# Virtual detector images of a 4D-STEM scan, built batch by batch from ScanTensorOp's tensors.
# ---------------------------------------------------------------------------------------------

class VirtualDetectorOp(Operator):
    """Inputs ``frames`` and ``index`` of a ``ScanTensorOp``, on its ``device``.

    Each diffraction pattern is integrated over a bright-field disc (radius ``bf_radius`` pixels)
    and an annular dark-field ring (``adf_radii`` inner, outer), both centred on the detector, and
    the two values are put at the image's scan position. At the end of a series the (2, height,
    width) float32 BF / ADF images are complete; the example only prints them and keeps the last
    one (``last``: series id, image). Positions without an image stay 0. A series whose end never
    arrives is dropped when the next one starts.
    """

    def __init__(self, fragment, *args, device: int, scan: ScanShape, bf_radius: float = 16.0,
                 adf_radii: tuple = (24.0, 48.0), max_batch: int = 512, **kwargs):
        self.device = cp.cuda.Device(device)
        self.scan = scan
        self.max_batch = max_batch  # images of the largest batch, for the warm-up
        self.bf_radius = bf_radius
        self.adf_radii = adf_radii
        self.series = None
        self.image = None  # (2, height, width) BF / ADF of the series
        self.placed = 0  # images put into it
        self.last = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("frames").condition(ConditionType.NONE)  # not emitted when there are no images
        spec.input("index")

    def initialize(self):
        super().initialize()
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            y, x = cp.mgrid[:IMAGE_SHAPE[0], :IMAGE_SHAPE[1]].astype(cp.float32)
            r = cp.hypot(y - (IMAGE_SHAPE[0] - 1) / 2, x - (IMAGE_SHAPE[1] - 1) / 2)
            inner, outer = self.adf_radii
            masks = cp.stack([r <= self.bf_radius, (r >= inner) & (r <= outer)])
            self.masks = masks.reshape(2, -1).T.astype(cp.float32)  # (pixels, 2)
            self._warm_up()

    def _warm_up(self):
        """Run a full batch into a full-size image now, not on the first (timed) batches: compiles
        the kernels, sets up cuBLAS and grows the memory pool (the image's memory goes back to it
        and is reused by the first series)."""
        k = self.max_batch
        with self.stream:
            values = self._integrate(cp.ones((k, *IMAGE_SHAPE), dtype=cp.uint16))
            positions, _ = scan_positions(np.arange(k) % self.scan.size, self.scan)
            positions = cp.asarray(positions)  # made on the host and copied, as ScanTensorOp does
            image = cp.zeros((2, self.scan.height, self.scan.width), dtype=cp.float32)
            image[:, positions[:, 0], positions[:, 1]] = values.T
            image.sum(axis=(1, 2)).get()
        self.stream.synchronize()

    def _integrate(self, diffraction):
        """(k, 2) BF / ADF sums of the (k, 96, 96) patterns."""
        return diffraction.reshape(diffraction.shape[0], -1).astype(cp.float32) @ self.masks

    def _begin(self, series):
        if self.image is not None:
            print(f"virtual detector: series {self.series.id} has no end, dropped", flush=True)
        self.series = series
        self.image = cp.zeros((2, self.scan.height, self.scan.width), dtype=cp.float32)
        self.placed = 0

    def _add(self, series, positions, values):
        if self.series is None or series.id != self.series.id:
            self._begin(series)
        self.image[:, positions[:, 0], positions[:, 1]] = values.T
        self.placed += positions.shape[0]

    def _finish(self, end: End):
        if self.series is None or end.series.id != self.series.id:
            self._begin(end.series)  # no image of this series came
        image, placed = self.image, self.placed
        self.series = self.image = None
        bf, adf = (float(s) for s in image.sum(axis=(1, 2)).get())
        print(f"virtual detector series {end.series.id}: {placed} of {self.scan.size} positions "
              f"({self.scan.width}x{self.scan.height}), BF sum={bf:.6g}, ADF sum={adf:.6g}",
              flush=True)
        self.last = (end.series.id, image)

    def compute(self, op_input, op_output, context):
        index = op_input.receive("index")
        frames = op_input.receive("frames")
        with self.device, self.stream:
            if frames is not None:
                positions = cp.asarray(frames["positions"])
                values = self._integrate(cp.asarray(frames["diffraction"]))
            for item in index:
                if isinstance(item, Start):
                    self._begin(item.series)
                elif isinstance(item, Segment):
                    self._add(item.series, positions[item.start:item.stop],
                              values[item.start:item.stop])
                elif isinstance(item, End):
                    self._finish(item)
            # The input tensors go back to their memory pool after this tick: be done with them.
            self.stream.synchronize()


def add_virtual_detectors(app, source, scan: ScanShape):
    """A ScanTensorOp and a VirtualDetectorOp per ring of ``source`` (an RdmaRxOp), on the ring's
    GPU. With several rings each one only sees the scan positions of its images."""
    for path in source.paths:
        tensors = ScanTensorOp(app, name=f"scan_tensors{path.index}", device=path.device,
                               scan=scan, max_batch=source.max_batch)
        detector = VirtualDetectorOp(app, name=f"detector{path.index}", device=path.device,
                                     scan=scan, max_batch=source.max_batch)
        app.add_flow(source, tensors, {(f"output{path.index}", "input")})
        app.add_flow(tensors, detector, {("frames", "frames"), ("index", "index")})

