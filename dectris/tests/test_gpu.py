"""GPU side of the receive path: dispatch from a ring, event-ordered work, 4D-STEM tensors and
virtual detector images.

Needs a GPU, CuPy and Holoscan (run it in the image with the NVIDIA runtime); skipped otherwise.
Image i of the test ring has every pixel equal to i + 1, so its pixel sum is (i + 1) * 9216.
"""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import cupy as cp
    from holoscan.core import Application

    HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # no CuPy, Holoscan or driver
    HAVE_GPU = False

if HAVE_GPU:
    from dectris.dispatch import (DeviceRing, Dispatcher, flush_gpudirect_writes,
                                  ring_location, set_sync_memops)
    from dectris.imagemux import FRAME_HEADER_LEN, FrameHeader, FrameKind
    # the example app's operators (examples/pipeline/operators.py)
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples",
                                 "pipeline"))
    from operators import VirtualDetectorOp
    from dectris.messages import IMAGE_BYTES, Batch, End, ScanShape, Segment, Series, Start
    from dectris.operators import ScanTensorOp

N_IMAGES = 600
PIXEL_SUM = 96 * 96


def expected(ids):
    return (np.asarray(ids, dtype=np.uint64) + 1) * PIXEL_SUM


class In:
    """Input ports: ``items`` for every port, or a dict port name -> items."""

    def __init__(self, items):
        self.items = items

    def receive(self, name):
        return self.items[name] if isinstance(self.items, dict) else self.items


class Out:
    def __init__(self):
        self.messages = []
        self.ports = {}  # port name -> messages

    def emit(self, message, port):
        self.messages.append(message)
        self.ports.setdefault(port, []).append(message)


@unittest.skipUnless(HAVE_GPU, "needs a GPU, CuPy and Holoscan")
class GpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pinned = cp.cuda.alloc_pinned_memory(N_IMAGES * IMAGE_BYTES)
        images = np.frombuffer(cls.pinned, dtype=np.uint16, count=N_IMAGES * PIXEL_SUM)
        images[:] = np.repeat(np.arange(N_IMAGES, dtype=np.uint16) + 1, PIXEL_SUM)
        cls.device_ring = cp.asarray(images)  # the same images in GPU memory

    def addrs(self, ids, base):
        return np.uint64(base) + np.asarray(ids, dtype=np.uint64) * np.uint64(IMAGE_BYTES)

    def copied(self, dispatched):
        """(ids, host copy of the images) once the event has happened."""
        dispatched.event.synchronize()
        return dispatched.ids, dispatched.payload.get()

    def assert_images(self, images, ids):
        """Image i has every pixel equal to i + 1."""
        want = (np.asarray(ids, dtype=np.uint16) + 1)[:, None, None]
        np.testing.assert_array_equal(images, np.broadcast_to(want, (len(want), 96, 96)))

    def test_dispatch_copies_the_images(self):
        ids = np.arange(0, 500, dtype=np.uint64)
        got_ids, images = self.copied(Dispatcher(0).dispatch(self.addrs(ids, self.pinned.ptr), ids))
        np.testing.assert_array_equal(got_ids, ids)
        self.assertEqual(images.dtype, np.uint16)
        self.assert_images(images, ids)

    def test_dispatch_does_not_wait_and_events_order_the_data(self):
        ids = np.arange(0, 512, dtype=np.uint64)
        dispatched = Dispatcher(0).dispatch(self.addrs(ids, self.pinned.ptr), ids)
        stream = cp.cuda.Stream(non_blocking=True)
        stream.wait_event(dispatched.event)  # what ScanTensorOp does
        with stream:
            total = int(dispatched.payload.sum(dtype=cp.uint64))
        self.assertEqual(total, int(expected(ids).sum()))
        self.assertTrue(dispatched.event.done)

    def test_ring_in_gpu_memory(self):
        base = self.device_ring.data.ptr
        self.assertEqual(ring_location(base), 0)
        self.assertIsNone(ring_location(self.pinned.ptr))
        ids = np.arange(100, 300, dtype=np.uint64)
        got_ids, images = self.copied(Dispatcher(0).dispatch(self.addrs(ids, base), ids))
        self.assert_images(images, got_ids)
        self.assertIn(flush_gpudirect_writes(0), (0, 801))  # done, or not supported by the GPU

    def test_sync_memops_can_be_set_on_gpu_memory(self):
        ring = cp.zeros(1 << 20, dtype=cp.uint8)
        self.assertEqual(set_sync_memops(ring.data.ptr, 0), 0)

    def test_device_ring_slices_are_copied_to_the_host(self):
        header = FrameHeader(FrameKind.START, 7, 4096, 5).encode()
        ring = cp.zeros(8192, dtype=cp.uint8)
        ring[4096:4096 + FRAME_HEADER_LEN] = cp.asarray(np.frombuffer(header, dtype=np.uint8))
        view = DeviceRing(ring.data.ptr, 8192, 0)
        decoded = FrameHeader.decode(view[4096:4096 + FRAME_HEADER_LEN])
        self.assertEqual((decoded.kind, decoded.sequence, decoded.position), (FrameKind.START, 7, 4096))
        self.assertEqual(len(view[100:100]), 0)

    def batch(self, series, ids):
        ids = np.asarray(ids, dtype=np.uint64)
        dispatched = Dispatcher(0).dispatch(self.addrs(ids, self.pinned.ptr), ids)
        # The pointer lists must outlive the kernels: they are kept until the event happened.
        dispatched.event.synchronize()
        return Batch(series, dispatched.ids, dispatched.event, dispatched.payload)

    def test_dispatcher_warm_up(self):
        dispatcher = Dispatcher(0)
        dispatcher.warm_up(64)
        ids = np.arange(10, 74, dtype=np.uint64)
        _, images = self.copied(dispatcher.dispatch(self.addrs(ids, self.pinned.ptr), ids))
        self.assert_images(images, ids)

    def test_warm_ups_leave_no_state(self):
        for scan, max_batch in ((ScanShape(10, 10), 512), (ScanShape(2, 2), 3),
                                (ScanShape(1, 1), 1)):
            tensors = ScanTensorOp(Application(), name="scan_tensors", device=0, scan=scan,
                                   max_batch=max_batch)
            tensors.initialize()
            self.assertIsNone(tensors.warned)
            detector = VirtualDetectorOp(Application(), name="detector", device=0, scan=scan,
                                         max_batch=max_batch)
            detector.initialize()
            self.assertEqual((detector.series, detector.image, detector.last), (None, None, None))

    def scan_tensors(self, scan):
        op = ScanTensorOp(Application(), name="scan_tensors", device=0, scan=scan)
        op.initialize()
        return op

    def scan_tensors_out(self, scan, items):
        """(frames or None, index) that ScanTensorOp emits for ``items``."""
        out = Out()
        self.scan_tensors(scan).compute(In(items), out, None)
        (index,) = out.ports["index"]
        frames = out.ports.get("frames", [None])
        self.assertEqual(len(frames), 1)  # one message per port and tick
        return frames[0], index

    def test_scan_tensors_and_positions(self):
        series = Series(5)
        ids = np.concatenate([np.arange(0, 100), np.arange(120, 170)])  # 20 images lost
        frames, index = self.scan_tensors_out(ScanShape(20, 10), [
            Start(series), self.batch(series, ids[:100]),
            self.batch(series, ids[100:]), End(series, 170)])
        self.assertEqual(index, [Start(series), Segment(series, 0, 100),
                                 Segment(series, 100, 150), End(series, 170)])
        diffraction = frames["diffraction"]
        self.assertEqual((diffraction.shape, diffraction.dtype), ((150, 96, 96), cp.uint16))
        np.testing.assert_array_equal(diffraction[:, 5, 7].get(), ids + 1)  # image i is i + 1
        np.testing.assert_array_equal(frames["ids"].get(), ids)
        self.assertEqual(frames["positions"].dtype, cp.int32)
        np.testing.assert_array_equal(frames["positions"].get(), np.stack(divmod(ids, 20), 1))

    def test_scan_tensors_leaves_out_images_beyond_the_scan(self):
        series = Series(6)
        frames, index = self.scan_tensors_out(ScanShape(10, 10), [
            self.batch(series, np.arange(50, 150))])
        self.assertEqual(index, [Segment(series, 0, 50)])
        np.testing.assert_array_equal(frames["ids"].get(), np.arange(50, 100))
        np.testing.assert_array_equal(frames["diffraction"][:, 0, 0].get(), np.arange(51, 101))
        frames, index = self.scan_tensors_out(ScanShape(10, 10), [
            self.batch(series, np.arange(100, 120)), End(series, 120)])
        self.assertIsNone(frames)  # nothing inside the scan: only the index
        self.assertEqual(index, [End(series, 120)])

    def detector(self, scan):
        op = VirtualDetectorOp(Application(), name="detector", device=0, scan=scan)
        op.initialize()
        return op

    def through(self, scan, items, detector=None):
        """``items`` through a ScanTensorOp into a VirtualDetectorOp; returns the detector."""
        frames, index = self.scan_tensors_out(scan, items)
        detector = detector or self.detector(scan)
        detector.compute(In({"frames": frames, "index": index}), Out(), None)
        return detector

    def bright_field(self, detector, ids):
        """BF value of images ``ids``: every pixel of image i is i + 1."""
        disc = int(detector.masks[:, 0].sum())
        return (np.asarray(ids, dtype=np.float64) + 1) * disc

    def test_virtual_detector_places_values_by_scan_position(self):
        scan, series = ScanShape(20, 10), Series(7)
        ids = np.concatenate([np.arange(0, 60), np.arange(80, 200)])  # 20 images lost
        detector = self.through(scan, [Start(series),
                                       self.batch(series, ids),
                                       End(series, 200)])
        series_id, image = detector.last
        self.assertEqual((series_id, image.shape), (7, (2, 10, 20)))
        want = np.zeros(200)
        want[ids] = self.bright_field(detector, ids)
        np.testing.assert_allclose(image[0].get().ravel(), want)
        ring = int(detector.masks[:, 1].sum())
        np.testing.assert_allclose(image[1, 9, 19].get(), 200 * ring)  # ADF of image 199

    def test_virtual_detector_two_series_in_one_tick(self):
        scan, first, second = ScanShape(10, 10), Series(1), Series(2)
        finished = []
        detector = self.detector(scan)
        finish = detector._finish
        detector._finish = lambda end: (finish(end), finished.append(detector.last))
        self.through(scan, [Start(first), self.batch(first, np.arange(0, 30)),
                            End(first, 30), Start(second),
                            self.batch(second, np.arange(0, 5)), End(second, 5)],
                     detector)
        self.assertEqual([series_id for series_id, _ in finished], [1, 2])
        np.testing.assert_allclose(finished[0][1][0, 2, :].get(),
                                   self.bright_field(detector, np.arange(20, 30)))
        np.testing.assert_allclose(finished[1][1][0, 0, :5].get(),
                                   self.bright_field(detector, np.arange(5)))
        self.assertEqual(float(finished[1][1][0, 0, 5:].sum().get()), 0.0)

    def test_virtual_detector_series_over_several_ticks(self):
        scan, series = ScanShape(10, 10), Series(3)
        detector = self.through(scan, [Start(series),
                                       self.batch(series, np.arange(0, 40))])
        self.assertIsNone(detector.last)
        self.through(scan, [self.batch(series, np.arange(40, 100)),
                            End(series, 100)], detector)
        np.testing.assert_allclose(detector.last[1][0].get().ravel(),
                                   self.bright_field(detector, np.arange(100)))


if __name__ == "__main__":
    unittest.main()
