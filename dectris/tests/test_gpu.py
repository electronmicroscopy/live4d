"""GPU side of the receive path: dispatch from a ring, event-ordered work, reconstruction.

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
    from operators import AggregateOp, WorkerOp
    from dectris.messages import IMAGE_BYTES, Batch, End, Series, Start

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

    def emit(self, message, port):
        self.messages.append(message)


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

    def sums_of(self, dispatched, reduce):
        """(ids, sums) once the event has happened."""
        dispatched.event.synchronize()
        payload = dispatched.payload
        sums = payload if reduce else payload.sum(axis=(1, 2), dtype=cp.uint64)
        return dispatched.ids, sums.get()

    def test_sums_and_copied_images_agree(self):
        ids = np.arange(0, 500, dtype=np.uint64)
        for reduce in (True, False):
            dispatched = Dispatcher(0, reduce=reduce).dispatch(self.addrs(ids, self.pinned.ptr), ids)
            got_ids, sums = self.sums_of(dispatched, reduce)
            np.testing.assert_array_equal(got_ids, ids)
            np.testing.assert_array_equal(sums, expected(ids))

    def test_dispatch_does_not_wait_and_events_order_the_data(self):
        ids = np.arange(0, 512, dtype=np.uint64)
        dispatched = Dispatcher(0).dispatch(self.addrs(ids, self.pinned.ptr), ids)
        stream = cp.cuda.Stream(non_blocking=True)
        stream.wait_event(dispatched.event)  # what the worker does
        with stream:
            total = int(dispatched.payload.sum())
        self.assertEqual(total, int(expected(ids).sum()))
        self.assertTrue(dispatched.event.done)

    def test_ring_in_gpu_memory(self):
        base = self.device_ring.data.ptr
        self.assertEqual(ring_location(base), 0)
        self.assertIsNone(ring_location(self.pinned.ptr))
        ids = np.arange(100, 300, dtype=np.uint64)
        got_ids, sums = self.sums_of(Dispatcher(0, reduce=True).dispatch(
            self.addrs(ids, base), ids), True)
        np.testing.assert_array_equal(sums, expected(got_ids))
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

    def worker(self):
        op = WorkerOp(Application(), name="worker", device=0)
        op.initialize()
        return op

    def feed(self, op, items):
        out = Out()
        op.compute(In(items), out, None)
        return out.messages

    def batch(self, series, ids, reduce=True):
        ids = np.asarray(ids, dtype=np.uint64)
        dispatched = Dispatcher(0, reduce=reduce).dispatch(self.addrs(ids, self.pinned.ptr), ids)
        # The pointer lists must outlive the kernels: they are kept until the event happened.
        dispatched.event.synchronize()
        data = {"sums": dispatched.payload} if reduce else {"images": dispatched.payload}
        return Batch(series, dispatched.ids, dispatched.event, **data)

    def test_worker_consecutive_and_gapped_batches(self):
        series = Series(5)
        for reduce in (True, False):
            op = self.worker()
            # 0..199 in two batches, then 220..299 after a gap of 20 lost images.
            items = [self.batch(series, np.arange(0, 100), reduce),
                     self.batch(series, np.arange(100, 200), reduce),
                     self.batch(series, np.arange(220, 300), reduce),
                     End(series, 300)]
            (messages,) = self.feed(op, items)
            (series_id, values), = messages
            self.assertEqual(series_id, 5)
            want = np.zeros(300, dtype=np.uint64)
            for lo, hi in ((0, 200), (220, 300)):
                want[lo:hi] = expected(np.arange(lo, hi))
            np.testing.assert_array_equal(values.get(), want)

    def test_lost_tail_keeps_its_place(self):
        series = Series(7)
        (messages,) = self.feed(self.worker(), [self.batch(series, np.arange(0, 100)),
                                                End(series, 120)])
        (_, values), = messages
        self.assertEqual(len(values), 120)  # 20 images lost at the end stay 0
        np.testing.assert_array_equal(values.get()[100:], np.zeros(20, dtype=np.uint64))

    def test_start_and_end_without_images_give_zeros(self):
        series = Series(3, expected_images=50)
        (messages,) = self.feed(self.worker(), [Start(series), End(series, 80)])
        (series_id, values), = messages
        self.assertEqual(series_id, 3)
        np.testing.assert_array_equal(values.get(), np.zeros(80, dtype=np.uint64))

    def test_start_does_not_change_the_result(self):
        series = Series(4)
        items = [self.batch(series, np.arange(0, 100)), End(series, 100)]
        (without,) = self.feed(self.worker(), list(items))
        (with_start,) = self.feed(self.worker(), [Start(series)] + items)
        np.testing.assert_array_equal(with_start[0][1].get(), without[0][1].get())

    def test_new_start_drops_a_series_without_end(self):
        first, second = Series(5), Series(6)
        messages = self.feed(self.worker(), [Start(first), self.batch(first, np.arange(0, 10)),
                                             Start(second), End(second, 10)])
        self.assertEqual([series_id for (series_id, _), in messages], [6])

    def aggregate(self, series_width=None):
        op = AggregateOp(Application(), name="aggregate", n_workers=2, device=0,
                         series_width=series_width)
        op.initialize()
        return op

    def test_aggregate_merges_by_image_id(self):
        # Two rings with blocks of ids that are not id % 2, and image 150 lost.
        series = Series(8)
        results = []
        for ids in (np.arange(0, 100), np.concatenate([np.arange(100, 150), np.arange(151, 200)])):
            (messages,) = self.feed(self.worker(), [self.batch(series, ids), End(series, 199)])
            results.append(messages[0][1])
        image = self.aggregate()._aggregate(8, dict(enumerate(results)))
        want = expected(np.arange(200))
        want[150] = 0
        np.testing.assert_array_equal(image.get().ravel(), want)
        self.assertEqual(image.shape, (1, 200))

    def test_aggregate_shapes_the_series_with_series_width(self):
        values = cp.asarray(expected(np.arange(200)))
        self.assertEqual(self.aggregate(series_width=20)._aggregate(9, {0: values}).shape, (10, 20))
        self.assertEqual(self.aggregate(series_width=30)._aggregate(9, {0: values}).shape, (1, 200))


if __name__ == "__main__":
    unittest.main()
