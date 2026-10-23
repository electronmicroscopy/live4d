"""GPU ring and dispatch checks for the reusable imagemux transport package.

Needs a CUDA GPU and CuPy. Image i has every pixel equal to i + 1.
"""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import cupy as cp

    HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # no CuPy or driver
    HAVE_GPU = False

if HAVE_GPU:
    from dectris.dispatch import (DeviceRing, Dispatcher, flush_gpudirect_writes,
                                 mapped_device_pointer, ring_location, set_sync_memops)

N_IMAGES = 600
PIXEL_SUM = 96 * 96


def expected(ids):
    return (np.asarray(ids, dtype=np.uint64) + 1) * PIXEL_SUM


@unittest.skipUnless(HAVE_GPU, "requires CUDA and CuPy")
class GpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pinned = cp.cuda.alloc_pinned_memory(N_IMAGES * PIXEL_SUM * np.dtype(np.uint16).itemsize)
        images = np.frombuffer(cls.pinned, dtype=np.uint16, count=N_IMAGES * PIXEL_SUM)
        images[:] = np.repeat(np.arange(N_IMAGES, dtype=np.uint16) + 1, PIXEL_SUM)
        cls.device_ring = cp.asarray(images)

    def addrs(self, ids, base):
        return np.uint64(base) + np.asarray(ids, dtype=np.uint64) * np.uint64(PIXEL_SUM * 2)

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
        stream.wait_event(dispatched.event)
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
        self.assertIn(flush_gpudirect_writes(0), (0, 801))

    def test_mapped_host_ring_images(self):
        count = 4
        memory = cp.cuda.PinnedMemory(count * PIXEL_SUM * 2, cp.cuda.runtime.hostAllocMapped)
        buffer = cp.cuda.PinnedMemoryPointer(memory, 0)
        pixels = np.frombuffer(buffer, dtype=np.uint16, count=count * PIXEL_SUM).reshape(
            count, 96, 96)
        pixels[:] = np.arange(1, count + 1, dtype=np.uint16)[:, None, None]
        alias = mapped_device_pointer(memory.ptr, 0)
        ids = np.arange(count, dtype=np.uint64)
        got_ids, images = self.copied(Dispatcher(0).dispatch(self.addrs(ids, alias), ids))
        np.testing.assert_array_equal(got_ids, ids)
        self.assert_images(images, ids)

    def test_sync_memops_can_be_set_on_gpu_memory(self):
        ring = cp.zeros(1 << 20, dtype=cp.uint8)
        self.assertEqual(set_sync_memops(ring.data.ptr, 0), 0)

    def test_device_ring_slices_are_copied_to_the_host(self):
        ring = cp.arange(256, dtype=cp.uint8)
        view = DeviceRing(ring.data.ptr, ring.nbytes, 0)
        self.assertEqual(view[100:110], bytes(range(100, 110)))
        self.assertEqual(len(view[100:100]), 0)

    def test_dispatcher_warm_up(self):
        dispatcher = Dispatcher(0)
        dispatcher.warm_up(64)
        ids = np.arange(10, 74, dtype=np.uint64)
        _, images = self.copied(dispatcher.dispatch(self.addrs(ids, self.pinned.ptr), ids))
        self.assert_images(images, ids)


if __name__ == "__main__":
    unittest.main()
