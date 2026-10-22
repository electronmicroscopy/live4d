"""GPU side of the receive path: dispatch from a ring, event-ordered work, 4D-STEM tensors and
virtual detector images.

Needs a GPU, CuPy and Holoscan (run it in the image with the NVIDIA runtime); skipped otherwise.
Image i of the test ring has every pixel equal to i + 1, so its pixel sum is (i + 1) * 9216.
"""

import os
import signal
import sys
import threading
import time
from datetime import timedelta
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import cupy as cp
    import holoscan as hs

    HAVE_GPU = (cp.cuda.runtime.getDeviceCount() > 0 and hasattr(hs, "Graph")
                and hasattr(hs, "compile") and hasattr(hs, "run_async"))
except Exception:  # no CuPy, Holoscan or driver
    HAVE_GPU = False

if HAVE_GPU:
    from dectris.dispatch import (DeviceRing, Dispatcher, flush_gpudirect_writes,
                                 mapped_device_pointer, ring_location, set_sync_memops)
    from dectris.imagemux import FRAME_HEADER_LEN, FrameHeader, FrameKind, RemoteRegion
    # the example app's operators (examples/pipeline/operators.py)
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples",
                                 "pipeline"))
    from app import run_plan
    from operators import VirtualDetectorOp
    from dectris.messages import IMAGE_BYTES, Batch, End, ScanShape, Segment, Series, Start
    from dectris.operators import RdmaRxOp, RingPath, ScanTensorOp

N_IMAGES = 600
PIXEL_SUM = 96 * 96


def expected(ids):
    return (np.asarray(ids, dtype=np.uint64) + 1) * PIXEL_SUM


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
        path = RingPath(0, RemoteRegion(base, self.device_ring.nbytes, 0), RemoteRegion(0, 8, 0))
        ids = np.arange(100, 300, dtype=np.uint64)
        got_ids, images = self.copied(Dispatcher(path.device).dispatch(
            self.addrs(ids, path.device_address), ids))
        self.assert_images(images, got_ids)
        self.assertIn(flush_gpudirect_writes(0), (0, 801))  # done, or not supported by the GPU

    def test_mapped_host_ring_headers_and_gpu_images(self):
        count = 4
        size = FRAME_HEADER_LEN + count * IMAGE_BYTES
        memory = cp.cuda.PinnedMemory(size, cp.cuda.runtime.hostAllocMapped)
        buffer = cp.cuda.PinnedMemoryPointer(memory, 0)
        header = FrameHeader(FrameKind.START, 7, 0, 5).encode()
        np.frombuffer(buffer, dtype=np.uint8, count=FRAME_HEADER_LEN)[:] = np.frombuffer(
            header, dtype=np.uint8)
        pixels = np.frombuffer(buffer, dtype=np.uint16, offset=FRAME_HEADER_LEN).reshape(count, 96, 96)
        pixels[:] = np.arange(1, count + 1, dtype=np.uint16)[:, None, None]
        alias = mapped_device_pointer(memory.ptr, 0)
        path = RingPath(0, RemoteRegion(memory.ptr, size, 0), RemoteRegion(0, 8, 0),
                        ring_device=0, ring_device_address=alias)
        decoded = FrameHeader.decode(path.view[:FRAME_HEADER_LEN])
        self.assertEqual((decoded.kind, decoded.sequence), (FrameKind.START, 7))
        ids = np.arange(count, dtype=np.uint64)
        got_ids, images = self.copied(Dispatcher(path.device).dispatch(
            self.addrs(ids, path.device_address + FRAME_HEADER_LEN), ids))
        np.testing.assert_array_equal(got_ids, ids)
        self.assert_images(images, ids)
        with self.assertRaises(ValueError):
            RingPath(0, RemoteRegion(memory.ptr, size, 0), RemoteRegion(0, 8, 0),
                     ring_device=0, ring_device_address=alias + 128)

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

    def test_failed_completion_requests_hsdk5_graph_completion(self):
        class FailedConnection:
            endpoint = SimpleNamespace(qp_num=7)
            errors = ()
            stats = {}

            def poll(self, imms):
                return 0, 0, 1

        connection = FailedConnection()
        ring_storage = cp.zeros(1 << 20, dtype=cp.uint8)
        ring = RemoteRegion(ring_storage.data.ptr, ring_storage.nbytes, 1)
        credit = RemoteRegion(0, 8, 2)

        class ConnectedReceiver(RdmaRxOp):
            def __init__(self):
                super().__init__(conn=connection, ring=ring, credit=credit)

        graph = hs.Graph("failed-completion-test", default_clock=hs.RealtimeClock)
        graph.op(ConnectedReceiver, "rx", max_batch=4, receive_only=True)
        graph.partition("pipeline").include_all()
        plan = hs.compile(graph)
        self.assertTrue(plan.ok, plan.summary())
        with hs.run_async(plan) as session:
            self.assertTrue(session.wait(timeout=5), session.summary())
            self.assertEqual(session.termination, hs.RunTermination.REQUESTED_COMPLETED)

    def test_run_plan_waits_for_started_event_before_ready(self):
        import io
        from contextlib import redirect_stdout

        downstream_started = threading.Event()
        early_compute = threading.Event()
        started = threading.Event()

        class Source(hs.Operator):
            output = hs.Output("output", hs.pyobject)

            def __init__(self):
                super().__init__()
                self.sent = False

            def setup(self, spec):
                spec.output(self.output)

            def contract(self):
                return hs.Contract().trigger(
                    hs.OnClock(period=timedelta(milliseconds=2)))

            def compute(self, context):
                if not downstream_started.is_set():
                    early_compute.set()
                if not self.sent:
                    self.output.emit("startup")
                    self.sent = True
                started.set()

        class DelayedSink(hs.Operator):
            inp = hs.Input("input", hs.pyobject)

            def __init__(self):
                super().__init__()

            def setup(self, spec):
                spec.input(self.inp)

            def contract(self):
                return hs.Contract().trigger(hs.OnEach(self.inp))

            def start(self):
                time.sleep(0.3)
                downstream_started.set()

            def compute(self, context):
                self.inp.receive_data()

        graph = hs.Graph("ready-test", default_clock=hs.RealtimeClock)
        source = graph.op(Source, "source")
        sink = graph.op(DelayedSink, "sink")
        graph.add_flow(source.output, sink.input)
        graph.partition("source").add(source)
        graph.partition("sink").add(sink)
        plan = hs.compile(graph)
        self.assertTrue(plan.ok, plan.summary())
        output = io.StringIO()
        with redirect_stdout(output):
            termination = run_plan(plan, 0.05, threading.Event(), started=started)
        self.assertTrue(downstream_started.is_set())
        self.assertFalse(early_compute.is_set())
        self.assertTrue(started.is_set())
        self.assertEqual(output.getvalue().strip(), "[holoscan] ready")
        self.assertIn(termination, (hs.RunTermination.REQUESTED_COMPLETED,
                                    hs.RunTermination.COMPLETED, hs.RunTermination.STOPPED))

    def test_run_plan_accepts_sigterm_during_startup(self):
        import io
        from contextlib import redirect_stdout

        class Source(hs.Operator):
            output = hs.Output("output", hs.pyobject)

            def __init__(self):
                super().__init__()

            def setup(self, spec):
                spec.output(self.output)

            def contract(self):
                return hs.Contract().trigger(
                    hs.OnClock(period=timedelta(milliseconds=2)))

            def compute(self, context):
                pass

        class DelayedSink(hs.Operator):
            inp = hs.Input("input", hs.pyobject)

            def __init__(self):
                super().__init__()

            def setup(self, spec):
                spec.input(self.inp)

            def contract(self):
                return hs.Contract().trigger(hs.OnEach(self.inp))

            def start(self):
                time.sleep(0.4)

            def compute(self, context):
                self.inp.receive_data()

        graph = hs.Graph("startup-signal-test", default_clock=hs.RealtimeClock)
        source = graph.op(Source, "source")
        sink = graph.op(DelayedSink, "sink")
        graph.add_flow(source.output, sink.input)
        graph.partition("source").add(source)
        graph.partition("sink").add(sink)
        plan = hs.compile(graph)
        self.assertTrue(plan.ok, plan.summary())
        started = threading.Event()
        output = io.StringIO()
        timer = threading.Timer(0.15, signal.raise_signal, args=(signal.SIGTERM,))
        with redirect_stdout(output):
            timer.start()
            termination = run_plan(plan, 0, threading.Event(), started=started)
            timer.join(timeout=1)
        self.assertFalse(started.is_set())
        self.assertNotIn("[holoscan] ready", output.getvalue())
        self.assertIn(termination, (hs.RunTermination.REQUESTED_COMPLETED,
                                    hs.RunTermination.COMPLETED, hs.RunTermination.STOPPED))

    def run_stages(self, scan, items, *, polls=None, threads=1, max_batch=8, observe=True):
        """Pass received image messages through the compiled scan and detector graph."""
        polls = [items] if polls is None else polls
        received_pairs = []
        completed = []
        done = threading.Event()

        class Source(hs.Operator):
            output0 = hs.Output("output0", hs.pyobject)

            def __init__(self):
                super().__init__()
                self.poll = 0

            def setup(self, spec):
                spec.output(self.output0)

            def contract(self):
                return hs.Contract().trigger(
                    hs.OnClock(period=timedelta(milliseconds=5)))

            def compute(self, context):
                if self.poll < len(polls):
                    self.output0.emit(polls[self.poll])
                    self.poll += 1

        class RecordingDetector(VirtualDetectorOp):
            def _finish(self, end):
                super()._finish(end)
                completed.append(self.last)
                done.set()

        class Observer(hs.Operator):
            inp = hs.Input("input", hs.pyobject)

            def __init__(self):
                super().__init__()

            def setup(self, spec):
                spec.input(self.inp)

            def contract(self):
                return hs.Contract().trigger(hs.OnEach(self.inp))

            def compute(self, context):
                received_pairs.append(self.inp.receive_data())

        graph = hs.Graph("gpu-test", default_clock=hs.RealtimeClock)
        source = graph.op(Source, "source")
        tensors = graph.op(ScanTensorOp, "scan_tensors", device_id=0,
                           scan_width=scan.width, scan_height=scan.height,
                           max_batch=max_batch)
        detector = graph.op(RecordingDetector, "detector", device_id=0,
                            scan_width=scan.width, scan_height=scan.height,
                            max_batch=max_batch)
        graph.add_flow(source.output0, tensors.input)
        graph.add_flow(tensors.output, detector.input)
        observer = graph.op(Observer, "observer") if observe else None
        if observer is not None:
            graph.add_flow(tensors.output, observer.input)
        if threads <= 1:
            graph.partition("pipeline").include_all()
        else:
            graph.partition("receiver").add(source)
            if threads == 2:
                processing = graph.partition("processing")
                processing.add(tensors).add(detector)
                if observer is not None:
                    processing.add(observer)
            else:
                scanning = graph.partition("scan").add(tensors)
                if observer is not None:
                    scanning.add(observer)
                graph.partition("detector").add(detector)

        plan = hs.compile(graph)
        self.assertTrue(plan.ok, plan.summary())
        observed_count = len(polls) if observe else 0
        with hs.run_async(plan) as session:
            deadline = time.monotonic() + 15
            while ((not done.is_set() or len(received_pairs) < observed_count)
                   and time.monotonic() < deadline):
                if session.wait(timeout=0.05):
                    break
            if not done.is_set() or len(received_pairs) < observed_count:
                session.request_stop()
                self.fail(f"Timed out waiting for GPU pipeline output: {session.summary()}")
            session.request_stop()
            self.assertTrue(session.wait(timeout=5), session.summary())
        if observe:
            self.assertGreaterEqual(len(received_pairs), len(polls),
                                    "ScanTensorOp did not emit the expected batch")
        return received_pairs, completed

    def test_virtual_detector_places_values_by_scan_position(self):
        scan, series = ScanShape(20, 10), Series(7)
        ids = np.concatenate([np.arange(0, 60), np.arange(80, 200)])  # 20 images lost
        pairs, completed = self.run_stages(scan, [Start(series), self.batch(series, ids),
                                                  End(series, 200)])
        series_id, image = completed[-1]
        self.assertEqual((series_id, image.shape), (7, (2, 10, 20)))
        want = np.zeros(200)
        y, x = cp.mgrid[:96, :96].astype(cp.float32)
        radius = cp.hypot(y - 47.5, x - 47.5)
        disc = int((radius <= 16.0).sum().get())
        ring = int(((radius >= 24.0) & (radius <= 48.0)).sum().get())
        want[ids] = (np.asarray(ids, dtype=np.float64) + 1) * disc
        np.testing.assert_allclose(image[0].get().ravel(), want)
        np.testing.assert_allclose(image[1, 9, 19].get(), 200 * ring)  # ADF of image 199
        observed_index, frames = pairs[0]
        self.assertEqual(observed_index, [Start(series), Segment(series, 0, 180), End(series, 200)])
        self.assertEqual((frames["diffraction"].shape, frames["diffraction"].dtype),
                         ((180, 96, 96), cp.uint16))
        np.testing.assert_array_equal(frames["ids"].get(), ids)
        self.assertEqual(frames["positions"].dtype, cp.int32)
        np.testing.assert_array_equal(frames["positions"].get(), np.stack(divmod(ids, 20), 1))

    def test_virtual_detector_two_series_in_one_tick(self):
        scan, first, second = ScanShape(10, 10), Series(1), Series(2)
        pairs, finished = self.run_stages(
            scan, [Start(first), self.batch(first, np.arange(0, 30)), End(first, 30),
                   Start(second), self.batch(second, np.arange(0, 5)), End(second, 5)])
        self.assertEqual([series_id for series_id, _ in finished], [1, 2])
        _, image1 = finished[0]
        _, image2 = finished[1]
        disc = int((cp.hypot(cp.mgrid[:96, :96][0] - 47.5,
                             cp.mgrid[:96, :96][1] - 47.5) <= 16.0).sum().get())
        np.testing.assert_allclose(image1[0, 2, :].get(),
                                   (np.arange(20, 30) + 1) * disc)
        np.testing.assert_allclose(image2[0, 0, :5].get(),
                                   (np.arange(5) + 1) * disc)
        self.assertEqual(float(image2[0, 0, 5:].sum().get()), 0.0)
        index, frames = pairs[0]
        self.assertEqual(index, [Start(first), Segment(first, 0, 30), End(first, 30),
                                 Start(second), Segment(second, 30, 35), End(second, 5)])
        np.testing.assert_array_equal(frames["ids"].get(), np.r_[np.arange(30), np.arange(5)])

    def test_scan_clips_outside_images_and_keeps_control_only_end(self):
        scan, series = ScanShape(10, 10), Series(4)
        polls = [[Start(series), self.batch(series, np.arange(50, 150))],
                 [self.batch(series, np.arange(100, 120)), End(series, 120)]]
        pairs, completed = self.run_stages(scan, [], polls=polls)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0], [Start(series), Segment(series, 0, 50)])
        np.testing.assert_array_equal(pairs[0][1]["ids"].get(), np.arange(50, 100))
        self.assertEqual(pairs[1], ([End(series, 120)], None))
        self.assertEqual(len(completed), 1)
        series_id, image = completed[0]
        self.assertEqual(series_id, series.id)
        self.assertEqual(image.shape, (2, 10, 10))
        y, x = cp.mgrid[:96, :96].astype(cp.float32)
        disc = int((cp.hypot(y - 47.5, x - 47.5) <= 16.0).sum().get())
        want = np.zeros(100)
        want[50:100] = (np.arange(50, 100) + 1) * disc
        np.testing.assert_allclose(image[0].get().ravel(), want)

    def test_control_only_pair_and_partition_layouts(self):
        series = Series(10)
        pairs, completed = self.run_stages(
            ScanShape(2, 2), [Start(series), End(series, 0)], max_batch=4)
        self.assertEqual(pairs[0], ([Start(series), End(series, 0)], None))
        series_id, image = completed[-1]
        self.assertEqual(series_id, series.id)
        self.assertEqual(image.shape, (2, 2, 2))
        self.assertEqual(float(image.sum().get()), 0.0)

        for threads in (1, 2, 3):
            with self.subTest(threads=threads):
                series = Series(10 + threads)
                pairs, completed = self.run_stages(
                    ScanShape(2, 2), [Start(series), End(series, 0)], threads=threads,
                    max_batch=4, observe=False)
                series_id, image = completed[-1]
                self.assertEqual(series_id, series.id)
                self.assertEqual(image.shape, (2, 2, 2))
                self.assertEqual(float(image.sum().get()), 0.0)

    def test_virtual_detector_series_over_several_ticks(self):
        scan, series = ScanShape(10, 10), Series(3)
        first, second = self.batch(series, np.arange(0, 40)), self.batch(series, np.arange(40, 100))
        pairs, completed = self.run_stages(
            scan, [], polls=[[Start(series), first], [second, End(series, 100)]])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0], [Start(series), Segment(series, 0, 40)])
        self.assertEqual(pairs[1][0], [Segment(series, 0, 60), End(series, 100)])
        series_id, image = completed[-1]
        self.assertEqual(series_id, 3)
        y, x = cp.mgrid[:96, :96].astype(cp.float32)
        disc = int((cp.hypot(y - 47.5, x - 47.5) <= 16.0).sum().get())
        np.testing.assert_allclose(image[0].get().ravel(),
                                   (np.arange(100) + 1) * disc)


if __name__ == "__main__":
    unittest.main()
