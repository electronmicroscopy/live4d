"""Native imagemux loopback checks for the HSDK RDMA source operator."""

import os
from pathlib import Path
import threading
import time

import pytest


LOCAL_ADDR = os.environ.get("IMXRDMA_LOCAL_ADDR")
IN_HSDK_IMAGE = os.environ.get("RDMA_TEST_IN_IMAGE") == "1"
pytestmark = pytest.mark.skipif(
    not IN_HSDK_IMAGE or not LOCAL_ADDR or not Path("/dev/infiniband").exists(),
    reason="requires the HSDK 5 GPU/RDMA test image and IMXRDMA_LOCAL_ADDR",
)


class RingRecord:
    def __init__(self, wire, sequence, start, end):
        self.wire = wire
        self.sequence = sequence
        self.start = start
        self.end = end


class RingWriter:
    """Build the aligned frame stream that imagemux writes into a ring."""

    def __init__(self, ring_bytes):
        self.ring_bytes = ring_bytes
        self.position = 0
        self.sequence = 0
        self.records = []
        self.wrap_count = 0

    def write(self, kind, data=b"", *, series_id=0, image=None):
        from dectris.imagemux import FrameHeader, FrameKind, frame_span

        offset = self.position % self.ring_bytes
        span = frame_span(len(data))
        if kind != FrameKind.WRAP and offset + span > self.ring_bytes:
            self.wrap_count += 1
            self.write(FrameKind.WRAP)
            offset = 0

        header = FrameHeader(kind, self.sequence, self.position, len(data), series_id,
                             image=image) if image is not None else FrameHeader(
                                 kind, self.sequence, self.position, len(data), series_id)
        start = self.position
        wire = header.encode() + data
        self.position += self.ring_bytes - offset if kind == FrameKind.WRAP else span
        self.records.append(RingRecord(wire, self.sequence, start, self.position))
        self.sequence += 1
        return self.records[-1]


class BatchWindow:
    def __init__(self, capacity):
        import ctypes
        from live4d import operators

        library_path = Path(operators.__file__).with_name("libdectris_holoscan_ops.so")
        self.library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        self.library.dectris_batch_window_create.argtypes = [ctypes.c_uint32]
        self.library.dectris_batch_window_create.restype = ctypes.c_void_p
        self.library.dectris_batch_window_destroy.argtypes = [ctypes.c_void_p]
        self.library.dectris_batch_window_destroy.restype = None
        self.library.dectris_batch_window_release.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.library.dectris_batch_window_release.restype = None
        self.handle = self.library.dectris_batch_window_create(capacity)
        if not self.handle:
            raise RuntimeError("native batch window allocation failed")

    def release(self, batch_id):
        self.library.dectris_batch_window_release(self.handle, batch_id)

    def close(self):
        if self.handle:
            self.library.dectris_batch_window_destroy(self.handle)
            self.handle = None


def _start_sender(loopback, records, **kwargs):
    errors = []

    def send():
        try:
            loopback.send(records, **kwargs)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=send)
    thread.start()
    return thread, errors


def _join_sender(thread, errors, timeout):
    thread.join(timeout=timeout)
    assert not thread.is_alive()
    assert not errors, errors


def _start(writer, series_id, announced):
    import cbor2
    from dectris.imagemux import FrameKind

    metadata = cbor2.dumps(cbor2.CBORTag(55799, {
        "type": "start", "series_id": series_id, "image_size_x": 96,
        "image_size_y": 96, "image_dtype": "uint16", "number_of_images": announced,
    }))
    record = writer.write(FrameKind.START, metadata, series_id=series_id)
    return record, metadata


def _image(writer, series_id, image_ids, values):
    import numpy as np
    from dectris.imagemux import Compression, DataType, FrameKind, ImageHeader

    frames = np.stack([
        np.full((96, 96), value, dtype=np.uint16) for value in values
    ])
    payload = frames.tobytes()
    image = ImageHeader(
        image_id=image_ids[0], batch_id=0, batch_size=len(image_ids), dim_x=96, dim_y=96,
        dtype=DataType.UINT16, compression=Compression.NONE,
        uncompressed_size=len(payload), channel=0, channel_count=1,
    )
    return writer.write(FrameKind.IMAGE, payload, series_id=series_id, image=image), frames


def _end(writer, series_id):
    from dectris.imagemux import FrameKind

    return writer.write(FrameKind.END, series_id=series_id)


class Loopback:
    """Two local queue pairs with a registered receiver ring."""

    def __init__(self, ring_bytes, ring_memory="host_mapped"):
        import cupy as cp
        import numpy as np
        from dectris import imxrdma
        from dectris.dispatch import mapped_device_pointer

        self.cp = cp
        self.np = np
        self.ctx = imxrdma.Context(local_addr=LOCAL_ADDR)
        if ring_memory == "host_mapped":
            self.ring_mem = cp.cuda.PinnedMemory(ring_bytes, cp.cuda.runtime.hostAllocMapped)
            self.ring_address = self.ring_mem.ptr
            self.ring_device_address = mapped_device_pointer(self.ring_address, 0)
        elif ring_memory == "gpu":
            with cp.cuda.Device(0):
                self.ring_mem = cp.empty(ring_bytes, dtype=cp.uint8)
            self.ring_address = self.ring_mem.data.ptr
            self.ring_device_address = self.ring_address
        else:
            raise ValueError(f"unsupported ring memory: {ring_memory}")
        self.ring_mr = self.ctx.register(self.ring_address, ring_bytes, remote_write=True)
        self.credit = np.zeros(1, dtype=np.uint64)
        self.credit_mr = self.ctx.register(self.credit.ctypes.data, self.credit.nbytes,
                                           remote_write=True)
        self.rx = self.ctx.create_qp(recv_depth=64, send_depth=64, signal_every=1)
        self.tx = self.ctx.create_qp(recv_depth=1, send_depth=64, signal_every=1)
        self.rx.connect(self.tx.endpoint)
        self.tx.connect(self.rx.endpoint)
        assert self.rx.post_recvs(64) == 64
        self.source = None
        self.source_mr = None
        self.imms = np.empty(64, dtype=np.uint32)
        self.ring_bytes = ring_bytes
        self.ring_memory = ring_memory

    @property
    def credit_position(self):
        import ctypes

        return ctypes.c_uint64.from_address(self.credit.ctypes.data).value

    def send(self, records, *, on_wait=None, timeout=20.0):
        import numpy as np

        payload = bytearray()
        offsets = []
        for record in records:
            offsets.append(len(payload))
            payload.extend(record.wire)
        self.source = np.frombuffer(payload, dtype=np.uint8)
        self.source_mr = self.ctx.register(self.source.ctypes.data, self.source.nbytes)

        def poll_sender():
            _, _, failed = self.tx.poll(self.imms)
            if failed:
                raise RuntimeError(f"sender completion failed: {self.tx.errors}")

        for index, (record, source_offset) in enumerate(zip(records, offsets)):
            deadline = time.monotonic() + timeout
            waited = False
            while record.end - self.credit_position > self.ring_bytes:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"credit did not reach frame {record.sequence}")
                if not waited and on_wait is not None:
                    on_wait(index, self.credit_position)
                    waited = True
                poll_sender()
                time.sleep(0.0001)
            while not self.tx.write(
                self.source_mr, source_offset, len(record.wire),
                self.ring_mr.address + (record.start % self.ring_bytes), self.ring_mr.rkey,
                imm=record.sequence & 0xFFFFFFFF,
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError("sender queue remained full")
                poll_sender()

        deadline = time.monotonic() + timeout
        while self.tx.stats["writes_outstanding"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("sender writes did not complete")
            poll_sender()
            time.sleep(0.0001)

    def send_bad_rkey(self):
        import numpy as np

        self.source = np.zeros(128, dtype=np.uint8)
        self.source_mr = self.ctx.register(self.source.ctypes.data, self.source.nbytes)
        if not self.tx.write(self.source_mr, 0, 128, self.ring_mr.address,
                             self.ring_mr.rkey ^ 0xFFFF, imm=0):
            raise RuntimeError("sender queue unexpectedly full")

    def close(self):
        for name in ("tx", "rx", "source_mr", "credit_mr", "ring_mr"):
            resource = getattr(self, name, None)
            if resource is not None:
                resource.close()
                setattr(self, name, None)
        self.source = None
        self.ring_mem = None
        self.ctx = None


def _loopback_or_skip(ring_bytes, ring_memory):
    try:
        return Loopback(ring_bytes, ring_memory=ring_memory)
    except RuntimeError as error:
        message = str(error)
        if (ring_memory == "gpu" and "ibv_reg_mr" in message
                and "Bad address" in message and "nvidia_peermem" in message):
            pytest.skip("GPU ring registration requires nvidia_peermem on the RDMA device")
        raise


def _build_receiver(loopback, batch_size, max_batch=32, queue_depth=2,
                    stop_after_first_data=False, publication_window=0, release=None,
                    sink_delay_s=0, ring_memory=None):
    import cupy as cp
    import holoscan as hs
    import numpy as np

    from live4d.operators import RdmaRxOp
    from live4d.schemas.decoded_batch import DecodedBatch

    records = []
    ready = threading.Event()
    first_data = threading.Event()
    all_ends = threading.Event()

    class TypedSink(hs.Operator):
        inp = hs.Input("input", DecodedBatch)

        def __init__(self):
            super().__init__()

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            batch = self.inp.receive_data()
            with cp.cuda.Device(0), ctx.stream():
                tensor = cp.from_dlpack(batch.frames)
                pixels = cp.asnumpy(tensor[:len(batch.image_ids)]).copy() \
                    if batch.image_ids else None
            item = {
                "batch_id": int(batch.batch_id),
                "pipeline_ready": bool(batch.pipeline_ready),
                "start": bool(batch.start_of_series),
                "end": bool(batch.end_of_series),
                "series_id": int(batch.series_id),
                "series_images": int(batch.series_images),
                "start_metadata": bytes(batch.start_metadata),
                "image_ids": [int(value) for value in batch.image_ids],
                "pixels": pixels,
            }
            records.append(item)
            if item["pipeline_ready"]:
                ready.set()
            if item["image_ids"]:
                first_data.set()
                if stop_after_first_data:
                    ctx.request_completion()
            if sink_delay_s:
                time.sleep(sink_delay_s)
            if release is not None:
                release(item["batch_id"])
            if item["end"] and sum(record["end"] for record in records) >= 1:
                all_ends.set()

    graph = hs.Graph("rdma-source-loopback", default_clock=hs.RealtimeClock)
    source = RdmaRxOp(
        graph, "rdma_rx", qp_handle=loopback.rx.native_handle,
        ring_address=loopback.ring_address,
        ring_device_address=loopback.ring_device_address,
        ring_bytes=loopback.ring_bytes, ring_memory=ring_memory or loopback.ring_memory,
        credit_address=loopback.credit_mr.address, credit_rkey=loopback.credit_mr.rkey,
        batch_size=batch_size, max_batch=max_batch, device=0,
        publication_window=publication_window,
    )
    sink = graph.op(TypedSink, "typed_sink")
    graph.add_flow(source.output, sink.input, queue_depth=queue_depth)
    graph.partition("receiver").add(source)
    graph.partition("sink").add(sink)
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_tensor_output_device(
        operator_path="rdma_rx", output_port="output", device=hs.execution.DeviceId(0))
    plan = hs.compile(graph, hs.CompileOptions(deployment=deployment))
    if not plan.ok:
        raise RuntimeError(plan.summary())
    return plan, records, ready, first_data, all_ends


class AsyncBatchReleaser:
    def __init__(self, window, consumer_count):
        self.window = window
        self.consumer_count = consumer_count
        self.pending = {}
        self.released = []
        self.max_pending = 0
        self.errors = []
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def add(self, batch_id, event):
        with self.lock:
            events = self.pending.setdefault(batch_id, [])
            events.append(event)
            if len(events) > self.consumer_count:
                raise AssertionError(f"too many consumer events for batch {batch_id}")
            self.max_pending = max(self.max_pending, len(self.pending))

    def _poll(self):
        while not self.stopping.is_set():
            with self.lock:
                ready = [
                    (batch_id, tuple(events))
                    for batch_id, events in self.pending.items()
                    if len(events) == self.consumer_count
                ]
            for batch_id, events in ready:
                try:
                    if all(event.done for event in events):
                        self.window.release(batch_id)
                        with self.lock:
                            self.pending.pop(batch_id, None)
                            self.released.append(batch_id)
                except BaseException as error:
                    self.errors.append(error)
                    self.stopping.set()
                    return
            time.sleep(0.001)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "asynchronous RDMA release poller did not stop"
        assert not self.errors, self.errors


def _async_copy_kernel():
    import cupy as cp

    return cp.RawKernel(r"""
        extern "C" __global__ void delayed_copy(
                const unsigned short* source, unsigned short* destination,
                int count, unsigned long long delay_cycles) {
            unsigned long long started = clock64();
            while (clock64() - started < delay_cycles) { }
            for (int index = threadIdx.x; index < count; index += blockDim.x) {
                destination[index] = source[index];
            }
        }
    """, "delayed_copy")


def _build_async_receiver(loopback, batch_size, *, queue_depth=1,
                          publication_window=0, releaser=None, fanout=False,
                          stop_after_first_data=False, delay_cycles=100_000_000,
                          copy_kernel=None, output_buffers=None,
                          record_input_reuse=False):
    import cupy as cp
    import holoscan as hs

    from live4d.operators import RdmaRxOp
    from live4d.schemas.decoded_batch import DecodedBatch

    records = {"slow": []}
    if fanout:
        records["fast"] = []
    ready = threading.Event()
    first_data = threading.Event()
    kernel = copy_kernel or _async_copy_kernel()
    output_indices = {name: 0 for name in records}

    class AsyncSink(hs.Operator):
        inp = hs.Input("input", DecodedBatch)

        def __init__(self):
            super().__init__()

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            batch = self.inp.receive_data()
            consumer_name = "fast" if "fast_sink" in ctx.operator_path else "slow"
            ids = [int(value) for value in batch.image_ids]
            tensor = None
            view = None
            output = None
            input_ptr = None
            prior_pending_input_ptrs = ()
            with cp.cuda.Device(0), ctx.stream():
                stream = cp.cuda.get_current_stream()
                if ids:
                    tensor = cp.from_dlpack(batch.frames)
                    view = tensor[:len(ids)]
                    if record_input_reuse:
                        input_ptr = int(view.data.ptr)
                        prior_pending_input_ptrs = tuple(
                            previous["input_ptr"]
                            for previous in records[consumer_name]
                            if previous["image_ids"] and not previous["event"].done
                        )
                    if output_buffers is None:
                        output = cp.empty_like(view)
                    else:
                        output = output_buffers[consumer_name][output_indices[consumer_name]]
                        output_indices[consumer_name] += 1
                    kernel((1,), (256,), (view, output, len(view.ravel()),
                                          delay_cycles if consumer_name == "slow" else 0),
                           stream=stream)
                event = cp.cuda.Event(disable_timing=True)
                event.record(stream)
                callback_pending = not event.done
            item = {
                "batch_id": int(batch.batch_id),
                "image_ids": ids,
                "output": output,
                "event": event,
                "callback_pending": callback_pending,
                "end": bool(batch.end_of_series),
            }
            if record_input_reuse and ids:
                item["input_ptr"] = input_ptr
                item["prior_pending_input_ptrs"] = prior_pending_input_ptrs
            if releaser is not None:
                releaser.add(item["batch_id"], event)
            records[consumer_name].append(item)
            if batch.pipeline_ready:
                ready.set()
            if ids:
                first_data.set()
                if stop_after_first_data:
                    ctx.request_completion()
            del tensor, view

    graph = hs.Graph("rdma-async-lifetime", default_clock=hs.RealtimeClock)
    source = RdmaRxOp(
        graph, "rdma_rx", qp_handle=loopback.rx.native_handle,
        ring_address=loopback.ring_address,
        ring_device_address=loopback.ring_device_address,
        ring_bytes=loopback.ring_bytes, ring_memory=loopback.ring_memory,
        credit_address=loopback.credit_mr.address, credit_rkey=loopback.credit_mr.rkey,
        batch_size=batch_size, max_batch=32, device=0,
        publication_window=publication_window,
    )
    sink_types = [("slow", AsyncSink)]
    if fanout:
        sink_types.append(("fast", AsyncSink))
    for name, sink_type in sink_types:
        sink = graph.op(sink_type, f"{name}_sink")
        graph.add_flow(source.output, sink.input, queue_depth=queue_depth)
        graph.partition(f"{name}_partition").add(sink)
    graph.partition("receiver").add(source)
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_tensor_output_device(
        operator_path="rdma_rx", output_port="output", device=hs.execution.DeviceId(0))
    plan = hs.compile(graph, hs.CompileOptions(deployment=deployment))
    if not plan.ok:
        raise RuntimeError(plan.summary())
    return plan, records, ready, first_data


def _assert_async_pixels(records, expected_batches, *, require_pending=True):
    import numpy as np

    data = [record for record in records if record["image_ids"]]
    assert [record["image_ids"] for record in data] == [ids for ids, _ in expected_batches]
    for record, (_, values) in zip(data, expected_batches):
        if require_pending:
            assert record["callback_pending"], "the GPU read completed before the callback returned"
        record["event"].synchronize()
        pixels = record["output"].get()
        expected = np.broadcast_to(
            np.asarray(values, dtype=np.uint16)[:, None, None], pixels.shape)
        np.testing.assert_array_equal(pixels, expected)


def _wait_until(session, predicate, timeout, detail):
    deadline = time.monotonic() + timeout
    while not predicate():
        if session.wait(timeout=0.02):
            raise AssertionError(f"receiver stopped before {detail}: {session.summary()}")
        if time.monotonic() >= deadline:
            session.request_stop()
            raise AssertionError(f"timed out waiting for {detail}: {session.summary()}")


def _run_until_output(plan, ready, predicate, timeout=30.0):
    import holoscan as hs

    with hs.run_async(plan) as session:
        _wait_until(session, ready.is_set, timeout, "READY")
        _wait_until(session, predicate, timeout, "expected RDMA output")
        # END is an acquisition marker. The service source must keep running until asked to stop.
        assert not session.wait(timeout=0.1)
        session.request_stop()
        assert session.wait(timeout=10), session.summary()
        assert session.termination in (
            hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.COMPLETED,
            hs.RunTermination.STOPPED,
        ), session.summary()


def _assert_pixels(records, expected_batches):
    import numpy as np

    data = [record for record in records if record["image_ids"]]
    assert [record["image_ids"] for record in data] == [ids for ids, _ in expected_batches]
    for record, (_, values) in zip(data, expected_batches):
        assert record["pixels"].dtype == np.uint16
        expected = np.broadcast_to(
            np.asarray(values, dtype=np.uint16)[:, None, None], record["pixels"].shape)
        np.testing.assert_array_equal(record["pixels"], expected)


def test_small_ring_credits_before_full_batch_and_preserves_pixels():
    ring_bytes = 37_376  # START and two one-image frames fit; B is three.
    loopback = Loopback(ring_bytes)
    try:
        writer = RingWriter(ring_bytes)
        start_record, _ = _start(writer, 51, 7)
        values = [7, 19, 31, 47, 63, 79, 101]
        for image_id, value in enumerate(values):
            _image(writer, 51, [image_id], [value])
        _end(writer, 51)

        plan, received, ready, _, ends = _build_receiver(loopback, batch_size=3,
                                                         max_batch=1, queue_depth=2)
        check = {"credit_before_full": False}

        def observe_wait(index, credit):
            # The third IMAGE cannot fit unless earlier ring reads were credited. At this
            # point fewer than B images have been emitted, so credit progress is independent
            # of output-batch completion.
            if index == 3 and not check["credit_before_full"]:
                assert ready.wait(10), "receiver did not publish READY before the ring filled"
                deadline = time.monotonic() + 10
                while loopback.credit_position <= start_record.end:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.0001)
                check["snapshot"] = [
                    (record["pipeline_ready"], record["start"], record["end"],
                     record["image_ids"]) for record in received
                ]
                credit = loopback.credit_position
                assert credit > start_record.end, f"credit={credit}, outputs={check['snapshot']}"
                assert not any(record["image_ids"] for record in received)
                check["credit_before_full"] = True

        sender, sender_errors = _start_sender(loopback, writer.records, on_wait=observe_wait)
        try:
            _run_until_output(plan, ready,
                              lambda: ends.is_set() and check["credit_before_full"], timeout=6)
        finally:
            _join_sender(sender, sender_errors, timeout=25)
        assert [len(record["image_ids"]) for record in received if record["image_ids"]] == [3, 3, 1]
        _assert_pixels(received, [([0, 1, 2], values[:3]), ([3, 4, 5], values[3:6]),
                                  ([6], values[6:])])
        assert loopback.credit_position >= writer.position
        batch_ids = [record["batch_id"] for record in received]
        assert all(left < right for left, right in zip(batch_ids, batch_ids[1:]))
    finally:
        loopback.close()


def test_one_transport_frame_splits_at_batch_boundaries():
    import numpy as np

    ring_bytes = 262_144
    loopback = Loopback(ring_bytes)
    window = BatchWindow(1)
    try:
        writer = RingWriter(ring_bytes)
        _start(writer, 72, 9)
        ids = list(range(201, 210))
        values = [5, 17, 33, 65, 129, 11, 23, 41, 87]
        image_record, frame_pixels = _image(writer, 72, ids, values)
        _end(writer, 72)
        plan, received, ready, first_data, ends = _build_receiver(
            loopback, batch_size=2, queue_depth=2, publication_window=window.handle,
            release=window.release, sink_delay_s=0.02)

        sender, sender_errors = _start_sender(loopback, writer.records)
        try:
            import holoscan as hs

            with hs.run_async(plan) as session:
                _wait_until(session, ready.is_set, 20, "READY")
                _wait_until(session, first_data.is_set, 20, "first split batch")
                _wait_until(session, ends.is_set, 30, "END")
                assert not session.wait(timeout=0.1)
                session.request_stop()
                assert session.wait(timeout=10), session.summary()
                assert session.termination in (
                    hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.COMPLETED,
                    hs.RunTermination.STOPPED,
                ), session.summary()
        finally:
            _join_sender(sender, sender_errors, timeout=25)
        data = [record for record in received if record["image_ids"]]
        assert [record["image_ids"] for record in data] == [ids[i:i + 2] for i in range(0, 9, 2)]
        assert [record["pixels"].shape[0] for record in data] == [2, 2, 2, 2, 1]
        pixels = np.concatenate([record["pixels"] for record in data])
        np.testing.assert_array_equal(pixels, frame_pixels)
        assert len(received) == 8  # READY, START, five data batches, and END.
        assert loopback.credit_position >= image_record.end
        assert loopback.credit_position >= writer.position
    finally:
        window.close()
        loopback.close()


def test_shutdown_waits_for_partial_frame_gather_before_ring_release():
    import numpy as np

    loopback = Loopback(1 << 20)
    try:
        writer = RingWriter(loopback.ring_bytes)
        _start(writer, 88, 8)
        image_record, frame_pixels = _image(writer, 88, list(range(8)),
                                            [2, 4, 8, 16, 32, 64, 128, 256])
        plan, received, ready, _, _ = _build_receiver(
            loopback, batch_size=2, stop_after_first_data=True, queue_depth=1)
        sender = None
        sender_errors = []
        try:
            import holoscan as hs

            with hs.run_async(plan) as session:
                _wait_until(session, ready.is_set, 20, "READY")
                sender, sender_errors = _start_sender(loopback, writer.records)
                assert session.wait(timeout=20), "source did not stop after the first output batch"
                assert session.termination in (
                    hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.COMPLETED,
                    hs.RunTermination.STOPPED,
                ), session.summary()
        finally:
            if sender is not None:
                _join_sender(sender, sender_errors, timeout=20)
        data = [record for record in received if record["image_ids"]]
        assert [record["image_ids"] for record in data] == [[0, 1]]
        np.testing.assert_array_equal(data[0]["pixels"], frame_pixels[:2])
        # The frame remainder was not consumed, so stopping must leave its bytes uncredited.
        assert loopback.credit_position <= image_record.start
        assert not any(record["end"] for record in received)
    finally:
        loopback.close()


def test_consecutive_empty_and_partial_series_keep_service_source_active():
    ring_bytes = 1 << 20
    loopback = Loopback(ring_bytes)
    try:
        writer = RingWriter(ring_bytes)
        starts = {}
        starts[10] = _start(writer, 10, 0)
        _end(writer, 10)
        starts[11] = _start(writer, 11, 3)
        _image(writer, 11, [40, 41, 42], [3, 11, 29])
        _end(writer, 11)
        starts[12] = _start(writer, 12, 0)
        _end(writer, 12)

        # Queue every completion before starting the graph so the first CQ poll can drain the
        # whole bounded set. The 1 MiB ring holds the complete sequence without credit pacing.
        loopback.send(writer.records)
        plan, received, ready, _, ends = _build_receiver(loopback, batch_size=2, max_batch=32)
        _run_until_output(plan, ready, lambda: sum(item["end"] for item in received) == 3)

        assert sum(item["pipeline_ready"] for item in received) == 1
        assert [(item["start"], item["end"], item["series_id"]) for item in received
                if item["start"] or item["end"]] == [
                    (True, False, 10), (False, True, 10),
                    (True, False, 11), (False, True, 11),
                    (True, False, 12), (False, True, 12),
                ]
        for series_id, (_, expected_metadata) in starts.items():
            start, = [item for item in received if item["start"] and item["series_id"] == series_id]
            assert start["start_metadata"] == expected_metadata
        data = [item for item in received if item["image_ids"]]
        assert [item["image_ids"] for item in data] == [[40, 41], [42]]
        _assert_pixels(received, [([40, 41], [3, 11]), ([42], [29])])
        assert [item["series_images"] for item in received if item["end"]] == [0, 3, 0]
        assert loopback.credit_position >= writer.position
    finally:
        loopback.close()


def test_failed_receive_completion_fails_the_graph():
    loopback = Loopback(1 << 20)
    try:
        plan, _, ready, _, _ = _build_receiver(loopback, batch_size=2, max_batch=8)
        import holoscan as hs

        with hs.run_async(plan) as session:
            _wait_until(session, ready.is_set, 20, "READY")
            loopback.send_bad_rkey()
            deadline = time.monotonic() + 20
            while not session.wait(timeout=0.02) and time.monotonic() < deadline:
                pass
            assert session.termination not in (
                hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.COMPLETED,
                hs.RunTermination.STOPPED,
            ), session.summary()
            assert "RDMA connection failed" in session.summary()
    finally:
        loopback.close()


@pytest.mark.parametrize("ring_memory", ["gpu", "host_mapped"])
def test_async_fanout_reads_survive_repeated_ring_wraps_and_partial_batch(ring_memory):
    ring_bytes = 37_376
    loopback = _loopback_or_skip(ring_bytes, ring_memory)
    window = BatchWindow(1)
    releaser = AsyncBatchReleaser(window, consumer_count=2)
    try:
        writer = RingWriter(ring_bytes)
        _start(writer, 93, 17)
        values = [((image_id * 29) + 7) % 251 for image_id in range(17)]
        for image_id, value in enumerate(values):
            _image(writer, 93, [image_id], [value])
        _end(writer, 93)
        plan, received, ready, _ = _build_async_receiver(
            loopback, batch_size=3, queue_depth=1, publication_window=window.handle,
            releaser=releaser, fanout=True)

        sender, sender_errors = _start_sender(loopback, writer.records)
        try:
            import holoscan as hs

            with hs.run_async(plan) as session:
                _wait_until(session, ready.is_set, 20, "READY")
                _wait_until(session, lambda: any(item["end"] for item in received["slow"]),
                            45, "END after asynchronous fanout")
                assert not session.wait(timeout=0.1)
                session.request_stop()
                assert session.wait(timeout=10), session.summary()
                assert session.termination in (
                    hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.COMPLETED,
                    hs.RunTermination.STOPPED,
                ), session.summary()
        finally:
            _join_sender(sender, sender_errors, timeout=25)

        expected = [(list(range(start, min(start + 3, 17))),
                     values[start:min(start + 3, 17)]) for start in range(0, 17, 3)]
        _assert_async_pixels(received["slow"], expected)
        _assert_async_pixels(received["fast"], expected, require_pending=False)
        assert writer.wrap_count >= 3
        assert releaser.max_pending <= 1
        assert len(releaser.released) >= len(expected)
        assert loopback.credit_position >= writer.position
    finally:
        releaser.close()
        window.close()
        loopback.close()


def test_async_split_transport_frame_preserves_partial_tail():
    loopback = Loopback(1 << 20, ring_memory="host_mapped")
    window = BatchWindow(1)
    releaser = AsyncBatchReleaser(window, consumer_count=1)
    try:
        writer = RingWriter(loopback.ring_bytes)
        _start(writer, 96, 9)
        ids = list(range(101, 110))
        values = [5, 17, 33, 65, 129, 11, 23, 41, 87]
        _image(writer, 96, ids, values)
        _end(writer, 96)
        plan, received, ready, _ = _build_async_receiver(
            loopback, batch_size=2, queue_depth=1, publication_window=window.handle,
            releaser=releaser)
        sender, sender_errors = _start_sender(loopback, writer.records)
        try:
            import holoscan as hs

            with hs.run_async(plan) as session:
                _wait_until(session, ready.is_set, 20, "READY")
                _wait_until(session, lambda: any(item["end"] for item in received["slow"]),
                            30, "END after asynchronous frame split")
                assert not session.wait(timeout=0.1)
                session.request_stop()
                assert session.wait(timeout=10), session.summary()
                assert session.termination in (
                    hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.COMPLETED,
                    hs.RunTermination.STOPPED,
                ), session.summary()
        finally:
            _join_sender(sender, sender_errors, timeout=20)

        expected = [(ids[index:index + 2], values[index:index + 2])
                    for index in range(0, len(ids), 2)]
        _assert_async_pixels(received["slow"], expected)
        assert [len(item["image_ids"]) for item in received["slow"]
                if item["image_ids"]] == [2, 2, 2, 2, 1]
        assert loopback.credit_position >= writer.position
    finally:
        releaser.close()
        window.close()
        loopback.close()


@pytest.mark.parametrize("ring_memory", ["gpu", "host_mapped"])
def test_stop_drains_delayed_gpu_read_after_callback_returns(ring_memory):
    loopback = _loopback_or_skip(1 << 20, ring_memory)
    window = BatchWindow(1)
    releaser = AsyncBatchReleaser(window, consumer_count=1)
    try:
        writer = RingWriter(loopback.ring_bytes)
        _start(writer, 94, 8)
        image_record, _ = _image(writer, 94, list(range(8)),
                                 [2, 4, 8, 16, 32, 64, 128, 256])
        plan, received, ready, first_data = _build_async_receiver(
            loopback, batch_size=2, queue_depth=1, publication_window=window.handle,
            releaser=releaser, stop_after_first_data=True)
        import holoscan as hs

        with hs.run_async(plan) as session:
            _wait_until(session, ready.is_set, 20, "READY")
            sender, sender_errors = _start_sender(loopback, writer.records)
            try:
                assert session.wait(timeout=25), "source did not stop after the first GPU read"
                assert session.termination in (
                    hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.COMPLETED,
                    hs.RunTermination.STOPPED,
                ), session.summary()
                assert first_data.is_set()
            finally:
                _join_sender(sender, sender_errors, timeout=25)

        _assert_async_pixels(received["slow"], [([0, 1], [2, 4])])
        data = [item for item in received["slow"] if item["image_ids"]]
        assert len(data) == 1 and data[0]["callback_pending"]
        assert loopback.credit_position <= image_record.start
        assert not any(item["end"] for item in received["slow"])
    finally:
        releaser.close()
        window.close()
        loopback.close()


def test_receive_error_preserves_async_gpu_read_until_graph_teardown():
    loopback = Loopback(1 << 20, ring_memory="host_mapped")
    try:
        writer = RingWriter(loopback.ring_bytes)
        _start(writer, 95, 2)
        _image(writer, 95, [50, 51], [17, 29])
        plan, received, ready, first_data = _build_async_receiver(
            loopback, batch_size=2, queue_depth=1, delay_cycles=2_000_000_000)
        import holoscan as hs

        sender, sender_errors = _start_sender(loopback, writer.records)
        try:
            with hs.run_async(plan) as session:
                _wait_until(session, ready.is_set, 20, "READY")
                _wait_until(session, first_data.is_set, 20, "first asynchronous GPU read")
                _join_sender(sender, sender_errors, timeout=20)
                data_item = next(item for item in received["slow"] if item["image_ids"])
                assert not data_item["event"].done
                loopback.send_bad_rkey()
                deadline = time.monotonic() + 20
                while not session.wait(timeout=0.02) and time.monotonic() < deadline:
                    pass
                assert session.termination not in (
                    hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.COMPLETED,
                    hs.RunTermination.STOPPED,
                ), session.summary()
                assert "RDMA connection failed" in session.summary()
        finally:
            if sender.is_alive():
                _join_sender(sender, sender_errors, timeout=20)

        _assert_async_pixels(received["slow"], [([50, 51], [17, 29])])
        assert data_item["callback_pending"]
    finally:
        loopback.close()


def test_async_slow_fanout_exhausts_bounded_sdk_pool_without_reusing_live_inputs():
    ring_bytes = 2 << 20
    loopback = Loopback(ring_bytes, ring_memory="host_mapped")
    try:
        writer = RingWriter(ring_bytes)
        image_ids = list(range(300, 364))
        values = [(image_id * 17 + 3) % 251 for image_id in image_ids]
        _start(writer, 98, len(image_ids))
        for image_id, value in zip(image_ids, values):
            _image(writer, 98, [image_id], [value])
        _end(writer, 98)

        import cupy as cp

        kernel = _async_copy_kernel()
        kernel.compile()
        output_buffers = {
            consumer: [cp.empty((1, 96, 96), dtype=cp.uint16) for _ in image_ids]
            for consumer in ("slow", "fast")
        }
        # No publication window: the source may publish independently of the slow consumer's
        # callback-stream events. Its finite SDK tensor pool must reject exhaustion rather than
        # recycle storage still being read by those asynchronous kernels.
        plan, received, ready, _ = _build_async_receiver(
            loopback, batch_size=1, queue_depth=8, fanout=True,
            delay_cycles=500_000_000, copy_kernel=kernel,
            output_buffers=output_buffers, record_input_reuse=True)
        import holoscan as hs

        sender, sender_errors = _start_sender(loopback, writer.records)
        try:
            with hs.run_async(plan) as session:
                _wait_until(session, ready.is_set, 20, "READY")
                deadline = time.monotonic() + 12
                completed = False
                while time.monotonic() < deadline:
                    if session.wait(timeout=0.01):
                        completed = True
                        break
                    if (any(item["end"] for item in received["slow"])
                            or sum(bool(item["image_ids"]) for item in received["slow"])
                            == len(image_ids)):
                        break
                failure_summary = session.summary()
                if not completed:
                    session.request_stop()
                    assert session.wait(timeout=20), session.summary()
                _join_sender(sender, sender_errors, timeout=20)
                slow_data = [item for item in received["slow"] if item["image_ids"]]
                fast_data = [item for item in received["fast"] if item["image_ids"]]
                progress = {
                    "slow_ids": [item["image_ids"] for item in slow_data],
                    "slow_pending": [item["callback_pending"] for item in slow_data],
                    "slow_prior_pending": [item["prior_pending_input_ptrs"]
                                           for item in slow_data],
                    "fast_ids": [item["image_ids"] for item in fast_data],
                }
                assert completed and "kResourceExhausted" in failure_summary, (
                    f"source did not report bounded tensor-pool exhaustion: "
                    f"{failure_summary}; progress={progress}")
        finally:
            if sender.is_alive():
                _join_sender(sender, sender_errors, timeout=20)

        assert slow_data, "the slow partition did not receive asynchronous data"
        assert fast_data, "the fast partition did not receive asynchronous data"
        assert len(slow_data) <= len(image_ids)
        assert len(fast_data) <= len(image_ids)
        assert all(item["callback_pending"] for item in slow_data)
        assert any(item["prior_pending_input_ptrs"] for item in slow_data)
        for item in slow_data:
            assert item["input_ptr"] not in item["prior_pending_input_ptrs"]
        values_by_id = dict(zip(image_ids, values))
        for consumer, data in (("slow", slow_data), ("fast", fast_data)):
            expected = [(item["image_ids"], [values_by_id[index]
                                               for index in item["image_ids"]])
                        for item in data]
            _assert_async_pixels(received[consumer], expected, require_pending=False)
    finally:
        loopback.close()
