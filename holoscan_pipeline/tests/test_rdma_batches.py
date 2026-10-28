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
    def __init__(self, wire, sequence, start, end, ring=0, imm=None):
        self.wire = wire
        self.sequence = sequence
        self.start = start
        self.end = end
        self.ring = ring
        self.imm = sequence & 0xFFFFFFFF if imm is None else imm


class RingWriter:
    """Build the aligned frame stream that imagemux writes into a ring.

    ``version=3`` writes ring ``ring`` of a multi-ring receiver. ``records`` is shared by the
    writers of all rings, so it keeps imagemux's write order across rings."""

    def __init__(self, ring_bytes, *, version=2, ring=0, records=None):
        self.ring_bytes = ring_bytes
        self.version = version
        self.ring = ring
        self.position = 0
        self.sequence = 0
        self.records = [] if records is None else records

    def write(self, kind, data=b"", *, series_id=0, image=None):
        from dectris.imagemux import FrameHeader, FrameKind, ImageHeader, frame_span, immediate

        offset = self.position % self.ring_bytes
        span = frame_span(len(data))
        if kind != FrameKind.WRAP and offset + span > self.ring_bytes:
            self.write(FrameKind.WRAP, series_id=series_id)
            offset = 0

        header = FrameHeader(kind, self.sequence, self.position, len(data), series_id,
                             image if image is not None else ImageHeader(), self.ring, self.version)
        start = self.position
        wire = header.encode() + data
        self.position += self.ring_bytes - offset if kind == FrameKind.WRAP else span
        self.records.append(RingRecord(wire, self.sequence, start, self.position, self.ring,
                                       immediate(self.ring, self.sequence, self.version)))
        self.sequence += 1
        return self.records[-1]


def _ring_writers(ring_bytes, rings):
    records = []
    return [RingWriter(ring_bytes, version=3, ring=i, records=records) for i in range(rings)], records


class BatchWindow:
    def __init__(self, capacity):
        import ctypes
        from live4d import operators

        library_path = Path(operators.__file__).with_name("libdectris_holoscan_ops.so")
        self.library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        self.library.dectris_batch_window_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self.library.dectris_batch_window_create.restype = ctypes.c_void_p
        self.library.dectris_batch_window_destroy.argtypes = [ctypes.c_void_p]
        self.library.dectris_batch_window_destroy.restype = None
        self.library.dectris_batch_window_release.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                                             ctypes.c_uint32]
        self.library.dectris_batch_window_release.restype = None
        self.handle = self.library.dectris_batch_window_create(capacity, 1)
        if not self.handle:
            raise RuntimeError("native batch window allocation failed")

    def release(self, batch_id):
        self.library.dectris_batch_window_release(self.handle, batch_id, 0)

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


def _start(writer, series_id, announced, shape=(96, 96), dtype="uint16"):
    import cbor2
    from dectris.imagemux import FrameKind

    metadata = cbor2.dumps(cbor2.CBORTag(55799, {
        "type": "start", "series_id": series_id, "image_size_x": shape[1],
        "image_size_y": shape[0], "image_dtype": dtype, "number_of_images": announced,
    }))
    record = writer.write(FrameKind.START, metadata, series_id=series_id)
    return record, metadata


def _image(writer, series_id, image_ids, values, shape=(96, 96)):
    import numpy as np
    from dectris.imagemux import Compression, DataType, FrameKind, ImageHeader

    frames = np.stack([
        np.full(shape, value, dtype=np.uint16) for value in values
    ])
    payload = frames.tobytes()
    image = ImageHeader(
        image_id=image_ids[0], batch_id=0, batch_size=len(image_ids), dim_x=shape[1], dim_y=shape[0],
        dtype=DataType.UINT16, compression=Compression.NONE,
        uncompressed_size=len(payload), channel=0, channel_count=1,
    )
    return writer.write(FrameKind.IMAGE, payload, series_id=series_id, image=image), frames


def _end(writer, series_id):
    from dectris.imagemux import FrameKind

    return writer.write(FrameKind.END, series_id=series_id)


class Loopback:
    """Two local queue pairs with registered host-mapped receiver rings.

    ``rings`` equal rings share one registration, ring i at ``i * ring_bytes``, with one
    credit word each, as ``VerbsConn`` and imagemux protocol 3 lay them out."""

    def __init__(self, ring_bytes, rings=1):
        import cupy as cp
        import numpy as np
        from dectris import imxrdma
        from dectris.dispatch import mapped_device_pointer

        self.cp = cp
        self.np = np
        self.ctx = imxrdma.Context(local_addr=LOCAL_ADDR)
        self.ring_mem = cp.cuda.PinnedMemory(ring_bytes * rings, cp.cuda.runtime.hostAllocMapped)
        self.ring_address = self.ring_mem.ptr
        self.ring_device_address = mapped_device_pointer(self.ring_address, 0)
        self.ring_mr = self.ctx.register(self.ring_address, ring_bytes * rings, remote_write=True)
        self.rings = rings
        self.credit = np.zeros(rings, dtype=np.uint64)
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

    def ring_credit(self, ring):
        import ctypes

        return ctypes.c_uint64.from_address(self.credit.ctypes.data + 8 * ring).value

    @property
    def credit_position(self):
        return self.ring_credit(0)

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
            while record.end - self.ring_credit(record.ring) > self.ring_bytes:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"credit did not reach frame {record.sequence} "
                                       f"of ring {record.ring}")
                if not waited and on_wait is not None:
                    on_wait(index, self.ring_credit(record.ring))
                    waited = True
                poll_sender()
                time.sleep(0.0001)
            while not self.tx.write(
                self.source_mr, source_offset, len(record.wire),
                self.ring_mr.address + record.ring * self.ring_bytes
                + (record.start % self.ring_bytes), self.ring_mr.rkey, imm=record.imm,
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


def _build_receiver(loopback, batch_size, max_batch=32, queue_depth=2,
                    stop_after_first_data=False, publication_window=0, release=None,
                    sink_delay_s=0, session_control=None, protocol=2):
    import cupy as cp
    import holoscan as hs
    import numpy as np

    from live4d.operators import RdmaRingsRxOp, RdmaRxOp
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
                "frame_shape": tuple(batch.frames.shape),
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
    common = dict(
        qp_handle=0 if session_control else loopback.rx.native_handle,
        ring_address=loopback.ring_address,
        ring_device_address=loopback.ring_device_address,
        ring_bytes=loopback.ring_bytes, ring_memory="host_mapped",
        credit_address=0 if session_control else loopback.credit_mr.address,
        credit_rkey=0 if session_control else loopback.credit_mr.rkey,
        batch_size=batch_size, max_batch=max_batch, device=0,
        publication_window=publication_window,
    )
    if protocol == 3:
        source = RdmaRingsRxOp(graph, "rdma_rx", ring_count=loopback.rings, **common)
    else:
        source = RdmaRxOp(graph, "rdma_rx",
                          session_handle=session_control.handle if session_control else 0, **common)
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
        starts[11] = _start(writer, 11, 3, (192, 192))
        _image(writer, 11, [40, 41, 42], [3, 11, 29], (192, 192))
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
        assert [item["frame_shape"] for item in received] == [
            (1, 96, 96), (1, 96, 96), (1, 96, 96), (1, 192, 192),
            (2, 192, 192), (2, 192, 192), (1, 192, 192), (1, 96, 96), (1, 96, 96),
        ]
        assert [item["series_images"] for item in received if item["end"]] == [0, 3, 0]
        assert loopback.credit_position >= writer.position
    finally:
        loopback.close()


@pytest.mark.parametrize("start_shape,image_shape,dtype,expected", [
    ((128, 128), (128, 128), "uint16", "Invalid START image geometry"),
    ((96, 96), (192, 192), "uint16", "RDMA image geometry does not match START"),
    ((96, 96), (192, 48), "uint16", "RDMA image geometry does not match START"),
    ((96, 96), (96, 96), "uint32", "Invalid START image geometry"),
])
def test_rdma_rejects_invalid_start_or_image_geometry(start_shape, image_shape, dtype, expected):
    loopback = Loopback(1 << 20)
    try:
        writer = RingWriter(loopback.ring_bytes)
        _start(writer, 99, 1, start_shape, dtype)
        if image_shape is not None:
            _image(writer, 99, [0], [7], image_shape)
        plan, _, ready, _, _ = _build_receiver(loopback, batch_size=2, max_batch=8)
        import holoscan as hs

        with hs.run_async(plan) as session:
            _wait_until(session, ready.is_set, 20, "READY")
            sender, sender_errors = _start_sender(loopback, writer.records)
            deadline = time.monotonic() + 20
            while not session.wait(timeout=0.02) and time.monotonic() < deadline:
                pass
            _join_sender(sender, sender_errors, timeout=20)
            assert session.termination not in (
                hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.COMPLETED,
                hs.RunTermination.STOPPED,
            ), session.summary()
            assert expected in session.summary()
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


def test_session_reconnect_reuses_ring_without_restarting_operator():
    """Fresh QP/credit epochs over one ring; batch IDs and bounded publication persist."""
    import holoscan as hs
    from dectris.imagemux import RemoteRegion
    from live4d.rdma_session import RdmaSession

    loopback = Loopback(1 << 20)
    control = RdmaSession()
    window = BatchWindow(1)
    original_ring = (loopback.ring_address, loopback.ring_mr.rkey)
    try:
        plan, records, ready, _, _ = _build_receiver(loopback, batch_size=2,
            session_control=control, publication_window=window.handle,
            release=window.release, sink_delay_s=0.01)
        expected = []
        with hs.run_async(plan) as session:
            _wait_until(session, ready.is_set, 20, "READY with no attached QP")
            assert control.snapshot()["starts"] == 1
            for index, (shape, count) in enumerate([((96, 96), 0), ((192, 192), 3), ((96, 96), 5)]):
                if index:
                    # Only after detach acknowledgment may either old QP be destroyed.
                    loopback.rx.close()
                    loopback.tx.close()
                    loopback.credit[:] = 0
                    loopback.rx = loopback.ctx.create_qp(recv_depth=64, send_depth=64, signal_every=1)
                    loopback.tx = loopback.ctx.create_qp(recv_depth=1, send_depth=64, signal_every=1)
                    loopback.rx.connect(loopback.tx.endpoint)
                    loopback.tx.connect(loopback.rx.endpoint)
                    assert loopback.rx.post_recvs(64) == 64
                credit = RemoteRegion(loopback.credit_mr.address, 8, loopback.credit_mr.rkey)
                generation = control.attach(loopback.rx, credit)
                with pytest.raises(RuntimeError):
                    control.attach(loopback.rx, credit)
                with pytest.raises(RuntimeError):
                    control.detach(generation)
                _wait_until(session, lambda: control.snapshot()["attached"] == generation,
                            20, "session attach")
                with pytest.raises(RuntimeError):
                    control.detach(generation - 1)
                writer = RingWriter(loopback.ring_bytes)
                _start(writer, 100 + index, count, shape)
                if count:
                    _image(writer, 100 + index, list(range(count)), list(range(20, 20+count)), shape)
                    expected.extend([(list(range(i, min(i+2, count))),
                                      list(range(20+i, 20+min(i+2, count))))
                                     for i in range(0, count, 2)])
                _end(writer, 100 + index)
                sender, errors = _start_sender(loopback, writer.records)
                _wait_until(session, lambda: control.snapshot()["drained"] == generation,
                            20, "GPU reads and credit sends drained")
                _join_sender(sender, errors, 20)
                status = control.snapshot()
                assert status["position"] == status["safe_credit"] == status["sent_credit"] == writer.position
                assert status["series_id"] == 100 + index
                assert loopback.rx.stats["writes_outstanding"] == 0
                control.detach(generation)
                _wait_until(session, lambda: control.snapshot()["detached"] == generation,
                            20, "detach acknowledgment")
                control.release_detached(generation)
                _wait_until(session, lambda: sum(r["end"] for r in records) == index + 1,
                            20, "downstream END")
                assert not session.wait(timeout=0)
                assert (loopback.ring_address, loopback.ring_mr.rkey) == original_ring
            assert sum(r["pipeline_ready"] for r in records) == 1
            assert [r["batch_id"] for r in records] == sorted({r["batch_id"] for r in records})
            _assert_pixels(records, expected)
            session.request_stop()
            assert session.wait(timeout=10), session.summary()
        assert control.snapshot()["starts"] == control.snapshot()["stops"] == 1
    finally:
        control.close()
        window.close()
        loopback.close()


def test_session_geometry_failure_never_reports_drained():
    import holoscan as hs
    from dectris.imagemux import RemoteRegion
    from live4d.rdma_session import RdmaSession

    loopback = Loopback(1 << 20)
    control = RdmaSession()
    try:
        plan, _, ready, _, _ = _build_receiver(loopback, batch_size=2, session_control=control)
        with hs.run_async(plan) as session:
            _wait_until(session, ready.is_set, 20, "READY")
            generation = control.attach(loopback.rx,
                RemoteRegion(loopback.credit_mr.address, 8, loopback.credit_mr.rkey))
            _wait_until(session, lambda: control.snapshot()["attached"] == generation, 20, "attach")
            writer = RingWriter(loopback.ring_bytes)
            _start(writer, 201, 1, (96, 96))
            _image(writer, 201, [0], [7], (192, 192))
            sender, errors = _start_sender(loopback, writer.records)
            assert session.wait(timeout=20), "malformed IMAGE did not fail the source"
            _join_sender(sender, errors, 20)
            assert "RDMA image geometry does not match START" in session.summary()
            assert control.snapshot()["drained"] == 0
        assert control.snapshot()["state"] == "stopped"
    finally:
        control.close()


# ---- Protocol 3: several rings, series written round robin ----------------------------------


def _wait_for_failure(session, timeout=20):
    import holoscan as hs

    deadline = time.monotonic() + timeout
    while not session.wait(timeout=0.02) and time.monotonic() < deadline:
        pass
    assert session.termination not in (
        hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.COMPLETED,
        hs.RunTermination.STOPPED,
    ), session.summary()
    return session.summary()


def test_rings_deliver_round_robin_series_in_order_at_both_sizes():
    ring_bytes = 1 << 20
    loopback = Loopback(ring_bytes, rings=3)
    try:
        writers, records = _ring_writers(ring_bytes, 3)
        # Series go to rings 0, 1, 2, 0, ...: sizes change between series, one is empty, and
        # every non-empty series ends with a partial batch of B=2.
        series = [(20, (96, 96), [0, 1, 2]), (21, (192, 192), []),
                  (22, (192, 192), [5, 6, 7, 8, 9]), (23, (96, 96), [4])]
        starts = {}
        expected = []
        for index, (series_id, shape, ids) in enumerate(series):
            writer = writers[index % 3]
            starts[series_id] = _start(writer, series_id, len(ids), shape)
            values = [series_id * 10 + image_id for image_id in ids]
            if ids:
                _image(writer, series_id, ids, values, shape)
            _end(writer, series_id)
            expected += [(ids[i:i + 2], values[i:i + 2]) for i in range(0, len(ids), 2)]

        # All completions queue before the graph starts, so later rings arrive while earlier
        # series are still unread.
        loopback.send(records)
        plan, received, ready, _, _ = _build_receiver(loopback, batch_size=2, protocol=3)
        _run_until_output(plan, ready, lambda: sum(item["end"] for item in received) == 4)

        assert [(item["start"], item["end"], item["series_id"]) for item in received
                if item["start"] or item["end"]] == [
                    (start, not start, series_id) for series_id, _, _ in series
                    for start in (True, False)]
        for series_id, (_, metadata) in starts.items():
            start, = [item for item in received if item["start"] and item["series_id"] == series_id]
            assert start["start_metadata"] == metadata
        _assert_pixels(received, expected)
        shapes = {series_id: shape for series_id, shape, _ in series}
        assert received[0]["pipeline_ready"]
        for item in received[1:]:
            assert item["frame_shape"][1:] == shapes[item["series_id"]]
        assert [len(item["image_ids"]) for item in received if item["image_ids"]] == [2, 1, 2, 2, 1, 1]
        assert [item["series_images"] for item in received if item["end"]] == [3, 0, 5, 1]
        assert [loopback.ring_credit(i) for i in range(3)] == [w.position for w in writers]
    finally:
        loopback.close()


def test_rings_return_credit_to_each_rings_word():
    # Each ring holds START plus two images, so series 40 needs ring 0 credits before its first
    # batch of four is full; series 41 then goes to ring 1, credited through its own word.
    ring_bytes = 37_376
    loopback = Loopback(ring_bytes, rings=2)
    try:
        writers, records = _ring_writers(ring_bytes, 2)
        _start(writers[0], 40, 6)
        values = [3, 5, 7, 11, 13, 17]
        for image_id, value in enumerate(values):
            _image(writers[0], 40, [image_id], [value])
        _end(writers[0], 40)
        _start(writers[1], 41, 1)
        _image(writers[1], 41, [0], [99])
        _end(writers[1], 41)

        plan, received, ready, _, _ = _build_receiver(loopback, batch_size=4, max_batch=1,
                                                      protocol=3)
        sender, sender_errors = _start_sender(loopback, records)
        try:
            _run_until_output(plan, ready, lambda: sum(item["end"] for item in received) == 2,
                              timeout=20)
        finally:
            _join_sender(sender, sender_errors, timeout=25)
        _assert_pixels(received, [([0, 1, 2, 3], values[:4]), ([4, 5], values[4:]), ([0], [99])])
        assert loopback.ring_credit(0) == writers[0].position > ring_bytes
        assert loopback.ring_credit(1) == writers[1].position
    finally:
        loopback.close()


def test_rings_fail_when_the_sender_moves_on_without_end():
    loopback = Loopback(1 << 20, rings=2)
    try:
        writers, records = _ring_writers(loopback.ring_bytes, 2)
        _start(writers[0], 30, 2)
        _image(writers[0], 30, [0], [1])  # imagemux dropped the rest of series 30
        _start(writers[1], 31, 1)
        plan, _, ready, _, _ = _build_receiver(loopback, batch_size=2, protocol=3)
        import holoscan as hs

        with hs.run_async(plan) as session:
            _wait_until(session, ready.is_set, 20, "READY")
            sender, sender_errors = _start_sender(loopback, records)
            summary = _wait_for_failure(session)
            _join_sender(sender, sender_errors, timeout=20)
        assert "before END of series 30 in ring 0" in summary, summary
    finally:
        loopback.close()


@pytest.mark.parametrize("ring,imm,expected", [
    (1, 0x0200_0000, "immediate names ring 2 of 2"),
    (1, 0x0000_0000, "frame header in ring 0 does not match"),
    (0, 0x0000_0001, "ring 0 announced sequence 1, expected 0"),
])
def test_rings_reject_misannounced_frames(ring, imm, expected):
    loopback = Loopback(1 << 20, rings=2)
    try:
        writers, records = _ring_writers(loopback.ring_bytes, 2)
        _start(writers[ring], 50, 0)
        records[0].imm = imm
        plan, _, ready, _, _ = _build_receiver(loopback, batch_size=2, protocol=3)
        import holoscan as hs

        with hs.run_async(plan) as session:
            _wait_until(session, ready.is_set, 20, "READY")
            sender, sender_errors = _start_sender(loopback, records)
            summary = _wait_for_failure(session)
            _join_sender(sender, sender_errors, timeout=20)
        assert expected in summary, summary
    finally:
        loopback.close()


@pytest.mark.parametrize("start_shape,image_shape,expected", [
    ((96, 96), (192, 192), "RDMA image geometry does not match START"),
    ((192, 192), (96, 96), "RDMA image geometry does not match START"),
    ((128, 128), None, "Invalid START image geometry"),
])
def test_rings_reject_invalid_geometry_in_a_later_ring(start_shape, image_shape, expected):
    loopback = Loopback(1 << 20, rings=2)
    try:
        writers, records = _ring_writers(loopback.ring_bytes, 2)
        _start(writers[0], 60, 1)
        _image(writers[0], 60, [0], [1])
        _end(writers[0], 60)
        _start(writers[1], 61, 1, start_shape)
        if image_shape is not None:
            _image(writers[1], 61, [0], [2], image_shape)
        plan, _, ready, _, _ = _build_receiver(loopback, batch_size=2, protocol=3)
        import holoscan as hs

        with hs.run_async(plan) as session:
            _wait_until(session, ready.is_set, 20, "READY")
            sender, sender_errors = _start_sender(loopback, records)
            summary = _wait_for_failure(session)
            _join_sender(sender, sender_errors, timeout=20)
        assert expected in summary, summary
    finally:
        loopback.close()


# ---- Protocol 3 with GPU lanes: host-staged batches, one upload operator per lane ------------


def _build_ring_lanes(loopback, lanes, batch_size, max_batch=32, queue_depth=4, hold=None,
                      devices=None):
    import cupy as cp
    import holoscan as hs

    from live4d.operators import RawUploadGpuOp, RdmaRingsHostRxOp
    from live4d.schemas.decoded_batch import DecodedBatch

    devices = devices or (0,) * lanes
    records = [[] for _ in range(lanes)]
    ready_lanes = set()
    ready_lock = threading.Lock()
    all_ready = threading.Event()
    finished = [0] * lanes
    inputs = [BatchWindow(queue_depth - 1) for _ in range(lanes)]
    outputs = [BatchWindow(queue_depth - 1) for _ in range(lanes)]
    admission = [BatchWindow(1) for _ in range(lanes)]

    def make_sink(lane):
        class LaneSink(hs.Operator):
            inp = hs.Input("input", DecodedBatch)

            def __init__(self):
                super().__init__()

            def setup(self, spec):
                spec.input(self.inp)

            def contract(self):
                return hs.Contract().trigger(hs.OnEach(self.inp))

            def compute(self, ctx):
                batch = self.inp.receive_data()
                with cp.cuda.Device(devices[lane]), ctx.stream():
                    tensor = cp.from_dlpack(batch.frames)
                    pixels = cp.asnumpy(tensor[:len(batch.image_ids)]).copy() \
                        if batch.image_ids else None
                    tensor_device = int(tensor.device.id)
                item = {
                    "device": tensor_device,
                    "batch_id": int(batch.batch_id),
                    "pipeline_ready": bool(batch.pipeline_ready),
                    "start": bool(batch.start_of_series),
                    "end": bool(batch.end_of_series),
                    "series_id": int(batch.series_id),
                    "series_images": int(batch.series_images),
                    "image_ids": [int(value) for value in batch.image_ids],
                    "frame_shape": tuple(batch.frames.shape),
                    "pixels": pixels,
                }
                records[lane].append(item)
                if item["pipeline_ready"]:
                    with ready_lock:
                        ready_lanes.add(lane)
                        if len(ready_lanes) == lanes:
                            all_ready.set()
                outputs[lane].release(item["batch_id"])
                if item["end"]:
                    if hold and hold.get(lane) is not None:
                        hold[lane].wait(30)
                    finished[lane] += 1
                    admission[lane].release(finished[lane])

        return LaneSink

    graph = hs.Graph("rdma-rings-lanes-loopback", default_clock=hs.RealtimeClock)
    source = RdmaRingsHostRxOp(
        graph, "rdma_rx", qp_handle=loopback.rx.native_handle, ring_address=loopback.ring_address,
        ring_bytes=loopback.ring_bytes, ring_count=loopback.rings,
        credit_address=loopback.credit_mr.address, credit_rkey=loopback.credit_mr.rkey,
        publication_windows=[window.handle for window in inputs],
        acquisition_windows=[window.handle for window in admission], acquisition_limit=0,
        batch_size=batch_size, max_batch=max_batch)
    graph.partition("receiver").add(source)
    deployment = hs.execution.DeploymentBinding()
    for lane in range(lanes):
        upload = RawUploadGpuOp(graph, f"upload{lane}", publication_window=outputs[lane].handle,
                                input_window=inputs[lane].handle, destination=lane,
                                device=devices[lane])
        sink = graph.op(make_sink(lane), f"sink{lane}")
        graph.add_flow(source.output, upload.input, queue_depth=queue_depth)
        graph.add_flow(upload.output, sink.input, queue_depth=queue_depth)
        graph.partition(f"upload{lane}").add(upload)
        graph.partition(f"sink{lane}").add(sink)
        deployment.bind_tensor_output_device(operator_path=f"upload{lane}", output_port="output",
                                             device=hs.execution.DeviceId(devices[lane]))
    plan = hs.compile(graph, hs.CompileOptions(deployment=deployment))
    if not plan.ok:
        raise RuntimeError(plan.summary())
    return plan, records, all_ready, inputs + outputs + admission


def _close_windows(windows):
    for window in windows or ():
        window.close()


@pytest.mark.parametrize("lanes", [2, 3])
def test_ring_lanes_route_series_round_robin_in_order_at_both_sizes(lanes):
    ring_bytes = 1 << 20
    loopback = Loopback(ring_bytes, rings=3)
    windows = None
    try:
        writers, sent = _ring_writers(ring_bytes, 3)
        series = [(20, (96, 96), [0, 1, 2]), (21, (192, 192), []),
                  (22, (192, 192), [5, 6, 7, 8, 9]), (23, (96, 96), [4]), (24, (96, 96), [0, 1])]
        expected = {lane: [] for lane in range(lanes)}
        for index, (series_id, shape, ids) in enumerate(series):
            writer = writers[index % 3]
            _start(writer, series_id, len(ids), shape)
            values = [series_id * 10 + image_id for image_id in ids]
            if ids:
                _image(writer, series_id, ids, values, shape)
            _end(writer, series_id)
            expected[index % lanes] += [(ids[i:i + 2], values[i:i + 2])
                                        for i in range(0, len(ids), 2)]

        loopback.send(sent)
        plan, received, ready, windows = _build_ring_lanes(loopback, lanes, batch_size=2)
        _run_until_output(plan, ready,
                          lambda: sum(item["end"] for lane in received for item in lane) == len(series))

        for lane in range(lanes):
            mine = [entry for index, entry in enumerate(series) if index % lanes == lane]
            assert received[lane][0]["pipeline_ready"]
            marks = [(item["start"], item["end"], item["series_id"]) for item in received[lane]
                     if item["start"] or item["end"]]
            assert marks == [(start, not start, entry[0]) for entry in mine for start in (True, False)]
            _assert_pixels(received[lane], expected[lane])
            shapes = {entry[0]: entry[1] for entry in mine}
            for item in received[lane][1:]:
                assert item["frame_shape"][1:] == shapes[item["series_id"]]
            assert [item["series_images"] for item in received[lane] if item["end"]] == [
                len(entry[2]) for entry in mine]
        assert [loopback.ring_credit(i) for i in range(3)] == [w.position for w in writers]
    finally:
        _close_windows(windows)
        loopback.close()


def test_ring_lanes_hold_the_next_series_until_its_lane_is_free():
    import holoscan as hs

    ring_bytes = 1 << 20
    loopback = Loopback(ring_bytes, rings=3)
    hold = {0: threading.Event()}
    windows = None
    try:
        writers, sent = _ring_writers(ring_bytes, 3)
        for index, series_id in enumerate((30, 31, 32)):  # lanes 0, 1, 0
            _start(writers[index], series_id, 2)
            _image(writers[index], series_id, [0, 1], [series_id, series_id])
            _end(writers[index], series_id)
        loopback.send(sent)
        plan, received, ready, windows = _build_ring_lanes(loopback, 2, batch_size=2, hold=hold)

        def ended(lane, series_id):
            return any(item["end"] and item["series_id"] == series_id for item in received[lane])

        with hs.run_async(plan) as session:
            _wait_until(session, ready.is_set, 30, "READY")
            _wait_until(session, lambda: ended(0, 30) and ended(1, 31), 30, "series 30 and 31")
            time.sleep(0.5)
            # Lane 0 has not released series 30, so series 32 stays in its ring.
            assert not any(item["series_id"] == 32 for item in received[0])
            hold[0].set()
            _wait_until(session, lambda: ended(0, 32), 30, "series 32 once lane 0 is free")
            session.request_stop()
            assert session.wait(timeout=10), session.summary()
        _assert_pixels(received[0], [([0, 1], [30, 30]), ([0, 1], [32, 32])])
        _assert_pixels(received[1], [([0, 1], [31, 31])])
    finally:
        hold[0].set()
        _close_windows(windows)
        loopback.close()


def test_ring_lanes_return_credit_on_the_host_copy():
    # Each ring holds START plus two images, so series 40 needs ring 0 credits before its first
    # batch of four is full. The host copy returns them without any GPU event.
    ring_bytes = 37_376
    loopback = Loopback(ring_bytes, rings=2)
    windows = None
    try:
        writers, sent = _ring_writers(ring_bytes, 2)
        _start(writers[0], 40, 6)
        values = [3, 5, 7, 11, 13, 17]
        for image_id, value in enumerate(values):
            _image(writers[0], 40, [image_id], [value])
        _end(writers[0], 40)
        _start(writers[1], 41, 1)
        _image(writers[1], 41, [0], [99])
        _end(writers[1], 41)

        plan, received, ready, windows = _build_ring_lanes(loopback, 2, batch_size=4, max_batch=1)
        sender, sender_errors = _start_sender(loopback, sent)
        try:
            _run_until_output(plan, ready,
                              lambda: sum(item["end"] for lane in received for item in lane) == 2,
                              timeout=20)
        finally:
            _join_sender(sender, sender_errors, timeout=25)
        _assert_pixels(received[0], [([0, 1, 2, 3], values[:4]), ([4, 5], values[4:])])
        _assert_pixels(received[1], [([0], [99])])
        assert loopback.ring_credit(0) == writers[0].position > ring_bytes
        assert loopback.ring_credit(1) == writers[1].position
    finally:
        _close_windows(windows)
        loopback.close()


def test_ring_lanes_upload_each_lane_to_its_own_gpu():
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two visible GPUs")
    ring_bytes = 1 << 20
    loopback = Loopback(ring_bytes, rings=2)
    windows = None
    try:
        writers, sent = _ring_writers(ring_bytes, 2)
        series = [(50, (96, 96), [0, 1, 2]), (51, (192, 192), [0, 1]),
                  (52, (96, 96), [5]), (53, (192, 192), [0, 1, 2, 3])]
        expected = {0: [], 1: []}
        for index, (series_id, shape, ids) in enumerate(series):
            writer = writers[index % 2]
            _start(writer, series_id, len(ids), shape)
            values = [series_id * 10 + image_id for image_id in ids]
            _image(writer, series_id, ids, values, shape)
            _end(writer, series_id)
            expected[index % 2] += [(ids[i:i + 2], values[i:i + 2])
                                    for i in range(0, len(ids), 2)]

        loopback.send(sent)
        plan, received, ready, windows = _build_ring_lanes(loopback, 2, batch_size=2,
                                                           devices=(0, 1))
        _run_until_output(plan, ready,
                          lambda: sum(item["end"] for lane in received for item in lane) == len(series))

        for lane in range(2):
            assert {item["device"] for item in received[lane]} == {lane}
            _assert_pixels(received[lane], expected[lane])
        assert [loopback.ring_credit(i) for i in range(2)] == [w.position for w in writers]
    finally:
        _close_windows(windows)
        loopback.close()
