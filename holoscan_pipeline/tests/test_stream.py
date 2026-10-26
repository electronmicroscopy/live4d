"""Public ZMQ series, decoded pixels, and shared 4D-STEM output contracts."""
import signal
import json
from datetime import datetime, timezone
import threading
import time

import cbor2
import cupy as cp
import numpy as np
import pytest
import zmq

from live4d.app import PipelineApp
from live4d.config import PipelineConfig
from live4d.algorithms.virtual_detectors import build_masks, integrate


@pytest.mark.parametrize("shape", [(96, 96), (192, 192)])
def test_virtual_detector_algorithms_match_independent_sums(shape):
    height, width = shape
    values = np.arange(3 * height * width, dtype=np.uint32).reshape(3, height, width)
    frames = ((values * 13 + values // 7) % 11).astype(np.uint16)
    bf_radius, adf_inner, adf_outer = 16.0, 24.0, 48.0
    masks = build_masks(bf_radius, adf_inner, adf_outer, shape=shape)
    actual = cp.asnumpy(integrate(cp.asarray(frames), masks))

    y, x = np.ogrid[:height, :width]
    radius_squared = (y - (height - 1) / 2) ** 2 + (x - (width - 1) / 2) ** 2
    bf_mask = radius_squared <= bf_radius ** 2
    adf_mask = ((radius_squared >= adf_inner ** 2)
                & (radius_squared <= adf_outer ** 2))
    expected = np.stack([
        frames[:, bf_mask].astype(np.uint64).sum(axis=1),
        frames[:, adf_mask].astype(np.uint64).sum(axis=1),
    ], axis=1).astype(np.float32)

    assert masks.dtype == cp.float32
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def start_message(series_id, count, shape=(96, 96), dtype="uint16"):
    return dict(type="start", series_id=series_id, image_size_x=shape[1], image_size_y=shape[0],
                image_dtype=dtype, number_of_images=count, frame_time=0.001,
                channels=["threshold_1"], acquisition_note="preserved")


def image_message(image_id, value, shape=(96, 96)):
    height, width = shape
    elements_left = height * width
    data = bytearray((height * width * 2).to_bytes(8, "big") + (8192).to_bytes(4, "big"))
    while elements_left:
        elements = min(4096, elements_left)
        shuffled = b"".join(bytes([255 if (value >> bit) & 1 else 0]) * (elements // 8)
                             for bit in range(16))
        length = len(shuffled) - 15
        block = bytes([0xf0]) + bytes([255]) * (length // 255) + bytes([length % 255]) + shuffled
        data += len(block).to_bytes(4, "big") + block
        elements_left -= elements
    payload = cbor2.CBORTag(40, [[height, width], cbor2.CBORTag(69,
               cbor2.CBORTag(56500, ["bslz4", 2, bytes(data)]))])
    return dict(type="image", image_id=image_id, data={"threshold_1": payload})


def run_series(series, *, batch_size=4, scan=(4, 2), service=False, delay=False,
               processing_delay=0, queue_depth=4, defer_readback=False, expect_error=False,
               devices=None, extra_observer=None, inspect_app=None):
    observed, results, errors, starts, batch_shapes = [], [], [], [], []
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.setsockopt(zmq.LINGER, 0)
    sender.setsockopt(zmq.SNDTIMEO, 10000)
    port = sender.bind_to_random_port("tcp://127.0.0.1")

    def observer(batch, positioned):
        batch_shapes.append((int(batch.series_id), bool(batch.start_of_series),
                             bool(batch.end_of_series), tuple(batch.frames.shape)))
        if batch.start_of_series:
            starts.append(bytes(batch.start_metadata))
        if len(batch.image_ids):
            if defer_readback:
                pixels = cp.array(positioned.images, copy=True)
                observed.append((int(batch.batch_id), int(batch.series_id), list(batch.image_ids), pixels))
            else:
                pixels = cp.asnumpy(positioned.images)
                observed.append((int(batch.series_id), list(batch.image_ids), pixels))
            time.sleep(processing_delay)
        if batch.end_of_series:
            results.append(positioned.result)
        if extra_observer is not None:
            extra_observer(batch, positioned)

    config = PipelineConfig(batch_size=batch_size, queue_depth=queue_depth, devices=devices,
                            scan=dict(width=scan[0], height=scan[1]),
                            execute=dict(mode="service" if service else "finite", seconds=3 if service else 10),
                            zmq=dict(endpoint=f"tcp://127.0.0.1:{port}"))
    app = PipelineApp(config, observer=observer)
    if inspect_app is not None:
        inspect_app(app)
    stopping = threading.Event()

    def receive():
        try:
            app.run(stop_event=stopping)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=receive)
    thread.start()
    try:
        assert app.ready.wait(20), errors
        for entry in series:
            sid, ids = entry[:2]
            shape = entry[2] if len(entry) > 2 else (96, 96)
            image_shape = entry[3] if len(entry) > 3 else shape
            dtype = entry[4] if len(entry) > 4 else "uint16"
            sender.send(cbor2.dumps(cbor2.CBORTag(
                55799, start_message(sid, len(ids), shape, dtype))))
            for image_id in ids:
                sender.send(cbor2.dumps(image_message(image_id, image_id + 1, image_shape)))
                if delay:
                    time.sleep(0.005)
            sender.send(cbor2.dumps(dict(type="end", series_id=sid)))
        thread.join(timeout=15)
        assert not thread.is_alive()
        if expect_error:
            assert errors, errors
            return observed, results, app.plan, batch_shapes
        assert not errors, errors
        assert sorted(starts) == sorted([cbor2.dumps(cbor2.CBORTag(
            55799, start_message(entry[0], len(entry[1]),
                                 entry[2] if len(entry) > 2 else (96, 96),
                                 entry[4] if len(entry) > 4 else "uint16")))
                          for entry in (series if service else series[:1])])
        if defer_readback:
            observed[:] = [(batch_id, sid, ids, cp.asnumpy(pixels))
                           for batch_id, sid, ids, pixels in observed]
    finally:
        stopping.set()
        thread.join(timeout=5)
        sender.close()
        context.term()
    return observed, results, app.plan, batch_shapes


@pytest.mark.parametrize("count", [0, 1, 4, 9, 33])
def test_order_pixels_and_end_of_series(count):
    observed, results, _, _ = run_series([(7, list(range(count)))], delay=count != 33,
                                        processing_delay=0.02 if count == 33 else 0,
                                        queue_depth=2 if count == 33 else 4)
    assert [len(ids) for _, ids, _ in observed] == [4] * (count // 4) + ([count % 4] if count % 4 else [])
    for sid, ids, pixels in observed:
        assert sid == 7 and pixels.dtype == np.uint16
        np.testing.assert_array_equal(pixels, np.broadcast_to(np.array(ids)[:, None, None] + 1, pixels.shape))
    result, = results
    assert result["images"] == count and result["gap_events"] == 0
    placed = min(count, 8)
    assert result["positions"] == placed
    assert result["bf_sum"] == 812 * sum(range(1, placed + 1))
    assert result["adf_sum"] == 5428 * sum(range(1, placed + 1))


def test_consecutive_empty_partial_series_and_scan_gaps(capfd):
    observed, results, _, batch_shapes = run_series(
        [(1, []), (2, [0, 3, 6, 9, 10], (192, 192)), (3, [0]),
         (4, [], (192, 192)), (5, [0, 1])], service=True)
    assert [(sid, ids) for sid, ids, _ in observed] == [
        (2, [0, 3, 6, 9]), (2, [10]), (3, [0]), (5, [0, 1])]
    assert [result["series_id"] for result in results] == [1, 2, 3, 4, 5]
    assert [result["positions"] for result in results] == [0, 3, 1, 0, 2]
    assert [result["announced_images"] for result in results] == [0, 5, 1, 0, 2]
    assert [pixels.shape[1:] for _, _, pixels in observed] == [
        (192, 192), (192, 192), (96, 96), (96, 96)]
    for _, ids, pixels in observed:
        expected = np.broadcast_to(np.asarray(ids, dtype=np.uint16)[:, None, None] + 1,
                                   pixels.shape)
        np.testing.assert_array_equal(pixels, expected)
    for sid, is_start, is_end, shape in batch_shapes:
        expected = (192, 192) if sid in (2, 4) else (96, 96)
        if is_start or is_end:
            assert shape == (1, *expected)
    assert results[1]["bf_sum"] == 812 * (1 + 4 + 7)
    y, x = np.ogrid[:192, :192]
    radius_squared = (y - 95.5) ** 2 + (x - 95.5) ** 2
    adf_pixels = ((radius_squared >= 24 ** 2) & (radius_squared <= 48 ** 2)).sum()
    assert results[1]["adf_sum"] == adf_pixels * (1 + 4 + 7)
    assert results[2]["adf_sum"] == 5428
    records = []
    for line in capfd.readouterr().out.splitlines():
        if line.startswith(("[series-start] ", "[series-end] ", "[series-result] ")):
            event, payload = line.split(" ", 1)
            records.append((event, json.loads(payload)))
    assert len(records) == 15
    for index, count in enumerate((0, 5, 1, 0, 2)):
        sid = index + 1
        entries = [(event, row) for event, row in records if row["series_id"] == sid]
        assert [event for event, _ in entries] == [
            "[series-start]", "[series-end]", "[series-result]"]
        start, end, result = [row for _, row in entries]
        assert start["image_shape"] == ([192, 192] if sid in (2, 4) else [96, 96])
        assert (start["scan_width"], start["scan_height"]) == (4, 2)
        assert start["announced_images"] == end["received_images"] == count
        assert result["processed_images"] == result["images"] == count
        assert end["positions"] == result["positions"] == results[index]["positions"]
        assert all(row["device"] == 0 for _, row in entries)
        assert start["stage"] == end["stage"] == "processing"
        timestamps = [datetime.fromisoformat(row["timestamp_utc"]) for _, row in entries]
        assert all(stamp.utcoffset() == timezone.utc.utcoffset(stamp) for stamp in timestamps)
        assert timestamps == sorted(timestamps)


def test_deferred_readback_preserves_many_batches_and_totals():
    count = 257
    observed, results, _, _ = run_series(
        [(8, list(range(count)))], batch_size=4, scan=(17, 16), queue_depth=2,
        defer_readback=True)

    assert len(observed) == 65
    assert [len(ids) for _, _, ids, _ in observed] == [4] * 64 + [1]
    batch_ids = [batch_id for batch_id, _, _, _ in observed]
    assert all(current < following for current, following in zip(batch_ids, batch_ids[1:]))
    assert [image_id for _, _, ids, _ in observed for image_id in ids] == list(range(count))
    for _, sid, ids, pixels in observed:
        assert sid == 8 and pixels.dtype == np.uint16
        np.testing.assert_array_equal(
            pixels, np.broadcast_to(np.array(ids)[:, None, None] + 1, pixels.shape))

    result, = results
    total = sum(range(1, count + 1))
    assert result["images"] == count
    assert result["positions"] == count
    assert result["gap_events"] == 0 and result["missing_images"] == 0
    assert result["batch_sizes"] == {4: 64, 1: 1}
    assert result["bf_sum"] == 812 * total
    assert result["adf_sum"] == 5428 * total


def test_idle_sigterm(capsys):
    app = PipelineApp(PipelineConfig(scan=dict(width=2, height=2)))
    def stop():
        assert app.ready.wait(20)
        signal.raise_signal(signal.SIGTERM)
    thread = threading.Thread(target=stop)
    thread.start()
    termination = app.run()
    thread.join(timeout=5)
    assert termination.name == "STOPPED"
    assert "[holoscan] ready" in capsys.readouterr().out


@pytest.mark.parametrize("shape,dtype", [((128, 128), "uint16"), ((96, 192), "uint16"),
                                          ((96, 96), "uint32")])
def test_unsupported_start_geometry_is_rejected(shape, dtype):
    run_series([(9, [], shape, shape, dtype)], expect_error=True)


@pytest.mark.parametrize("start_shape,image_shape", [
    ((192, 192), (96, 96)),
    ((96, 96), (192, 48)),
])
def test_image_geometry_must_match_start(start_shape, image_shape):
    run_series([(10, [0], start_shape, image_shape)], expect_error=True)


@pytest.mark.parametrize("devices", [(0,), (0, 0), (0, 1), (1, 0)])
def test_zmq_routes_whole_acquisitions_and_reuses_branches(devices):
    if cp.cuda.runtime.getDeviceCount() <= max(devices):
        pytest.skip("requires two visible GPUs")
    series = [(101, [], (96, 96)), (102, [0, 1, 2, 3, 4], (192, 192)),
              (103, [0], (192, 192)), (104, [0, 1, 2], (96, 96)),
              (105, [], (192, 192)), (106, [0, 1], (96, 96))]
    seen, last_ids, finished = {}, {}, []
    lanes = {entry[0]: index % len(devices) for index, entry in enumerate(series)}

    def observe(batch, positioned):
        device = cp.cuda.runtime.getDevice()
        bid, sid = int(batch.batch_id), int(batch.series_id)
        if sid not in lanes:  # READY precedes any acquisition.
            return
        lane = lanes[sid]
        assert bid > last_ids.get(lane, 0)
        last_ids[lane] = bid
        seen.setdefault(sid, set()).add(device)

    def inspect(app):
        # Repeated device ordinals still own independent branches and admission.
        finished.append(app.zmq.finished)
        for resources in (app.branch_ready, app.windows, app.zmq.results,
                          app.zmq.inputs, app.zmq.acquisitions):
            assert len(resources) == len({id(item) for item in resources}) == len(devices)

    observed, results, plan, _ = run_series(series, batch_size=2, service=True,
        devices=devices, queue_depth=2, extra_observer=observe, inspect_app=inspect)
    assert plan.ok, plan.summary()
    assert finished[0] == [sum(lane == index for lane in lanes.values())
                           for index in range(len(devices))]
    for index, (sid, ids, shape) in enumerate(series):
        device = devices[index % len(devices)]
        assert seen[sid] == {device}
        result, = [r for r in results if r["series_id"] == sid]
        assert result["device"] == device
        assert result["images"] == result["announced_images"] == result["positions"] == len(ids)
        assert result["gap_events"] == result["missing_images"] == 0
        y, x = np.ogrid[:shape[0], :shape[1]]
        r2 = (2*y-(shape[0]-1))**2 + (2*x-(shape[1]-1))**2
        assert result["bf_sum"] == int((r2 <= 4*16**2).sum()) * sum(i+1 for i in ids)
        assert result["adf_sum"] == int(((r2 >= 4*24**2) & (r2 <= 4*48**2)).sum()) * sum(i+1 for i in ids)
        for s, batch_ids, pixels in observed:
            if s == sid:
                assert pixels.shape[1:] == shape
                np.testing.assert_array_equal(pixels, np.broadcast_to(
                    np.asarray(batch_ids, dtype=np.uint16)[:, None, None] + 1, pixels.shape))


def test_zmq_second_gpu_overlaps_and_first_gpu_revisit_waits_for_completion():
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two visible GPUs")
    hold = cp.RawKernel(r'''
        extern "C" __global__ void hold(unsigned long long cycles) {
            unsigned long long start = clock64();
            while (clock64()-start < cycles) { }
        }
    ''', "hold")
    with cp.cuda.Device(0):
        hold((1,), (1,), (np.uint64(1),))
        cp.cuda.get_current_stream().synchronize()
        rate = cp.cuda.runtime.getDeviceProperties(0)["clockRate"]
    event = cp.cuda.Event(disable_timing=True)
    times, pending = {}, []

    def observe(batch, positioned):
        sid = int(batch.series_id)
        if sid == 201 and len(batch.image_ids):
            hold((1,), (1,), (np.uint64(rate * 1000 * 2),))
            event.record()
        if sid == 202 and len(batch.image_ids):
            with cp.cuda.Device(0):
                pending.append(not event.done)
        if batch.start_of_series:
            times[(sid, "start")] = time.monotonic()
        if batch.end_of_series:
            times[(sid, "done")] = time.monotonic()

    run_series([(201, [0]), (202, [0]), (203, [0])], devices=[0, 1],
               service=True, batch_size=1, queue_depth=2, extra_observer=observe)
    assert pending == [True]
    assert times[(202, "done")] < times[(201, "done")]
    assert times[(203, "start")] >= times[(201, "done")]


@pytest.mark.parametrize("image_id", [55335, 55336, 55337])
def test_parser_does_not_interpret_image_id_bytes_as_tensor_tag(image_id):
    # D8 28 inside uint16(55336) used to be mistaken for the tensor tag 40.
    observed, results, _, _ = run_series([(301, [image_id])])
    assert observed[0][1] == [image_id]
    assert results[0]["images"] == 1


@pytest.mark.parametrize("noise", [b"\xd8\x28\xd9\xdc\xb4", cbor2.CBORTag(40, [[1, 2], b"fake"])])
def test_parser_ignores_tag_patterns_in_unrelated_metadata(noise):
    import ctypes as c
    from live4d.operators import _library
    image = image_message(55336, 42)
    message = dict(type=image["type"], image_id=image["image_id"], noise=noise, data=image["data"])
    wire = cbor2.dumps(cbor2.CBORTag(55799, message))
    raw = c.create_string_buffer(wire)
    payload = c.c_void_p()
    size, original, element = c.c_size_t(), c.c_size_t(), c.c_size_t()
    width, height, image_id = c.c_uint32(), c.c_uint32(), c.c_uint64()
    result = _library.fast_extract_compressed_bslz4(raw, c.c_size_t(len(wire)),
        c.byref(payload), c.byref(size), c.byref(original), c.byref(element),
        c.byref(width), c.byref(height), c.byref(image_id))
    assert result == 0
    assert (width.value, height.value, image_id.value, original.value, element.value) == (96, 96, 55336, 18432, 2)


def test_sample_validator_accepts_completed_async_result(tmp_path, monkeypatch):
    import json
    import sys
    from live4d import validate
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.setsockopt(zmq.LINGER, 0)
    sender.setsockopt(zmq.SNDTIMEO, 10000)
    port = sender.bind_to_random_port("tcp://127.0.0.1")
    reference = tmp_path / "reference.npz"
    report = tmp_path / "report.json"
    np.savez(reference, ids=np.array([0]), frames=np.ones((1, 96, 96), dtype=np.uint16))
    producers = []

    def construct(config, **kwargs):
        app = PipelineApp(config, **kwargs)
        def send():
            assert app.ready.wait(20)
            for item in (start_message(401, 1), image_message(0, 1), dict(type="end")):
                sender.send(cbor2.dumps(item))
        thread = threading.Thread(target=send)
        producers.append(thread)
        thread.start()
        return app

    monkeypatch.setattr(validate, "PipelineApp", construct)
    monkeypatch.setattr(sys, "argv", ["validate", "--reference", str(reference),
        "--expected-frames", "1", "--report", str(report), "--endpoint", f"tcp://127.0.0.1:{port}"])
    try:
        with pytest.raises(SystemExit) as done:
            validate.main()
        assert done.value.code == 0
        result = json.loads(report.read_text())
        assert result["success"] and result["cpu_reference_samples_checked"] == 1
        assert result["images"] == 1 and result["termination"] == "STOPPED"
    finally:
        for thread in producers:
            thread.join(20)
            assert not thread.is_alive()
        sender.close()
        context.term()


@pytest.mark.parametrize("devices", [(0,), (0, 1)])
def test_zmq_maximum_compressed_metadata_batch_fits_channel_envelope(devices):
    if cp.cuda.runtime.getDeviceCount() <= max(devices):
        pytest.skip("requires two visible GPUs")
    observed, results, _, _ = run_series([(501, list(range(1000)))],
        devices=devices, batch_size=1000, scan=(40, 25))
    assert len(observed) == 1 and len(observed[0][1]) == 1000
    result, = results
    assert result["images"] == result["positions"] == 1000
    assert result["bf_sum"] == 812 * sum(range(1, 1001))
    assert result["adf_sum"] == 5428 * sum(range(1, 1001))



def test_finite_zmq_processes_first_acquisition_even_if_later_gpu_could_finish_first():
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two visible GPUs")
    hold = cp.RawKernel(r'''
        extern "C" __global__ void hold(unsigned long long cycles) {
            unsigned long long start = clock64();
            while (clock64()-start < cycles) { }
        }
    ''', "hold")
    devices = [1, 0]
    with cp.cuda.Device(devices[0]):
        hold((1,), (1,), (np.uint64(1),))
        cp.cuda.get_current_stream().synchronize()
        rate = cp.cuda.runtime.getDeviceProperties(devices[0])["clockRate"]
    seen = []

    def observe(batch, positioned):
        seen.append(int(batch.series_id))
        if len(batch.image_ids):
            hold((1,), (1,), (np.uint64(rate * 1000),))

    observed, results, _, _ = run_series([(601, [0]), (602, [])],
        devices=devices, batch_size=1, queue_depth=2, extra_observer=observe)
    assert set(seen) == {601}
    assert [r["series_id"] for r in results] == [601]
    assert results[0]["device"] == devices[0]
    assert results[0]["images"] == 1
    assert [sid for sid, _, _ in observed] == [601]
