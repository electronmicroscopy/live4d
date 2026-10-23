"""Public ZMQ series, decoded pixels, and shared 4D-STEM output contracts."""
import signal
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


def test_virtual_detector_algorithms_match_independent_sums():
    values = np.arange(3 * 96 * 96, dtype=np.uint32).reshape(3, 96, 96)
    frames = ((values * 13 + values // 7) % 11).astype(np.uint16)
    bf_radius, adf_inner, adf_outer = 16.0, 24.0, 48.0
    masks = build_masks(bf_radius, adf_inner, adf_outer)
    actual = cp.asnumpy(integrate(cp.asarray(frames), masks))

    y, x = np.ogrid[:96, :96]
    radius_squared = (y - 47.5) ** 2 + (x - 47.5) ** 2
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


def start_message(series_id, count):
    return dict(type="start", series_id=series_id, image_size_x=96, image_size_y=96,
                image_dtype="uint16", number_of_images=count, frame_time=0.001,
                channels=["threshold_1"], acquisition_note="preserved")


def image_message(image_id, value):
    data = bytearray((18432).to_bytes(8, "big") + (8192).to_bytes(4, "big"))
    for elements in (4096, 4096, 1024):
        shuffled = b"".join(bytes([255 if (value >> bit) & 1 else 0]) * (elements // 8)
                             for bit in range(16))
        length = len(shuffled) - 15
        block = bytes([0xf0]) + bytes([255]) * (length // 255) + bytes([length % 255]) + shuffled
        data += len(block).to_bytes(4, "big") + block
    payload = cbor2.CBORTag(40, [[96, 96], cbor2.CBORTag(69,
               cbor2.CBORTag(56500, ["bslz4", 2, bytes(data)]))])
    return dict(type="image", image_id=image_id, data={"threshold_1": payload})


def run_series(series, *, batch_size=4, scan=(4, 2), service=False, delay=False,
               processing_delay=0, queue_depth=4, observer=None):
    observed, results, errors, starts = [], [], [], []
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.setsockopt(zmq.LINGER, 0)
    sender.setsockopt(zmq.SNDTIMEO, 10000)
    port = sender.bind_to_random_port("tcp://127.0.0.1")

    def observe(batch, positioned):
        if batch.start_of_series:
            starts.append(bytes(batch.start_metadata))
        if len(batch.image_ids):
            if observer is None:
                pixels = cp.asnumpy(positioned.images)
                observed.append((int(batch.series_id), list(batch.image_ids), pixels))
            else:
                observer(batch, positioned)
            time.sleep(processing_delay)
        if batch.end_of_series:
            results.append(positioned.result)

    config = PipelineConfig(batch_size=batch_size, queue_depth=queue_depth,
                            scan=dict(width=scan[0], height=scan[1]),
                            execute=dict(mode="service" if service else "finite", seconds=3 if service else 10),
                            zmq=dict(endpoint=f"tcp://127.0.0.1:{port}"))
    app = PipelineApp(config, observer=observe)
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
        for sid, ids in series:
            sender.send(cbor2.dumps(cbor2.CBORTag(55799, start_message(sid, len(ids)))))
            for image_id in ids:
                sender.send(cbor2.dumps(image_message(image_id, image_id + 1)))
                if delay:
                    time.sleep(0.005)
            sender.send(cbor2.dumps(dict(type="end", series_id=sid)))
        thread.join(timeout=15)
        assert not thread.is_alive()
        assert not errors, errors
        assert starts == [cbor2.dumps(cbor2.CBORTag(55799, start_message(sid, len(ids)))) for sid, ids in series]
    finally:
        stopping.set()
        thread.join(timeout=5)
        sender.close()
        context.term()
    return observed, results, app.plan


@pytest.mark.parametrize("count", [0, 1, 4, 9, 33])
def test_order_pixels_and_end_of_series(count):
    observed, results, _ = run_series([(7, list(range(count)))], delay=count != 33,
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


def test_consecutive_empty_partial_series_and_scan_gaps():
    observed, results, _ = run_series([(1, []), (2, [0, 3, 6, 9, 10]), (3, [0])], service=True)
    assert [(sid, ids) for sid, ids, _ in observed] == [(2, [0, 3, 6, 9]), (2, [10]), (3, [0])]
    assert [result["series_id"] for result in results] == [1, 2, 3]
    assert [result["positions"] for result in results] == [0, 3, 1]
    assert results[1]["bf_sum"] == 812 * (1 + 4 + 7)
    assert results[2]["adf_sum"] == 5428


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


def _delayed_copy_kernel():
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


def _queue_copy(kernel, source, destination, stream, delay_cycles):
    kernel((1,), (256,), (source, destination, int(source.size), int(delay_cycles)),
           stream=stream)


def test_async_observer_retains_sdk_tensor_and_cupy_views_across_callbacks():
    kernel = _delayed_copy_kernel()
    retained = []
    reimports = []

    def observer(batch, positioned):
        ids = [int(value) for value in batch.image_ids]
        if not ids:
            return
        stream = cp.cuda.get_current_stream()
        if retained:
            previous = retained[-1]
            tensor = cp.from_dlpack(previous["tensor"])
            view = tensor[:len(previous["ids"])]
            output = cp.empty_like(view)
            _queue_copy(kernel, view, output, stream, 0)
            event = cp.cuda.Event(disable_timing=True)
            event.record(stream)
            reimports.append({"ids": previous["ids"], "output": output, "event": event})

        view = positioned.images
        output = cp.empty_like(view)
        _queue_copy(kernel, view, output, stream, 200_000_000)
        event = cp.cuda.Event(disable_timing=True)
        event.record(stream)
        retained.append({
            "ids": ids,
            "tensor": positioned.tensor,
            "view": view,
            "output": output,
            "event": event,
            "callback_pending": not event.done,
        })

    observed, _, _ = run_series(
        [(19, list(range(9)))], batch_size=3, scan=(4, 3), queue_depth=2,
        observer=observer)

    assert not observed
    assert len(retained) == 3
    assert all(item["callback_pending"] for item in retained)
    assert [item["ids"] for item in retained] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert [item["ids"] for item in reimports] == [[0, 1, 2], [3, 4, 5]]
    for item in retained:
        item["event"].synchronize()
        expected = np.broadcast_to(
            (np.asarray(item["ids"], dtype=np.uint16) + 1)[:, None, None],
            item["output"].shape)
        np.testing.assert_array_equal(cp.asnumpy(item["output"]), expected)
        np.testing.assert_array_equal(cp.asnumpy(item["view"]), expected)
    for item in reimports:
        item["event"].synchronize()
        expected = np.broadcast_to(
            (np.asarray(item["ids"], dtype=np.uint16) + 1)[:, None, None],
            item["output"].shape)
        np.testing.assert_array_equal(cp.asnumpy(item["output"]), expected)


def test_completion_tracker_rejects_over_capacity_until_drain():
    from live4d.completion import CompletionTracker

    tracker = CompletionTracker(capacity=1, device=0)
    tracker.start()
    kernel = _delayed_copy_kernel()
    source = cp.arange(256, dtype=cp.uint16)
    first_output = cp.empty_like(source)
    second_output = cp.empty_like(source)
    first_stream = cp.cuda.Stream(non_blocking=True)
    second_stream = cp.cuda.Stream(non_blocking=True)
    completed = []
    try:
        with first_stream:
            _queue_copy(kernel, source, first_output, first_stream, 600_000_000)
        tracker.track(1, first_stream, owners=(source, first_output),
                      on_complete=lambda completed_at: completed.append(1))

        with second_stream:
            _queue_copy(kernel, source, second_output, second_stream, 1_000_000)
        with pytest.raises(RuntimeError, match="Asynchronous completion capacity exhausted"):
            tracker.track(2, second_stream, owners=(source, second_output),
                          on_complete=lambda completed_at: completed.append(2))

        assert completed == []
        tracker.drain()
        tracker.check()
        assert completed == [1]

        with second_stream:
            _queue_copy(kernel, source, second_output, second_stream, 0)
        tracker.track(3, second_stream, owners=(source, second_output),
                      on_complete=lambda completed_at: completed.append(3))
        tracker.drain()
        tracker.check()
        assert completed == [1, 3]
    finally:
        tracker.close()


def test_completion_tracker_reports_callback_error_during_close():
    from live4d.completion import CompletionTracker

    tracker = CompletionTracker(capacity=1, device=0)
    tracker.start()
    stream = cp.cuda.Stream(non_blocking=True)
    source = cp.arange(32, dtype=cp.uint16)
    output = cp.empty_like(source)
    kernel = _delayed_copy_kernel()

    def fail_completion(completed_at):
        raise ValueError("completion callback failed")

    with stream:
        _queue_copy(kernel, source, output, stream, 1_000_000)
    tracker.track(4, stream, owners=(source, output), on_complete=fail_completion)
    with pytest.raises(RuntimeError, match="Asynchronous GPU completion failed") as error:
        tracker.close()
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "completion callback failed"
