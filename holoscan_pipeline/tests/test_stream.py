"""Public ZMQ series, decoded pixels, and shared 4D-STEM output contracts."""
import signal
import threading
import time

import cbor2
import cupy as cp
import numpy as np
import pytest
import zmq

from dectris_holoscan.app import PipelineApp
from dectris_holoscan.config import PipelineConfig


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
               processing_delay=0, queue_depth=4):
    observed, results, errors, starts = [], [], [], []
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.setsockopt(zmq.LINGER, 0)
    sender.setsockopt(zmq.SNDTIMEO, 10000)
    port = sender.bind_to_random_port("tcp://127.0.0.1")

    def observer(batch, positioned):
        if batch.start_of_series:
            starts.append(bytes(batch.start_metadata))
        if len(batch.image_ids):
            pixels = cp.asnumpy(positioned.images)
            observed.append((int(batch.series_id), list(batch.image_ids), pixels))
            time.sleep(processing_delay)
        if batch.end_of_series:
            results.append(positioned.result)

    config = PipelineConfig(batch_size=batch_size, queue_depth=queue_depth,
                            scan=dict(width=scan[0], height=scan[1]),
                            execute=dict(mode="service" if service else "finite", seconds=3 if service else 10),
                            zmq=dict(endpoint=f"tcp://127.0.0.1:{port}"))
    app = PipelineApp(config, observer=observer)
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
