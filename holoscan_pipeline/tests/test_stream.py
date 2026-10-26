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


def start_message(series_id, count, shape=(96, 96), dtype="uint16", appendix=None):
    message = dict(type="start", series_id=series_id, image_size_x=shape[1], image_size_y=shape[0],
                   image_dtype=dtype, number_of_images=count, frame_time=0.001,
                   channels=["threshold_1"], acquisition_note="preserved")
    if appendix is not None:
        message["header_appendix"] = appendix
    return message


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
               appendices=None, placements=None):
    observed, results, errors, starts, batch_shapes = [], [], [], [], []
    appendices = appendices or {}
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
        if placements is not None and len(batch.image_ids):
            ids = np.asarray(batch.image_ids)[positioned.selected].tolist()
            placements.extend((int(batch.series_id), image_id, tuple(position))
                              for image_id, position in zip(ids, positioned.positions.tolist()))
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
        for entry in series:
            sid, ids = entry[:2]
            shape = entry[2] if len(entry) > 2 else (96, 96)
            image_shape = entry[3] if len(entry) > 3 else shape
            dtype = entry[4] if len(entry) > 4 else "uint16"
            sender.send(cbor2.dumps(cbor2.CBORTag(
                55799, start_message(sid, len(ids), shape, dtype, appendices.get(sid)))))
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
        assert starts == [cbor2.dumps(cbor2.CBORTag(
            55799, start_message(entry[0], len(entry[1]),
                                 entry[2] if len(entry) > 2 else (96, 96),
                                 entry[4] if len(entry) > 4 else "uint16",
                                 appendices.get(entry[0]))))
                          for entry in series]
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


def test_image_id_containing_tag_40_bytes_is_accepted():
    # 55336 is CBOR 0x19 0xD8 0x28; the shape scan must not take it for the tag-40 array.
    sender_ids = [55335, 55336, 55337]
    observed, results, _, _ = run_series([(8, sender_ids)], scan=(55338, 1))
    assert [ids for _, ids, _ in observed] == [sender_ids]
    for _, ids, pixels in observed:
        np.testing.assert_array_equal(
            pixels, np.broadcast_to((np.array(ids) + 1).astype(np.uint16)[:, None, None], pixels.shape))
    assert results[0]["images"] == 3


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
        assert start["scan_source"] == "config"
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


def test_announced_scan_follows_each_series(capfd):
    # Scan shapes change between series and image sizes independently of each other.
    placements = []
    _, results, _, _ = run_series(
        [(1, [0, 1, 2, 3, 4, 5, 6]), (2, [0, 3, 6, 9, 10], (192, 192)), (3, [0, 5, 7, 8]),
         (4, [], (192, 192)), (5, [0, 1, 2], (192, 192))],
        appendices={1: {"scan_width": 3, "scan_height": 2, "operator": "x"},
                    2: {"scan_width": 5, "scan_height": 2}, 3: {"operator": "x"},
                    4: {"scan_width": 7, "scan_height": 7}, 5: {"scan_width": 1, "scan_height": 2}},
        service=True, placements=placements)
    # Independent reference: row-major placement in the announced scan, else config (4, 2).
    scans = {1: (3, 2), 2: (5, 2), 3: (4, 2), 4: (7, 7), 5: (1, 2)}
    sent = {1: range(7), 2: [0, 3, 6, 9, 10], 3: [0, 5, 7, 8], 4: [], 5: range(3)}
    expected = [(sid, i, (i // scans[sid][0], i % scans[sid][0]))
                for sid in sent for i in sent[sid] if i < scans[sid][0] * scans[sid][1]]
    assert placements == expected
    assert [result["scan_positions"] for result in results] == [6, 10, 8, 49, 2]
    assert [result["positions"] for result in results] == [6, 4, 3, 0, 2]
    assert [result["images"] for result in results] == [7, 5, 4, 0, 3]
    assert results[0]["bf_sum"] == 812 * sum(range(1, 7))
    assert results[2]["bf_sum"] == 812 * (1 + 6 + 8)
    starts = [json.loads(line.split(" ", 1)[1]) for line in capfd.readouterr().out.splitlines()
              if line.startswith("[series-start] ")]
    assert [(row["scan_width"], row["scan_height"], row["scan_source"]) for row in starts] == [
        (3, 2, "header_appendix"), (5, 2, "header_appendix"), (4, 2, "config"),
        (7, 7, "header_appendix"), (1, 2, "header_appendix")]


@pytest.mark.parametrize("appendix", [{"scan_width": 4}, {"scan_width": 0, "scan_height": 2},
                                      {"scan_width": "4", "scan_height": 2}])
def test_invalid_announced_scan_is_rejected(appendix):
    run_series([(11, [0])], appendices={11: appendix}, expect_error=True)
