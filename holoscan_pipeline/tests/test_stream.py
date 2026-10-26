"""Public ZMQ series, decoded pixels, and shared 4D-STEM output contracts."""
import signal
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
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


def image_message(image_id, value, shape=(96, 96), pixels=None):
    height, width = shape
    elements_left = height * width
    data = bytearray((height * width * 2).to_bytes(8, "big") + (8192).to_bytes(4, "big"))
    flat = None if pixels is None else np.ascontiguousarray(pixels, dtype=np.uint16).ravel()
    while elements_left:
        elements = min(4096, elements_left)
        if flat is None:
            shuffled = b"".join(bytes([255 if (value >> bit) & 1 else 0]) * (elements // 8)
                                 for bit in range(16))
        else:
            # Bit plane b holds bit b of consecutive elements, least significant element first.
            block = flat[height * width - elements_left:][:elements]
            planes = (block[None, :] >> np.arange(16, dtype=np.uint16)[:, None]) & 1
            shuffled = np.packbits(planes.astype(np.uint8), axis=1, bitorder="little").tobytes()
        length = len(shuffled) - 15
        block = bytes([0xf0]) + bytes([255]) * (length // 255) + bytes([length % 255]) + shuffled
        data += len(block).to_bytes(4, "big") + block
        elements_left -= elements
    payload = cbor2.CBORTag(40, [[height, width], cbor2.CBORTag(69,
               cbor2.CBORTag(56500, ["bslz4", 2, bytes(data)]))])
    return dict(type="image", image_id=image_id, data={"threshold_1": payload})


def run_series(series, *, batch_size=4, scan=(4, 2), service=False, delay=False,
               processing_delay=0, queue_depth=4, defer_readback=False, expect_error=False,
               appendices=None, placements=None, writer=None, pixels=None):
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
                            zmq=dict(endpoint=f"tcp://127.0.0.1:{port}"),
                            writer=dict(writer or {}, enabled=writer is not None))
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
                sender.send(cbor2.dumps(image_message(
                    image_id, image_id + 1, image_shape,
                    None if pixels is None else pixels(sid, image_id, image_shape))))
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
@pytest.mark.parametrize("writing", [False, True])
def test_image_geometry_must_match_start(start_shape, image_shape, writing, tmp_path):
    run_series([(10, [0], start_shape, image_shape)], expect_error=True,
               writer=dict(output_dir=str(tmp_path)) if writing else None)
    assert not list(tmp_path.glob("*_data_*.h5"))


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


def series_pixels(sid, image_id, shape):
    rng = np.random.default_rng(1000 * sid + image_id)
    pixels = rng.poisson(2.0, shape).astype(np.uint16)
    pixels[0, :3] = [image_id, 65534, 4096]
    return pixels


# Read in a separate interpreter: h5py bundles its own HDF5, and the pipeline library has the
# system HDF5 loaded globally. hdf5plugin is the standard bitshuffle/zstd decoder.
READ_SERIES = """
import json, sys
import h5py, hdf5plugin, numpy as np
prefix, out = sys.argv[1], sys.argv[2]
with h5py.File(prefix + "_master.h5", "r") as m:
    d = m["/entry/data/data"]
    sources = sorted({s.file_name for s in d.virtual_sources()})
    meta = dict(shape=list(d.shape), virtual=d.is_virtual, sources=sources,
                axes=[str(a) for a in m["/entry/data"].attrs["axes"]],
                definition=m["/entry/definition"][()].decode(),
                frame_time=float(m["/entry/instrument/detector/frame_time"][()]))
    np.save(out, d[()])
filters = {}
for name in sources:
    try:
        with h5py.File(prefix.rsplit("/", 1)[0] + "/" + name, "r") as f:
            fid, _, cd, _ = f["/entry/data/data"].id.get_create_plist().get_filter(0)
            filters[name] = [fid, list(cd), f["/entry/data/data"].chunks]
    except FileNotFoundError:
        pass
meta["filters"] = filters
print(json.dumps(meta))
"""


def read_series(prefix, tmp_path):
    out = tmp_path / (prefix.name + ".npy")
    result = subprocess.run([sys.executable, "-I", "-c", READ_SERIES, str(prefix), str(out)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout), np.load(out)


def h5_series_records(text):
    # Under fd capture, print() writes text and newline separately, so a record from the
    # writer thread can land mid-line; decode each record wherever it starts.
    decoder, records, at = json.JSONDecoder(), [], text.find("[h5-series] ")
    while at >= 0:
        record, end = decoder.raw_decode(text, at + len("[h5-series] "))
        records.append(record)
        at = text.find("[h5-series] ", end)
    return records


def test_hdf5_writer_both_sizes_partial_empty_and_dropped_images(tmp_path, capfd):
    # Series 1: 96x96, ten images in three data files of four, batches 4+4+2.
    # Series 2: empty. Series 3: 192x192 with a missing image 1 and image 3 beyond the
    # announced count of three (start_message announces len(ids)).
    series = [(1, list(range(10))), (2, []), (3, [0, 2, 3], (192, 192))]
    observed, results, _, _ = run_series(
        series, service=True, pixels=series_pixels,
        writer=dict(output_dir=str(tmp_path), images_per_file=4, name_pattern="run_$id"))
    assert [r["images"] for r in results] == [10, 0, 3]

    meta, data = read_series(tmp_path / "run_1", tmp_path)
    assert meta["shape"] == [10, 1, 96, 96] and meta["virtual"]
    assert meta["sources"] == [f"run_1_data_{n:06d}.h5" for n in (1, 2, 3)]
    assert meta["axes"] == ["image_id", "channel", ".", "."] and meta["definition"] == "NXmx"
    assert meta["frame_time"] == 0.001
    assert all(f == [32008, [0, 5, 2, 0, 3], [1, 1, 96, 96]] for f in meta["filters"].values())
    assert len(meta["filters"]) == 3
    for image_id in range(10):
        np.testing.assert_array_equal(data[image_id, 0], series_pixels(1, image_id, (96, 96)))

    assert not list(tmp_path.glob("run_2_*"))  # an empty series announces no images

    meta, data = read_series(tmp_path / "run_3", tmp_path)
    assert meta["shape"] == [3, 1, 192, 192]
    assert meta["filters"] == {"run_3_data_000001.h5": [32008, [0, 5, 2, 0, 3], [1, 1, 192, 192]]}
    np.testing.assert_array_equal(data[0, 0], series_pixels(3, 0, (192, 192)))
    np.testing.assert_array_equal(data[1, 0], np.full((192, 192), 65535, np.uint16))
    np.testing.assert_array_equal(data[2, 0], series_pixels(3, 2, (192, 192)))

    records = h5_series_records(capfd.readouterr().out)
    assert [(r["series_id"], r["written_images"], r["dropped_images"], r["data_files"],
             r["image_shape"], r["ended"]) for r in records] == [
        (1, 10, 0, 3, [96, 96], True), (2, 0, 0, 0, [96, 96], True),
        (3, 2, 1, 1, [192, 192], True)]
    assert records[1]["master_file"] is None
    assert sorted(p.name for p in tmp_path.glob("*.h5")) == sorted(
        ["run_1_master.h5", "run_1_data_000001.h5", "run_1_data_000002.h5",
         "run_1_data_000003.h5", "run_3_master.h5", "run_3_data_000001.h5"])
    assert not list(tmp_path.glob("*.tmp"))


def test_hdf5_writer_finite_mode_completes_after_last_write(tmp_path, capfd):
    # A slow detector and a writer both see END; completion waits for the later one.
    count = 33
    _, results, _, _ = run_series([(5, list(range(count)))], pixels=series_pixels,
                                  processing_delay=0.02, queue_depth=2,
                                  writer=dict(output_dir=str(tmp_path)))
    assert results[0]["images"] == count
    meta, data = read_series(tmp_path / "series_5", tmp_path)
    assert meta["shape"] == [count, 1, 96, 96] and meta["sources"] == ["series_5_data_000001.h5"]
    for image_id in range(count):
        np.testing.assert_array_equal(data[image_id, 0], series_pixels(5, image_id, (96, 96)))
    record, = h5_series_records(capfd.readouterr().out)
    assert record["written_images"] == count and record["ended"]


def test_hdf5_writer_name_pattern_from_header_appendix(tmp_path, capfd):
    # A non-empty header_appendix name_pattern overrides writer.name_pattern for its series;
    # an empty or missing one keeps the configured pattern.
    _, results, _, _ = run_series(
        [(1, [0, 1]), (2, [0]), (3, [0, 1, 2], (192, 192)), (4, [0])], service=True,
        pixels=series_pixels, appendices={
            1: {"name_pattern": "runA/scan_$id"}, 2: {"name_pattern": ""},
            3: {"name_pattern": "custom", "scan_width": 3, "scan_height": 1}},
        writer=dict(output_dir=str(tmp_path), name_pattern="cfg_$id"))
    assert [r["images"] for r in results] == [2, 1, 3, 1]
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.h5")) == sorted([
        "runA/scan_1_master.h5", "runA/scan_1_data_000001.h5",
        "cfg_2_master.h5", "cfg_2_data_000001.h5",
        "custom_master.h5", "custom_data_000001.h5",
        "cfg_4_master.h5", "cfg_4_data_000001.h5"])
    meta, data = read_series(tmp_path / "runA" / "scan_1", tmp_path)
    assert meta["shape"] == [2, 1, 96, 96]
    for image_id in range(2):
        np.testing.assert_array_equal(data[image_id, 0], series_pixels(1, image_id, (96, 96)))
    meta, data = read_series(tmp_path / "custom", tmp_path)
    assert meta["shape"] == [3, 1, 192, 192]
    np.testing.assert_array_equal(data[2, 0], series_pixels(3, 2, (192, 192)))
    records = h5_series_records(capfd.readouterr().out)
    assert [(r["series_id"], r["name_source"], r["master_file"]) for r in records] == [
        (1, "header_appendix", f"{tmp_path}/runA/scan_1_master.h5"),
        (2, "config", f"{tmp_path}/cfg_2_master.h5"),
        (3, "header_appendix", f"{tmp_path}/custom_master.h5"),
        (4, "config", f"{tmp_path}/cfg_4_master.h5")]


@pytest.mark.parametrize("pattern", ["../escape_$id", "/tmp/absolute_$id", "runs/..", ".", 5])
def test_hdf5_writer_rejects_unsafe_header_appendix_name_pattern(pattern, tmp_path):
    output = tmp_path / "out"
    run_series([(12, [0])], appendices={12: {"name_pattern": pattern}}, expect_error=True,
               writer=dict(output_dir=str(output)))
    assert not list(tmp_path.rglob("*.h5"))
    assert not list(Path("/tmp").glob("absolute_12*"))


def test_hdf5_writer_disabled_per_series_by_header_appendix(tmp_path, capfd):
    # enable_h5_writer false skips a series; true or missing writes it.
    _, results, _, _ = run_series(
        [(1, [0, 1, 2]), (2, [0, 1], (192, 192)), (3, [0])], service=True, pixels=series_pixels,
        appendices={1: {"enable_h5_writer": False, "name_pattern": "skipped_$id"},
                    2: {"enable_h5_writer": True}},
        writer=dict(output_dir=str(tmp_path)))
    assert [r["images"] for r in results] == [3, 2, 1]
    assert sorted(p.name for p in tmp_path.rglob("*.h5")) == sorted([
        "series_2_master.h5", "series_2_data_000001.h5",
        "series_3_master.h5", "series_3_data_000001.h5"])
    meta, data = read_series(tmp_path / "series_2", tmp_path)
    assert meta["shape"] == [2, 1, 192, 192]
    for image_id in range(2):
        np.testing.assert_array_equal(data[image_id, 0], series_pixels(2, image_id, (192, 192)))
    records = h5_series_records(capfd.readouterr().out)
    assert [(r["series_id"], r["enabled"], r["written_images"], r["series_images"])
            for r in records] == [(1, False, 0, 3), (2, True, 2, 2), (3, True, 1, 1)]
    assert records[0]["master_file"] is None and records[0]["name_source"] is None


@pytest.mark.parametrize("flag", ["false", 0, 1, None])
def test_hdf5_writer_rejects_non_boolean_enable_h5_writer(flag, tmp_path):
    run_series([(13, [0])], appendices={13: {"enable_h5_writer": flag}}, expect_error=True,
               writer=dict(output_dir=str(tmp_path)))
    assert not list(tmp_path.rglob("*.h5"))
