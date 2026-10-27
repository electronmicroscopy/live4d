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
               appendices=None, placements=None, writer=None, pixels=None, encoder=None):
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
                            writer=dict(writer or {}, enabled=writer is not None),
                            encoder=dict(enabled=encoder is not None))
    app = PipelineApp(config, observer=observer, encoder_sink=encoder)
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


READ_EM = """
import json, sys
import h5py
master_name, metadata_name = sys.argv[1], sys.argv[2]
out = {}
with h5py.File(master_name, "r") as m:
    link = m["/entry/instrument"].get("electron_microscope", getlink=True)
    out["link"] = [link.filename, link.path] if isinstance(link, h5py.ExternalLink) else None

def text(v):
    return v.decode() if isinstance(v, bytes) else v

def visit(name, obj):
    attrs = {k: text(v) for k, v in obj.attrs.items()}
    if isinstance(obj, h5py.Group):
        out["groups"][name] = attrs
        return
    string = h5py.check_string_dtype(obj.dtype)
    if string:
        kind = "str" if string.length is None and string.encoding == "utf-8" else repr(obj.dtype)
        value = text(obj[()])
    else:
        kind = obj.dtype.name + ("[]" if obj.shape else "")
        value = obj[()].tolist()
    out["fields"][name] = dict(kind=kind, value=value, units=attrs.get("units"))

if metadata_name != "-":
    with h5py.File(metadata_name, "r") as f:
        out["groups"], out["fields"] = {}, {}
        out["groups"]["electron_microscope"] = {k: text(v) for k, v in f["electron_microscope"].attrs.items()}
        f["electron_microscope"].visititems(lambda n, o: visit("electron_microscope/" + n, o))
print(json.dumps(out))
"""


def read_em(master, metadata=None):
    result = subprocess.run([sys.executable, "-I", "-c", READ_EM, str(master), str(metadata or "-")],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


# The EM metadata format 1.0 as novena writes it (em_metadata/EmMetadata.h): group -> NX_class
# and field -> (HDF5 type, units). Written out independently of the writer's own table.
F, S, B, U = "float64", "str", "int8", "uint64"
EM_LAYOUT = {
    "": ("NXem_instrument", dict(instrument_type=(S, None), manufacturer=(S, None),
                                 model=(S, None), location=(S, None))),
    "experiment": ("NXcollection", dict(persons=(S, None), time=(S, None), sample=(S, None),
                                        description=(S, None))),
    "electron_source": ("NXebeam_column", dict(gun_type=(S, None), accelerating_voltage=(F, "V"),
                                               beam_current=(F, "A"), spot_size=(U, None))),
    "illumination_system": ("NXem_optical_system", dict(
        mode=(S, None), submode=(S, None), scan_magnification=(F, None),
        scan_field_of_view_x=(F, "m"), scan_field_of_view_y=(F, "m"), working_distance=(F, "m"),
        beam_diameter=(F, "m"), semi_convergence_angle=(F, "mrad"), defocus=(F, "m"),
        beam_defining_aperture_type=(S, None), beam_defining_aperture_size=(F, "m"))),
    "imaging_system": ("NXem_optical_system", dict(
        mode=(S, None), image_magnification=(F, None), pixel_size_x=(F, "m"),
        pixel_size_y=(F, "m"), camera_length=(F, "m"), reciprocal_pixel_size_x=(F, "mrad"),
        reciprocal_pixel_size_y=(F, "mrad"), rotation=(F, "degrees"), defocus=(F, "m"))),
    "scan_controller": ("NXscan_controller", dict(
        manufacturer=(S, None), model=(S, None), enabled=(B, None), rotation=(F, "degrees"),
        rotation_offset=(F, "degrees"), scan_type=(S, None))),
    "scan_controller/regular_scan": ("NXparameters", dict(
        scan_pattern=(S, None), dwell_time=(F, "s"), flyback_time_line=(F, "s"),
        flyback_time_frame=(F, "s"), frame_interval=(F, "s"), n_frames=(U, None),
        n_pixels_x=(U, None), n_pixels_y=(U, None), pixel_size_x=(F, "m"), pixel_size_y=(F, "m"))),
    "scan_controller/irregular_scan": ("NXparameters", dict(
        n_scan_positions=(U, None), dwell_time=("float64[]", "s"), position_x=("float64[]", "m"),
        position_y=("float64[]", "m"))),
    **{f"scan_controller/aux_detector_{n}": ("NXdetector", dict(
        description=(S, None), gain=(F, None), offset=(F, None), data=("float32[]", None)))
       for n in range(1, 5)},
    "sample_stage": ("NXcollection", dict(
        manufacturer=(S, None), model=(S, None), position_x=(F, "m"), position_y=(F, "m"),
        position_z=(F, "m"), tilt_alpha=(F, "degrees"), tilt_beta=(F, "degrees"))),
}


def em_path(group, name=None):
    return "/".join(p for p in ("electron_microscope", group, name) if p)


def em_default(group, name, kind):
    if (group, name) == ("scan_controller", "scan_type"):
        return "regular"
    if (group, name) == ("scan_controller/regular_scan", "scan_pattern"):
        return "raster"
    return {S: "", "float64[]": [0.0], "float32[]": [0.0]}.get(kind, 0)


def assert_em_file(tree, given):
    # Every group and field of the format, with novena's types and units; given values where
    # announced and novena's defaults elsewhere.
    expected_groups = {em_path(g): dict(NX_class=c) for g, (c, _) in EM_LAYOUT.items()}
    expected_groups["electron_microscope"]["format_version"] = "1.0"
    assert tree["groups"] == expected_groups
    expected = {}
    for g, (_, fields) in EM_LAYOUT.items():
        for name, (kind, units) in fields.items():
            value = given
            for part in [*filter(None, g.split("/")), name]:
                value = value.get(part, {}) if isinstance(value, dict) else {}
            value = em_default(g, name, kind) if value == {} else value
            if kind == B:
                value = int(value)
            expected[em_path(g, name)] = dict(kind=kind, value=value, units=units)
    assert tree["fields"] == expected


EM_FULL = {
    "instrument_type": "TEM", "manufacturer": "FEI Co", "model": "Tecnai", "location": "Baden",
    "experiment": {"persons": "A. Person", "time": "2026-10-27 12:00:00", "sample": "Au",
                   "description": "µ-STEM test"},
    "electron_source": {"gun_type": "Schottky FEG", "accelerating_voltage": 200000,
                        "beam_current": 5.0e-11, "spot_size": 7},
    "illumination_system": {"mode": "STEM", "submode": "scanning", "scan_magnification": 225000.0,
                            "scan_field_of_view_x": 2.56e-7, "scan_field_of_view_y": 1.28e-7,
                            "semi_convergence_angle": 20.5, "beam_defining_aperture_type": "C2",
                            "beam_defining_aperture_size": 5.0e-5},
    "imaging_system": {"mode": "diffraction", "camera_length": 0.12,
                       "reciprocal_pixel_size_x": 0.46, "reciprocal_pixel_size_y": 0.47},
    "scan_controller": {
        "manufacturer": "Point Electronic GmbH", "model": "Revolon", "enabled": True,
        "rotation": -12.5, "rotation_offset": 3.0, "scan_type": "irregular",
        "regular_scan": {"scan_pattern": "serpentine", "dwell_time": 8.3e-6, "n_frames": 1,
                         "n_pixels_x": 4, "n_pixels_y": 2, "pixel_size_x": 1.0e-9,
                         "pixel_size_y": 2.0e-9},
        "irregular_scan": {"n_scan_positions": 3, "dwell_time": [1e-6, 2e-6, 3e-6],
                           "position_x": [0.0, 1e-9, 2e-9], "position_y": [0, 0, 1e-9]},
        "aux_detector_2": {"description": "HAADF", "gain": 2.0, "offset": -1.0,
                           "data": [0.5, 1.5, 2.5]}},
    "sample_stage": {"position_z": -1.0e-6, "tilt_alpha": 15.0},
}


def test_hdf5_writer_em_format_writes_metadata_from_header_appendix(tmp_path, capfd):
    # writer.format em: a header_appendix electron_microscope map becomes <name>_em_metadata.h5
    # in novena's layout, linked from the master file, for both image sizes. Series 1 announces
    # a full map with two unknown keys, series 2 (192x192) a partial map under an appendix name
    # pattern, series 3 none: it gets only the link, for the controller to fill.
    em_partial = {"electron_source": {"accelerating_voltage": 300000.0},
                  "scan_controller": {"regular_scan": {"n_pixels_x": 4}}}
    em_unknown = dict(EM_FULL, dose=12.0, experiment=dict(EM_FULL["experiment"], operator_id=7))
    series = [(1, [0, 1]), (2, [0], (192, 192)), (3, [0])]
    run_series(series, service=True, pixels=series_pixels,
               appendices={1: {"electron_microscope": em_unknown},
                           2: {"name_pattern": "sub/custom_$id", "electron_microscope": em_partial},
                           3: {"scan_width": 4, "scan_height": 2}},
               writer=dict(output_dir=str(tmp_path), format="em", name_pattern="run_$id"))
    captured = capfd.readouterr()

    tree = read_em(tmp_path / "run_1_master.h5", tmp_path / "run_1_em_metadata.h5")
    assert tree["link"] == ["run_1_em_metadata.h5", "/electron_microscope"]
    assert_em_file(tree, EM_FULL)
    log = captured.out + captured.err
    for key in ("electron_microscope.dose", "electron_microscope.experiment.operator_id"):
        assert f"series 1: header_appendix {key} is not part of the EM metadata format" in log

    sub = tmp_path / "sub"
    tree = read_em(sub / "custom_2_master.h5", sub / "custom_2_em_metadata.h5")
    assert tree["link"] == ["custom_2_em_metadata.h5", "/electron_microscope"]
    assert_em_file(tree, em_partial)

    assert read_em(tmp_path / "run_3_master.h5")["link"] == ["run_3_em_metadata.h5", "/electron_microscope"]
    assert not (tmp_path / "run_3_em_metadata.h5").exists()
    records = {r["series_id"]: r for r in h5_series_records(captured.out)}
    assert records[1]["em_metadata_file"] == str(tmp_path / "run_1_em_metadata.h5")
    assert records[2]["em_metadata_file"] == str(sub / "custom_2_em_metadata.h5")
    assert records[3]["em_metadata_file"] is None
    for prefix, shape, image_id in ((tmp_path / "run_1", (96, 96), 1), (sub / "custom_2", (192, 192), 0)):
        meta, data = read_series(prefix, tmp_path)
        np.testing.assert_array_equal(data[image_id, 0], series_pixels(int(prefix.name[-1]), image_id, shape))
    assert not list(tmp_path.rglob("*.tmp"))

    # nxmx ignores the map, even an invalid one: no link and no metadata file.
    nxmx = tmp_path / "nxmx"
    run_series([(4, [0])], pixels=series_pixels, writer=dict(output_dir=str(nxmx)),
               appendices={4: {"electron_microscope": {"electron_source": {"accelerating_voltage": "x"}}}})
    assert read_em(nxmx / "series_4_master.h5")["link"] is None
    assert not list(nxmx.glob("*_em_metadata.h5"))


@pytest.mark.parametrize("em", [
    "TEM", {"experiment": "x"}, {"scan_controller": {"regular_scan": 1}},
    {"electron_source": {"accelerating_voltage": "200 kV"}},
    {"electron_source": {"accelerating_voltage": True}},
    {"scan_controller": {"enabled": 1}}, {"electron_source": {"spot_size": -1}},
    {"electron_source": {"spot_size": 1.5}}, {"model": 3},
    {"scan_controller": {"scan_type": "spiral"}},
    {"scan_controller": {"regular_scan": {"scan_pattern": "Raster"}}},
    {"scan_controller": {"aux_detector_1": {"data": [1.0, "x"]}}},
    {"scan_controller": {"irregular_scan": {"position_x": 1.0}}}])
def test_hdf5_writer_em_format_rejects_invalid_metadata(em, tmp_path):
    # An invalid announced value fails the run before any file of the series is written.
    run_series([(14, [0])], appendices={14: {"electron_microscope": em}}, expect_error=True,
               writer=dict(output_dir=str(tmp_path), format="em"))
    assert not list(tmp_path.rglob("*.h5")) and not list(tmp_path.rglob("*.tmp"))


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


def encoded(dataset):
    """Every pattern of an ANS-encoded series, decoded on the GPU, in scan order."""
    rows, cols, *shape = dataset.shape
    return cp.asnumpy(dataset.data.decode_scan_range_device(0, rows * cols)).reshape(-1, *shape)


@pytest.mark.parametrize("writing", [False, True])
def test_ans_encoder_both_sizes_and_partial_series(writing, tmp_path, capfd):
    # The encoder is one more decoded consumer, beside the detector and, optionally, the writer.
    pytest.importorskip("quantem.gpu.streaming")
    from live4d.algorithms.series_store import SeriesStore
    sink = SeriesStore(keep=0)
    series = [(1, list(range(16)), (96, 96)), (2, list(range(9)), (192, 192)), (3, list(range(16)), (96, 96))]
    run_series(series, batch_size=5, service=True, pixels=series_pixels, encoder=sink,
               writer=dict(output_dir=str(tmp_path)) if writing else None)
    assert not sink.dropped
    assert [dataset.shape for dataset in sink.datasets] == [(4, 4, 96, 96), (3, 3, 192, 192), (4, 4, 96, 96)]
    for (sid, ids, shape), dataset in zip(series, sink.datasets):
        expected = np.stack([series_pixels(sid, image_id, shape) for image_id in ids])
        np.testing.assert_array_equal(encoded(dataset), expected)
    records = [json.loads(line.split(" ", 1)[1]) for line in capfd.readouterr().out.splitlines()
               if line.startswith("[ans-result] ")]
    assert [(r["series_id"], r["images"], r["kept"]) for r in records] == [(1, 16, True), (2, 9, True), (3, 16, True)]


def test_ans_encoder_drops_a_series_with_a_gap_and_keeps_the_next(capfd):
    pytest.importorskip("quantem.gpu.streaming")
    from live4d.algorithms.series_store import SeriesStore
    sink = SeriesStore(keep=0)
    series = [(1, [0, 1, 2, 5, 6], (96, 96)), (2, list(range(4)), (96, 96))]
    run_series(series, batch_size=2, scan=(5, 1), service=True, pixels=series_pixels, encoder=sink)
    assert [d["series"] for d in sink.dropped] == ["series_1"]
    assert len(sink.datasets) == 1 and sink.datasets[0].shape == (2, 2, 96, 96)
    expected = np.stack([series_pixels(2, image_id, (96, 96)) for image_id in range(4)])
    np.testing.assert_array_equal(encoded(sink.datasets[0]), expected)


def test_ans_encoder_finite_mode_completes_after_every_consumer(tmp_path, capfd):
    # Detector, writer and encoder all see END; the run completes after the last of them.
    pytest.importorskip("quantem.gpu.streaming")
    from live4d.algorithms.series_store import SeriesStore
    sink = SeriesStore()
    run_series([(1, list(range(9)))], batch_size=4, pixels=series_pixels, encoder=sink,
               writer=dict(output_dir=str(tmp_path)))
    expected = np.stack([series_pixels(1, image_id, (96, 96)) for image_id in range(9)])
    np.testing.assert_array_equal(encoded(sink.datasets[0]), expected)
