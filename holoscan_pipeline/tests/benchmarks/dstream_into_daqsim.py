"""Put a .dstream recording's frames into an HDF5 series that daqsim's own filewriter wrote.

    python3 holoscan_pipeline/tests/benchmarks/dstream_into_daqsim.py <series_master.h5> \
        <recording.dstream>... [--check 2000]

Inside the parallax image, which has h5py, hdf5plugin and cbor2. daqsim writes the series first
(its filewriter enabled, the stream disabled, `PUT /daqsim/arm_test_images` with the frame count,
frame time, binning and compression, then a trigger); `PUT /daqsim/arm_hdf5` replays it after.

The series keeps daqsim's layout, metadata and filters (its frame time, binning and compression
are what `PUT /daqsim/arm_hdf5` configures). Only each frame's chunk is replaced, by the
recordings' bslz4 blobs: Stream2 reuses bitshuffle's HDF5 chunk format, so nothing is
recompressed. The series takes the recordings' frames in image-id order (a capture over several
connections arrives interleaved, and daqsim numbers frames in the order it sends them) and cycles
the last one to its end, so with lengths that are multiples of the scan size every frame keeps
its raster position.
A sample of frames is then read back through the HDF5 bitshuffle filter and compared, pixel for
pixel, with live4d's own decoding of the recorded messages.
"""

import argparse
import time
from pathlib import Path

import cbor2
import h5py
import hdf5plugin  # noqa: F401  (registers the bitshuffle filter for the read-back)
import numpy as np

from live4d.parallax.ingest.codec import decode_message
from live4d.parallax.ingest.dstream import iter_records

COMPRESSION_TAG, MULTIDIM_TAG = 56500, 40


def blob_of(raw):
    msg = cbor2.loads(raw)
    msg = msg.value if isinstance(msg, cbor2.CBORTag) else msg
    if msg.get("type") != "image":
        return None, None
    node = msg["data"]["threshold_1"]
    assert node.tag == MULTIDIM_TAG, node
    shape, typed = node.value
    payload = typed.value
    assert payload.tag == COMPRESSION_TAG and payload.value[0] == "bslz4", payload.value[:2]
    return msg["image_id"], (tuple(shape), typed.tag, bytes(payload.value[2]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("master", type=Path)
    p.add_argument("recordings", type=Path, nargs="+")
    p.add_argument("--check", type=int, default=2000, help="frames to verify pixel for pixel")
    a = p.parse_args()

    t0 = time.perf_counter()
    raws, blobs = [], []
    for recording in a.recordings:
        first = len(blobs)
        records = []
        for raw in iter_records(recording):
            image_id, blob = blob_of(raw)
            if blob is not None:
                records.append((image_id, raw, blob))
        # a live capture over several connections interleaves them: daqsim numbers frames in
        # the order sent, so send them by image id (each one at its own scan position)
        ordered = [r[0] for r in records] == list(range(len(records)))
        records.sort(key=lambda r: r[0])
        assert [r[0] for r in records] == list(range(len(records))), f"{recording}: ids not 0..n-1"
        raws += [r[1] for r in records]
        blobs += [r[2] for r in records]
        print(f"recording {recording.name}: {len(records)} frames"
              + ("" if ordered else ", sorted by image id (captured out of order)"), flush=True)
    head, cycle = first, len(blobs) - first             # before the last recording, and its length
    source = lambda k: k if k < head else head + (k - head) % cycle
    shapes = {b[0] for b in blobs}
    typed = {b[1] for b in blobs}
    n = len(blobs)
    print(f"recordings: {n} frames, shapes {shapes}, typed-array tags {typed}, "
          f"{sum(len(b[2]) for b in blobs) / n:.0f} B per blob, read in {time.perf_counter() - t0:.1f} s",
          flush=True)
    assert len(shapes) == 1 and typed == {69}, "expected one shape of uint16 (tag 69) frames"
    (shape,) = shapes

    with h5py.File(a.master, "r") as master:
        group = master["entry/data"]
        names = sorted(k for k in group if k.startswith("data_"))
        files = [(name, group.get(name, getlink=True)) for name in names]
    t0 = time.perf_counter()
    k = 0
    layout = []
    for name, link in files:
        path = a.master.parent / link.filename
        with h5py.File(path, "r+") as f:
            ds = f[link.path]
            assert ds.dtype == np.uint16 and ds.shape[1:] == shape, (ds.dtype, ds.shape, shape)
            assert ds.chunks == (1, *shape), ds.chunks
            frames = ds.shape[0]
            mask, _ = ds.id.read_direct_chunk((0, 0, 0))
            for j in range(frames):
                ds.id.write_direct_chunk((j, 0, 0), blobs[source(k + j)][2], filter_mask=mask)
            layout.append((path, link.path, k, frames))
            k += frames
    print(f"series: {k} frames in {len(files)} data files: {min(k, head)} of the first recordings, "
          f"then {max(0, k - head) / cycle:.3f} cycles of the last, written in "
          f"{time.perf_counter() - t0:.1f} s", flush=True)

    rng = np.random.default_rng(0)
    picks = np.unique(np.concatenate([[0, k - 1], rng.integers(0, k, a.check)]))
    bad = 0
    for path, dpath, start, frames in layout:
        mine = picks[(picks >= start) & (picks < start + frames)]
        if not mine.size:
            continue
        with h5py.File(path, "r") as f:
            ds = f[dpath]
            for g in mine:
                got = ds[g - start]
                _, _, want = decode_message(raws[source(g)])
                if got.dtype != np.uint16 or not np.array_equal(got, want):
                    bad += 1
    print(f"check: {picks.size} frames read back through the HDF5 filter, {bad} differ from the "
          f"recording", flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
