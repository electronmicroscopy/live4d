# Live scans: ANS-encoded on the GPU as they arrive

With `encoder.enabled`, `PipelineApp` adds `AnsEncoderOp` (`live4d.live.encoding`) as one more
consumer of the decoded batches, beside the virtual detectors and the HDF5 writer:

```
RdmaRxOp / DecompressGpuOp ─┬─► ScanTensorOp ─► VirtualDetectorOp   (window consumer 0)
                            ├─► H5WriterOp                         (consumer 1, optional)
                            └─► AnsEncoderOp                       (next consumer, own partition)
```

Each activation copies the batch's images once, on the runtime's CUDA stream, into a staging chunk
of 4096 patterns and releases the batch to the publication window; no other host synchronization.
A full chunk is corrected for flagged pixels and ANS-encoded in place by quantem.gpu's
`LiveEncoder`, which waits for its encode once per chunk; the encoder's own partition keeps that
wait off the receiver and the detectors. The scan raster of each series is the START
`header_appendix` scan, else a square raster of `number_of_images`, else `scan` from the config.
A series with a gap, repeat or overrun is dropped; the graph keeps running.

`LiveFrames` is the encoder's sink and the user-facing session: it runs `PipelineApp` in service
mode, keeps each completed series as the `Dataset4dstemGPU` that `quantem.gpu.io.load` returns,
saves it, and serves the live Show4DSTEM page.

```python
from live4d.live import LiveFrames, LiveScans

live = LiveFrames(remote="<detector>:31003", local="<this host>", folder="/data/live")
live.start()                 # connects; nothing is armed or triggered
live.viewer()                # Show4DSTEM, following arriving scans (quantem.widget)
live.datasets                # completed scans on the GPU (latest `keep`)
live.stop()

scans = LiveScans("/data/live")   # in any process: saved scans by name
data = scans["series_297"]
```

- **Scan raster:** from the START `header_appendix` (`scan_height`, `scan_width`, as live4d reads
  them), else a square raster of `number_of_images`.
- **Faulty series** (a gap, a repeat, no raster) are dropped and listed in `live.series.dropped`;
  the receiver keeps running.
- **Saving:** each completed scan is written as `<folder>/<date>/series_<id>_<rows>x<cols>_<det>px.qem`
  (the exact ANS bytes) and is released from the GPU only after it is saved.
- **One receiver per detector stream:** a lock file (`LIVE4D_LOCK`) records who receives;
  `page()` offers Take over.

Files in `holoscan_pipeline/python/live4d/live/`: `encoding.py` (`AnsEncoderOp`, `SeriesEncoder`),
`receiver.py` (`LiveFrames`), `recording.py` (`Recording`), `scans.py` (`LiveScans`).

Image: `docker/Dockerfile.live` (the pipeline image plus pinned quantem.gpu, with `LiveEncoder`, and
quantem.widget for the viewer). Tests: `python -m pytest holoscan_pipeline/tests/test_live_encoding.py`.

`test_live_encoding.py` sends batches through `SeriesEncoder` into quantem.gpu's `LiveEncoder` on a
CUDA device and decodes every kept series bit-exactly (chunk boundaries, 96 and 192 pixel images, a
gap, a short series). The full graph with `AnsEncoderOp` has not yet been run against the 120,000 fps
stress test.
