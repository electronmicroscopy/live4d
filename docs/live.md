# Live scans: ANS-encoded on the GPU as they arrive

`live4d.live` turns the detector stream received by live4d's operators (`RdmaRxOp`, or `ZmqRxOp` →
`DecompressGpuOp`) into lossless, GPU-resident datasets. Each detector series is one scan; each
batch is corrected for flagged pixels and ANS-encoded by quantem.gpu's `LiveEncoder` as it lands,
so a scan is complete and encoded when its last image arrives. The result is the
`Dataset4dstemGPU` that `quantem.gpu.io.load` returns for the same counts.

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

Files in `holoscan_pipeline/python/live4d/live/`: `graph.py` (receiver session and sink operator), `receiver.py` (`LiveFrames`,
`SeriesFeed`), `feed.py` (`BatchFeed`), `recording.py` (`record`, `Recording`), `scans.py`
(`LiveScans`).

Image: `docker/Dockerfile.live` (the pipeline image plus pinned quantem.gpu, with `LiveEncoder`, and
quantem.widget for the viewer). Tests: `python -m pytest holoscan_pipeline/tests/test_live_series.py`.

Checked by replaying three 128×128 scans of 96×96 uint16 images over RDMA (same host) at
120,000 frames/s: all three received, saved, and decoded bit-exactly to the source.
