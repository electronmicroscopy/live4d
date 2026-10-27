"""Replay recorded frames through parallax on CPU, MPS, or CUDA without Holoscan."""
import argparse
from pathlib import Path
import signal
import time

import numpy as np

from live4d.config import load_config
from .runtime import ParallaxRuntime


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m live4d.parallax.offline", description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Shared live4d YAML; parallax must be enabled.")
    parser.add_argument("--source", type=Path, required=True, help="A .dstream recording or (..., H, W) .npy frames.")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu", help="PyTorch replay device; no Holoscan required.")
    parser.add_argument("--source-rate", type=float, default=0, help="Optional replay pacing in frames/s.")
    return parser


def replay(runtime, source, *, source_rate=0):
    from .ingest.codec import decode_message
    from .ingest.dstream import iter_records
    from .ingest.python_ingest import _Batcher
    import cbor2

    configured_scan = (runtime.config.scan["height"], runtime.config.scan["width"])
    count = 0
    started = time.perf_counter()

    def pace(n):
        if source_rate > 0:
            delay = n / source_rate - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)

    if source.suffix == ".npy":
        data = np.load(source, mmap_mode="r")
        if data.ndim < 3:
            raise ValueError("Replay expects (..., H, W) frames")
        data = data.reshape(-1, *data.shape[-2:])
        runtime.args.dtype = str(data.dtype)
        runtime.start_series(data.shape[1:], configured_scan)
        for first in range(0, len(data), runtime.config.batch_size):
            last = min(first + runtime.config.batch_size, len(data))
            runtime.sink.feed(np.array(data[first:last]), np.arange(first, last))
            pace(last)
        runtime.finish_series()
        return
    batcher = None
    series_id = None
    for raw in iter_records(source):
        kind, image_id, frame = decode_message(raw)
        if kind == "start":
            if batcher is not None:
                raise ValueError("Replay START arrived before END")
            metadata = cbor2.loads(raw)
            appendix = metadata.get("header_appendix", {})
            scan = (appendix.get("scan_height", configured_scan[0]),
                    appendix.get("scan_width", configured_scan[1]))
            shape = (metadata["image_size_y"], metadata["image_size_x"])
            series_id = metadata["series_id"]
            runtime.args.dtype = metadata["image_dtype"]
            if runtime.args.dtype not in ("uint16", "uint32"):
                raise ValueError("Replay START requires uint16 or uint32 images")
            runtime.start_series(shape, scan)
            batcher = _Batcher(runtime.sink, runtime.acc, runtime.config.batch_size, 0.05, runtime.args.dtype)
        elif kind == "image":
            if batcher is None:
                raise ValueError("Replay image has no matching START")
            metadata = cbor2.loads(raw)
            if "series_id" in metadata and metadata["series_id"] != series_id:
                raise ValueError("Replay image series differs from START")
            batcher.push(image_id, frame)
            count += 1
            pace(count)
        elif kind == "end":
            if batcher is None or cbor2.loads(raw).get("series_id") != series_id:
                raise ValueError("Replay END has no matching START")
            batcher.flush()
            runtime.finish_series()
            batcher = None
    if batcher is not None:
        batcher.flush()
        raise ValueError("Recording ended without END")


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if not config.parallax["enabled"]:
        parser.error("parallax.enabled must be true")
    if args.source_rate < 0:
        parser.error("--source-rate must be nonnegative")
    runtime = ParallaxRuntime(config, device=args.device)
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        runtime.open()
        try:
            replay(runtime, args.source, source_rate=args.source_rate)
        except KeyboardInterrupt:
            print("[parallax] replay interrupted", flush=True)
        # An enabled viewer remains available for inspecting and tuning the final image.
        if runtime.viewer is not None:
            print("[parallax] replay finished; viewer remains available until Ctrl-C", flush=True)
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
    finally:
        runtime.close()
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
