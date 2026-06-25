"""
Entry point for the real-time virtual bright-field (VBF) pipeline.

Connects to the DCU and accumulates the VBF image as frames stream in, on the chunked streaming
pipeline (`ingest/pipeline.py`) and the GPU-cast VBF engine (`vbf.py`):

    python virtual_detectors/main.py                            # live view (GUI)
    python virtual_detectors/main.py --headless --decode gpu    # headless, GPU decode + timing

Tuning knobs: --chunk-size (frames per chunk = GPU batch) and --decode-threads.
"""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

# Make the package root importable so `python main.py` works from any directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ingest.pipeline import (
    PipelineStats,
    validate_start,
    BatchedLiveSource,
    run_pipeline_batched,
)
from visualize.visualize_virtual_detector import LiveImage
from virtual_detectors.config import CONFIG
from virtual_detectors.vbf import build_weight_map, VBFAccumulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time virtual bright-field (VBF).")
    p.add_argument("--decode-threads", type=int, default=None, metavar="N",
                   help="Override config.n_decode_threads (decoder-pool size).")
    p.add_argument("--chunk-size", type=int, default=None, metavar="C",
                   help="Override config.chunk_size (frames per chunk = GPU batch).")
    p.add_argument("--headless", action="store_true",
                   help="Skip the live GUI window (also enables per-stage timing).")
    p.add_argument("--decode", choices=("cpu", "gpu"), default="cpu",
                   help="Decode backend: 'cpu' (threaded bslz4 on the host) or 'gpu' (envelope "
                        "parse on the host, bslz4 decompress on the GPU). Default cpu.")
    p.add_argument("--binned", action="store_true",
                   help="Use the ARINA 2x2-binned preset (96x96, BF center (48,48) r15, 8.33 us / "
                        "120 kHz mode) instead of the config's unbinned 192x192.")
    return p.parse_args()


def load_mask(path: Path, detector_shape, name: str) -> np.ndarray:
    """Load a calibration mask from `path`, or default to all-ones if it's absent."""
    if path.exists():
        m = np.load(path)
        if m.shape != tuple(detector_shape):
            raise ValueError(f"{name} shape {m.shape} != detector {detector_shape} ({path})")
        return m.astype(np.float32)
    print(f"[masks] {name} not found at {path} -- defaulting to ones.")
    return np.ones(detector_shape, dtype=np.float32)


def report_stats(stats: PipelineStats, acc: VBFAccumulator, cfg) -> None:
    """Print the pipeline throughput summary and, if measured, the per-stage GPU/host timing."""
    print("\n" + stats.summary(queue_maxsize=cfg.chunk_queue_maxsize))

    elapsed = stats.elapsed_s
    our_rate = (stats.decoded / elapsed) if elapsed else float("nan")
    det_rate = cfg.detector_rate_hz
    print("=== Real-time check (informational -- not a gate) ===")
    print(f"  frame period (assumed): {cfg.frame_period_us:8.2f} us/frame  (externally supplied)")
    print(f"  detector frame rate:    {det_rate / 1e3:8.1f} kHz       (full-rate target for this mode)")
    print(f"  measured decode rate:   {our_rate / 1e3:8.1f} kHz")
    print(f"  real-time factor:       {our_rate / det_rate:8.2f}x        (>=1 keeps up at this exposure)")
    print(f"  acquisition wall-time:  {stats.decoded * cfg.frame_period_us / 1e6:8.1f} s  vs processed in {elapsed:.1f} s")

    perf = acc.perf_stats()
    if perf is None:
        return
    n = stats.batched or 1
    print("=== Per-stage cost (VBFAccumulator -- cast is on the GPU now) ===")
    print(f"  host prep (CPU):    {perf['host_prep_s']:8.3f} s  ({perf['host_prep_s'] / n * 1e6:8.1f} us/frame)")
    if "gpu_copy_s" in perf:
        print(f"  int H2D copy (GPU): {perf['gpu_copy_s']:8.3f} s  ({perf['gpu_copy_s'] / n * 1e6:8.1f} us/frame)")
        print(f"  cast+reduce (GPU):  {perf['gpu_compute_s']:8.3f} s  ({perf['gpu_compute_s'] / n * 1e6:8.1f} us/frame)")
    else:
        print(f"  cast+reduce (CPU):  {perf['compute_s']:8.3f} s  ({perf['compute_s'] / n * 1e6:8.1f} us/frame)")


def main() -> None:
    args = parse_args()
    cfg = CONFIG
    overrides = {}
    if args.decode_threads is not None:
        overrides["n_decode_threads"] = args.decode_threads
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.binned:
        overrides.update(detector_shape=(96, 96), bf_center_yx=(48.0, 48.0),
                         bf_radius_px=15.0, frame_period_us=8.33)   # 120 kHz binned mode
    if overrides:
        cfg = replace(cfg, **overrides)   # frozen dataclass -> replace()

    # 1) masks -> weight map (host, NumPy)
    flatfield = load_mask(cfg.flatfield_path, cfg.detector_shape, "flatfield")
    dead_px = load_mask(cfg.dead_px_path, cfg.detector_shape, "dead_px")
    W = build_weight_map(cfg.detector_shape, cfg.bf_center_yx, cfg.bf_radius_px, flatfield, dead_px)

    # 2) accumulator (GPU if available) + optional live display. Headless runs also measure timing.
    acc = VBFAccumulator(W, scan_shape=cfg.scan_shape, measure=args.headless)
    stats = PipelineStats()
    live = None if args.headless else LiveImage(cfg.scan_shape)
    print(f"[vbf]    device = {acc.device}")
    print(f"[stream] connecting to tcp://{cfg.dcu_ip}:{cfg.zmq_port}")
    print(f"[stream] scan {cfg.scan_shape}, detector {cfg.detector_shape}, "
          f"chunk {cfg.chunk_size}, decode threads {cfg.n_decode_threads}, decode={args.decode}")

    # 3) build the source. The on_start gate validates the stream against our preallocated config;
    #    the producer runs it before any frame chunk (no gate race).
    gate = lambda m: validate_start(m, cfg.detector_shape, cfg.num_positions)
    source = BatchedLiveSource(cfg, on_start=gate, stats=stats, decode_mode=args.decode)

    # 4) run the pipeline. With a GUI, run on a background thread; headless, run inline and block.
    if live is None:
        run_pipeline_batched(source, acc.process, stats)
    else:
        worker = threading.Thread(
            target=run_pipeline_batched, args=(source, acc.process, stats),
            name="pipeline", daemon=True,
        )
        worker.start()
        try:
            while worker.is_alive():
                live.update(acc.result())
        except KeyboardInterrupt:
            print("\n[main] interrupted -- showing the partial image.")
        worker.join()

    if source.start_error is not None:
        print(f"[main] START VALIDATION FAILED: {source.start_error}")
        return

    report_stats(stats, acc, cfg)

    image = acc.result()
    np.save(_REPO_ROOT / "vbf_result.npy", image)
    print(f"[main] done. saved vbf_result.npy to {_REPO_ROOT}")
    if live is not None:
        live.update(image)
        live.fig.savefig(_REPO_ROOT / "vbf_result.png", dpi=150)
        print("[main] saved vbf_result.png; close the window to exit.")
        live.keep_open()


if __name__ == "__main__":
    main()
