from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from holoscan.core import Application, Operator, OperatorSpec

from dectris_holoscan_ops import DecompressGpuOp, ZmqRxOp


@dataclass(frozen=True)
class BasicIngestConfig:
    endpoint: str = "tcp://127.0.0.1:5555"
    batch_size: int = 2048
    scheduler_threads: int = 4
    detector_shape: tuple[int, int] = (96, 96)  # (height, width); validated vs the stream
    dtype: str = "uint16"  # 'uint16' or 'uint32'; validated vs the stream


def _tensor_from_dlpack(tensor):
    """Convert a Holoscan Tensor-like object to torch.Tensor without a copy."""
    if hasattr(tensor, "__dlpack__"):
        return torch.from_dlpack(tensor)
    if hasattr(tensor, "to_dlpack"):
        return torch.from_dlpack(tensor.to_dlpack())
    raise TypeError(f"Object does not expose DLPack: {type(tensor)!r}")


def _get_tensor(message, name: str):
    if isinstance(message, dict):
        return message[name]
    if hasattr(message, "get"):
        value = message.get(name)
        if value is not None:
            return value
    return message[name]


class StatsSinkOp(Operator):
    """Python-only sink that converts decompressor output to PyTorch and prints throughput."""

    def __init__(self, fragment, *args, name: str = "stats_sink"):
        super().__init__(fragment, *args, name=name)
        self._start = None
        self._batches = 0
        self._frames = 0

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("input")

    def compute(self, op_input, op_output, context) -> None:
        message = op_input.receive("input")
        frames = _tensor_from_dlpack(_get_tensor(message, "frames"))
        image_ids = _tensor_from_dlpack(_get_tensor(message, "image_ids")).cpu()
        batch_id = int(_tensor_from_dlpack(_get_tensor(message, "batch_id")).cpu()[0].item())

        if self._start is None:
            self._start = time.perf_counter()

        num_frames = int(frames.shape[0])
        self._batches += 1
        self._frames += num_frames

        elapsed = max(time.perf_counter() - self._start, 1e-9)
        fps = self._frames / elapsed
        last_id = int(image_ids[-1].item()) if image_ids.numel() else -1

        print(
            "[stats] "
            f"batch={batch_id} batches={self._batches} frames={self._frames} "
            f"fps={fps:,.1f} last_id={last_id} "
            f"shape={tuple(frames.shape)} dtype={frames.dtype} device={frames.device}",
            flush=True,
        )


class BasicIngestApp(Application):
    def __init__(self, config: BasicIngestConfig):
        super().__init__()
        self.config = config

    def compose(self) -> None:
        if self.config.scheduler_threads > 0:
            try:
                from holoscan.schedulers import MultiThreadScheduler

                self.scheduler(
                    MultiThreadScheduler(
                        self,
                        worker_thread_number=self.config.scheduler_threads,
                    )
                )
            except Exception as exc:
                print(f"[holoscan] using default scheduler: {exc}", flush=True)

        height, width = self.config.detector_shape
        rx = ZmqRxOp(
            self,
            endpoint=self.config.endpoint,
            batch_size=self.config.batch_size,
            detector_width=width,
            detector_height=height,
            dtype=self.config.dtype,
            name="zmq_rx",
        )
        decompress = DecompressGpuOp(
            self,
            detector_width=width,
            detector_height=height,
            dtype=self.config.dtype,
            name="decompress_gpu",
        )
        sink = StatsSinkOp(self, name="stats_sink")

        self.add_flow(rx, decompress, {("output", "input")})
        self.add_flow(decompress, sink, {("output", "input")})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic DECTRIS Holoscan ingest pipeline.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--scheduler-threads", type=int, default=None)
    parser.add_argument("--detector-shape", type=int, nargs=2, default=None,
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--dtype", choices=("uint16", "uint32"), default=None)
    return parser.parse_args()


def load_config(path: Path | None) -> BasicIngestConfig:
    data = {}
    if path is not None:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
    return BasicIngestConfig(
        endpoint=data.get("endpoint", BasicIngestConfig.endpoint),
        batch_size=int(data.get("batch_size", BasicIngestConfig.batch_size)),
        scheduler_threads=int(data.get("scheduler_threads", BasicIngestConfig.scheduler_threads)),
        detector_shape=tuple(data.get("detector_shape", BasicIngestConfig.detector_shape)),
        dtype=str(data.get("dtype", BasicIngestConfig.dtype)),
    )


def main() -> None:
    args = parse_args()
    file_config = load_config(args.config)
    config = BasicIngestConfig(
        endpoint=args.endpoint or file_config.endpoint,
        batch_size=args.batch_size or file_config.batch_size,
        scheduler_threads=args.scheduler_threads or file_config.scheduler_threads,
        detector_shape=tuple(args.detector_shape) if args.detector_shape
        else file_config.detector_shape,
        dtype=args.dtype or file_config.dtype,
    )
    app = BasicIngestApp(config)
    app.run()


if __name__ == "__main__":
    main()
