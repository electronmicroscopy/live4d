from __future__ import annotations

import argparse
import signal
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import holoscan as hs
import yaml

from .operators import DecompressGpuOp, ZmqRxOp
from .schemas.decoded_batch import DecodedBatch


@dataclass(frozen=True)
class BasicIngestConfig:
    endpoint: str = "tcp://127.0.0.1:5555"
    batch_size: int = 1000
    device: int = 0
    queue_depth: int = 4

    def __post_init__(self):
        if not 1 <= self.batch_size <= 1000:
            raise ValueError("batch_size must be between 1 and 1000")
        if self.device < 0 or self.queue_depth < 2:
            raise ValueError("device must be nonnegative and queue_depth must be at least 2")
        if not self.endpoint:
            raise ValueError("endpoint must not be empty")


class StatsSinkOp(hs.Operator):
    """Report device Tensor metadata without copying image pixels to the CPU."""
    inp = hs.Input("input", DecodedBatch)

    def setup(self, spec):
        spec.input(self.inp)
        self._start = None
        self._batches = 0
        self._frames = 0
        self._first_batch_frames = 0

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def process_batch(self, batch, ctx):
        count = len(batch.image_ids)
        if count:
            if self._start is None:
                self._start = time.perf_counter()
                self._first_batch_frames = count
            self._batches += 1
            self._frames += count
            elapsed = max(time.perf_counter() - self._start, 1e-9)
            fps = ((self._frames - self._first_batch_frames) / elapsed
                   if self._batches > 1 else None)
            fps_text = f"{fps:,.1f}" if fps is not None else "n/a"
            print(f"[stats] batch={batch.batch_id} batches={self._batches} "
                  f"frames={self._frames} fps={fps_text} "
                  f"last_id={batch.image_ids[-1]} shape={batch.frames.shape} "
                  f"dtype={batch.frames.dtype} device={batch.frames.device}", flush=True)

    def compute(self, ctx):
        batch = self.inp.receive_data()
        self.process_batch(batch, ctx)
        # This marker follows all data on both FIFO edges, including a partial batch.
        if batch.end_of_stream:
            ctx.request_completion()


class BasicIngestApp:
    def __init__(self, config: BasicIngestConfig, sink_type=StatsSinkOp):
        self.config = config
        self.graph = hs.Graph("dectris-ingest", default_clock=hs.RealtimeClock)
        rx = ZmqRxOp(self.graph, "zmq_rx", endpoint=config.endpoint, batch_size=config.batch_size)
        decompress = DecompressGpuOp(self.graph, "decompress_gpu")
        sink = self.graph.op(sink_type, "stats_sink")
        self.graph.add_flow(rx.output, decompress.input, queue_depth=config.queue_depth)
        self.graph.add_flow(decompress.output, sink.input, queue_depth=config.queue_depth)
        # Preserve FIFO and bounded callback ordering across this linear graph.
        self.graph.partition("ingest").include_all()
        deployment = hs.execution.DeploymentBinding()
        deployment.bind_tensor_output_device(
            operator_path="decompress_gpu", output_port="output",
            device=hs.execution.DeviceId(config.device))
        self.plan = hs.compile(self.graph, hs.CompileOptions(deployment=deployment))
        if not self.plan.ok:
            raise RuntimeError(self.plan.summary())

    def run(self):
        stopping = threading.Event()
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, lambda *_: stopping.set())
        try:
            with hs.run_async(self.plan) as session:
                print("[holoscan] ready", flush=True)
                while not session.wait(timeout=0.1):
                    if stopping.is_set():
                        session.request_stop()
                if session.termination not in (hs.RunTermination.REQUESTED_COMPLETED,
                                                hs.RunTermination.COMPLETED,
                                                hs.RunTermination.STOPPED):
                    raise RuntimeError(f"Holoscan run failed: {session.summary()}")
                return session.termination
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def load_config(path: Path | None) -> BasicIngestConfig:
    if path is None:
        return BasicIngestConfig()
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    # Fixed detector properties describe the supported input, not runtime parameters.
    for key, expected in (("detector_shape", [96, 96]), ("dtype", "uint16"), ("compression", "bslz4")):
        if data.pop(key, expected) != expected:
            raise ValueError(f"unsupported {key}; expected {expected!r}")
    return BasicIngestConfig(**data)


def main():
    parser = argparse.ArgumentParser(description="DECTRIS ingest pipeline for Holoscan SDK 5.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--queue-depth", type=int)
    parser.add_argument("--plan", type=Path, help="Save the compiled execution plan.")
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {key: getattr(args, key) for key in ("endpoint", "batch_size", "device", "queue_depth")
                 if getattr(args, key) is not None}
    app = BasicIngestApp(replace(config, **overrides))
    if args.plan:
        args.plan.write_text(app.plan.json(), encoding="utf-8")
    app.run()


if __name__ == "__main__":
    main()
