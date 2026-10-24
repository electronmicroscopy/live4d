"""Run either DECTRIS transport through the shared Holoscan 4D-STEM pipeline."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import signal
import threading
import time

import holoscan as hs

from .config import PipelineConfig, load_config
from .operators import DecompressGpuOp, RdmaRxOp, ZmqRxOp, _PublicationWindow
from .processing import ScanTensorOp, VirtualDetectorOp


class PipelineApp:
    def __init__(self, config: PipelineConfig, *, observer=None, receiver_ready=None):
        self.config = config
        self.ready = threading.Event()
        self.rdma = self.zmq = self.receiver = None
        self._run_started = False
        self.graph = hs.Graph("dectris-4dstem", default_clock=hs.RealtimeClock)
        self.branch_ready, self.windows = [], []
        deployment = hs.execution.DeploymentBinding()
        try:
            if config.transport == "rdma":
                from .rdma import RdmaReceiver
                self.rdma = self.receiver = RdmaReceiver(config, receiver_ready)
            else:
                from .zmq import ZmqReceiver
                self.zmq = self.receiver = ZmqReceiver(config)
                rx = ZmqRxOp(self.graph, "zmq_rx", endpoint=config.zmq["endpoint"],
                    batch_size=config.batch_size,
                    publication_windows=[window.handle for window in self.zmq.inputs],
                    acquisition_windows=[window.handle for window in self.zmq.acquisitions],
                    acquisition_limit=1 if config.execute["mode"] == "finite" else 0)
                self.graph.partition("receiver").add(rx)
            for index, device in enumerate(config.gpu_devices):
                prefix = f"gpu{index}_"
                warmed = threading.Event()
                window = _PublicationWindow(config.queue_depth - 1)
                self.branch_ready.append(warmed)
                self.windows.append(window)
                if self.rdma is None:
                    source_path = prefix + "decompress_gpu"
                    decoded = DecompressGpuOp(self.graph, source_path,
                        publication_window=window.handle,
                        input_window=self.zmq.inputs[index].handle, destination=index, device=device)
                    self.graph.add_flow(rx.output, decoded.input, queue_depth=config.queue_depth)
                    self.graph.partition(prefix + "decode").add(decoded)
                    completion = self.zmq.completion(index)
                else:
                    ring = self.rdma.rings[index]
                    source_path = prefix + "rdma_rx"
                    decoded = RdmaRxOp(self.graph, source_path,
                        session_handle=self.rdma.sessions[index].handle,
                        ring_address=ring.region.addr, ring_device_address=ring.device_address,
                        ring_bytes=ring.region.length, ring_memory=config.rdma["ring_memory"],
                        device=device, batch_size=config.batch_size, max_batch=config.rdma["max_batch"],
                        publication_window=window.handle)
                    self.graph.partition(prefix + "receiver").add(decoded)
                    completion = self.rdma.results[index].submit
                self._add_processing(decoded, prefix, device, warmed, window, observer, completion)
                deployment.bind_tensor_output_device(operator_path=source_path, output_port="output",
                                                       device=hs.execution.DeviceId(device))
            self.plan = hs.compile(self.graph, hs.CompileOptions(deployment=deployment))
            if not self.plan.ok:
                raise RuntimeError(self.plan.summary())
        except BaseException:
            if self.receiver is not None:
                self.receiver.close()
            raise

    def _add_processing(self, decoded, prefix, device, ready, window, observer, completion):
        # Both transports and every GPU use these same operators and flows.
        config, graph = self.config, self.graph
        scan = graph.op(ScanTensorOp, prefix + "scan", scan_width=config.scan["width"],
                        scan_height=config.scan["height"])
        finite = config.execute["mode"] == "finite"

        class BoundDetector(VirtualDetectorOp):
            def __init__(self):
                super().__init__(ready=ready, finite=finite, observer=observer,
                                 release=window.release, completion=completion)

        detector = graph.op(BoundDetector, prefix + "detector", device_id=device,
            scan_width=config.scan["width"], scan_height=config.scan["height"],
            batch_size=config.batch_size, **config.detector)
        graph.add_flow(decoded.output, scan.input, queue_depth=config.queue_depth)
        graph.add_flow(decoded.output, detector.input, queue_depth=config.queue_depth)
        graph.add_flow(scan.output, detector.positions, queue_depth=config.queue_depth)
        graph.partition(prefix + "scan").add(scan)
        graph.partition(prefix + "detector").add(detector)

    def run(self, sender_gone=None, *, stop_event=None):
        if self._run_started:
            raise RuntimeError(f"Construct a new {self.config.transport.upper()} PipelineApp for another run")
        self._run_started = True
        stopping = stop_event if stop_event is not None else threading.Event()
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, lambda *_: stopping.set())
        try:
            with hs.run_async(self.plan) as session:
                deadline = None
                reported = False
                while not session.wait(timeout=0.001):
                    if all(event.is_set() for event in self.branch_ready):
                        self.receiver.tick()
                        if self.rdma is None or self.rdma.last_ready is not None:
                            self.ready.set()
                    if self.ready.is_set() and not reported:
                        print("[holoscan] ready", flush=True)
                        reported = True
                        seconds = self.config.execute["seconds"]
                        deadline = time.monotonic() + seconds if seconds else None
                    finite_done = (self.receiver.completed
                                   and self.config.execute["mode"] == "finite")
                    if (finite_done or stopping.is_set()
                            or (sender_gone is not None and sender_gone.is_set())
                            or (deadline is not None and time.monotonic() >= deadline)):
                        session.request_stop()
                if session.termination not in (hs.RunTermination.REQUESTED_COMPLETED,
                                                hs.RunTermination.COMPLETED,
                                                hs.RunTermination.STOPPED):
                    raise RuntimeError(f"Holoscan run failed: {session.summary()}")
                return session.termination
        finally:
            if self.receiver is not None:
                self.receiver.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--transport", choices=("zmq", "rdma"))
    parser.add_argument("--endpoint", help="Override zmq.endpoint")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--devices", type=int, nargs="+", help="Ordered CUDA device list")
    parser.add_argument("--queue-depth", type=int)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {key: getattr(args, key) for key in ("transport", "batch_size", "device", "queue_depth")
                 if getattr(args, key) is not None}
    if args.devices is not None:
        overrides["devices"] = args.devices
    elif args.device is not None:
        overrides["devices"] = None
    if args.endpoint:
        overrides["zmq"] = config.zmq | dict(endpoint=args.endpoint)
    config = replace(config, **overrides)
    app = PipelineApp(config)
    try:
        if args.plan:
            args.plan.write_text(app.plan.json(), encoding="utf-8")
        app.run()
    finally:
        if app.receiver is not None:
            app.receiver.close()


if __name__ == "__main__":
    main()
