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
from .completion import CompletionTracker
from .operators import DecompressGpuOp, RdmaRxOp, ZmqRxOp, _PublicationWindow
from .processing import ScanTensorOp, VirtualDetectorOp


class PipelineApp:
    def __init__(self, config: PipelineConfig, *, conn=None, ring=None, credit=None,
                 ring_device_address=None, observer=None, observer_host_ready=False):
        self.config = config
        self.ready = threading.Event()
        # One spare queue slot covers a released input until its callback commits consumption.
        self.decoded_window = _PublicationWindow(config.queue_depth - 1,
                                                 consumers=("scan", "detector"))
        self.completions = CompletionTracker(config.queue_depth - 1, config.device,
                                             self.decoded_window.consumer("detector"))
        self.graph = hs.Graph("dectris-4dstem", default_clock=hs.RealtimeClock)
        graph = self.graph
        if config.transport == "zmq":
            self.compressed_window = _PublicationWindow(config.queue_depth - 1)
            rx = ZmqRxOp(graph, "zmq_rx", endpoint=config.zmq["endpoint"], batch_size=config.batch_size,
                         publication_window=self.compressed_window.handle)
            decoded = DecompressGpuOp(graph, "decompress_gpu",
                                      publication_window=self.decoded_window.handle,
                                      input_window=self.compressed_window.handle)
            graph.add_flow(rx.output, decoded.input, queue_depth=config.queue_depth)
            graph.partition("receiver").add(rx)
            graph.partition("decode").add(decoded)
        else:
            if conn is None or ring is None or credit is None:
                raise ValueError("RDMA graph requires a connected receiver and registered ring")
            # These owners must outlive the compiled graph and its runtime session.
            self.conn = conn
            decoded = RdmaRxOp(graph, "rdma_rx", qp_handle=conn.native_handle,
                               ring_address=ring.addr, ring_device_address=ring_device_address,
                               ring_bytes=ring.length, ring_memory=config.rdma["ring_memory"],
                               credit_address=credit.addr, credit_rkey=credit.rkey,
                               batch_size=config.batch_size, max_batch=config.rdma["max_batch"],
                               device=config.device, publication_window=self.decoded_window.handle)
            graph.partition("receiver").add(decoded)
        scan_release = self.decoded_window.consumer("scan")

        class BoundScan(ScanTensorOp):
            def __init__(self):
                super().__init__(release=scan_release)

        scan = graph.op(BoundScan, "scan", scan_width=config.scan["width"],
                        scan_height=config.scan["height"])
        graph.add_flow(decoded.output, scan.input, queue_depth=config.queue_depth)
        ready, finite = self.ready, config.execute["mode"] == "finite"
        completions = self.completions

        class BoundDetector(VirtualDetectorOp):
            def __init__(self):
                super().__init__(ready=ready, finite=finite, observer=observer,
                                 completions=completions, observer_host_ready=observer_host_ready)

        detector = graph.op(BoundDetector, "detector", device_id=config.device,
                            scan_width=config.scan["width"], scan_height=config.scan["height"],
                            batch_size=config.batch_size, **config.detector)
        graph.add_flow(decoded.output, detector.input, queue_depth=config.queue_depth)
        graph.add_flow(scan.output, detector.positions, queue_depth=config.queue_depth)
        graph.partition("scan").add(scan)
        graph.partition("detector").add(detector)
        deployment = hs.execution.DeploymentBinding()
        deployment.bind_tensor_output_device(
            operator_path="decompress_gpu" if config.transport == "zmq" else "rdma_rx",
            output_port="output", device=hs.execution.DeviceId(config.device))
        self.plan = hs.compile(graph, hs.CompileOptions(deployment=deployment))
        if not self.plan.ok:
            raise RuntimeError(self.plan.summary())

    def run(self, sender_gone=None, *, stop_event=None):
        stopping = stop_event if stop_event is not None else threading.Event()
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, lambda *_: stopping.set())
        try:
            with hs.run_async(self.plan) as session:
                deadline = None
                reported = False
                while not session.wait(timeout=0.05):
                    self.completions.check()
                    if self.ready.is_set() and not reported:
                        reported = True
                        seconds = self.config.execute["seconds"]
                        deadline = time.monotonic() + seconds if seconds else None
                    if (stopping.is_set() or (sender_gone is not None and sender_gone.is_set())
                            or (deadline is not None and time.monotonic() >= deadline)):
                        session.request_stop()
                if session.termination not in (hs.RunTermination.REQUESTED_COMPLETED,
                                                hs.RunTermination.COMPLETED,
                                                hs.RunTermination.STOPPED):
                    raise RuntimeError(f"Holoscan run failed: {session.summary()}")
                return session.termination
        finally:
            try:
                self.completions.close()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--transport", choices=("zmq", "rdma"))
    parser.add_argument("--endpoint", help="Override zmq.endpoint")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--queue-depth", type=int)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {key: getattr(args, key) for key in ("transport", "batch_size", "device", "queue_depth")
                 if getattr(args, key) is not None}
    if args.endpoint:
        overrides["zmq"] = config.zmq | dict(endpoint=args.endpoint)
    config = replace(config, **overrides)
    if config.transport == "zmq":
        app = PipelineApp(config)
        if args.plan:
            args.plan.write_text(app.plan.json(), encoding="utf-8")
        app.run()
        return
    from dectris import ImagemuxCodec, LocalInfo, TcpConnector, VerbsConn, watch_liveness
    rdma = config.rdma
    host, _, port = rdma["remote_address"].rpartition(":")
    verbs = VerbsConn(rdma["local_address"], config.device, rdma["ring_mib"] << 20,
                      rdma["device"] or None, ring_memory=rdma["ring_memory"])
    try:
        link = TcpConnector(host, ImagemuxCodec(), lambda: LocalInfo(verbs.ring, verbs.endpoint),
                            verbs.connect, port=int(port), timeout_s=rdma["timeout"]).connect()
        closing, sender_gone = watch_liveness(link)
        try:
            app = PipelineApp(config, conn=link.conn, ring=verbs.ring, credit=link.peer.credit,
                              ring_device_address=verbs.ring_device_address)
            if args.plan:
                args.plan.write_text(app.plan.json(), encoding="utf-8")
            app.run(sender_gone)
        finally:
            closing.set()
            link.close()
    finally:
        verbs.close()


if __name__ == "__main__":
    main()
