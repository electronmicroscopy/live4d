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
from .operators import (DecompressGpuOp, H5WriterOp, RdmaRingsRxOp, RdmaRxOp, ZmqRxOp,
                        _CompletionLatch, _PublicationWindow)
from .processing import AnsEncoderOp, ScanTensorOp, VirtualDetectorOp


def _multi_lane(config: PipelineConfig) -> bool:
    return config.devices is not None


class PipelineApp:
    def __init__(self, config: PipelineConfig, *, conn=None, ring=None, credit=None,
                 ring_device_address=None, observer=None, encoder_sink=None, receiver_ready=None):
        self.config = config
        self.ready = threading.Event()
        self.lane_mode = _multi_lane(config)
        if self.lane_mode:
            self._init_lanes(conn, ring, credit, observer, encoder_sink, receiver_ready)
            return
        writing = config.writer["enabled"]
        encoding = config.encoder["enabled"]
        if encoding and encoder_sink is None:
            from .algorithms.ans_encoding import ResidentSeries
            encoder_sink = ResidentSeries(config.device, config.encoder["keep"])
        self.encoder_sink = encoder_sink
        finite = config.execute["mode"] == "finite"
        # One spare queue slot covers a released input until its callback commits consumption.
        # The HDF5 writer is decoded consumer 1; the ANS encoder is the next one.
        consumers = 1 + writing + encoding
        self.decoded_window = _PublicationWindow(config.queue_depth - 1, consumers)
        # Requested completion does not drain queues; the last END consumer requests it.
        self.completion = _CompletionLatch(consumers) if consumers > 1 and finite else None
        self.graph = hs.Graph("dectris-4dstem", default_clock=hs.RealtimeClock)
        graph = self.graph
        if config.transport == "zmq":
            self.compressed_window = _PublicationWindow(config.queue_depth - 1)
            rx = ZmqRxOp(graph, "zmq_rx", endpoint=config.zmq["endpoint"], batch_size=config.batch_size,
                         publication_window=self.compressed_window.handle)
            decoded = DecompressGpuOp(graph, "decompress_gpu",
                                      publication_window=self.decoded_window.handle,
                                      input_window=self.compressed_window.handle,
                                      device=config.device)
            graph.add_flow(rx.output, decoded.input, queue_depth=config.queue_depth)
            graph.partition("receiver").add(rx)
            graph.partition("decode").add(decoded)
        else:
            if conn is None or ring is None or credit is None:
                raise ValueError("RDMA graph requires a connected receiver and registered ring")
            # These owners must outlive the compiled graph and its runtime session.
            self.conn = conn
            rdma = config.rdma
            common = dict(qp_handle=conn.native_handle, ring_address=ring.addr,
                          ring_device_address=ring_device_address,
                          ring_memory=rdma["ring_memory"], credit_address=credit.addr,
                          credit_rkey=credit.rkey, batch_size=config.batch_size,
                          max_batch=rdma["max_batch"], device=config.device,
                          publication_window=self.decoded_window.handle)
            if rdma["protocol"] == 3:
                # ``ring`` is the registration holding every ring back to back.
                decoded = RdmaRingsRxOp(graph, "rdma_rx", ring_bytes=ring.length // rdma["rings"],
                                        ring_count=rdma["rings"], **common)
            else:
                decoded = RdmaRxOp(graph, "rdma_rx", ring_bytes=ring.length, **common)
            graph.partition("receiver").add(decoded)
        scan = graph.op(ScanTensorOp, "scan", scan_width=config.scan["width"],
                        scan_height=config.scan["height"])
        graph.add_flow(decoded.output, scan.input, queue_depth=config.queue_depth)
        ready, completion = self.ready, self.completion
        release = self.decoded_window.release

        class BoundDetector(VirtualDetectorOp):
            def __init__(self):
                super().__init__(ready=ready, finite=finite, observer=observer, release=release,
                                 completion=completion)

        detector = graph.op(BoundDetector, "detector", device_id=config.device,
                            scan_width=config.scan["width"], scan_height=config.scan["height"],
                            batch_size=config.batch_size, **config.detector)
        graph.add_flow(decoded.output, detector.input, queue_depth=config.queue_depth)
        graph.add_flow(scan.output, detector.positions, queue_depth=config.queue_depth)
        graph.partition("scan").add(scan)
        graph.partition("detector").add(detector)
        if writing:
            writer = H5WriterOp(graph, "h5_writer", output_dir=config.writer["output_dir"],
                                name_pattern=config.writer["name_pattern"],
                                format=config.writer["format"],
                                images_per_file=config.writer["images_per_file"],
                                staging_batches=config.writer["staging_batches"],
                                batch_size=config.batch_size, device=config.device,
                                publication_window=self.decoded_window.handle, window_consumer=1,
                                completion_latch=self.completion.handle if self.completion else 0)
            graph.add_flow(decoded.output, writer.input, queue_depth=config.queue_depth)
            graph.partition("writer").add(writer)
        if encoding:
            class BoundEncoder(AnsEncoderOp):
                def __init__(self):
                    super().__init__(sink=encoder_sink, release=release, consumer=1 + writing,
                                     finite=finite, completion=completion)

            encoder = graph.op(BoundEncoder, "ans_encoder", device_id=config.device,
                               scan_width=config.scan["width"], scan_height=config.scan["height"])
            graph.add_flow(decoded.output, encoder.input, queue_depth=config.queue_depth)
            # Its own partition: the once-per-chunk encode wait stays off the receiver and detector threads.
            graph.partition("encoder").add(encoder)
        deployment = hs.execution.DeploymentBinding()
        deployment.bind_tensor_output_device(
            operator_path="decompress_gpu" if config.transport == "zmq" else "rdma_rx",
            output_port="output", device=hs.execution.DeviceId(config.device))
        self.plan = hs.compile(graph, hs.CompileOptions(deployment=deployment))
        if not self.plan.ok:
            raise RuntimeError(self.plan.summary())

    def _init_lanes(self, conn, ring, credit, observer, encoder_sink, receiver_ready):
        config = self.config
        if conn is not None or ring is not None or credit is not None:
            raise ValueError("Multiple devices own their RDMA connection; do not pass conn, ring or credit")
        if config.writer["enabled"] or config.encoder["enabled"] or encoder_sink is not None:
            raise ValueError("The HDF5 writer and the ANS encoder are not supported with `devices` "
                             "yet; omit `devices` to use them")
        writing = config.writer["enabled"]
        self.finite = config.execute["mode"] == "finite"
        self.lane_done = threading.Event()
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
                             acquisition_limit=1 if self.finite else 0)
                self.graph.partition("receiver").add(rx)
            for index, device in enumerate(config.gpu_devices):
                prefix = f"gpu{index}_"
                warmed = threading.Event()
                consumers = 1 + writing
                window = _PublicationWindow(config.queue_depth - 1, consumers)
                latch = _CompletionLatch(consumers) if consumers > 1 and self.finite else None
                self.branch_ready.append(warmed)
                self.windows.append(window)
                if self.rdma is None:
                    source_path = prefix + "decompress_gpu"
                    decoded = DecompressGpuOp(self.graph, source_path,
                                              publication_window=window.handle,
                                              input_window=self.zmq.inputs[index].handle,
                                              destination=index, device=device)
                    self.graph.add_flow(rx.output, decoded.input, queue_depth=config.queue_depth)
                    self.graph.partition(prefix + "decode").add(decoded)
                    results = self.zmq.completion(index)
                else:
                    lane_ring = self.rdma.rings[index]
                    source_path = prefix + "rdma_rx"
                    decoded = RdmaRxOp(self.graph, source_path,
                                       session_handle=self.rdma.sessions[index].handle,
                                       ring_address=lane_ring.region.addr,
                                       ring_device_address=lane_ring.device_address,
                                       ring_bytes=lane_ring.region.length,
                                       ring_memory=config.rdma["ring_memory"],
                                       device=device, batch_size=config.batch_size,
                                       max_batch=config.rdma["max_batch"],
                                       publication_window=window.handle)
                    self.graph.partition(prefix + "receiver").add(decoded)
                    results = self.rdma.results[index].submit
                self._add_lane_processing(decoded, prefix, device, warmed, window, latch, results,
                                          observer)
                deployment.bind_tensor_output_device(operator_path=source_path, output_port="output",
                                                     device=hs.execution.DeviceId(device))
            self.plan = hs.compile(self.graph, hs.CompileOptions(deployment=deployment))
            if not self.plan.ok:
                raise RuntimeError(self.plan.summary())
        except BaseException:
            self.close()
            raise

    def _add_lane_processing(self, decoded, prefix, device, ready, window, latch, results, observer):
        config, graph = self.config, self.graph
        scan = graph.op(ScanTensorOp, prefix + "scan", scan_width=config.scan["width"],
                        scan_height=config.scan["height"])
        finite, done = self.finite, self.lane_done.set

        class BoundDetector(VirtualDetectorOp):
            def __init__(self):
                super().__init__(ready=ready, finite=finite, observer=observer,
                                 release=window.release, completion=latch, results=results,
                                 done=done, announce_ready=False)

        detector = graph.op(BoundDetector, prefix + "detector", device_id=device,
                            scan_width=config.scan["width"], scan_height=config.scan["height"],
                            batch_size=config.batch_size, **config.detector)
        graph.add_flow(decoded.output, scan.input, queue_depth=config.queue_depth)
        graph.add_flow(decoded.output, detector.input, queue_depth=config.queue_depth)
        graph.add_flow(scan.output, detector.positions, queue_depth=config.queue_depth)
        graph.partition(prefix + "scan").add(scan)
        graph.partition(prefix + "detector").add(detector)
        if config.writer["enabled"]:
            writer = H5WriterOp(graph, prefix + "h5_writer", output_dir=config.writer["output_dir"],
                                name_pattern=config.writer["name_pattern"],
                                format=config.writer["format"],
                                images_per_file=config.writer["images_per_file"],
                                staging_batches=config.writer["staging_batches"],
                                batch_size=config.batch_size, device=device,
                                publication_window=window.handle, window_consumer=1,
                                completion_latch=latch.handle if latch else 0)
            graph.add_flow(decoded.output, writer.input, queue_depth=config.queue_depth)
            graph.partition(prefix + "writer").add(writer)

    def _run_lanes(self, sender_gone, stop_event):
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
                    if ((self.finite and self.lane_done.is_set()) or stopping.is_set()
                            or (sender_gone is not None and sender_gone.is_set())
                            or (deadline is not None and time.monotonic() >= deadline)):
                        session.request_stop()
                if session.termination not in (hs.RunTermination.REQUESTED_COMPLETED,
                                                hs.RunTermination.COMPLETED,
                                                hs.RunTermination.STOPPED):
                    raise RuntimeError(f"Holoscan run failed: {session.summary()}")
                return session.termination
        finally:
            self.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def close(self):
        receiver = getattr(self, "receiver", None)
        if receiver is not None:
            receiver.close()

    def run(self, sender_gone=None, *, stop_event=None):
        if self.lane_mode:
            return self._run_lanes(sender_gone, stop_event)
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
    if _multi_lane(config):
        app = PipelineApp(config)
        try:
            if args.plan:
                args.plan.write_text(app.plan.json(), encoding="utf-8")
            app.run()
        finally:
            app.close()
        return
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
                      rdma["device"] or None, ring_memory=rdma["ring_memory"], rings=rdma["rings"])
    try:
        link = TcpConnector(host, ImagemuxCodec(rdma["protocol"]),
                            lambda: LocalInfo(verbs.ring, verbs.endpoint, verbs.rings),
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
