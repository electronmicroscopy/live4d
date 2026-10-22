"""4D-STEM receiver pipeline for the RDMA transport of the imagemux bstream output.

    imagemux ==WRITE_WITH_IMM==> frame ring ==> RdmaRx -> ScanTensor -> VirtualDetector
        ^------------- consumed ring position (RDMA WRITE) ----'          (per ring)

The images become 4D-STEM tensors positioned in the configured raster scan.

We are the TCP client of imagemux's RDMA handshake address: we send our queue pair and ring
(Hello), imagemux answers with its queue pair and credit word (Accept), and once our QP is
connected and RECEIVEs are posted we say Ready (imagemux.py). The ring uses GPU memory by default,
or mapped host memory with ``connection.ring_memory`` set to ``host_mapped``. Kernels read images
through the ring's CUDA address in either mode.

Its operator after ScanTensorOp (VirtualDetectorOp) is in operators.py next to it; everything
else comes from the imx-receiver wheel (the base image,
dectris/tools/build_base_image.sh); run it from anywhere:

    python3 examples/pipeline/app.py [--config examples/pipeline/pipeline.yaml]
"""

import argparse
import os
import signal
import sys
import threading
import time

import holoscan as hs
import yaml

from dectris import (ImagemuxCodec, LocalInfo, RdmaRxOp, ScanShape, TcpConnector, VerbsConn,
                     watch_liveness)

from operators import add_virtual_detectors
from settings import resolve

HERE = os.path.dirname(os.path.abspath(__file__))


def load_settings(path):
    with open(path, encoding="utf-8") as stream:
        return resolve(yaml.safe_load(stream) or {})


def compile_graph(settings, conn, ring, credit, *, ring_device=None,
                  ring_device_address=None, ready=None):
    graph = hs.Graph("dectris-rdma", default_clock=hs.RealtimeClock)

    class ConnectedRdmaRxOp(RdmaRxOp):
        def __init__(self):
            super().__init__(conn=conn, ring=ring, credit=credit,
                             ring_device=ring_device,
                             ring_device_address=ring_device_address)
            self.reported_ready = False

        def compute(self, context):
            super().compute(context)
            if ready is not None and not self.reported_ready:
                self.reported_ready = True
                if not self.lost:
                    ready.set()

    receiver = graph.op(ConnectedRdmaRxOp, "rx", **settings["receiver"])
    tensors, detector = add_virtual_detectors(
        graph, receiver, ScanShape(**settings["scan"]),
        device_id=settings["connection"]["ring_gpu"],
        max_batch=settings["receiver"]["max_batch"])

    threads = settings["execute"]["threads"]
    if threads <= 1:
        graph.partition("pipeline").include_all()
    else:
        graph.partition("receiver").add(receiver)
        if threads == 2:
            processing = graph.partition("processing")
            processing.add(tensors).add(detector)
        else:
            graph.partition("scan").add(tensors)
            graph.partition("detector").add(detector)

    plan = hs.compile(graph)
    if not plan.ok:
        raise RuntimeError(f"Holoscan plan rejected: {plan.summary()}")
    return plan


def run_plan(plan, seconds, sender_gone, started=None):
    stopping = threading.Event()
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stopping.set())
    try:
        with hs.run_async(plan) as session:
            if started is None:
                print("[holoscan] ready", flush=True)
                deadline = time.monotonic() + seconds if seconds > 0 else None
            else:
                deadline = None
                while not started.is_set() and not session.wait(timeout=0.1):
                    if stopping.is_set() or sender_gone.is_set():
                        session.request_stop()
                if started.is_set():
                    print("[holoscan] ready", flush=True)
                    deadline = time.monotonic() + seconds if seconds > 0 else None
            while not session.wait(timeout=0.1):
                if (stopping.is_set() or sender_gone.is_set()
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(HERE, "pipeline.yaml"),
                    help="Holoscan YAML configuration (default: pipeline.yaml next to this file)")
    config = ap.parse_args().config

    try:
        settings = load_settings(config)
    except ValueError as e:
        sys.exit(f"{config}: {e}")
    conn = settings["connection"]
    address = conn["remote_address"]
    host, _, port = address.rpartition(":")

    verbs = VerbsConn(conn["local_address"], conn["ring_gpu"], conn["ring_mib"] << 20,
                      conn["device"] or None, ring_memory=conn["ring_memory"])
    try:
        endpoint, ring = verbs.endpoint, verbs.ring
        link = TcpConnector(host, ImagemuxCodec(), lambda: LocalInfo(ring, endpoint),
                            verbs.connect, port=int(port),
                            timeout_s=conn["timeout"]).connect()
        print(f"connected to {address} via {verbs.ctx.device} port {verbs.ctx.port} "
              f"(gid index {verbs.ctx.gid_index}): ring {ring.length >> 20} MiB "
              f"addr={ring.addr:#x} rkey={ring.rkey:#x}; our qp {endpoint.qp_num} "
              f"psn {endpoint.psn} mtu {endpoint.mtu}, imagemux qp {link.peer.endpoint.qp_num} "
              f"psn {link.peer.endpoint.psn} mtu {link.peer.endpoint.mtu}, "
              f"credit addr={link.peer.credit.addr:#x} rkey={link.peer.credit.rkey:#x}",
              flush=True)

        closing, sender_gone = watch_liveness(link)
        try:
            started = threading.Event()
            plan = compile_graph(settings, link.conn, ring, link.peer.credit,
                                 ring_device=conn["ring_gpu"],
                                 ring_device_address=verbs.ring_device_address, ready=started)
            run_plan(plan, settings["execute"]["seconds"], sender_gone, started=started)
        finally:
            closing.set()
            link.close()
    finally:
        verbs.close()


if __name__ == "__main__":
    main()
