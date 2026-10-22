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
import sys

from holoscan.core import Application

from dectris import (ImagemuxCodec, LocalInfo, RdmaRxOp, ScanShape, TcpConnector, VerbsConn, run,
                     watch_liveness)

from operators import add_virtual_detectors  # this app's operators, next to this file
from settings import DEFAULTS, resolve

HERE = os.path.dirname(os.path.abspath(__file__))


class RdmaApp(Application):
    """Settings come from the YAML file (app.config); the connection is attached once made."""

    def attach(self, settings, conn, ring, credit, *, ring_device=None,
               ring_device_address=None):
        self.settings = settings
        self.conn, self.ring, self.credit = conn, ring, credit
        self.ring_device = ring_device
        self.ring_device_address = ring_device_address

    def compose(self):
        rx = RdmaRxOp(self, name="rx", conn=self.conn, ring=self.ring,
                      credit=self.credit, ring_device=self.ring_device,
                      ring_device_address=self.ring_device_address,
                      **self.settings["receiver"])
        add_virtual_detectors(self, rx, ScanShape(**self.settings["scan"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(HERE, "pipeline.yaml"),
                    help="Holoscan YAML configuration (default: pipeline.yaml next to this file)")
    config = ap.parse_known_args()[0].config

    app = RdmaApp()
    app.config(config)
    try:
        settings = resolve({name: app.kwargs(name) for name in DEFAULTS})
    except ValueError as e:
        sys.exit(f"{config}: {e}")
    conn, execute = settings["connection"], settings["execute"]
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

        app.attach(settings, link.conn, ring, link.peer.credit,
                   ring_device=conn["ring_gpu"],
                   ring_device_address=verbs.ring_device_address)
        closing = watch_liveness(app, link)
        try:
            run(app, execute["seconds"], execute["threads"])
        finally:
            closing.set()
            link.close()
    finally:
        verbs.close()


if __name__ == "__main__":
    main()
