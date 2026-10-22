"""Receive-only pipeline for the RDMA transport of the imagemux bstream output.

    imagemux ==WRITE_WITH_IMM==> frame ring ==> RdmaRx -> Worker x N -> Aggregate
        ^------------- consumed ring position (RDMA WRITE) -------------'

We are the TCP client of imagemux's RDMA handshake address: we send our queue pair and ring
(Hello), imagemux answers with its queue pair and credit word (Accept), and once our QP is
connected and RECEIVEs are posted we say Ready (imagemux.py). The ring is in GPU memory
(GPUDirect RDMA): the CPU copies the frame headers out of it and the kernels read the images in place.

Its operators after the source (WorkerOp, AggregateOp) are in operators.py next to it; everything
else comes from the imx-receiver wheel (the base image,
dectris/tools/build_base_image.sh); run it from anywhere:

    python3 examples/pipeline/pipeline.py 10.42.41.10:31003 --local-addr 10.42.41.11
"""

import argparse

from holoscan.core import Application

from dectris import (ImagemuxCodec, LocalInfo, RdmaRxOp, TcpConnector, VerbsConn, run,
                     watch_liveness)

from operators import add_workers  # this app's operators, next to this file


def add_run_args(ap: argparse.ArgumentParser):
    ap.add_argument("--seconds", type=float, default=10.0, help="run time; 0 = until Ctrl+C")
    ap.add_argument("--series-width", type=int, default=None,
                    help="optional: images per series row, used to shape a series' image if it "
                         "divides the number of images the series turned out to have "
                         "(default: a single row)")


THREADS = 3  # source, worker and aggregator run concurrently


class RdmaApp(Application):
    def __init__(self, conn, ring, credit, series_width, reduce, receive_only=False):
        super().__init__()
        self.conn, self.ring, self.credit = conn, ring, credit
        self.series_width, self.reduce = series_width, reduce
        self.receive_only = receive_only

    def compose(self):
        rx = RdmaRxOp(self, name="rx", conn=self.conn, ring=self.ring,
                      credit=self.credit, reduce=self.reduce,
                      receive_only=self.receive_only)
        add_workers(self, rx, self.series_width)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local-addr", required=True,
                    help="address of the RoCE interface to use")
    ap.add_argument("--device", default=None,
                    help="RDMA device (default: the one holding --local-addr)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU holding the ring")
    ap.add_argument("--ring-mib", type=int, default=64, help="ring size in MiB")
    ap.add_argument("--connect-timeout", type=float, default=10,
                    help="seconds for the TCP connection to imagemux")
    ap.add_argument("--receive-only", action="store_true",
                    help="only receive: count images and return the credit at once, no GPU work "
                         "(the raw rate the sender delivers)")
    ap.add_argument("--keep-images", action="store_true",
                    help="copy the images to GPU memory instead of reducing them to pixel sums "
                         "straight from the ring (what a real reconstruction needs)")
    ap.add_argument("address", help="imagemux bstream RDMA handshake address, e.g. 10.0.0.1:31003")
    add_run_args(ap)
    args = ap.parse_args()
    host, _, port = args.address.rpartition(":")

    verbs = VerbsConn(args.local_addr, args.gpu, args.ring_mib << 20, args.device)
    try:
        endpoint, ring = verbs.endpoint, verbs.ring
        link = TcpConnector(host, ImagemuxCodec(), lambda: LocalInfo(ring, endpoint),
                            verbs.connect, port=int(port),
                            timeout_s=args.connect_timeout).connect()
        print(f"connected to {args.address} via {verbs.ctx.device} port {verbs.ctx.port} "
              f"(gid index {verbs.ctx.gid_index}): ring {ring.length >> 20} MiB "
              f"addr={ring.addr:#x} rkey={ring.rkey:#x}; our qp {endpoint.qp_num} "
              f"psn {endpoint.psn} mtu {endpoint.mtu}, imagemux qp {link.peer.endpoint.qp_num} "
              f"psn {link.peer.endpoint.psn} mtu {link.peer.endpoint.mtu}, "
              f"credit addr={link.peer.credit.addr:#x} rkey={link.peer.credit.rkey:#x}",
              flush=True)

        app = RdmaApp(link.conn, ring, link.peer.credit, args.series_width,
                      reduce=not args.keep_images, receive_only=args.receive_only)
        closing = watch_liveness(app, link)
        try:
            run(app, args.seconds, THREADS)
        finally:
            closing.set()
            link.close()
    finally:
        verbs.close()


if __name__ == "__main__":
    main()
