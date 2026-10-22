"""Receiver of imagemux's RDMA transport for Holoscan: the source operator (RdmaRxOp), the wire
protocol (imagemux), and the RDMA verbs (imxrdma, on libibverbs). The operators that process the
images belong to the app (examples/pipeline/operators.py has WorkerOp and AggregateOp).

Everything below is imported on first use, so ``import dectris`` and the protocol modules
work without holoscan or cupy.

Building your own operators
---------------------------
``RdmaRxOp`` is the source. Set up the queue pair and the GPU ring with ``VerbsConn``, connect to
imagemux with ``TcpConnector(host, ImagemuxCodec(), lambda: LocalInfo(conn.ring, conn.endpoint),
conn.connect, port=...)``, and pass the link's queue pair, the ring and the peer's credit word to
``RdmaRxOp``. It has one output ``output{i}`` per ring (``RdmaRxOp.paths``, one for now), and each
emission is a list, in arrival order, of:

- ``Start(series)``: a series begins (sent to every output);
- ``Batch(series, ids, ready, images | sums)``: images of the series on the ring's GPU, as
  (k, 96, 96) uint16 ``images`` (``reduce=False``) or (k,) uint64 pixel ``sums``
  (``reduce=True``); wait for the CUDA event ``ready`` on your stream before reading them;
- ``End(series, images)``: the series is over, with the number of images received.

A custom worker is a Holoscan ``Operator`` with an input that receives these lists, connected with
``app.add_flow(rx, worker, {(f"output{path.index}", "input")})``. ``run(app, seconds, threads)`` and
``watch_liveness(app, link)`` are the scheduler and stop-on-disconnect helpers.
examples/pipeline/app.py (the receiver) is a complete app built this way.
"""

import importlib

__version__ = "0.1.0"

# public name -> submodule that defines it
_EXPORTS = {
    # operators
    "RdmaRxOp": "operators", "RingPath": "operators",
    # setup and running
    "VerbsConn": "connection", "run": "connection", "watch_liveness": "connection",
    # what the operators pass along a series
    "Series": "messages", "Start": "messages", "Batch": "messages", "End": "messages",
    "IMAGE_SHAPE": "messages", "IMAGE_BYTES": "messages",
    # wire protocol and connection
    "FrameReader": "imagemux", "FrameHeader": "imagemux", "FrameKind": "imagemux",
    "CreditTracker": "imagemux", "TcpConnector": "imagemux", "ImagemuxCodec": "imagemux",
    "LocalInfo": "imagemux", "Link": "imagemux", "Peer": "imagemux",
    "RemoteRegion": "imagemux", "QpEndpoint": "imagemux",
    # GPU dispatch
    "Dispatcher": "dispatch", "DeviceRing": "dispatch",
}
_SUBMODULES = ("imxrdma", "imagemux", "messages", "dispatch", "operators", "connection")


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_EXPORTS, *_SUBMODULES])


__all__ = list(_EXPORTS)
