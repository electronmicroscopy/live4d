"""Imagemux RDMA reception and standalone Holoscan SDK 5 operators.

The package provides the native verbs binding, wire protocol, registered-ring ownership,
GPU dispatch, and scan-position processing. Imports are lazy so protocol modules can be
used without Holoscan or CuPy.

``VerbsConn`` owns the queue pair and a GPU or mapped-host ring. ``TcpConnector`` performs
imagemux's handshake. The example app binds those live resources to a graph-local
``RdmaRxOp`` subclass, then compiles and runs an HSDK 5 graph in one process.

``RdmaRxOp.output0`` carries an ``hs.pyobject`` list containing ordered ``Start``, ``Batch``,
and ``End`` values. ``Batch.images`` is a CuPy array; wait for ``Batch.ready`` on the consuming
CUDA stream before reading it. The receiver returns ring credits after the copy completes.

``ScanTensorOp`` consumes those lists and emits one ``(index, frames)`` pair on ``output``.
``index`` preserves ``Start``, ``Segment``, and ``End`` ordering. ``frames`` is a dictionary
containing ``diffraction`` (k, 96, 96) uint16, ``positions`` (k, 2) int32 raster coordinates,
and ``ids`` (k,) int64 CuPy arrays, or ``None`` when there are no images. Arrays are complete
when emitted. Keeping both values in one message preserves their pairing across series.

``watch_liveness`` returns local-close and sender-disconnect events for the application's
session loop. ``examples/pipeline/app.py`` shows graph construction, configuration, virtual
detector processing, and cooperative shutdown.
"""

import importlib

__version__ = "0.1.0"

# public name -> submodule that defines it
_EXPORTS = {
    # operators
    "RdmaRxOp": "operators", "RingPath": "operators", "ScanTensorOp": "operators",
    # setup and running
    "VerbsConn": "connection", "watch_liveness": "connection",
    # what the operators pass along a series
    "Series": "messages", "Start": "messages", "Batch": "messages", "End": "messages",
    "IMAGE_SHAPE": "messages", "IMAGE_BYTES": "messages",
    "ScanShape": "messages", "Segment": "messages", "scan_positions": "messages",
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
