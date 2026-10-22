"""Reusable imagemux RDMA transport, registered rings, and GPU dispatch helpers.

``VerbsConn`` owns the native queue pair and a GPU or mapped-host ring.
``TcpConnector`` performs the imagemux handshake. Protocol and geometry modules
can be imported without Holoscan or CuPy. The shared HSDK 5 application lives in
``dectris_holoscan``; it owns graph construction and 4D-STEM processing.
"""

import importlib

__version__ = "0.1.0"

# public name -> submodule that defines it
_EXPORTS = {
    # setup and running
    "VerbsConn": "connection", "watch_liveness": "connection",
    # series messages and image geometry
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
_SUBMODULES = ("imxrdma", "imagemux", "messages", "dispatch", "connection")


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_EXPORTS, *_SUBMODULES])


__all__ = list(_EXPORTS)
