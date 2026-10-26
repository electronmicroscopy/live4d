"""Load native SDK 5 registrations and expose reflected operator proxies."""
import ctypes
from pathlib import Path
import holoscan as hs

# Payload identities must exist before registry discovery reflects native ports.
from .schemas.compressed_batch import CompressedBatch
from .schemas.decoded_batch import DecodedBatch

_library = ctypes.CDLL(str(Path(__file__).with_name("libdectris_holoscan_ops.so")),
                       mode=ctypes.RTLD_GLOBAL)
ZmqRxOp = hs.registry.resolve("holoscan.dectris.ZmqRxOp")
DecompressGpuOp = hs.registry.resolve("holoscan.dectris.DecompressGpuOp")
RdmaRxOp = hs.registry.resolve("holoscan.dectris.RdmaRxOp")
H5WriterOp = hs.registry.resolve("holoscan.dectris.H5WriterOp")

_library.dectris_batch_window_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_library.dectris_batch_window_create.restype = ctypes.c_void_p
_library.dectris_batch_window_destroy.argtypes = [ctypes.c_void_p]
_library.dectris_batch_window_destroy.restype = None
_library.dectris_batch_window_release.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                                  ctypes.c_uint32]
_library.dectris_batch_window_release.restype = None
_library.dectris_completion_latch_create.argtypes = [ctypes.c_uint32]
_library.dectris_completion_latch_create.restype = ctypes.c_void_p
_library.dectris_completion_latch_destroy.argtypes = [ctypes.c_void_p]
_library.dectris_completion_latch_destroy.restype = None
_library.dectris_completion_latch_arrive.argtypes = [ctypes.c_void_p]
_library.dectris_completion_latch_arrive.restype = ctypes.c_bool


class _PublicationWindow:
    """Bound native publications until every downstream consumer releases each batch.

    Consumer 0 is the Python detector; native consumers release their own index."""

    def __init__(self, capacity, consumers=1):
        self.handle = _library.dectris_batch_window_create(capacity, consumers)
        if not self.handle:
            raise RuntimeError("Invalid batch publication window")

    def release(self, batch_id, consumer=0):
        _library.dectris_batch_window_release(self.handle, batch_id, consumer)

    def __del__(self):
        handle = getattr(self, "handle", None)
        if handle:
            _library.dectris_batch_window_destroy(handle)
            self.handle = None


class _CompletionLatch:
    """Let the last of several END consumers request finite-mode completion."""

    def __init__(self, parties):
        self.handle = _library.dectris_completion_latch_create(parties)
        if not self.handle:
            raise RuntimeError("Invalid completion latch")

    def arrive(self):
        return _library.dectris_completion_latch_arrive(self.handle)

    def __del__(self):
        handle = getattr(self, "handle", None)
        if handle:
            _library.dectris_completion_latch_destroy(handle)
            self.handle = None
