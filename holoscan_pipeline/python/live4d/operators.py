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

_library.dectris_batch_window_create.argtypes = [ctypes.c_uint32]
_library.dectris_batch_window_create.restype = ctypes.c_void_p
_library.dectris_batch_window_destroy.argtypes = [ctypes.c_void_p]
_library.dectris_batch_window_destroy.restype = None
_library.dectris_batch_window_release.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
_library.dectris_batch_window_release.restype = None


class _PublicationWindow:
    """Bound native publications until the downstream callback releases each batch."""

    def __init__(self, capacity):
        self.handle = _library.dectris_batch_window_create(capacity)
        if not self.handle:
            raise RuntimeError("Invalid batch publication window")

    def release(self, batch_id):
        _library.dectris_batch_window_release(self.handle, batch_id)

    def __del__(self):
        handle = getattr(self, "handle", None)
        if handle:
            _library.dectris_batch_window_destroy(handle)
            self.handle = None
