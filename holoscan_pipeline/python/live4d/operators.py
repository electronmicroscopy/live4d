"""Load native SDK 5 registrations and expose reflected operator proxies."""
import ctypes
from pathlib import Path
import threading
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
_library.dectris_batch_window_create_for_consumers.argtypes = [ctypes.c_uint32,
                                                               ctypes.c_uint32]
_library.dectris_batch_window_create_for_consumers.restype = ctypes.c_void_p
_library.dectris_batch_window_acknowledge.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                      ctypes.c_uint64]
_library.dectris_batch_window_acknowledge.restype = ctypes.c_int


class _PublicationWindow:
    """Bound admission by exact completion from every registered consumer."""

    def __init__(self, capacity, consumers=("default",)):
        if not isinstance(capacity, int) or capacity <= 0 or capacity > 0xFFFFFFFF:
            raise RuntimeError("Invalid batch publication window")
        names = tuple(consumers)
        try:
            unique = len(set(names)) == len(names)
        except TypeError:
            unique = False
        if not unique:
            raise ValueError("Publication consumers must have unique names")

        self._lock = threading.Lock()
        self._consumers = {name: index for index, name in enumerate(names)}
        try:
            self.handle = _library.dectris_batch_window_create_for_consumers(
                capacity, len(names))
        except (TypeError, OverflowError, ctypes.ArgumentError):
            self.handle = None
        if not self.handle:
            raise RuntimeError("Invalid batch publication window")

    def release(self, batch_id):
        self._acknowledge(0, batch_id)

    def consumer(self, name):
        """Return an acknowledgement callback for one registered consumer."""
        try:
            consumer_id = self._consumers[name]
        except (KeyError, TypeError):
            raise ValueError("Invalid publication acknowledgement") from None

        owner = self

        def acknowledge(batch_id):
            owner._acknowledge(consumer_id, batch_id)

        return acknowledge

    def _acknowledge(self, consumer_id, batch_id):
        with self._lock:
            if not self.handle:
                raise RuntimeError("Publication window is closed")
            try:
                acknowledged = _library.dectris_batch_window_acknowledge(
                    self.handle, consumer_id, batch_id)
            except (TypeError, OverflowError, ctypes.ArgumentError):
                acknowledged = 0
            if not acknowledged:
                raise ValueError("Invalid publication acknowledgement")

    def close(self):
        """Destroy the native window after all publishers and consumers stop."""
        with self._lock:
            handle = self.handle
            if handle:
                _library.dectris_batch_window_destroy(handle)
                self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
