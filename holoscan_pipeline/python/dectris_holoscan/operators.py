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
