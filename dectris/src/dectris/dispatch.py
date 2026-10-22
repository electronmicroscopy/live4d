"""Copy images from an RDMA ring into GPU tensors.

Kernels read the ring through its CUDA address, whether it is GPU memory or mapped host memory.
"""

import ctypes
from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

from .messages import IMAGE_BYTES, IMAGE_SHAPE

_THREADS = 128

# ptrs holds k source addresses, followed by k destination addresses for dispatch_images.
_KERNELS = cp.RawModule(code=r"""
extern "C" __global__ void dispatch_images(const unsigned long long* ptrs, int k,
                                         int image_bytes) {
  const int f = blockIdx.x;
  const uint4* src = reinterpret_cast<const uint4*>(ptrs[f]);
  uint4* dst = reinterpret_cast<uint4*>(ptrs[k + f]);
  for (int i = threadIdx.x; i < image_bytes / 16; i += blockDim.x) dst[i] = src[i];
}
""")

_pinned_pool = cp.cuda.PinnedMemoryPool()


def _pointer_list(values: np.ndarray):
    """``values`` (uint64 addresses) in pinned memory, which kernels read directly. The returned
    object must be kept alive until the kernels have finished."""
    mem = _pinned_pool.malloc(values.nbytes)
    np.frombuffer(mem, dtype=np.uint64, count=values.size)[:] = values
    return mem


@dataclass
class Dispatched:
    """What a dispatch launched. Nothing has necessarily finished: ``payload`` is only complete
    once ``event`` has happened (wait for it on the stream that reads it)."""

    payload: Any  # (k, 96, 96) uint16 images
    ids: np.ndarray  # (k,) host image ids, in arrival order
    event: Any  # happened = the ring bytes the kernel reads are free
    keep: list  # host memory the kernel reads until then


class Dispatcher:
    """Gets the images of one ring onto the ring's GPU: one kernel per batch, on its own stream,
    copies them out of the ring into a (k, 96, 96) uint16 array. The ring is in pinned host memory
    or in the memory of ``device``: a GPU only ever reads its own ring.
    """

    def __init__(self, device: int):
        self.device = device
        with cp.cuda.Device(device):
            self.stream = cp.cuda.Stream(non_blocking=True)
            # Load the module on this device now, not on the first (timed) batch.
            self.kernel = _KERNELS.get_function("dispatch_images")

    def dispatch(self, addrs: np.ndarray, ids: np.ndarray) -> Dispatched:
        """Launch the copy of the images at ``addrs`` (host array of ring addresses) with ids
        ``ids`` (host array). Returns at once."""
        k = len(addrs)
        keep = []
        with cp.cuda.Device(self.device), self.stream:
            payload = cp.empty((k, *IMAGE_SHAPE), dtype=cp.uint16)
            dst = np.uint64(payload.data.ptr) + np.arange(k, dtype=np.uint64) * np.uint64(
                IMAGE_BYTES)
            ptrs = _pointer_list(np.concatenate([addrs.astype(np.uint64), dst]))
            self.kernel((k,), (_THREADS,), (np.uint64(ptrs.ptr), np.int32(k),
                                            np.int32(IMAGE_BYTES)))
            keep.append(ptrs)
            event = cp.cuda.Event(block=False, disable_timing=True)
            event.record(self.stream)
        return Dispatched(payload, ids, event, keep)

    def warm_up(self, k: int):
        """Dispatch ``k`` images from a scratch buffer and wait: the pinned pointer-list pool and
        the device memory pool grow to a full batch now, not on the first (timed) batches."""
        with cp.cuda.Device(self.device):
            scratch = cp.zeros(k * IMAGE_BYTES, dtype=cp.uint8)
            addrs = np.uint64(scratch.data.ptr) + np.arange(k, dtype=np.uint64) * np.uint64(
                IMAGE_BYTES)
            self.dispatch(addrs, np.arange(k, dtype=np.uint64)).event.synchronize()


def ring_location(addr: int):
    """GPU index if ``addr`` is GPU memory, None if it is host memory."""
    attrs = cp.cuda.runtime.pointerGetAttributes(addr)
    return attrs.device if attrs.type == 2 else None  # cudaMemoryTypeDevice


class DeviceRing:
    """A ring in GPU memory, which the CPU cannot dereference: slices are copied to the host.
    Only used for the frame headers and the small start messages, never for image data."""

    def __init__(self, addr: int, length: int, device: int):
        self.addr, self.length, self.device = addr, length, device

    def __len__(self):
        return self.length

    def __getitem__(self, key: slice) -> bytes:
        start, stop, step = key.indices(self.length)
        if step != 1:
            raise ValueError("DeviceRing supports contiguous slices only")
        n = max(0, stop - start)
        buf = (ctypes.c_char * n)()
        with cp.cuda.Device(self.device):
            cp.cuda.runtime.memcpy(ctypes.addressof(buf), self.addr + start, n,
                                   cp.cuda.runtime.memcpyDeviceToHost)
        return bytes(buf)


def set_sync_memops(addr: int, device: int) -> int:
    """Set CU_POINTER_ATTRIBUTE_SYNC_MEMOPS on the GPU memory at ``addr``, so that synchronous CUDA
    operations on it are ordered with the NIC's GPUDirect RDMA writes. Returns the CUresult."""
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuPointerSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    with cp.cuda.Device(device):
        cp.zeros(1)  # make sure the context exists
        flag = ctypes.c_uint(1)
        return cuda.cuPointerSetAttribute(ctypes.byref(flag), 6, addr)  # ..._SYNC_MEMOPS = 6


_flush_state = {"fn": None, "warned": False}


def load_cudart() -> ctypes.CDLL:
    """The CUDA runtime library of the image (CUDA 12 or 13)."""
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise OSError("no CUDA runtime library (libcudart.so.13 / .12) found")


def mapped_device_pointer(addr: int, device: int) -> int:
    """Return the CUDA alias of a mapped host allocation for ``device``."""
    cuda = load_cudart()
    fn = cuda.cudaHostGetDevicePointer
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    alias = ctypes.c_void_p()
    with cp.cuda.Device(device):
        status = fn(ctypes.byref(alias), addr, 0)
    if status != 0 or alias.value is None:
        raise RuntimeError(f"cudaHostGetDevicePointer failed with CUDA status {status}")
    return alias.value


def flush_gpudirect_writes(device: int):
    """Make the NIC's writes into the memory of GPU ``device`` visible to the GPU
    (cudaDeviceFlushGPUDirectRDMAWrites, to the owner). Call after a completion announced new data
    and before reading it. Warns once if the GPU cannot do it."""
    state = _flush_state
    if state["fn"] is None:
        state["fn"] = load_cudart().cudaDeviceFlushGPUDirectRDMAWrites
        state["fn"].argtypes = [ctypes.c_int, ctypes.c_int]
    with cp.cuda.Device(device):
        status = state["fn"](0, 100)  # target: the current device; scope: to the owner
    if status != 0 and not state["warned"]:
        state["warned"] = True
        print(f"cudaDeviceFlushGPUDirectRDMAWrites returned {status}: the NIC's writes into GPU "
              f"memory may not be visible to the GPU in time", flush=True)
    return status
