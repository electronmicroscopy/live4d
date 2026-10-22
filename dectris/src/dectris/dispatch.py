"""Get the images of a ring onto its GPU: copy (or reduce them in place).

The ring is GPU memory that the NIC writes into (GPUDirect RDMA); the kernels read the images
where they are, with no staging copy.
"""

import ctypes
from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

from .messages import IMAGE_BYTES, IMAGE_SHAPE

N_PIXELS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
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

// Sum of the pixels of image f, straight from the ring.
extern "C" __global__ void image_sums(const unsigned long long* ptrs, int n_pixels,
                                      unsigned long long* out) {
  const int f = blockIdx.x;
  const unsigned short* px = reinterpret_cast<const unsigned short*>(ptrs[f]);
  unsigned long long sum = 0;
  for (int i = threadIdx.x; i < n_pixels; i += blockDim.x) sum += px[i];
  __shared__ unsigned long long part[128];  // blockDim.x == 128
  part[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[f] = part[0];
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

    payload: Any  # (k,) uint64 pixel sums, or (k, 96, 96) uint16 images
    ids: np.ndarray  # (k,) host image ids, in arrival order
    event: Any  # happened = the ring bytes the kernel reads are free
    keep: list  # host memory the kernel reads until then


class Dispatcher:
    """Gets the images of one ring onto the ring's GPU: one kernel per batch, on its own stream.

    The payload is either the images, a (k, 96, 96) uint16 array (``reduce=False``), or just their
    pixel sums, a (k,) uint64 array (``reduce=True``, which skips the copy of the images
    altogether). The ring is in pinned host memory or in the memory of ``device``: a GPU only ever
    reads its own ring.
    """

    def __init__(self, device: int, reduce: bool = True):
        self.device = device
        self.reduce = reduce
        with cp.cuda.Device(device):
            self.stream = cp.cuda.Stream(non_blocking=True)
            # Load the module on this device now, not on the first (timed) batch.
            self.kernel = _KERNELS.get_function("image_sums" if reduce else "dispatch_images")

    def dispatch(self, addrs: np.ndarray, ids: np.ndarray) -> Dispatched:
        """Launch the work for the images at ``addrs`` (host array of ring addresses) with ids
        ``ids`` (host array). Returns at once."""
        k = len(addrs)
        keep = []
        with cp.cuda.Device(self.device), self.stream:
            if self.reduce:
                payload = cp.empty(k, dtype=cp.uint64)
                ptrs = _pointer_list(addrs.astype(np.uint64))
                self.kernel((k,), (_THREADS,), (np.uint64(ptrs.ptr), np.int32(N_PIXELS), payload))
            else:
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
