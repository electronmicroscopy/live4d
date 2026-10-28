"""Check that copies between pinned host memory and the GPU deliver the right bytes.

The ingest stages every batch in one large pinned (page-locked) host buffer and copies it to the
GPU. On one test machine (an IGX Orin) a region of some such buffers came back wrong after the
copy, without any error, which silently corrupted the reconstruction. This check needs no
detector and no pipeline: it allocates pinned buffers of the sizes the ingest uses (and a few
others), fills them with random bytes, copies them to the GPU and back, and compares.

Run it inside the image, on the GPU the pipeline will use (takes about a minute):

    docker run --rm --gpus all -v "$PWD":/work -w /work --entrypoint python3 \\
      realtime-stem:latest -m tools.check_gpu_copies

Exit code 0 and "all copies correct" means the machine is fine; anything else lists the bad
regions (in MiB from the start of each buffer).
"""
from __future__ import annotations

import ctypes
import sys

import numpy as np
import torch

H2D, D2H = 1, 2   # cudaMemcpyKind


def _cudart():
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise SystemExit("cannot load the CUDA runtime (libcudart.so); run this inside the image")


def ingest_buffer_bytes(height: int, width: int, elem_size: int, max_batch: int = 2048,
                        pool: int = 3, block: int = 8192) -> int:
    """The pinned size ZmqRxOp allocates (zmq_rx_operator.cu, start())."""
    image = height * width * elem_size
    blocks = (image + block - 1) // block
    per_image = 12 + image + blocks * (4 + block // 255 + 16)
    return per_image * max_batch * pool


def alloc_pinned(cudart, n_bytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    if cudart.cudaMallocHost(ctypes.byref(ptr), ctypes.c_size_t(n_bytes)) != 0:
        raise SystemExit(f"cudaMallocHost({n_bytes}) failed")
    return ptr


def check_buffer(cudart, ptr: ctypes.c_void_p, n_bytes: int, seed: int
                 ) -> tuple[list[int], list[int]]:
    """Bad MiB indices of one pinned buffer: (host→GPU, GPU→host)."""
    host = np.frombuffer((ctypes.c_uint8 * n_bytes).from_address(ptr.value), dtype=np.uint8)
    ref = np.random.default_rng(seed).integers(0, 256, n_bytes, dtype=np.uint8)
    host[:] = ref
    dev = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    if cudart.cudaMemcpy(ctypes.c_void_p(dev.data_ptr()), ptr, ctypes.c_size_t(n_bytes), H2D) != 0:
        raise SystemExit("cudaMemcpy host→GPU failed")
    torch.cuda.synchronize()
    up = dev.cpu().numpy() != ref                     # what arrived on the GPU
    host[:] = 0
    if cudart.cudaMemcpy(ptr, ctypes.c_void_p(dev.data_ptr()), ctypes.c_size_t(n_bytes), D2H) != 0:
        raise SystemExit("cudaMemcpy GPU→host failed")
    down = host != ref                                # what came back into pinned memory
    return _bad_mib(up), _bad_mib(down)


def _bad_mib(mask: np.ndarray) -> list[int]:
    mib = 1 << 20
    padded = np.zeros(-(-mask.size // mib) * mib, dtype=bool)
    padded[:mask.size] = mask
    return np.flatnonzero(padded.reshape(-1, mib).any(axis=1)).tolist()


def main() -> int:
    cudart = _cudart()
    torch.zeros(1, device="cuda")
    print(f"[check] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    sizes = {
        "ingest 96x96 uint16": ingest_buffer_bytes(96, 96, 2),
        "ingest 96x96 uint32": ingest_buffer_bytes(96, 96, 4),
        "ingest 192x192 uint16": ingest_buffer_bytes(192, 192, 2),
        "ingest 192x192 uint32": ingest_buffer_bytes(192, 192, 4),
        "512 MiB": 512 << 20,
        "1 GiB": 1 << 30,
        "2 GiB": 2 << 30,
    }
    failures = 0

    def report(label: str, n: int, result: tuple[list[int], list[int]]) -> None:
        nonlocal failures
        up, down = result
        ok = not up and not down
        failures += not ok
        detail = "ok" if ok else f"BAD MiB host→GPU {up[:20]}, GPU→host {down[:20]}"
        print(f"[check] {label:<26} {n / 2**20:7.0f} MiB: {detail}", flush=True)

    for seed, (label, n) in enumerate(sizes.items()):
        ptr = alloc_pinned(cudart, n)
        try:
            report(label, n, check_buffer(cudart, ptr, n, seed))
        finally:
            cudart.cudaFreeHost(ptr)
    # Where a buffer lands in memory decides whether it is affected, so also hold several at
    # once (on the affected machine, the third of three such buffers was bad).
    n = 290 << 20
    held = [alloc_pinned(cudart, n) for _ in range(4)]
    try:
        for k, ptr in enumerate(held):
            report(f"held together, buffer {k + 1}", n, check_buffer(cudart, ptr, n, 50 + k))
    finally:
        for ptr in held:
            cudart.cudaFreeHost(ptr)
    if failures:
        print(f"[check] FAILED: {failures} copies came back wrong. This machine corrupts copies "
              f"between pinned memory and the GPU; results of the pipeline cannot be trusted. "
              f"Report it to the system's vendor.", flush=True)
        return 1
    print("[check] all copies correct", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
