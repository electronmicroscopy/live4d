"""
GPU bslz4 decoder -- the full-rate "data in" decode path, torch-only.

Decompresses a *batch* of DECTRIS bslz4 frame blobs entirely on the GPU, so the CPU only parses the
small message envelope (the heavy decompress leaves the host).

Path (all on-device after the small compressed bytes are shipped over):

    parse_bslz4_blocks(encoded)              # CPU, microseconds, on compressed bytes only (ingest/bslz4.py)
    nvcompBatchedLZ4DecompressAsync          # GPU: one batched LZ4 launch for ALL blocks
    unshuffle CUDA kernel (LSB-first)        # GPU: custom bit-transpose, torch load_inline

Two design choices, both established empirically:
  - nvCOMP's *built-in* bitshuffle uses a different convention than DECTRIS, so we use nvCOMP for LZ4
    only and do the unshuffle ourselves.
  - the high-level nvCOMP Python `Codec` is host-side marshalling-bound (a Python object per block);
    the *low-level* batched C API (device pointer/size arrays) is ~13x faster. We bind it via ctypes
    and build the pointer arrays with torch -- no cupy (PyTorch is the single GPU-array library).

Requires: nvcc for torch's `load_inline` (system CUDA) + `ninja`. Run with
`CUDA_HOME=/usr/local/cuda`, the venv `bin` on PATH, and `TORCH_CUDA_ARCH_LIST` set to the target
GPU architecture. Assumes each block's n_elem is a multiple of 8 (true for real ARINA/QUADRO frames
and the standard fixtures); only the last block of a frame may be short.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

# -- locate + bind the low-level nvCOMP batched LZ4 C API ------------------------

def _find_libnvcomp() -> str:
    import nvidia
    for root in list(getattr(nvidia, "__path__", [])):
        cands = sorted(Path(root).glob("libnvcomp/lib64/libnvcomp.so*"))
        if cands:
            return str(cands[-1])
    raise RuntimeError("libnvcomp.so not found under the nvidia package (is nvidia-nvcomp installed?)")


_NVCOMP_TYPE_CHAR = 0
_NVCOMP_BITSHUFFLE_NONE = 0


class _Opts(ctypes.Structure):
    _fields_ = [("data_type", ctypes.c_int),
                ("bitshuffle_mode", ctypes.c_int),
                ("reserved", ctypes.c_char * 48)]


class _Align(ctypes.Structure):
    _fields_ = [("input", ctypes.c_size_t), ("output", ctypes.c_size_t), ("temp", ctypes.c_size_t)]


def _bind_nvcomp():
    lib = ctypes.CDLL(_find_libnvcomp(), mode=ctypes.RTLD_GLOBAL)
    lib.nvcompBatchedLZ4DecompressGetRequiredAlignments.argtypes = [_Opts, ctypes.POINTER(_Align)]
    lib.nvcompBatchedLZ4DecompressGetRequiredAlignments.restype = ctypes.c_int
    lib.nvcompBatchedLZ4DecompressGetTempSizeAsync.argtypes = [
        ctypes.c_size_t, ctypes.c_size_t, _Opts, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
    lib.nvcompBatchedLZ4DecompressGetTempSizeAsync.restype = ctypes.c_int
    lib.nvcompBatchedLZ4DecompressAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, _Opts, ctypes.c_void_p, ctypes.c_void_p]
    lib.nvcompBatchedLZ4DecompressAsync.restype = ctypes.c_int
    return lib


# -- the unshuffle kernel (compiled once, shared) --------------------------------

_CPP = "torch::Tensor unshuffle(torch::Tensor in, int64_t block_size, int64_t elem_size);"
_CUDA = r"""
#include <torch/extension.h>
#include <cstdint>

__global__ void unshuffle_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                 long total, int block_size, int elem_size, int n_elem) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long blockbase = (idx / block_size) * block_size;   // start of this block
    int o = (int)(idx - blockbase);                     // output byte offset within block
    int j = o / elem_size;                              // element index within block
    int c = o % elem_size;                              // byte within the element
    int bbase = c * 8;
    uint8_t val = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        long src_bit = (long)(bbase + k) * n_elem + j;  // bit position within block (LSB-first)
        uint8_t bit = (in[blockbase + (src_bit >> 3)] >> (src_bit & 7)) & 1u;
        val |= (uint8_t)(bit << k);
    }
    out[idx] = val;
}

torch::Tensor unshuffle(torch::Tensor in, int64_t block_size, int64_t elem_size) {
    TORCH_CHECK(in.is_cuda() && in.dtype() == torch::kUInt8 && in.is_contiguous());
    TORCH_CHECK(in.numel() % block_size == 0, "size not a multiple of block_size");
    long total = in.numel();
    int n_elem = (int)(block_size / elem_size);
    auto out = torch::empty_like(in);
    const int threads = 256;
    long blocks = (total + threads - 1) / threads;
    unshuffle_kernel<<<blocks, threads>>>(
        in.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), total,
        (int)block_size, (int)elem_size, n_elem);
    return out;
}
"""

_ext = None


def _get_unshuffle_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        try:
            _ext = load_inline(name="bslz4_unshuffle", cpp_sources=[_CPP], cuda_sources=[_CUDA],
                               functions=["unshuffle"], verbose=False)
        except Exception as exc:  # nvcc / ninja missing, arch mismatch, ...
            raise RuntimeError(
                "failed to compile the unshuffle CUDA kernel -- need nvcc (system CUDA) + ninja, and "
                "CUDA_HOME / TORCH_CUDA_ARCH_LIST set for the target GPU") from exc
    return _ext


class GpuBslz4Decoder:
    """Decompress batches of DECTRIS bslz4 frame blobs on the GPU.

    Construct once (compiles the kernel, binds nvCOMP, queries alignment), then call `decode` per
    chunk. `decode` returns the decompressed frames as a contiguous CUDA `uint8` tensor of shape
    `(n, frame_bytes)`; reinterpreting to pixels (`pixel_dtype`) and casting to float is the
    consumer's job (e.g. the VBF accumulator).
    """

    def __init__(self):
        import torch  # local import so importing this module is cheap / torch-optional
        self._torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("GpuBslz4Decoder needs CUDA")
        self._lib = _bind_nvcomp()
        self._opts = _Opts(_NVCOMP_TYPE_CHAR, _NVCOMP_BITSHUFFLE_NONE, b"\x00" * 48)
        align = _Align()
        self._lib.nvcompBatchedLZ4DecompressGetRequiredAlignments(self._opts, ctypes.byref(align))
        self._align_in = max(align.input, 1)
        self._ext = _get_unshuffle_ext()

    # -- helpers --------------------------------------------------------------------
    @staticmethod
    def pixel_dtype(elem_size: int):
        import torch
        return {1: torch.uint8, 2: torch.uint16, 4: torch.uint32}[elem_size]

    def decode(self, encoded_list, elem_size: int):
        """Decompress a chunk of bslz4 blobs -> CUDA uint8 tensor `(n, frame_bytes)`.

        All device work runs on the default stream; we synchronize **once** at the end, so every
        device tensor below -- the pointer/size arrays nvCOMP reads asynchronously, the temp and I/O
        buffers -- stays alive until the GPU is done.
        """
        torch = self._torch
        if not encoded_list:
            return torch.empty((0, 0), dtype=torch.uint8, device="cuda")
        n = len(encoded_list)
        first = encoded_list[0]
        fb = int.from_bytes(first[0:8], "big")      # uncompressed bytes/frame (uniform geometry)
        bs = int.from_bytes(first[8:12], "big")     # block size (bytes)
        nb = (fb + bs - 1) // bs                     # blocks per frame

        # One C-level concat + one H2D of the compressed bytes; align_in == 1 lets us point straight
        # into it (no per-block copy -- the packing loop was the host-side bottleneck).
        cin = torch.frombuffer(bytearray(b"".join(encoded_list)), dtype=torch.uint8).cuda()
        blob_base = np.zeros(n, np.int64)
        np.cumsum(np.fromiter((len(e) for e in encoded_list), np.int64, n)[:-1], out=blob_base[1:])

        # Single-pass framing walk: only the per-block *compressed* offset+size are data-dependent.
        # One tight Python loop -- no namedtuples, no second pass (parse_bslz4_blocks is kept for
        # validation; this is the hot path).
        in_off, comp = [], []
        be = int.from_bytes
        for e, bb in zip(encoded_list, blob_base.tolist()):
            pos, remaining = 12, fb
            while remaining > 0:
                cs = be(e[pos:pos + 4], "big"); pos += 4
                in_off.append(bb + pos); comp.append(cs); pos += cs
                remaining -= bs if remaining >= bs else remaining
        n_blk = len(in_off)

        # Uncompressed layout is uniform across frames -> build by tiling (no per-block work).
        per_frame_uncomp = np.full(nb, bs, np.int64)
        per_frame_uncomp[-1] = fb - (nb - 1) * bs
        uncomp = np.tile(per_frame_uncomp, n)
        out_off = np.zeros(n_blk, np.int64)
        np.cumsum(uncomp[:-1], out=out_off[1:])

        out = torch.empty(n * fb, dtype=torch.uint8, device="cuda")
        in_ptrs = torch.from_numpy(cin.data_ptr() + np.asarray(in_off, np.int64)).cuda()
        out_ptrs = torch.from_numpy(out.data_ptr() + out_off).cuda()
        comp_bytes = torch.from_numpy(np.asarray(comp, np.int64)).cuda()
        buf_bytes = torch.from_numpy(uncomp).cuda()
        actual = torch.empty(n_blk, dtype=torch.int64, device="cuda")

        temp_bytes = ctypes.c_size_t()
        self._lib.nvcompBatchedLZ4DecompressGetTempSizeAsync(
            n_blk, bs, self._opts, ctypes.byref(temp_bytes), n * fb)
        temp = torch.empty(max(temp_bytes.value, 1), dtype=torch.uint8, device="cuda")

        status = self._lib.nvcompBatchedLZ4DecompressAsync(
            in_ptrs.data_ptr(), comp_bytes.data_ptr(), buf_bytes.data_ptr(), actual.data_ptr(),
            n_blk, temp.data_ptr(), temp_bytes.value, out_ptrs.data_ptr(), self._opts, None, None)
        if status != 0:
            raise RuntimeError(f"nvcompBatchedLZ4DecompressAsync status={status}")

        # `out` now holds LZ4-decoded but still bit-shuffled bytes; unshuffle on the GPU. Fast path:
        # fb % bs == 0 implies every block (incl. each frame's last) is full -> one launch.
        if fb % bs == 0:
            frames = self._ext.unshuffle(out, bs, elem_size)
        else:
            frames = self._unshuffle_per_frame(out, elem_size, n, bs, fb, nb)
        torch.cuda.synchronize()   # single sync: all async device work above is now complete
        return frames.view(n, fb)

    def decode_int_frames(self, encoded_list, elem_size: int, shape):
        """Decode + reinterpret to integer frames `(n, H, W)` on the GPU, ready for `.float()`.

        Mirrors the host int reinterpret the CPU path uses so the device frames are numerically
        identical to a CPU decode: uint32 -> int32 (counts < 2^31, exact, zero-copy view); uint16 ->
        widened to unsigned int32; uint8 -> int32. Returns an `int32` CUDA tensor `(n, H, W)`; the
        VBF accumulator casts it to float on the device.
        """
        torch = self._torch
        H, W = int(shape[0]), int(shape[1])
        out = self.decode(encoded_list, elem_size)        # (n, frame_bytes) uint8, contiguous
        n = out.shape[0]
        if n == 0:
            return torch.empty((0, H, W), dtype=torch.int32, device="cuda")
        if elem_size == 4:
            return out.view(torch.int32).view(n, H, W)    # uint32 counts < 2^31 -> reinterpret is exact
        if elem_size == 2:
            i16 = out.view(torch.int16).view(n, H, W)     # bytes -> int16 (values > 32767 go negative)
            return i16.to(torch.int32).bitwise_and_(0xFFFF)   # recover the unsigned uint16 value
        if elem_size == 1:
            return out.view(n, H, W).to(torch.int32)      # uint8 0..255
        raise ValueError(f"unsupported elem_size {elem_size} (expected 1, 2, or 4)")

    def _unshuffle_per_frame(self, shuffled, elem_size, n, bs, fb, nb):
        """Frames whose last block is short (fb % bs != 0): unshuffle the full-block region of ALL
        frames in one launch and the short tail of ALL frames in another -- two BATCHED launches, not
        a per-frame Python loop (the loop made small binned frames ~15x slower: ~190 vs ~13 us/frame).

        Uniform geometry, so the full/tail split is identical for every frame; each bslz4 block is
        independent, so concatenating the same-sized regions across frames and unshuffling them as
        one batch is exact. `full` is an exact multiple of `bs` (the (nb-1) full blocks); `tail` is
        the short last block, unshuffled with its own block size."""
        torch = self._torch
        tail = fb - (nb - 1) * bs          # short last block (< bs in this branch)
        full = fb - tail                   # full-block bytes per frame (an exact multiple of bs)
        s2 = shuffled.view(n, fb)
        frames = torch.empty_like(s2)
        if full:
            frames[:, :full] = self._ext.unshuffle(
                s2[:, :full].contiguous().view(-1), bs, elem_size).view(n, full)
        frames[:, full:] = self._ext.unshuffle(
            s2[:, full:].contiguous().view(-1), tail, elem_size).view(n, tail)
        return frames.reshape(-1)


# -- standalone validation + benchmark -------------------------------------------

if __name__ == "__main__":
    import torch
    from dectris.compression import decompress

    def fixture(elem_size, n_elem, bse):
        """Build a synthetic bslz4 blob (bitshuffle + LZ4) framed the way detector frames are."""
        import bitshuffle
        rng = np.random.default_rng(0)
        dt = {2: "<u2", 4: "<u4"}[elem_size]
        x = rng.integers(0, 256, size=n_elem * elem_size, dtype=np.uint8).view(dt)
        comp = bitshuffle.compress_lz4(x, bse).tobytes()
        return x.nbytes.to_bytes(8, "big") + (bse * elem_size).to_bytes(4, "big") + comp

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    dec = GpuBslz4Decoder()

    print("\n=== correctness vs reference CPU decoder ===")
    ok = True
    for name, blob, es in [("u2 single block", fixture(2, 4096, 4096), 2),
                           ("u4 single block", fixture(4, 2048, 2048), 4),
                           ("u2 multi-block + short tail", fixture(2, 10000, 4096), 2)]:
        truth = decompress(blob, "bslz4", elem_size=es)
        got = dec.decode([blob], es)[0].cpu().numpy().tobytes()
        good = got == truth
        ok &= good
        print(f"  [{name}] {'PASS' if good else 'FAIL'}")
    print("  ->", "ALL PASS" if ok else "FAILURES")
