"""
VBF engine: build the weight map (host, NumPy) and accumulate the virtual bright-field image
on the device (PyTorch), casting integer frames to float on the GPU.

Two pieces:
- `build_weight_map(...)` -- pure NumPy, no GPU. Folds the BF-disk region, the dead-pixel mask,
  and the flatfield gain into a single float32 weight map `W`. Fully unit-testable.
- `VBFAccumulator` -- holds `W` and the running output image on the *device*; each batch of
  *integer* frames is uploaded as-is, cast to float and reduced to one scalar per frame
  (sum of frame*W) on the GPU, then scattered into the image by scan id.

The integer->float cast runs on the GPU (not the host): a host float cast is the second-biggest
host cost after queue orchestration. We keep the integer frames on the host (cheap, often a
zero-copy reinterpret), copy the *integer* bytes to the device, and cast there, where it is
effectively free.

Per-frame math (no dark term -- counting detector):  S = sum_p frame[p]*W[p],
with  W = bf_region * good_pixel * flatfield_gain.

This module is self-contained on purpose; nothing here imports from the test/reference layer at
runtime.
"""

from __future__ import annotations

import time

import numpy as np
import torch

__all__ = ["build_weight_map", "VBFAccumulator"]


def build_weight_map(
    detector_shape: tuple[int, int],
    bf_center_yx: tuple[float, float],
    bf_radius_px: float,
    flatfield: np.ndarray | None = None,
    dead_px: np.ndarray | None = None,
) -> np.ndarray:
    """Fold BF region + dead-pixel mask + flatfield gain into one float32 weight map.

    Args:
        detector_shape: (Hy, Hx) detector pixel dimensions.
        bf_center_yx:   (cy, cx) center of the bright-field disk, in detector pixels.
        bf_radius_px:   bright-field disk radius, in detector pixels.
        flatfield:      per-pixel gain, shape == detector_shape. None -> all ones.
        dead_px:        1 = good, 0 = dead, shape == detector_shape. None -> all good.

    Returns:
        W: float32, shape detector_shape; zero outside the BF disk and at dead pixels.
    """
    hy, hx = detector_shape
    cy, cx = bf_center_yx

    # Disk mask. indexing="ij" => yy varies down rows (axis 0), xx across columns (axis 1),
    # matching NumPy's array layout.
    yy, xx = np.meshgrid(np.arange(hy), np.arange(hx), indexing="ij")
    bf_region = ((yy - cy) ** 2 + (xx - cx) ** 2) <= bf_radius_px**2

    if flatfield is None:
        flatfield = np.ones(detector_shape, dtype=np.float32)
    if dead_px is None:
        dead_px = np.ones(detector_shape, dtype=np.float32)

    if flatfield.shape != tuple(detector_shape):
        raise ValueError(f"flatfield shape {flatfield.shape} != detector {detector_shape}")
    if dead_px.shape != tuple(detector_shape):
        raise ValueError(f"dead_px shape {dead_px.shape} != detector {detector_shape}")

    return (
        bf_region.astype(np.float32)
        * dead_px.astype(np.float32)
        * flatfield.astype(np.float32)
    )


def _pick_device(device: str | None) -> torch.device:
    """Default to CUDA when available, else CPU (so tests run anywhere)."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VBFAccumulator:
    """Accumulate the VBF image on the device, casting integer frames to float on the GPU.

    Usage:
        acc = VBFAccumulator(W, scan_shape=(Ny, Nx))
        run_pipeline_batched(source, acc.process, stats)   # .process is the on_batch callback
        image = acc.result()                               # (Ny, Nx) host array
    """

    def __init__(self, weight_map: np.ndarray, scan_shape: tuple[int, int],
                 device: str | None = None, measure: bool = False):
        self.device = _pick_device(device)
        self.scan_shape = scan_shape
        self.W = torch.as_tensor(weight_map, dtype=torch.float32, device=self.device)
        # Flat output image (Ny*Nx,) on the device; reshaped to 2D only at the end. Flat makes
        # the scatter-by-id a single indexed assignment.
        n = scan_shape[0] * scan_shape[1]
        self.image = torch.zeros(n, dtype=torch.float32, device=self.device)

        # Instrumentation (off by default). We split: host prep (the cheap reinterpret/cast that
        # replaced the host float cast), the *integer* H2D copy, and the GPU cast+reduce. GPU
        # stages use CUDA events (no per-batch synchronize -> no perturbation).
        self.measure = measure
        self._cuda = self.device.type == "cuda"
        self._host_prep_s = 0.0
        self._compute_s = 0.0
        self._events: list[tuple] = []   # (copy0, copy1, compute1) per batch

    def _to_int_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """Wrap integer frames as a torch tensor for upload, avoiding a host float cast.

        uint32 (this ARINA mode) reinterprets to int32 with no copy -- counts are well under 2^31,
        so the bits are identical. torch handles uint8/int* directly. Other widths (e.g. uint16)
        take a small *integer* cast -- still far cheaper than a host float cast, and the float
        conversion itself happens on the device.
        """
        a = np.ascontiguousarray(frames)
        dt = a.dtype
        if dt == np.uint32:
            return torch.from_numpy(a.view(np.int32))
        if dt == np.uint8 or dt in (np.int8, np.int16, np.int32, np.float32, np.float64):
            return torch.from_numpy(a)
        return torch.from_numpy(a.astype(np.int32))  # e.g. uint16: safe widening, no host float cast

    def process(self, frames: np.ndarray, ids: np.ndarray) -> None:
        """Upload a batch (integer), cast to float + reduce on the GPU, scatter by scan id."""
        # Shape guard: W and the output image are preallocated from config, off the hot path. A
        # frame whose detector dims don't match W means config.detector_shape is wrong; fail loudly.
        if tuple(frames.shape[1:]) != tuple(self.W.shape):
            raise ValueError(
                f"frame detector shape {tuple(frames.shape[1:])} != weight-map shape "
                f"{tuple(self.W.shape)} -- the detector doesn't match config.detector_shape."
            )

        # GPU decode path: `frames` is already an integer tensor on the device (GpuBslz4Decoder
        # output). No host prep, no H2D -- just cast to float + reduce here.
        if isinstance(frames, torch.Tensor):
            f = frames if frames.device == self.device else frames.to(self.device)
            if not self.measure:
                self._reduce_into(f.float(), ids)
                return
            if self._cuda:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                self._reduce_into(f.float(), ids)
                e1.record()
                self._events.append((e0, e0, e1))   # decode already did the H2D -> copy~0 here
            else:
                t1 = time.perf_counter()
                self._reduce_into(f.float(), ids)
                self._compute_s += time.perf_counter() - t1
            return

        if not self.measure:
            f_int = self._to_int_tensor(frames).to(self.device, non_blocking=True)
            self._reduce_into(f_int.float(), ids)
            return

        # Measured path.
        t0 = time.perf_counter()
        t_int = self._to_int_tensor(frames)             # host prep (reinterpret / small int cast)
        self._host_prep_s += time.perf_counter() - t0

        if self._cuda:
            e_copy0 = torch.cuda.Event(enable_timing=True)
            e_copy1 = torch.cuda.Event(enable_timing=True)
            e_comp1 = torch.cuda.Event(enable_timing=True)
            e_copy0.record()
            g = t_int.to(self.device, non_blocking=True)  # integer H2D
            e_copy1.record()
            self._reduce_into(g.float(), ids)             # cast + reduce, both on the GPU
            e_comp1.record()
            self._events.append((e_copy0, e_copy1, e_comp1))
        else:
            t1 = time.perf_counter()
            self._reduce_into(t_int.float(), ids)
            self._compute_s += time.perf_counter() - t1

    def _reduce_into(self, f: torch.Tensor, ids: np.ndarray) -> None:
        # (B, Hy, Hx) * (Hy, Hx) broadcasts W over the batch; sum over detector axes -> (B,).
        s = (f * self.W).sum(dim=(1, 2))
        idx = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self.image[idx] = s   # each scan position visited once -> direct assignment, no atomics

    def perf_stats(self) -> dict | None:
        """Per-stage timing for the run, or None if measurement was off."""
        if not self.measure:
            return None
        out: dict = {"host_prep_s": self._host_prep_s}
        if self._cuda and self._events:
            torch.cuda.synchronize()
            out["gpu_copy_s"] = sum(a.elapsed_time(b) for a, b, _ in self._events) / 1000.0
            out["gpu_compute_s"] = sum(b.elapsed_time(c) for _, b, c in self._events) / 1000.0
            out["n_batches"] = len(self._events)
        else:
            out["compute_s"] = self._compute_s
        return out

    def result(self) -> np.ndarray:
        return self.image.reshape(self.scan_shape).cpu().numpy()
