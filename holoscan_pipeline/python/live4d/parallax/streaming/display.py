"""Rendering live frames: one panel per channel, robust contrast, metadata in the title.

Shared by the consumers of the live stream — the desktop viewer (`examples/viewer.py`) and the
Jupyter view (`streaming.notebook.LiveView`) — so they draw the same thing. numpy + matplotlib only;
the figure (and its backend) belongs to the caller.
"""

from __future__ import annotations

import numpy as np

__all__ = ["high_pass", "robust_clim", "meta_line", "PanelFigure"]


def high_pass(img, sigma_px: float):
    """`img` minus a Gaussian low-pass (sigma in real-space pixels). sigma 0 = identity.

    Pure-numpy FFT (no scipy dependency), NaN-safe. Useful for phase-like reconstructions where a
    slow background swamps the contrast; leave it off (0) for parallax bright-field.
    """
    a = np.asarray(img, dtype=np.float64)
    if sigma_px <= 0:
        return a.copy()
    if not np.isfinite(a).all():
        finite = a[np.isfinite(a)]
        fill = float(np.median(finite)) if finite.size else 0.0
        a = np.where(np.isfinite(a), a, fill)
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    gauss = np.exp(-2.0 * np.pi**2 * sigma_px**2 * (fy**2 + fx**2))
    lowpass = np.fft.ifft2(np.fft.fft2(a) * gauss).real
    return a - lowpass


def robust_clim(img, lo_pct: float = 0.5, hi_pct: float = 99.5):
    """Display range from percentiles — so a hot pixel doesn't wash out the image."""
    finite = np.asarray(img)[np.isfinite(img)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if not hi > lo:
        lo, hi = lo - 0.5, lo + 0.5
    return float(lo), float(hi)


def meta_line(header: dict) -> str:
    """A one-line summary of a frame header's metadata (progress, C10, rotation)."""
    meta = header.get("meta") or {}
    parts = []
    if "frames" in meta:
        parts.append(f"{meta['frames']:,} frames")
    repeating = meta.get("accumulate") in ("latest", "all") or meta.get("pass", 0) > 0
    if repeating and "progress" in meta:     # continuous scanning: where the scan cursor is
        parts.append(f"pass {meta.get('pass', 0) + 1} at {100 * meta['progress']:.0f}%"
                     f" ({meta.get('accumulate', 'all')})")
    elif "coverage" in meta:
        parts.append(f"{100 * meta['coverage']:.0f}% scanned")
    if "C10" in meta:
        parts.append(f"C10 {meta['C10']:.4g} Å")
    if "rotation_deg" in meta:
        parts.append(f"rot {meta['rotation_deg']:.3g}°")
    if header.get("final"):
        parts.append("final")
    return "  |  ".join(parts)


class PanelFigure:
    """Draw live frames into a matplotlib figure: one panel per channel.

    A frame is `(array, header)` as `live_stream.unpack` returns it: a (C, Ny, Nx) stack with
    `header["channels"]` naming each image, or a single (Ny, Nx) image. The panels are rebuilt
    whenever the channels or the shape change (e.g. a new scan with another upsampling).
    """

    def __init__(self, fig, *, cmap: str = "gray", high_pass_sigma: float = 0.0,
                 panel_size: float = 4.5, label: str = ""):
        self.fig = fig
        self.cmap = cmap
        self.hp = float(high_pass_sigma)
        self.panel_size = float(panel_size)
        self.label = label
        self.axes, self.ims, self.layout = [], [], None

    def _rebuild(self, names, shape) -> None:
        self.fig.clf()
        self.fig.set_layout_engine("compressed")       # panels fill the figure, title included
        n = len(names)
        self.fig.set_size_inches(self.panel_size * n, self.panel_size + 0.6, forward=True)
        self.axes = [self.fig.add_subplot(1, n, i + 1) for i in range(n)]
        self.ims = []
        for ax, name in zip(self.axes, names):
            self.ims.append(ax.imshow(np.zeros(shape), cmap=self.cmap))
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
        self.layout = (tuple(names), tuple(shape))

    def update(self, arr: np.ndarray, header: dict, title: str | None = None) -> list[tuple]:
        """Draw one frame; returns the (lo, hi) display range of each panel."""
        names = header.get("channels")
        stack = arr if names else arr[None]
        names = names or [self.label or "reconstruction"]
        if self.layout != (tuple(names), tuple(stack.shape[1:])):
            self._rebuild(names, stack.shape[1:])
        ranges = []
        for im, img in zip(self.ims, stack):
            disp = high_pass(img, self.hp) if self.hp > 0 else img
            lo, hi = robust_clim(disp)
            im.set_data(disp)
            im.set_clim(lo, hi)
            ranges.append((lo, hi))
        line = meta_line(header)
        if self.hp > 0:
            line += f"  |  high-pass σ={self.hp:g}px"
        self.fig.suptitle(line if title is None else f"{title}\n{line}", fontsize=10)
        return ranges
