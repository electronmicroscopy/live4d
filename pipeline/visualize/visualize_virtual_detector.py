"""
Visualize a virtual-detector image (virtual bright-field).

The minimal "core" imaging method -- it displays whatever 2D image it is handed. Per-method
visualizers live in this shared `visualize/` folder; richer features (live updating, multiple
panels, scalebars, shared display machinery) come later, once a second visualizer justifies
them.
"""

from __future__ import annotations

import numpy as np


def show_image(
    image,
    title: str = "Virtual Bright-Field",
    cmap: str = "gray",
    save_path: str | None = None,
):
    """Display a 2D image; optionally save it instead of showing.

    Args:
        image:     any 2D array-like -- e.g. the (Ny, Nx) array from `VBFAccumulator.result()`
                   (a torch tensor is also accepted and moved to host automatically).
        title:     plot title.
        cmap:      matplotlib colormap.
        save_path: if given, write the figure here (headless) instead of opening a window.

    Returns:
        (fig, ax) so callers can tweak or save further.
    """
    # Imported lazily so merely importing this module is cheap and doesn't require matplotlib
    # until something is actually drawn.
    import matplotlib.pyplot as plt

    arr = _to_numpy(image)

    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("scan x")
    ax.set_ylabel("scan y")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    return fig, ax


class LiveImage:
    """A reusable image panel that updates in place -- for live (during-scan) display.

    GUI calls must happen on the **main thread**, so the intended use is: run the streaming
    pipeline on a background thread, and on the main thread poll the always-current image and
    refresh at a human rate (a few Hz). The buffer being snapshot may be a batch behind or
    mid-update -- harmless for a live preview.

        live = LiveImage(scan_shape)
        # start the pipeline on a background thread, then:
        while pipeline_thread.is_alive():
            live.update(acc.result())     # snapshot + redraw (also throttles via `interval`)
        live.update(acc.result())         # final frame
        live.keep_open()                  # block so the window stays up after the scan
    """

    def __init__(
        self,
        scan_shape: tuple[int, int],
        title: str = "Virtual Bright-Field (live)",
        cmap: str = "gray",
        interval: float = 0.2,
    ):
        import matplotlib.pyplot as plt

        self._plt = plt
        plt.ion()  # interactive mode: draw without blocking
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.zeros(scan_shape, dtype=np.float32), cmap=cmap)
        self.ax.set_title(title)
        self.ax.set_xlabel("scan x")
        self.ax.set_ylabel("scan y")
        self.fig.colorbar(self.im, ax=self.ax)
        self.interval = interval

    def update(self, image) -> None:
        """Replace the displayed data, autoscale the color range, and pump the GUI."""
        arr = _to_numpy(image)
        self.im.set_data(arr)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:  # autoscale as the image fills in; skip while still all-zero
            self.im.set_clim(lo, hi)
        self.fig.canvas.draw_idle()
        # `pause` flushes GUI events (keeps the window responsive) and throttles the loop.
        self._plt.pause(self.interval)

    def keep_open(self) -> None:
        """Leave the finished window open (blocking) at the end of the scan."""
        self._plt.ioff()
        self._plt.show()


def _to_numpy(image) -> np.ndarray:
    """Coerce an array-like (including a torch tensor) to a 2D NumPy array."""
    if hasattr(image, "detach"):  # torch tensor -> host NumPy
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {arr.shape}")
    return arr
