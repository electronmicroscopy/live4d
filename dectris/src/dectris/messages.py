"""Image geometry and the messages passed along a series of the imagemux bstream output
(start, images, end), plus parsing of the CBOR start message and the scan geometry of 4D-STEM.

No holoscan / cupy imports.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cbor2
import numpy as np


IMAGE_SHAPE = (96, 96)
IMAGE_DTYPE = np.uint16
IMAGE_BYTES = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * np.dtype(IMAGE_DTYPE).itemsize  # 18,432


# ---------------------------------------------------------------------------------------------
# Series
#
# Series of the imagemux bstream output: bookmarked by a start and an end frame.
#
# The start message (Stream2, CBOR) carries ``series_id``, ``image_size_x/y``, ``image_dtype`` and
# ``number_of_images``. The series' shape is not known, and the number of images that actually
# arrive between the start and the end frame is what counts, so a series is only described by its
# id (and ``number_of_images`` as a size hint). Everything the pipeline passes along a series is
# here: the marker that opens it, the images, and the marker that closes it.
#
# Images are identified by their id within the series (it restarts at 0 with every series). Which
# GPU an image is processed on is decided by the sender, which writes it into that GPU's ring.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Series:
    id: int
    expected_images: int | None = None  # number_of_images of the start message; a hint only
    frame_time: float | None = None  # frame_time (s) of the start message: the announced rate


@dataclass(frozen=True)
class Start:
    """The start frame of a series; sent to every worker, also those that get no image of it."""

    series: Series


@dataclass(frozen=True)
class Batch:
    """Consecutive images of one series from one ring, on the ring's GPU.

    ``ids`` are the (k,) host image ids within the series (they restart at 0 with every series).
    The data is either ``images``, the (k, 96, 96) uint16 images, or ``sums``, their (k,) uint64
    pixel sums. It is produced asynchronously: it is only complete once the CUDA event ``ready``
    has happened, so wait for it on the stream that reads the data.
    """

    series: Series
    ids: Any
    ready: Any
    images: Any = None
    sums: Any = None


@dataclass(frozen=True)
class End:
    """The end frame of a series; ``images`` is the number of images received in it."""

    series: Series
    images: int


# ---------------------------------------------------------------------------------------------
# Scan
#
# A 4D-STEM scan is a raster of ``height`` rows of ``width`` probe positions; image id ``i`` of a
# series is at row ``i // width``, column ``i % width``. The shape is configured, not announced by
# the sender. Images beyond ``width * height`` have no position.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanShape:
    width: int
    height: int

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"scan {name} must be a positive integer, not {value!r}")

    @property
    def size(self) -> int:
        return self.width * self.height


def scan_positions(ids, scan: ScanShape):
    """``(positions, valid)`` of the image ids ``ids``: ``valid`` (bool, like ``ids``) marks the ids
    inside the scan, ``positions`` the (n, 2) int32 ``[row, column]`` of those, in order."""
    ids = np.asarray(ids, dtype=np.int64)
    valid = (ids >= 0) & (ids < scan.size)
    rows, cols = np.divmod(ids[valid], scan.width)
    return np.stack([rows, cols], axis=1).astype(np.int32), valid


@dataclass(frozen=True)
class Segment:
    """Rows ``start:stop`` of the tensors that ``StemOp`` emitted on ``frames`` in the same tick:
    images of ``series``, in arrival order."""

    series: Series
    start: int
    stop: int


def parse_start(data) -> Series:
    """Series of a start message (CBOR bytes). Raises ValueError if the message is not a start
    message or describes images the pipeline cannot process."""
    try:
        msg = cbor2.loads(bytes(data))
    except Exception as e:  # cbor2 raises its own decode errors
        raise ValueError(f"start message is not valid CBOR: {e}") from e
    if not isinstance(msg, Mapping) or msg.get("type") != "start":
        raise ValueError("not a start message")
    try:
        series_id = int(msg["series_id"])
        shape = (int(msg["image_size_y"]), int(msg["image_size_x"]))
        dtype = msg["image_dtype"]
    except KeyError as e:
        raise ValueError(f"start message has no {e.args[0]}") from e
    if shape != IMAGE_SHAPE or dtype != IMAGE_DTYPE.__name__:
        raise ValueError(f"series {series_id} has {shape[1]}x{shape[0]} {dtype} images, "
                         f"expected {IMAGE_SHAPE[1]}x{IMAGE_SHAPE[0]} {IMAGE_DTYPE.__name__}")
    expected = msg.get("number_of_images")
    frame_time = msg.get("frame_time")
    return Series(series_id, int(expected) if isinstance(expected, int) and expected > 0 else None,
                  float(frame_time) if isinstance(frame_time, (int, float)) and frame_time > 0
                  else None)
