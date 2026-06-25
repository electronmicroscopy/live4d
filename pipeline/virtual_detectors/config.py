"""
Configuration for the real-time virtual-detector (VBF) pipeline.

All tunable parameters live here as plain data so they can be edited without touching logic.
This is the single, self-contained config for the live app.

Values marked `# CONFIRM` are experiment/lab-specific placeholders -- set them to match the
actual acquisition before a live run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Directory of this file, so mask paths resolve regardless of working directory.
_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VBFConfig:
    # -- DCU connection ----------------------------------------------------------
    # Detector control unit address (ZMQ PUSH side); set to match the deployment.
    dcu_ip: str = "11.0.0.1"
    zmq_port: int = 31001

    # -- Detector geometry -------------------------------------------------------
    # Active: ARINA full resolution. Swap the active line to change camera/mode; keep
    # bf_center (below) and any real mask files in sync with the chosen shape.
    # detector_shape: tuple[int, int] = (96, 96)    # ARINA binned x2
    detector_shape: tuple[int, int] = (192, 192)    # ARINA full resolution (unbinned) -- active
    # detector_shape: tuple[int, int] = (512, 512)  # QUADRO

    # -- Scan geometry -----------------------------------------------------------
    # The raster scan as configured at the microscope. Frames arrive in raster
    # order and are placed by message id: (id // Nx, id % Nx).
    scan_shape: tuple[int, int] = (1024, 1024)  # 1,048,576 frames; a long-run dataset
    # scan_shape: tuple[int, int] = (32, 32)    # QUADRO 32x32 (1024 frames)

    # -- Bright-field region (virtual detector) ----------------------------------
    # A disk on the detector: center pixel + radius in pixels. Keep in sync with
    # detector_shape above.
    # bf_center_yx: tuple[float, float] = (48.0, 48.0)  # ARINA binned x2 center
    bf_center_yx: tuple[float, float] = (96.0, 96.0)    # ARINA full-res (unbinned) center -- active
    # bf_center_yx: tuple[float, float] = (256.0, 256.0)  # QUADRO center
    bf_radius_px: float = 30.0                          # CONFIRM: BF disk radius (~2x larger unbinned vs binned)

    # -- Acquisition timing (externally supplied -- NOT carried in the stream) --
    # The capture messages contain only frames; the per-frame period (dwell) must be
    # supplied here. Detector minimum frame periods (fastest mode):
    #   unbinned 192x192 -> 33.30 us  (~ 30 kHz)   |   binned 96x96 -> 8.33 us  (~ 120 kHz)
    # Printed at run end to compare our processing rate vs the detector rate; not enforced.
    frame_period_us: float = 33.30        # active: unbinned 192x192 minimum
    # frame_period_us: float = 8.33       # binned 96x96 minimum

    # -- Calibration mask files (same shape as detector) -------------------------
    flatfield_path: Path = _HERE / "masks" / "flatfield.npy"   # all ones by default
    dead_px_path: Path = _HERE / "masks" / "dead_px.npy"       # 1 = good, 0 = dead

    # -- Pipeline ----------------------------------------------------------------
    # Batch size for GPU hand-off. None -> one scan row (Nx).
    batch_size: int | None = None
    n_decode_threads: int = 4          # decoder-pool size (measured optimum; collapses at 6+ -- GIL convoy)
    queue_maxsize: int = 256           # bound on per-frame queues (general-purpose; unused by the chunked pipeline)

    # -- Chunked-pipeline knobs --------------------------------------------------
    # chunk_size = how many frames a decoder grinds per chunk; it is ALSO the GPU batch size
    # (each decoded chunk is one on_batch call). Measured optimum is ~64-128 (broad plateau);
    # counter-intuitively, *smaller* beats bigger because each decoded chunk allocates a fresh
    # (chunk*H*W*dtype) buffer and large allocs (e.g. 1024*192*192*4 ~ 150 MB) hit slower memory
    # paths. 128 is the round default near the peak.
    chunk_size: int = 128
    # Bound on the chunk queues (raw + decoded). Small is fine -- chunks are large, so a handful
    # already buffers plenty and keeps backpressure tight. Peak memory ~ a few x chunk bytes.
    chunk_queue_maxsize: int = 8

    # -- Derived helpers ---------------------------------------------------------
    @property
    def n_scan_y(self) -> int:
        return self.scan_shape[0]

    @property
    def n_scan_x(self) -> int:
        return self.scan_shape[1]

    @property
    def num_positions(self) -> int:
        return self.scan_shape[0] * self.scan_shape[1]

    @property
    def effective_batch_size(self) -> int:
        """Batch size actually used: explicit value, else one scan row."""
        return self.batch_size if self.batch_size is not None else self.n_scan_x

    @property
    def detector_rate_hz(self) -> float:
        """Detector frame rate implied by the (externally-supplied) frame period."""
        return 1e6 / self.frame_period_us


# Default instance imported by the rest of the pipeline. Construct a different
# VBFConfig(...) (e.g. in tests or for another scan) to override.
CONFIG = VBFConfig()
