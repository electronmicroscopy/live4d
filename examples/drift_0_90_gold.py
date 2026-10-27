"""Affine drift rate of a recorded 0°/90° 4D-STEM pair (gold, quantem-tutorials drift/4d_4dstem.ipynb).

    python examples/drift_0_90_gold.py ~/datasets/gold_4dstem [frame_time_us]

Data: the gold_4dstem folder linked from electronmicroscopy/quantem-tutorials
tutorials/imaging/drift/4d_4dstem.ipynb (gold_019 = 0°, gold_020 = 90°).
"""
import sys
import time
from pathlib import Path

from quantem.gpu.io import load

from live4d.algorithms.scan_drift import drift_0_90

folder = Path(sys.argv[1]).expanduser()
frame_time_s = float(sys.argv[2]) * 1e-6 if len(sys.argv) > 2 else None
PIXEL_SIZE_NM = 0.29771

# ANS-encoded on the GPU, as the live encoder keeps them; quantem decodes what the drift search reads.
scans = [load(folder / f"gold_{n}_master.h5", verbose=False) for n in ("019", "020")]
print(f"0° {tuple(scans[0].shape)}, 90° {tuple(scans[1].shape)}")
line_time_s = frame_time_s * scans[0].shape[1] if frame_time_s else None
start = time.perf_counter()
rate = drift_0_90(*scans, sampling_nm=PIXEL_SIZE_NM, line_time_s=line_time_s, verbose=True)
print(f"drift {rate.summary()}  ({time.perf_counter() - start:.1f} s)")
