# 0°/90° drift correction with DirectPtycho (branch `4dstem-drift`)

Status and plan for anyone picking this up. Design discussion: issue #14.

## Goal

Measure the linear sample drift live and send it to the microscope PC, which moves the piezo to cancel it:

1. Scan an area at 0° scan rotation; fit the probe aberrations once (DirectPtycho, through fifth order).
2. Scan the same area at +90°; reconstruct it with the same aberrations.
3. Find the drift rate from the two phase images (affine 0°/90° drift correction).
4. Publish the rate (ZMQ) to the microscope PC; the operator applies it to the piezo.
5. Run all of this as Holoscan operators in this pipeline, with a GUI on the microscope PC.

## What is on this branch

| File | What it does |
|---|---|
| `holoscan_pipeline/python/live4d/algorithms/scan_drift.py` | `drift_0_90(scan_0, scan_90, sampling_nm=, line_time_s=)` -> drift in px/line, nm/line, nm/s ((row, col), 0° frame). Takes 4D scans (`Dataset4dstemGPU` or arrays, uses the virtual image) or 2D phase images. Uses quantem's `DriftCorrection.from_images(...).correct_affine()`. |
| `holoscan_pipeline/python/live4d/algorithms/series_store.py` | `SeriesStore`: the encoder sink (begin/complete/drop). It is the only owner of the finished `Dataset4dstemGPU`s; downstream operators pass series names and hold a `lease()` while reading. |
| `holoscan_pipeline/tests/test_scan_drift.py`, `tests/test_series_store.py` | Synthetic 0°/90° pairs with known drift (recovered within 0.005 px/line); store retention and leases. |
| `examples/drift_0_90_gold.py` | Public gold pair (quantem-tutorials `gold_019` 0°, `gold_020` 90°, 0.29771 nm): (+0.022, -0.123) px/line in 2.8 s; the tutorial gives (+0.027, -0.123). |
| `docker/Dockerfile.ans` | Pins quantem.gpu `a8cb286` and quantem (`bobleesj/quantem`, branch `hackathon`) `749e5cb`, which carries the drift code. |

## Method (DirectPtycho, measured on a 512x512 scan, 96x96 detector)

1. Once per session: `detector.fit_bright_field_disk` -> `DirectPtycho(...)` -> `fit_aberrations()` (full search + Newton through fifth order), about 13 s.
2. Every scan: fast Newton from the session fit, `fit_aberrations(search="primary", tilt=True, trials=0)`, 0.12 s; then `reconstruct(fit, upsampling="auto")`, 0.03 s.
3. The 90° scan uses scan-detector rotation R+90°, every aberration angle (`phi*`) +90°, and tilt (row, col) -> (-col, row). Its image (i, j) sees the 0° frame at (j, n-1-i).
4. `drift_0_90(phase_0, phase_90)`, 0.16 s (0.95 s on the first call). px/line does not depend on the upsampling factor: do not divide by it.

Compute from the end of the 90° scan to the drift rate: about 0.3 s. A simulated 90° scan with known drift came back within about 0.001 px/line.

## Next steps

1. **Merge `main` into this branch** (it predates the merged ANS encoder PR #13 and #15); check `SeriesStore` still matches `AnsEncoderOp`'s sink.
2. Check that fast Newton keeps the fifth-order terms (its 0° phase correlates only 0.59 with the full-fit phase). If not, refit only C10/C12 and hold the rest, as quantem.gpu's per-frame stream refit does.
3. `live4d/algorithms/direct_ptycho.py` (no Holoscan imports): steps above; tests and example on the public gold pair only.
4. Operators: `ScanPairOp` (pairs `_rot0`/`_rot90` from the START `header_appendix`), `AberrationFitOp`, `PtychoPhaseOp`, `DriftRateOp`, `DriftPublishOp` (ZMQ PUB, non-blocking, high-water mark 2; subscriber with CONFLATE; set by `drift.live_endpoint`).
5. Rebuild the `Dockerfile.ans` image; measure end-to-end latency on encoded scans.
6. Open: the piezo interface on the microscope PC; whether the scan engine's raster follows `scan_rotation` (compare the first real `_rot0`/`_rot90` pair).

## Rules

- Software never moves the beam, arms the detector or starts a scan; only the operator does.
- No unpublished data or sample names in this repository; test on public data.
