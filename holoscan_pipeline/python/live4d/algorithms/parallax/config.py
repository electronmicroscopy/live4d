"""
Configuration for the real-time parallax / tcBF pipeline.

All tunable parameters live here as plain data so they can be edited without touching logic. On
top of the detector/scan geometry it carries the parallax-specific **acquisition physics** the
geometry preprocessing needs: beam energy, convergence semiangle, detector angular sampling, scan
sampling, and the aberration coefficients (defocus C10, optional astigmatism C12/phi12) that set
the parallax shifts.

Parallax needs a *defocus* to compute shifts: the microscope's reported defocus is the starting
point (override live with `--defocus`); the shift field itself is observable, so a future step can
refine C10 from the data.

The values here are plausible ARINA-scale examples for tests and the Python API, not measured
constants. The command line does not fall back on them for the acquisition values
(`REQUIRED_ACQUISITION_ARGS`): a run without them stops with a list of what is missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from live4d.algorithms.parallax.orientation import DETECTOR_LAYOUTS, layout_center
from live4d.algorithms.parallax.readout import ReadoutConfig


@dataclass(frozen=True)
class ParallaxConfig:
    # ── Detector geometry ─────────────────────────────────────────────────────
    # Swap the active line, or override with --detector / --binned.
    detector_shape: tuple[int, int] = (192, 192)    # ARINA full resolution (unbinned)
    # detector_shape: tuple[int, int] = (96, 96)    # ARINA binned ×2

    # ── Scan geometry ─────────────────────────────────────────────────────────
    # Frames arrive in raster order and are placed by message id: (id // Nx, id % Nx).
    scan_shape: tuple[int, int] = (1024, 1024)      # override with --scan NY NX

    # ── Parallax acquisition physics (for preprocess_geometry) ────────────────
    energy: float = 300e3                               # beam energy, eV
    semiangle_cutoff: float = 4.0                       # probe convergence semiangle, mrad
    angular_sampling: tuple[float, float] = (0.5, 0.5)  # detector pixel size, mrad/px
    scan_sampling: tuple[float, float] = (0.5, 0.5)     # scan step, Å
    # Aberrations in Å / Å / rad. C10 = defocus (the dominant term; negative = underfocus in the
    # reference's sign convention). Override C10 live with --defocus.
    aberration_coefs: dict = field(default_factory=lambda: {"C10": -1.5e4})
    rotation_angle: float | None = None                 # scan/detector rotation, rad
    upsampling_factor: int = 1                          # output-grid upsampling (finer shifts)
    # BF-disk center on the detector, (row, col) px — py4DSTEM get_probe_size's
    # (Center_X, Center_Y). None → frame center. Real beams sit off-center; override with --center.
    probe_center: tuple[float, float] | None = None
    # Detector columns reversed relative to the canonical (Q_rows, Q_cols) layout — LiberTEM's
    # `detector_flip_cols`; with rotation_angle it describes any transpose/flip. --flip-cols.
    flip_cols: bool = False

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def n_scan_y(self) -> int:
        return self.scan_shape[0]

    @property
    def n_scan_x(self) -> int:
        return self.scan_shape[1]

    @property
    def num_positions(self) -> int:
        return self.scan_shape[0] * self.scan_shape[1]


# Default instance imported by the rest of the pipeline. Construct a different
# ParallaxConfig(...) (e.g. in tests or for another scan) to override.
CONFIG = ParallaxConfig()


# The acquisition values that change with every microscope and session. No default is safe for
# them: a wrong one gives a plausible but wrong image, not an error. (flag, attribute, unit / hint)
REQUIRED_ACQUISITION_ARGS = (
    ("--energy", "energy", "beam energy in eV (300 kV = 300000)"),
    ("--semiangle", "semiangle", "probe convergence semiangle in mrad"),
    ("--angular-sampling", "angular_sampling",
     "detector pixel size in mrad/px for the mode running (binned or not)"),
    ("--scan-sampling", "scan_sampling", "scan step in Å"),
    ("--scan", "scan", "scan positions NY NX"),
    ("--defocus", "defocus", "defocus C10 in Å, negative = underfocus"),
    ("--rotation", "rotation", "scan/detector rotation in degrees"),
)   # --center is not here: the frame center is a sound default for a centered detector


def missing_acquisition_args(args) -> list[str]:
    """One message per required acquisition flag that was not given (empty: all there)."""
    missing = [f"    {flag:<20} {hint}" for flag, attr, hint in REQUIRED_ACQUISITION_ARGS
               if getattr(args, attr, None) is None]
    if not missing:
        return []
    return ["these acquisition values have no default (they differ per microscope and "
            "session):\n" + "\n".join(missing)]


# ── shared CLI tuning knobs ───────────────────────────────────────────────────
# The tuning-knob flag group and its unit conversions live HERE (not in run.py) so the parallax
# algorithm owns its CLI in one place — the same flags, with the same units, whether you tune a
# defocus offline against a replay or drive it live. CLI angles are degrees (the microscope readout).

def add_tuning_args(p) -> None:
    """Attach the shared parallax tuning flags to an argparse parser (each help names the
    config default it overrides)."""
    c = CONFIG
    p.add_argument("--defocus", type=float, default=None, metavar="C10_ANG",
                   help="Target defocus C10 (Å) the parallax shifts are built for (negative = "
                        "underfocus). Required.")
    p.add_argument("--rotation", type=float, default=None, metavar="DEG",
                   help="Scan/detector rotation (degrees). Required.")
    p.add_argument("--c12", type=float, default=None, metavar="C12_ANG",
                   help="Target astigmatism magnitude C12 (Å). Config default: 0.")
    p.add_argument("--phi12", type=float, default=None, metavar="DEG",
                   help="Target astigmatism angle phi12 (degrees). Config default: 0.")
    p.add_argument("--upsampling", type=int, default=None, metavar="U",
                   help="Output-grid upsampling (finer grid → finer shifts). "
                        f"Config default: {c.upsampling_factor}.")
    p.add_argument("--energy", type=float, default=None, metavar="EV",
                   help="Beam energy (eV; 300 kV = 300000). Required.")
    p.add_argument("--semiangle", type=float, default=None, metavar="MRAD",
                   help="Probe convergence semiangle, the bright-field cutoff (mrad). "
                        f"Required.")
    p.add_argument("--angular-sampling", type=float, default=None, metavar="MRAD_PX",
                   help="Detector angular sampling (mrad/px, isotropic) of the mode actually "
                        "running (binned or not). Required.")
    p.add_argument("--scan-sampling", type=float, default=None, metavar="ANG_PX",
                   help="Scan step (Å, isotropic). Required.")
    p.add_argument("--center", type=float, nargs=2, default=None, metavar=("ROW", "COL"),
                   help="BF-disk center on the raw detector frames (px), py4DSTEM's Center_X "
                        "Center_Y from get_probe_size. Default: the frame center (mapped through "
                        "--detector-layout). (The disk's radius is not an input: it is "
                        "--semiangle / --angular-sampling.)")
    p.add_argument("--scan", type=int, nargs=2, default=None, metavar=("NY", "NX"),
                   help="Scan grid; must match what the microscope drives. Required.")
    p.add_argument("--flip-cols", action="store_true",
                   help="The detector's columns are reversed relative to the canonical layout "
                        "(LiberTEM's detector_flip_cols). With --rotation this covers every "
                        "transpose/flip: e.g. a transposed detector is --rotation +90 --flip-cols. "
                        "Give --center as measured on the raw frames. Table in docs/GUIDE.md.")
    p.add_argument("--detector-layout", choices=list(DETECTOR_LAYOUTS), default=None,
                   metavar="LAYOUT",
                   help="What the raw frame's (axis 0, axis 1) hold relative to the canonical "
                        "(rows, cols) layout: " + " | ".join(DETECTOR_LAYOUTS) + ". E.g. "
                        "'cols,-rows' = transposed with the second axis reversed. Adds the "
                        "matching rotation and column flip to --rotation, and maps the default "
                        "disk center onto the raw frame. Exclusive with --flip-cols.")


def add_readout_args(p) -> None:
    """Attach the readout flags: what each live snapshot / saved image derives from the
    accumulated sum (no hot-path cost). See processing/parallax/readout.py."""
    g = p.add_argument_group("parallax readout (live channels and saved images)")
    g.add_argument("--no-phase-flip", dest="phase_flip", action="store_false",
                   help="Don't add the phase-flipped (sign(sin chi(q))) channel.")
    g.add_argument("--no-dc-correct", dest="dc_correct", action="store_false",
                   help="Don't remove each bright-field pixel's running scan mean (it removes the "
                        "static bright-field pattern at the scan front and on upsampled grids).")
    g.add_argument("--normalize", action="store_true",
                   help="Divide by the per-pixel deposit count (tapered below 1%% of its peak), "
                        "so a partial scan reads at full brightness.")
    g.add_argument("--coverage", action="store_true",
                   help="Add the per-pixel deposit count as a channel.")
    g.add_argument("--lowpass", type=float, default=None, metavar="Q",
                   help="Butterworth low-pass cutoff (1/Å).")
    g.add_argument("--highpass", type=float, default=None, metavar="Q",
                   help="Butterworth high-pass cutoff (1/Å).")
    g.add_argument("--butterworth-order", type=int, default=12, metavar="N",
                   help="Steepness of the Butterworth low/high-pass edges.")
    g.add_argument("--suppress-nyquist", choices=("auto", "on", "off"), default="auto",
                   help="Zero the Nyquist row/column of the output (auto: on when --upsampling > 1).")
    g.add_argument("--band-limit", action="store_true",
                   help="Clamp the low-pass at the parallax band limit 2*alpha/lambda.")


def readout_from_args(args) -> ReadoutConfig:
    """The `ReadoutConfig` for the readout flags (defaults when they were never added)."""
    nyq = getattr(args, "suppress_nyquist", "auto")
    return ReadoutConfig(
        phase_flip=getattr(args, "phase_flip", True),
        dc_correct=getattr(args, "dc_correct", True),
        normalize=getattr(args, "normalize", False),
        coverage_channel=getattr(args, "coverage", False),
        q_lowpass=getattr(args, "lowpass", None),
        q_highpass=getattr(args, "highpass", None),
        butterworth_order=getattr(args, "butterworth_order", 12),
        suppress_nyquist=None if nyq == "auto" else nyq == "on",
        band_limit=getattr(args, "band_limit", False),
    )


# Float-valued tuning flags whose natural values are often negative (e.g. underfocus C10).
_TUNING_FLOAT_FLAGS = {"--defocus", "--rotation", "--c12", "--phi12", "--energy",
                       "--semiangle", "--angular-sampling", "--scan-sampling"}


def join_negative_floats(argv: list[str]) -> list[str]:
    """Rewrite `--defocus -1.5e4` to `--defocus=-1.5e4` — argparse only auto-recognizes plain
    negative numbers (`-15000`), not scientific notation, so `-1.5e4` parses as a flag."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in _TUNING_FLOAT_FLAGS and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            try:
                float(argv[i + 1])
            except ValueError:
                pass
            else:
                out.append(f"{a}={argv[i + 1]}")
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def config_from_args(args, cfg: ParallaxConfig | None = None) -> ParallaxConfig:
    """Apply the CLI tuning knobs to the config — the one place units are converted.

    CLI angles are degrees (what the microscope UI shows) and become radians here; a single
    sampling value becomes an isotropic (y, x) tuple. `--binned` doubles the config-default angular
    sampling, but an explicit `--angular-sampling` describes the actual running mode and wins.
    """
    if cfg is None:
        cfg = CONFIG
    overrides = {}
    if getattr(args, "binned", False):
        # 2×2 binning halves the detector and doubles the per-pixel angular sampling.
        overrides.update(detector_shape=(96, 96),
                         angular_sampling=(cfg.angular_sampling[0] * 2, cfg.angular_sampling[1] * 2))
    if getattr(args, "detector", None) is not None:
        overrides["detector_shape"] = (int(args.detector[0]), int(args.detector[1]))

    aberr = dict(cfg.aberration_coefs)
    if args.defocus is not None:
        aberr["C10"] = args.defocus
    if args.c12 is not None:
        aberr["C12"] = args.c12
    if args.phi12 is not None:
        aberr["phi12"] = math.radians(args.phi12)
    if aberr != cfg.aberration_coefs:
        overrides["aberration_coefs"] = aberr

    if args.rotation is not None:
        overrides["rotation_angle"] = math.radians(args.rotation)
    if args.upsampling is not None:
        overrides["upsampling_factor"] = args.upsampling
    if args.energy is not None:
        overrides["energy"] = args.energy
    if args.semiangle is not None:
        overrides["semiangle_cutoff"] = args.semiangle
    if args.angular_sampling is not None:
        overrides["angular_sampling"] = (args.angular_sampling, args.angular_sampling)
    if args.scan_sampling is not None:
        overrides["scan_sampling"] = (args.scan_sampling, args.scan_sampling)
    if args.center is not None:
        overrides["probe_center"] = (args.center[0], args.center[1])
    if args.scan is not None:
        overrides["scan_shape"] = (int(args.scan[0]), int(args.scan[1]))
    if getattr(args, "flip_cols", False):
        overrides["flip_cols"] = True

    layout = getattr(args, "detector_layout", None)
    if layout is not None:
        if getattr(args, "flip_cols", False):
            raise SystemExit("--detector-layout already sets the column flip; drop --flip-cols.")
        rot_adjust, flip = DETECTOR_LAYOUTS[layout]
        base = overrides.get("rotation_angle", cfg.rotation_angle) or 0.0
        overrides["rotation_angle"] = base + rot_adjust
        overrides["flip_cols"] = flip
        if overrides.get("probe_center", cfg.probe_center) is None and layout != "rows,cols":
            det = overrides.get("detector_shape", cfg.detector_shape)
            overrides["probe_center"] = layout_center(det, layout)

    return replace(cfg, **overrides) if overrides else cfg   # frozen dataclass → replace()
