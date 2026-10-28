"""Parallax YAML settings; validation requires no GPU or PyTorch installation."""
import math

DEFAULTS = dict(
    enabled=False, energy=None, semiangle=None, angular_sampling=None, scan_sampling=None,
    defocus=None, rotation=None, center=None, c12=None, phi12=None, upsampling=1,
    flip_cols=False, detector_layout=None, subtract_frame_mean=False,
    phase_flip=True, dc_correct=True, normalize=False, coverage=False,
    lowpass=None, highpass=None, butterworth_order=12, suppress_nyquist="auto", band_limit=False,
    accumulate="scan", pass_average=1.0, save_passes=False, save_passes_dtype="float32",
    out_dir="out", live_endpoint="", live_every=0.2, live_every_rows=0,
    viewer_port=0, viewer_host="127.0.0.1")


def validate(settings):
    if not isinstance(settings, dict) or set(settings) - DEFAULTS.keys():
        raise ValueError("Invalid parallax settings")
    values = DEFAULTS | settings
    for key in ("enabled", "flip_cols", "subtract_frame_mean", "phase_flip", "dc_correct",
                "normalize", "coverage", "band_limit", "save_passes"):
        if type(values[key]) is not bool:
            raise ValueError(f"parallax.{key} must be true or false")
    required = ("energy", "semiangle", "angular_sampling", "scan_sampling", "defocus", "rotation")
    if values["enabled"]:
        missing = [key for key in required if values[key] is None]
        if missing:
            raise ValueError("parallax requires acquisition settings: " + ", ".join(missing))
    for key in (*required, "c12", "phi12", "lowpass", "highpass", "pass_average", "live_every"):
        value = values[key]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"parallax.{key} must be finite")
    for key in ("energy", "semiangle", "angular_sampling", "scan_sampling", "lowpass", "highpass"):
        if values[key] is not None and values[key] <= 0:
            raise ValueError(f"parallax.{key} must be positive")
    for key, low, high in (("upsampling", 1, 64), ("butterworth_order", 1, 128),
                            ("live_every_rows", 0, 2147483647), ("viewer_port", 0, 65535)):
        if type(values[key]) is not int or not low <= values[key] <= high:
            raise ValueError(f"Invalid parallax.{key}")
    if values["live_every"] is None or values["pass_average"] is None:
        raise ValueError("Parallax live_every and pass_average must be numbers")
    if values["live_every"] < 0 or not 0 < values["pass_average"] <= 1:
        raise ValueError("Invalid parallax snapshot interval or pass_average")
    if values["accumulate"] not in ("scan", "latest", "all", "reset", "swap"):
        raise ValueError("Invalid parallax.accumulate")
    if values["pass_average"] != 1 and values["accumulate"] != "swap":
        raise ValueError("parallax.pass_average requires accumulate: swap")
    if values["save_passes"] and values["accumulate"] == "scan":
        raise ValueError("parallax.save_passes requires repeated accumulation")
    if values["save_passes_dtype"] not in ("float32", "float16"):
        raise ValueError("Invalid parallax.save_passes_dtype")
    if values["suppress_nyquist"] not in ("auto", "on", "off"):
        raise ValueError("Invalid parallax.suppress_nyquist")
    layouts = ("rows,cols", "rows,-cols", "-rows,cols", "-rows,-cols",
               "cols,rows", "cols,-rows", "-cols,rows", "-cols,-rows")
    if values["detector_layout"] is not None and values["detector_layout"] not in layouts:
        raise ValueError("Invalid parallax.detector_layout")
    if values["detector_layout"] is not None and values["flip_cols"]:
        raise ValueError("parallax.detector_layout already sets the column flip")
    center = values["center"]
    if center is not None and (not isinstance(center, (list, tuple)) or len(center) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in center)):
        raise ValueError("parallax.center must contain two finite pixel coordinates")
    for key in ("out_dir", "live_endpoint", "viewer_host"):
        if not isinstance(values[key], str) or (key != "live_endpoint" and not values[key]):
            raise ValueError(f"Invalid parallax.{key}")
    return values
