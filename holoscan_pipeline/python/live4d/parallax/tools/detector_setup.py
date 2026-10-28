"""Set up a DECTRIS detector's DCU for the pipeline, arm it and end a series.

Talks to the DCU's SIMPLON REST API (http://<DCU>/{detector,stream,filewriter}/api/<version>/…).
One command per step of a scan:

    python -m live4d.parallax.tools.detector_setup check  --ip <DCU_IP> --scan 256 256 --binned     # read only
    python -m live4d.parallax.tools.detector_setup apply  --ip <DCU_IP> --scan 256 256 --binned --frame-time 8.334e-6
    python -m live4d.parallax.tools.detector_setup arm    --ip <DCU_IP>       # the pipeline must be running already
    python -m live4d.parallax.tools.detector_setup disarm --ip <DCU_IP>       # ends the series: the pipeline reports
    python -m live4d.parallax.tools.detector_setup scan   --ip <DCU_IP> --scan 256 256 --frame-time 8.334e-6

`check` compares each setting the pipeline depends on with what the run needs (stream on, cbor,
bslz4, binning, one image per scan position) and prints the matching `run.py` flags; it changes
nothing. `apply` sets what differs. `scan` arms, runs `start_scan` below, then disarms.

Order for every scan: start the pipeline, arm, start the scan, disarm. The DCU sends the series'
`start` message when it is armed and drops frames while nothing reads the stream.

The scan generator is not driven from here: `start_scan` is an empty slot for each site's own
code. `--ip` is the address of the DCU's REST API; the stream can come from another interface of
the DCU (the address given to `run.py --endpoint`). Needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def start_scan(ny: int, nx: int, frame_time: float) -> None:
    """Start the scan on YOUR scan generator and return when it has finished.

    This depends on the microscope and its scan generator, so it is left to you. Pseudo-code:

        connect to the scan generator
        set the scan size to ny x nx positions and the dwell time to frame_time
        set it to send one trigger per position to the detector
        start the scan
        wait until the scan has finished

    As long as this is not filled in, `scan` asks you to start the scan by hand.
    """
    raise NotImplementedError


class Simplon:
    """A minimal SIMPLON REST client: GET and PUT of {"value": …} documents."""

    def __init__(self, ip: str, port: int = 80, api: str | None = None, timeout: float = 5.0):
        self.base = f"http://{ip}:{port}"
        self.timeout = timeout
        self.api = api or self._version()

    def _call(self, method: str, path: str, value=None, timeout: float | None = None):
        data = None if value is None else json.dumps({"value": value}).encode()
        if method == "PUT" and data is None:
            data = b""
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            body = r.read()
        try:
            return json.loads(body) if body else None
        except ValueError:
            return body.decode(errors="replace")

    def _version(self) -> str:
        try:
            return self._call("GET", "/detector/api/version/")["value"]
        except (urllib.error.HTTPError, KeyError, TypeError):
            return "1.8.0"

    def get(self, module: str, task: str, param: str):
        """The parameter's value, or None when this DCU does not have it."""
        try:
            reply = self._call("GET", f"/{module}/api/{self.api}/{task}/{param}")
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404):
                return None
            raise
        return reply.get("value") if isinstance(reply, dict) else reply

    def put(self, module: str, param: str, value) -> None:
        self._call("PUT", f"/{module}/api/{self.api}/config/{param}", value)

    def command(self, module: str, command: str, timeout: float = 60.0):
        return self._call("PUT", f"/{module}/api/{self.api}/command/{command}", timeout=timeout)


def needed(args) -> list[tuple[str, str, object]]:
    """(module, parameter, value) the run needs, in the order `apply` sets them."""
    ny, nx = args.scan
    out = [("stream", "mode", "enabled"), ("stream", "format", "cbor"),
           ("detector", "compression", "bslz4")]
    if args.binned:
        out.append(("detector", "binning_mode", "2x2"))
    out += [("detector", "trigger_mode", args.trigger_mode), ("detector", "ntrigger", ny * nx),
            ("detector", "nimages", 1)]
    if not args.keep_filewriter:
        out.append(("filewriter", "mode", "disabled"))
    return out


def check(dcu: Simplon, args) -> tuple[list[tuple[str, str, object]], list[str]]:
    """Print the comparison; return the settings that differ and the problems `apply` cannot fix."""
    ny, nx = args.scan
    differ, problems = [], []
    print(f"DCU {dcu.base}, SIMPLON API {dcu.api}")
    for module, param, value in needed(args):
        now = dcu.get(module, "config", param)
        if now is None:
            print(f"  --    {module}/{param}: not on this DCU (skipped)")
            continue
        ok = now == value
        print(f"  {'OK  ' if ok else 'DIFF'}  {module}/{param}: {now!r}"
              + ("" if ok else f"  (needs {value!r})"))
        if not ok:
            differ.append((module, param, value))
    frame_time = dcu.get("detector", "config", "frame_time")
    if args.frame_time is not None and isinstance(frame_time, (int, float)) \
            and abs(frame_time - args.frame_time) > 1e-3 * args.frame_time:   # DCU may round it
        print(f"  DIFF  detector/frame_time: {frame_time!r}  (needs {args.frame_time!r})")
        differ.append(("detector", "frame_time", args.frame_time))
    if not args.binned and dcu.get("detector", "config", "binning_mode") == "2x2":
        problems.append("the detector bins 2x2 (96x96 frames): add --binned, or switch binning off")
    state = dcu.get("detector", "status", "state")
    if state == "na":
        problems.append("the detector is not initialized: initialize it first "
                        "(PUT detector/api/<version>/command/initialize, or the usual software)")
    bits = dcu.get("detector", "config", "bit_depth_image")
    dtype = {16: "uint16", 32: "uint32"}.get(bits)
    if isinstance(frame_time, (int, float)) and frame_time > 0:
        print(f"  info  frame_time {frame_time * 1e6:.3f} us ({1 / frame_time / 1e3:.1f} kHz)"
              + (f", {bits}-bit images" if bits else ""))
    print(f"  info  detector state: {state}, stream state: {dcu.get('stream', 'status', 'state')}, "
          f"frames dropped: {dcu.get('stream', 'status', 'dropped')}")
    flags = f"--scan {ny} {nx}" + (" --binned" if args.binned else "") + \
            (f" --dtype {dtype}" if dtype else "")
    print(f"  run.py flags for this setup: {flags}")
    return differ, problems


def apply(dcu: Simplon, differ) -> None:
    for module, param, value in differ:
        if param == "frame_time":   # the count time is the frame time minus the readout time
            readout = dcu.get("detector", "config", "detector_readout_time") or 0.0
            if value - readout <= 0:
                raise SystemExit(f"frame time {value} s is shorter than the readout time {readout} s")
            dcu.put("detector", "count_time", value - readout)
            print(f"  set   detector/count_time = {value - readout!r}")
        dcu.put(module, param, value)
        print(f"  set   {module}/{param} = {value!r}")


def arm(dcu: Simplon) -> None:
    reply = dcu.command("detector", "arm")
    print(f"Armed: {reply}. Detector state: {dcu.get('detector', 'status', 'state')}.")


def disarm(dcu: Simplon) -> None:
    dcu.command("detector", "disarm")
    print(f"Disarmed (the series' end message goes to the pipeline). "
          f"Frames dropped by the DCU: {dcu.get('stream', 'status', 'dropped')}.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m live4d.parallax.tools.detector_setup",
        description="Set up the DCU for the pipeline (check / apply), arm it and end a series. "
                    "Never drives the scan generator: start_scan is a slot for your own code.")
    p.add_argument("step", choices=("check", "apply", "arm", "disarm", "scan"),
                   help="check: compare the settings (read only); apply: set what differs; "
                        "arm; disarm: end the series; scan: arm, start_scan, disarm.")
    p.add_argument("--ip", required=True, help="Address of the DCU's REST API.")
    p.add_argument("--http-port", type=int, default=80, help="Port of the REST API.")
    p.add_argument("--api", default=None,
                   help="SIMPLON API version, e.g. 1.8.0 (default: asked from the DCU).")
    p.add_argument("--scan", type=int, nargs=2, metavar=("NY", "NX"),
                   help="Scan positions, as given to run.py --scan (check, apply, scan).")
    p.add_argument("--binned", action="store_true", help="2x2 binning (96x96 frames).")
    p.add_argument("--frame-time", type=float, default=None,
                   help="Seconds per frame, the scan generator's dwell time (e.g. 8.334e-6 = "
                        "120 kHz). Default: leave the DCU's as it is.")
    p.add_argument("--trigger-mode", default="exts",
                   help="exts = one external trigger per frame, from the scan generator.")
    p.add_argument("--keep-filewriter", action="store_true",
                   help="Leave the DCU's file writer as it is (default: disabled, live only).")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.step in ("check", "apply", "scan") and args.scan is None:
        parser.error(f"{args.step} needs --scan NY NX")
    try:
        dcu = Simplon(args.ip, args.http_port, args.api)
        if args.step == "arm":
            arm(dcu)
            return 0
        if args.step == "disarm":
            disarm(dcu)
            return 0
        differ, problems = check(dcu, args)
        if args.step == "apply" and differ and not problems:
            apply(dcu, differ)
            differ, problems = check(dcu, args)
        for problem in problems:
            print(f"PROBLEM: {problem}")
        if differ:
            print("Some settings differ from what the run needs"
                  + (": run `apply` to set them." if args.step == "check" else "."))
        if differ or problems:
            return 1
        print("The DCU is set up for this run.")
        if args.step == "scan":
            print("Is the pipeline running and waiting for a scan? It must be, before arming.")
            arm(dcu)
            try:
                start_scan(*args.scan, args.frame_time)
            except NotImplementedError:
                input("Start the scan now. Press Enter when it has finished … ")
            finally:
                disarm(dcu)
        return 0
    except (urllib.error.URLError, OSError) as exc:
        print(f"Cannot talk to the DCU's REST API at http://{args.ip}:{args.http_port} ({exc}). "
              f"Check the address: the REST API and the stream can be on different interfaces.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
