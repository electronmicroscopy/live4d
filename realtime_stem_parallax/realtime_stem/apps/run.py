"""Run a real-time reconstruction.

    python -m realtime_stem.apps.run --algorithm parallax [transport flags] [algorithm flags]

This is the one place ingest → processing → stream-out are wired together. It is deliberately
generic: it owns the transport flags (endpoint, batching, scheduler, live preview) and delegates
everything method-specific to the selected `Algorithm` (`--algorithm`). Adding a method needs no
change here — see `realtime_stem/processing/registry.py`.

Runs inside the image built by `docker/Dockerfile` (this package bind-mounted), e.g.:

    docker run --rm --network host --gpus "device=0" --ulimit stack=33554432:33554432 \
      -v "$PWD":/work -v "$PWD/out":/out --entrypoint python3 \
      realtime-stem:latest -m realtime_stem.apps.run \
      --algorithm parallax --endpoint tcp://<DCU_IP>:31001 --batch-size 2048 \
      --binned --scan 256 256 --energy <eV> --semiangle <mrad> --angular-sampling <mrad/px> \
      --scan-sampling <Å> --defocus <Å> --rotation <deg> --out-dir /out

The acquisition values (energy, semiangle, angular and scan sampling, scan, defocus, rotation)
have no defaults: without them the run stops and lists what is missing.

`--continuous` (optionally `--rounds N`) re-arms with a fresh accumulator after each scan's `end`,
so a whole session of scans runs against one container. See the README for the full walkthrough
and the offline replay demo (no detector required).

After every scan the runner reports whether each scan position received exactly one frame
("COMPLETE" or "WARNING: INCOMPLETE SCAN"); the result is saved either way.

With `--transport rdma` (Holoscan 5 image, `docker/Dockerfile.hsdk5`) the frames come over the
DCU's imagemux RDMA instead, uncompressed (DCU compression `none`), straight into GPU memory:

    python3 -m realtime_stem.apps.run --transport rdma --rdma-remote <DCU_RDMA_IP>:31003 \
      --rdma-local <this machine's RoCE IP> --binned --scan 256 256 ... --out-dir /out

Two ingests feed the same processing and stream-out (`--ingest`): the compiled Holoscan graph
(NVIDIA GPU, full detector rate, inside the container) and a pure-Python one (CPU decode, any torch
device incl. Apple's mps, tens of kHz) that also reads captures directly (`--source`):

    python -m realtime_stem.apps.run --ingest python --device mps \
      --source capture.dstream --source-rate 5000 --binned $(cat capture.flags) \
      --live-endpoint tcp://127.0.0.1:5601
"""

from __future__ import annotations

import argparse
import importlib.util
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

# Make the package importable when run as a bare script from a bind mount
# (python3 /work/realtime_stem/apps/run.py): parents[2] is the repo root that holds the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from realtime_stem import __version__   # noqa: E402
from realtime_stem.processing import registry   # noqa: E402
from realtime_stem.processing.base import Algorithm   # noqa: E402
from realtime_stem.ingest.sink import publish_snapshot   # noqa: E402



def holoscan_generation() -> int | None:
    """Which Holoscan ingest this environment has: 5 in the Holoscan 5 image
    (`docker/Dockerfile.hsdk5`: ZMQ and RDMA), 4 in the Holoscan 4 image (`docker/Dockerfile`:
    ZMQ), None outside a container."""
    if importlib.util.find_spec("holoscan") is None:
        return None
    if importlib.util.find_spec("live4d") is not None:
        return 5
    if importlib.util.find_spec("dectris_holoscan_ops") is not None:
        return 4
    return None


def endpoint_warning(endpoint: str, timeout: float = 2.0) -> str | None:
    """A warning when nothing accepts TCP connections at a `tcp://host:port` endpoint, else None.

    ZeroMQ connects lazily and retries forever, so a wrong DCU address, a disabled stream or a
    firewall otherwise looks exactly like "waiting for a scan". This opens and immediately closes
    one plain TCP connection: it carries no ZeroMQ handshake, so the sender never treats it as a
    receiver and no frame goes to it. Purely advisory: the pipeline starts either way. Local
    endpoints are not checked (a replay feeder there usually starts after the pipeline).
    """
    parts = urlsplit(endpoint)
    if parts.scheme != "tcp" or not parts.hostname or parts.port is None:
        return None
    if parts.hostname in ("localhost", "127.0.0.1", "::1"):
        return None
    try:
        with socket.create_connection((parts.hostname, parts.port), timeout=timeout):
            return None
    except OSError as exc:
        return (f"[run] WARNING: cannot reach {parts.hostname}:{parts.port} ({exc}). No frames "
                f"can arrive until this is fixed: check the DCU address and port, that its "
                f"stream interface is enabled, and the network / firewall "
                f"(`DCU={parts.hostname} scripts/check_install.sh`). The pipeline keeps trying; "
                f"Ctrl-C stops it.")


def watch_endpoint(endpoint: str, every: float = 30.0) -> threading.Thread | None:
    """Warn now when the endpoint is unreachable (`endpoint_warning`) and keep checking in the
    background: the warning repeats every `every` seconds until the endpoint answers, so a wrong
    address cannot pass for a quiet wait for the scan. Returns the watching thread, or None when
    there is nothing to watch."""
    warning = endpoint_warning(endpoint)
    if warning is None:
        return None
    print(warning, flush=True)
    where = urlsplit(endpoint).netloc

    def watch() -> None:
        t0 = time.monotonic()
        while True:
            time.sleep(every)
            if endpoint_warning(endpoint) is None:
                print(f"[run] {where} is reachable now.", flush=True)
                return
            print(f"[run] WARNING: still cannot reach {where} after {time.monotonic() - t0:.0f} s; "
                  f"no frames can arrive. Check the DCU address (Ctrl-C stops the pipeline).",
                  flush=True)

    thread = threading.Thread(target=watch, name="endpoint-watch", daemon=True)
    thread.start()
    return thread


def gpu_report() -> list[str]:
    """Which GPU the run uses (name, memory, PCI bus as in nvidia-smi's Bus-Id), plus warnings
    when several GPUs are visible or when other programs already hold much of its memory."""
    import torch
    if not torch.cuda.is_available():
        return []
    p = torch.cuda.get_device_properties(0)
    bus = getattr(p, "pci_bus_id", None)
    lines = [f"[run] GPU: {p.name}, {p.total_memory / 2**30:.0f} GiB"
             + (f", PCI bus {bus:02X}" if isinstance(bus, int) else "")]
    n = torch.cuda.device_count()
    if n > 1:
        lines.append(f"[run] WARNING: {n} GPUs are visible; this run uses the first in CUDA's "
                     f"order, which can differ from nvidia-smi's numbering. Choose one with "
                     f"docker run --gpus \"device=N\" (N as nvidia-smi numbers them).")
    free, total = torch.cuda.mem_get_info(0)
    if total - free > 2 * 2**30:
        lines.append(f"[run] note: {(total - free) / 2**30:.0f} of {total / 2**30:.0f} GiB of this "
                     f"GPU are already in use (nvidia-smi shows by what). Is this the right GPU?")
    return lines


def batch_sink(app):
    """The `BatchSink` driving an ingest app: the Python ingest's own, or the Holoscan graph's
    sink operator's (made only when the graph starts; None before)."""
    sink = getattr(app, "sink", None)
    return getattr(sink, "core", sink)


def wait_with_viewer(control, stop, algorithm, out_dir: Path) -> None:
    """After the last scan, keep serving the viewer (and its live control: with --accumulate
    latest every change still re-renders the frames held) until Ctrl-C or a stop signal. Results
    tuned meanwhile are saved again on exit."""
    print("[run] done; the viewer stays up for tuning (Ctrl-C to exit). Flags now: "
          f"{control.state()['flags']}", flush=True)
    seen = control.revision
    try:
        while not stop["requested"]:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    if control.revision != seen and control.acc is not None:
        algorithm.save(control.acc, out_dir)          # the image as tuned
    print(f"[run] exiting. Tuned flags: {control.state()['flags']}", flush=True)


def check_batch_for_accumulate(args, acc) -> None:
    """`--accumulate latest` replaces each position's frame once per batch, so a batch must not
    hold the same position twice: refuse batches longer than the scan."""
    n_pos = acc.scan_shape[0] * acc.scan_shape[1]
    if args.accumulate == "latest" and args.batch_size > n_pos:
        raise SystemExit(f"--accumulate latest needs --batch-size <= the scan's {n_pos} positions "
                         f"(got {args.batch_size}).")


def resolve_ingest(args) -> str:
    """Which ingest drives this run: "holoscan" or "python".

    `--source` (a file) is only read by the Python ingest. `auto` takes Holoscan when its modules
    are importable (inside a container) and the device can be CUDA, else the Python ingest.
    `--transport rdma` needs the Holoscan 5 image.
    """
    if args.transport == "rdma":
        if args.source is not None or args.ingest == "python":
            raise SystemExit("--transport rdma is received by the Holoscan 5 ingest only; drop "
                             "--source / --ingest python.")
        if holoscan_generation() != 5:
            raise SystemExit("--transport rdma needs the Holoscan 5 image "
                             "(docker/Dockerfile.hsdk5).")
        return "holoscan"
    if args.source is not None:
        if args.ingest == "holoscan":
            raise SystemExit("--source (a capture file) is read by the Python ingest; "
                             "drop --ingest holoscan, or replay it over ZMQ.")
        return "python"
    if args.ingest != "auto":
        return args.ingest
    have_holoscan = holoscan_generation() is not None
    return "holoscan" if have_holoscan and args.device in ("auto", "cuda") else "python"


def build_parser(algorithm: Algorithm) -> argparse.ArgumentParser:
    """The full CLI: generic transport flags + the selected algorithm's own flags."""
    p = argparse.ArgumentParser(
        prog="python -m realtime_stem.apps.run",
        description="Real-time STEM reconstruction (Holoscan or pure-Python ingest).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--algorithm", default=registry.names()[0], choices=registry.names(),
                   help="Which reconstruction algorithm to run.")

    # ── transport (algorithm-agnostic) ────────────────────────────────────────
    g = p.add_argument_group("ingest (where frames come from)")
    g.add_argument("--ingest", choices=("auto", "holoscan", "python"), default="auto",
                   help="Frame ingest: the compiled Holoscan graph (NVIDIA GPU, full rate) or the "
                        "pure-Python one (CPU decode, any device). auto = Holoscan when available.")
    g.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                   help="Torch device for the reconstruction (auto: cuda, then mps, then cpu). "
                        "The Holoscan ingest needs cuda.")
    g.add_argument("--transport", choices=("zmq", "rdma"), default="zmq",
                   help="How frames arrive. zmq: the DCU's stream (port 31001), bslz4-compressed. "
                        "rdma: the DCU's imagemux RDMA (handshake port 31003), uncompressed (DCU "
                        "compression none), straight into GPU memory; needs the Holoscan 5 image "
                        "(docker/Dockerfile.hsdk5) and an RDMA network card.")
    g.add_argument("--endpoint", default="tcp://127.0.0.1:5555",
                   help="ZMQ PULL endpoint (DCU: tcp://<dcu-ip>:31001; replay demo: localhost).")
    g.add_argument("--n-connections", type=int, default=3,
                   help="TCP connections to the stream. The DCU paces each connection (about "
                        "60k beam-on binned frames/s each), so 120 kHz needs 3; larger "
                        "(unbinned) frames can need more.")
    g.add_argument("--source", type=Path, default=None, metavar="PATH",
                   help="Read a capture instead of a ZMQ stream: a .dstream recording or a .npy "
                        "array of frames, (..., H, W). Implies the Python ingest.")
    g.add_argument("--source-rate", type=float, default=0.0, metavar="HZ",
                   help="With --source: replay at this many frames/s (0 = as fast as possible), "
                        "so the live view fills in as it would during a scan.")
    g.add_argument("--rdma-remote", default="", metavar="HOST:PORT",
                   help="--transport rdma: the DCU's RDMA handshake address, e.g. 10.10.10.2:31003.")
    g.add_argument("--rdma-local", default="", metavar="IP",
                   help="--transport rdma: this machine's address on the RDMA (RoCE) network.")
    g.add_argument("--rdma-device", default="",
                   help="--transport rdma: the verbs device (e.g. mlx5_0); empty = found from "
                        "--rdma-local.")
    g.add_argument("--rdma-ring-memory", choices=("gpu", "host_mapped"), default="gpu",
                   help="--transport rdma: where the DCU writes the frames. gpu = GPUDirect RDMA "
                        "into GPU memory; host_mapped = pinned host memory the GPU reads.")
    g.add_argument("--rdma-ring-mib", type=int, default=64,
                   help="--transport rdma: size of the receive ring, in MiB.")
    g.add_argument("--rdma-timeout", type=float, default=10.0, metavar="S",
                   help="--transport rdma: seconds to wait for the DCU's handshake.")
    g.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16",
                   help="Pixel dtype of the stream (ARINA can send uint32 in long-dwell modes).")
    g.add_argument("--batch-size", type=int, default=2048,
                   help="Frames per ingest batch (any size works; the last, partial batch of a "
                        "scan is processed too). The compiled ingest caps this (kMaxBatchSize in "
                        "the operators' constants.h; larger values are clamped, with a warning "
                        "in the log). With --accumulate latest it must not exceed the scan "
                        "positions.")
    g.add_argument("--scheduler-threads", type=int, default=0,
                   help="Holoscan ingest: 0 = the single-threaded (greedy) scheduler, the fastest "
                        "here; N > 0 = MultiThreadScheduler with N workers, whose idle workers "
                        "sleep up to 5 ms between checks for work (5-7 ms per batch).")

    g = p.add_argument_group("acquisition (one scan, a session of scans, continuous scanning)")
    g.add_argument("--accumulate", choices=("scan", "latest", "all", "reset", "swap"),
                   default="scan",
                   help="What the reconstruction holds. scan: one scan per detector series, ids "
                        "must stay within the scan (a fresh reconstruction per series with "
                        "--continuous). For repeated scans of the same area (one long series, or "
                        "a series per pass with --continuous): latest: the newest frame at every "
                        "scan position (live tuning: the image follows the knobs within one "
                        "pass; stores positions x bright-field pixels). all: every frame of "
                        "every pass (averaging a settled condition). reset: a fresh image each "
                        "pass, so the live view is the current pass filling in. swap: the "
                        "current pass on the rows it has completed and the last completed pass "
                        "elsewhere, never partial (with --pass-average, a running average of "
                        "passes). reset and swap save the last completed pass and need no store.")
    g.add_argument("--pass-average", type=float, default=1.0, metavar="BETA",
                   help="With --accumulate swap: each completed pass is blended into the shown "
                        "one with this weight, front = (1 - BETA) front + BETA pass (1 = no "
                        "averaging; 0.3 averages over ~3 passes while following drift).")
    g.add_argument("--save-passes", action="store_true",
                   help="With repeated scans (--accumulate latest/all/reset/swap): save each "
                        "completed pass's channels to <out-dir>/passes/, one file per pass and "
                        "channel (a movie of the session), written off the hot path.")
    g.add_argument("--save-passes-dtype", choices=("float32", "float16"), default="float32",
                   help="Data type of the --save-passes files (float16 halves them).")
    g.add_argument("--continuous", action="store_true",
                   help="Don't exit after one stream: on the series 'end', save results and re-arm "
                        "for the next series (Ctrl-C / docker stop to end); with --accumulate scan "
                        "each series starts a fresh reconstruction. Every scan must use the SAME "
                        "geometry.")
    g.add_argument("--rounds", type=int, default=0, metavar="N",
                   help="With --continuous (implied if set): exit cleanly after N streams "
                        "(0 = keep going forever).")

    g = p.add_argument_group("live stream (snapshots for a viewer or notebook)")
    g.add_argument("--live-endpoint", default="",
                   help="If set (e.g. tcp://127.0.0.1:5601, or tcp://0.0.0.0:5601 for other "
                        "machines), PUB live snapshots of the reconstruction. Empty = none.")
    g.add_argument("--live-every", type=float, default=0.2, metavar="S",
                   help="Minimum seconds between live snapshots (0.2 = ~5 Hz).")
    g.add_argument("--live-every-rows", type=int, default=0, metavar="N",
                   help="Publish a live snapshot every N scan rows instead of every --live-every "
                        "seconds (0 = time-based).")
    g.add_argument("--viewer-port", type=int, default=0, metavar="PORT",
                   help="Serve the browser viewer from this process on PORT (e.g. 8050), with live "
                        "control of the reconstruction: the algorithm's parameters (parallax: "
                        "C10, rotation, C12) start from the command line's and can be fine-tuned "
                        "from the page, and the --accumulate mode switched. With --accumulate "
                        "latest a change re-renders the frames held at once. The viewer stays up "
                        "after the run until Ctrl-C. 0 = none (a separate viewer can still "
                        "subscribe to --live-endpoint).")
    g.add_argument("--viewer-host", default="127.0.0.1",
                   help="Address the viewer listens on (0.0.0.0 for other machines; or forward "
                        "the port over ssh).")

    g = p.add_argument_group("output")
    g.add_argument("--out-dir", type=Path, default=Path("."),
                   help="Where results land (written at each series' end and on exit).")

    # ── algorithm-specific (geometry, physics, tuning knobs) ──────────────────
    algorithm.add_cli_args(p)
    return p


def finish_scan(app, acc, algorithm: Algorithm, out_dir: Path, publisher, tag: str,
                final: bool = True) -> None:
    """Report one scan (frame count, completeness), save its results, land the viewer on them."""
    if app.sink is not None:
        print(app.sink.summary(), flush=True)
        print(app.sink.completeness()[1], flush=True)
    algorithm.save(acc, out_dir, tag=tag)
    if publisher is not None:
        publish_snapshot(publisher, acc, final=final)   # land the viewer on the final frame


def main() -> None:
    argv = sys.argv[1:]

    # Peek --algorithm so the right algorithm can add its flags to the real parser.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--algorithm", default=registry.names()[0], choices=registry.names())
    known, _ = peek.parse_known_args(argv)
    algorithm = registry.get(known.algorithm)

    parser = build_parser(algorithm)
    args = parser.parse_args(algorithm.preprocess_argv(argv))
    problems = algorithm.check_args(args)
    if args.pass_average != 1.0 and args.accumulate != "swap":
        problems.append("--pass-average needs --accumulate swap")
    if not 0.0 < args.pass_average <= 1.0:
        problems.append("--pass-average must be in (0, 1]")
    if args.save_passes and args.accumulate == "scan":
        problems.append("--save-passes needs repeated scans: --accumulate latest, all, reset or "
                        "swap (with --accumulate scan --continuous every scan is saved anyway)")
    if problems:
        parser.error("\n".join(problems))

    ingest = resolve_ingest(args)
    generation = holoscan_generation() if ingest == "holoscan" else None
    if args.transport == "rdma":
        problems = [f"--transport rdma needs {flag}" for flag, value in
                    (("--rdma-remote HOST:PORT", args.rdma_remote), ("--rdma-local IP", args.rdma_local))
                    if not value]
        if problems:
            parser.error("\n".join(problems))
    if ingest == "holoscan":
        if args.device not in ("auto", "cuda"):
            raise SystemExit(f"the Holoscan ingest hands frames over as CUDA tensors; --device "
                             f"{args.device} needs --ingest python.")
        args.device = "cuda"
    if generation == 5:
        if args.dtype != "uint16":
            raise SystemExit("the Holoscan 5 ingest receives uint16 frames only (fast scans); "
                             "for uint32 (long dwells) use the Holoscan 4 image.")
        # holoscan lives only in the container — import lazily so --help / tests need none of it.
        from realtime_stem.ingest.hsdk5_graph import MAX_BATCH, Hsdk5Session
        if args.batch_size > MAX_BATCH:
            print(f"[run] --batch-size {args.batch_size} > {MAX_BATCH}, the Holoscan 5 receivers' "
                  f"largest: using {MAX_BATCH}", flush=True)
            args.batch_size = MAX_BATCH
        print(f"[run] realtime-stem {__version__}, Holoscan 5 {args.transport.upper()} ingest",
              flush=True)
    elif ingest == "holoscan":
        from realtime_stem.ingest.holoscan_graph import OPERATORS_VERSION, IngestApp
        print(f"[run] realtime-stem {__version__}, operators {OPERATORS_VERSION or 'before 2.0'}",
              flush=True)
        if OPERATORS_VERSION is None:
            print("[run] WARNING: this image's operators predate 2.0: no end-of-scan drain, no "
                  "integrity check, Ctrl-C may not work while idle. Rebuild the image from "
                  "docker/Dockerfile.", flush=True)
    else:
        from realtime_stem.ingest.python_ingest import PythonIngest, ZmqStream
        print(f"[run] realtime-stem {__version__}, Python ingest", flush=True)
    if args.device in ("auto", "cuda"):
        for line in gpu_report():
            print(line, flush=True)
    if args.source is None and args.transport == "zmq":
        watch_endpoint(args.endpoint)

    publisher = None
    if args.live_endpoint:
        from realtime_stem.streaming.live_stream import LivePublisher   # lazy: only if live is on
        publisher = LivePublisher(args.live_endpoint)
        print(f"[run] live preview PUB on {args.live_endpoint} (every {args.live_every}s)",
              flush=True)

    saver = None
    if args.save_passes:
        from concurrent.futures import ThreadPoolExecutor
        saver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="save-passes")

    def hook(acc):
        """The runner's hooks on a new accumulator (also one a live control builds)."""
        if saver is not None:
            acc.on_pass_complete = (lambda k, acc=acc: algorithm.save_pass(
                acc, args.out_dir, k, args.save_passes_dtype, executor=saver))

    control = None
    telemetry = None
    if args.viewer_port:
        from realtime_stem.apps.control import LiveControl
        from realtime_stem.streaming.live_stream import Fanout
        from realtime_stem.streaming.web.server import FrameStore, StorePublisher, serve
        store = FrameStore()
        publisher = Fanout(publisher, StorePublisher(store))
        from realtime_stem.streaming.telemetry import Telemetry
        telemetry = Telemetry(args.endpoint if args.transport == "zmq"
                              else f"tcp://{args.rdma_remote}")
        publisher.telemetry = telemetry
        telemetry.start()
        control = LiveControl(algorithm, args, publisher=publisher, on_new_accumulator=hook)
        viewer = serve(store, args.viewer_host, args.viewer_port, endpoint="run.py, in process",
                       control=control, telemetry=telemetry)
        host = "127.0.0.1" if args.viewer_host in ("", "0.0.0.0") else args.viewer_host
        print(f"[run] viewer on http://{host}:{viewer.server_address[1]}/ with live control of "
              f"the reconstruction", flush=True)

    # Ctrl-C (SIGINT) and `docker stop` (SIGTERM): Holoscan stops the running graph gracefully by
    # itself, and app.run() then returns as if the scan had ended. This flag tells the loop that
    # the run was stopped. (A KeyboardInterrupt is not enough: with the single-threaded scheduler
    # it is raised inside an operator and swallowed there, and SIGTERM raises nothing.) The
    # Python ingest stops on a KeyboardInterrupt, so there SIGTERM is turned into one.
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    if ingest == "holoscan":
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, request_stop)
    else:
        signal.signal(signal.SIGTERM, interrupt)

    # One iteration = one stream (the ingest graph shuts down at the series 'end'). Continuous
    # mode re-arms with a FRESH accumulator and a fresh app: Holoscan Applications are one-shot,
    # but building both takes well under a second — far shorter than the detector re-arm.
    continuous = args.continuous or args.rounds > 0
    # latest / all: one reconstruction across repeated scans and series (live tuning, averaging)
    persistent = args.accumulate != "scan"
    if args.source is not None and continuous:
        print("[run] --source replays the file once; ignoring --continuous/--rounds", flush=True)
        continuous = False
    # The Python ingest's ZMQ sockets outlive each round, so a message of the next scan that
    # arrives while the previous one drains is kept rather than lost.
    stream = None
    if ingest == "python" and args.source is None:
        stream = ZmqStream(args.endpoint, args.n_connections)
    # Holoscan 5: one connection to the DCU and one graph for the whole run; each scan is a round.
    session = None
    rounds_done = 0
    acc = None
    stopped = False
    try:
        while True:
            rounds_done += 1
            if control is not None and control.acc is not None:
                acc = control.acc             # the live control may have built a new one
            if acc is None or not persistent:
                acc = algorithm.build(args)   # builds geometry + accumulator, prints a summary
                hook(acc)
            elif hasattr(acc, "start_series"):
                acc.start_series()            # ids of the new series continue the pass count
            if rounds_done == 1:
                check_batch_for_accumulate(args, acc)
                origin = args.source if args.source is not None else f"listening on {args.endpoint}"
                if args.transport == "rdma":
                    origin = (f"RDMA from {args.rdma_remote} to {args.rdma_local} "
                              f"({args.rdma_ring_memory} ring)")
                line = (f"[run] {ingest} ingest, {origin}, scan {acc.scan_shape}, "
                        f"detector {acc.detector_shape} {args.dtype}, batch {args.batch_size}")
                if args.transport == "zmq":
                    line += f", connections {args.n_connections}"
                if generation == 4:
                    line += ", scheduler " + ("single-threaded" if args.scheduler_threads <= 0
                                              else f"{args.scheduler_threads} threads")
                print(line, flush=True)
            live = dict(live_publisher=publisher, live_every=args.live_every,
                        live_every_rows=args.live_every_rows)
            if generation == 5:
                if session is None:
                    session = Hsdk5Session(
                        transport=args.transport, detector_shape=acc.detector_shape,
                        batch_size=args.batch_size, endpoint=args.endpoint,
                        n_connections=args.n_connections, remote=args.rdma_remote,
                        local=args.rdma_local, rdma_device=args.rdma_device,
                        ring_memory=args.rdma_ring_memory, ring_mib=args.rdma_ring_mib,
                        timeout=args.rdma_timeout)
                    try:
                        session.open()
                    except ConnectionError as exc:
                        raise SystemExit(f"[run] {exc}") from None
                app = session.round(acc, should_stop=lambda: stop["requested"], **live)
            elif ingest == "holoscan":
                app = IngestApp(args.endpoint, args.batch_size, args.scheduler_threads, acc,
                                dtype=args.dtype, n_connections=args.n_connections, **live)
            else:
                app = PythonIngest(acc, stream=stream, source=args.source,
                                   batch_size=args.batch_size, dtype=args.dtype,
                                   source_rate=args.source_rate, **live)
            if control is not None:
                control.attach(acc, lambda app=app: batch_sink(app))
            if continuous:
                total = f"/{args.rounds}" if args.rounds else ""
                print(f"[run] ── round {rounds_done}{total}: armed, waiting for a scan …",
                      flush=True)
            # In continuous mode every scan also gets its own numbered result file.
            tag = f"scan{rounds_done:03d}" if continuous else ""
            if stop["requested"]:      # stopped between two scans: the last one is saved already
                print("[run] stopped.", flush=True)
                stopped = True
                break
            try:
                app.run()          # returns after the stream's `end` message, or on a stop signal
            except KeyboardInterrupt:
                acc = control.acc if control is not None else acc
                print("[run] interrupted — saving partial results.", flush=True)
                finish_scan(app, acc, algorithm, args.out_dir, publisher, tag)
                stopped = True
                break
            except Exception:
                acc = control.acc if control is not None else acc
                # e.g. a frame id beyond --scan: keep what was reconstructed, then fail loudly.
                print("[run] the pipeline stopped with an error — saving partial results.",
                      flush=True)
                finish_scan(app, acc, algorithm, args.out_dir, publisher, tag)
                raise
            if control is not None:
                acc = control.acc             # the mode may have changed during the scan
            if stop["requested"]:
                print("[run] interrupted — saving partial results.", flush=True)
                finish_scan(app, acc, algorithm, args.out_dir, publisher, tag)
                stopped = True
                break
            if hasattr(acc, "end_series"):
                acc.end_series()              # a pass that ended with the series is complete
            last = not continuous or (args.rounds and rounds_done >= args.rounds)
            # a persistent reconstruction is only final when the run ends (the next series keeps
            # updating the same image)
            finish_scan(app, acc, algorithm, args.out_dir, publisher, tag,
                        final=bool(last or not persistent))
            if last:
                break
        if control is not None and not stopped:
            wait_with_viewer(control, stop, algorithm, args.out_dir)
    finally:
        if telemetry is not None:
            telemetry.close()
        if session is not None:
            session.close()
        if saver is not None:
            saver.shutdown(wait=True)         # every completed pass is on disk before exiting
        if publisher is not None:
            publisher.close()
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
