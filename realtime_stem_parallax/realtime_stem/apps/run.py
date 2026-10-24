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

_HOLOSCAN_MODULES = ("holoscan", "dectris_holoscan_ops")


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
    are importable (inside the container) and the device can be CUDA, else the Python ingest.
    """
    if args.source is not None:
        if args.ingest == "holoscan":
            raise SystemExit("--source (a capture file) is read by the Python ingest; "
                             "drop --ingest holoscan, or replay it over ZMQ.")
        return "python"
    if args.ingest != "auto":
        return args.ingest
    have_holoscan = all(importlib.util.find_spec(m) is not None for m in _HOLOSCAN_MODULES)
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
    g.add_argument("--accumulate", choices=("scan", "latest", "all"), default="scan",
                   help="What the reconstruction holds. scan: one scan per detector series, ids "
                        "must stay within the scan (a fresh reconstruction per series with "
                        "--continuous). latest: the newest frame at every scan position, kept "
                        "across repeated scans and series (live tuning while the microscope scans "
                        "continuously: the image follows the knobs within one scan pass). all: "
                        "every frame of every repeat (averaging a settled condition).")
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
    if problems:
        parser.error("\n".join(problems))

    ingest = resolve_ingest(args)
    if ingest == "holoscan":
        if args.device not in ("auto", "cuda"):
            raise SystemExit(f"the Holoscan ingest hands frames over as CUDA tensors; --device "
                             f"{args.device} needs --ingest python.")
        args.device = "cuda"
        # holoscan lives only in the container — import lazily so --help / tests need none of it.
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
    if args.source is None:
        watch_endpoint(args.endpoint)

    publisher = None
    if args.live_endpoint:
        from realtime_stem.streaming.live_stream import LivePublisher   # lazy: only if live is on
        publisher = LivePublisher(args.live_endpoint)
        print(f"[run] live preview PUB on {args.live_endpoint} (every {args.live_every}s)",
              flush=True)

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
    rounds_done = 0
    acc = None
    try:
        while True:
            rounds_done += 1
            if acc is None or not persistent:
                acc = algorithm.build(args)   # builds geometry + accumulator, prints a summary
            elif hasattr(acc, "start_series"):
                acc.start_series()            # ids of the new series continue the pass count
            if rounds_done == 1:
                check_batch_for_accumulate(args, acc)
                origin = args.source if args.source is not None else f"listening on {args.endpoint}"
                line = (f"[run] {ingest} ingest, {origin}, scan {acc.scan_shape}, "
                        f"detector {acc.detector_shape} {args.dtype}, batch {args.batch_size}, "
                        f"connections {args.n_connections}")
                if ingest == "holoscan":
                    line += ", scheduler " + ("single-threaded" if args.scheduler_threads <= 0
                                              else f"{args.scheduler_threads} threads")
                print(line, flush=True)
            live = dict(live_publisher=publisher, live_every=args.live_every,
                        live_every_rows=args.live_every_rows)
            if ingest == "holoscan":
                app = IngestApp(args.endpoint, args.batch_size, args.scheduler_threads, acc,
                                dtype=args.dtype, n_connections=args.n_connections, **live)
            else:
                app = PythonIngest(acc, stream=stream, source=args.source,
                                   batch_size=args.batch_size, dtype=args.dtype,
                                   source_rate=args.source_rate, **live)
            if continuous:
                total = f"/{args.rounds}" if args.rounds else ""
                print(f"[run] ── round {rounds_done}{total}: armed, waiting for a scan …",
                      flush=True)
            # In continuous mode every scan also gets its own numbered result file.
            tag = f"scan{rounds_done:03d}" if continuous else ""
            if stop["requested"]:      # stopped between two scans: the last one is saved already
                print("[run] stopped.", flush=True)
                break
            try:
                app.run()          # returns after the stream's `end` message, or on a stop signal
            except KeyboardInterrupt:
                print("[run] interrupted — saving partial results.", flush=True)
                finish_scan(app, acc, algorithm, args.out_dir, publisher, tag)
                break
            except Exception:
                # e.g. a frame id beyond --scan: keep what was reconstructed, then fail loudly.
                print("[run] the pipeline stopped with an error — saving partial results.",
                      flush=True)
                finish_scan(app, acc, algorithm, args.out_dir, publisher, tag)
                raise
            if stop["requested"]:
                print("[run] interrupted — saving partial results.", flush=True)
                finish_scan(app, acc, algorithm, args.out_dir, publisher, tag)
                break
            last = not continuous or (args.rounds and rounds_done >= args.rounds)
            # a persistent reconstruction is only final when the run ends (the next series keeps
            # updating the same image)
            finish_scan(app, acc, algorithm, args.out_dir, publisher, tag,
                        final=bool(last or not persistent))
            if last:
                break
    finally:
        if publisher is not None:
            publisher.close()
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
