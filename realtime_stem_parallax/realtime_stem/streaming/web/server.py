"""Browser viewer for the live stream: a small HTTP server that subscribes to a `run.py
--live-endpoint` PUB and serves a WebGL page (`static/index.html`) showing the newest snapshot.

    python -m realtime_stem.streaming.web tcp://127.0.0.1:5601        # then open http://127.0.0.1:8050
    ssh -L 8050:127.0.0.1:8050 <server>                              # to watch from another computer

Only what is on screen crosses the wire. The page asks for the region it shows at the size it
shows it (`/view`): the server averages blocks of pixels down to about the screen's resolution,
and sends native pixels once zoomed in. So a 4096² snapshot costs the browser about what a 512²
one does. Contrast, gamma and the colour map are applied on the GPU in the browser, so the
sliders don't wait for the server. Python's standard library, numpy and pyzmq only. The page loads
nothing from the internet, since microscope computers are often offline.

Endpoints (all GET):
    /                the page;  /static/<file>  its scripts
    /events          Server-Sent Events: the state (below) of each new snapshot
    /state           the state of the newest snapshot (or ?seq=), as JSON
    /view            ?seq&ch&y0&x0&h&w&oh&ow[&hp&lp&order]: a region, block-averaged by the
                     largest integer that keeps it at least oh × ow, optionally high/low-passed
                     (Butterworth of `order`, 12 by default, cutoff periods in full-resolution
                     pixels), as float16 normalized by X-Offset/X-Scale
    /fft             ?seq&ch&y0&x0&h&w[&n&out&hp&lp&order]: log10 power spectrum of the central n × n
                     (≤ 2048) native pixels of a region, Hann-windowed, optionally filtered like
                     /view, averaged in X-Bin × X-Bin blocks to ≤ out (1024) a side, float16 like
                     /view; of a partly scanned image, only its finished rows (X-Rows: those used).
                     The page applies the filters itself (|H(q)|² on the spectrum it has)
    /npy             ?seq&ch: one channel at full resolution, as .npy
    /params          GET the live control's state; POST {name: value, …, "mode": …} or
                     {"reset": true} to change the running reconstruction (run.py --viewer-port)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

STATIC = Path(__file__).resolve().parent / "static"
_PCT = (0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 95.0, 98.0, 99.0, 99.5, 99.9, 99.95)


# ── the snapshots ──────────────────────────────────────────────────────────────

def channel_stats(img: np.ndarray, samples: int = 1 << 20, bins: int = 256) -> dict:
    """Percentiles and a histogram of one channel, from a strided sample of at most `samples`
    pixels (enough for display limits, cheap at 4096²)."""
    step = max(1, int(math.sqrt(img.size / samples)))
    s = img[::step, ::step]
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "pct": {}, "hist": [],
                "hist_range": [0.0, 1.0]}
    pct = np.percentile(s, _PCT)
    lo, hi = float(pct[0]), float(pct[-1])
    if not hi > lo:
        lo, hi = lo - 0.5, lo + 0.5
    hist, _ = np.histogram(s, bins=bins, range=(lo, hi))
    return {"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()),
            "std": float(s.std()), "pct": {f"{p:g}": float(v) for p, v in zip(_PCT, pct)},
            "hist": hist.tolist(), "hist_range": [lo, hi]}


@dataclass
class Frame:
    seq: int
    stack: np.ndarray          # (C, H, W) float32
    names: list
    header: dict
    stats: list
    received: float

    @property
    def nbytes(self) -> int:
        return self.stack.nbytes


class FrameStore:
    """The newest snapshots, up to `history_mb` (the newest is always kept), for the page and for
    scrubbing back. `wait` blocks until a snapshot newer than a given seq arrives."""

    def __init__(self, history_mb: float = 512.0):
        self.budget = int(history_mb * 2**20)
        self._frames: deque[Frame] = deque()
        self._cond = threading.Condition()
        self._next = 0
        self.received = 0

    def put(self, arr: np.ndarray, header: dict) -> Frame:
        names = header.get("channels")
        stack = np.asarray(arr, dtype=np.float32)
        if not names:
            stack = stack[None] if stack.ndim == 2 else stack
            names = [f"channel {i}" for i in range(stack.shape[0])] if stack.shape[0] > 1 \
                else ["reconstruction"]
        rows = (header.get("meta") or {}).get("complete_rows")     # a partly scanned image:
        part = stack[:, rows[0]:rows[1]] if rows and rows[1] > rows[0] else stack   # its finished rows
        frame = Frame(seq=self._next, stack=stack, names=list(names), header=header,
                      stats=[channel_stats(c) for c in part], received=time.time())
        with self._cond:
            self._frames.append(frame)
            self._next += 1
            self.received += 1
            total = sum(f.nbytes for f in self._frames)
            while len(self._frames) > 1 and total > self.budget:
                total -= self._frames.popleft().nbytes
            self._cond.notify_all()
        return frame

    def get(self, seq: int | None = None) -> Frame | None:
        with self._cond:
            if not self._frames:
                return None
            if seq is None:
                return self._frames[-1]
            for f in self._frames:
                if f.seq == seq:
                    return f
            return None

    def seqs(self) -> list[int]:
        with self._cond:
            return [f.seq for f in self._frames]

    def wait(self, after: int, timeout: float) -> Frame | None:
        """The newest snapshot if it is newer than `after`, waiting up to `timeout` s for one."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self._frames and self._frames[-1].seq > after,
                                     timeout)
            return self._frames[-1] if ok else None

    def state(self, frame: Frame) -> dict:
        h = frame.header
        return {"seq": frame.seq, "names": frame.names, "shape": list(frame.stack.shape[1:]),
                "meta": h.get("meta") or {}, "final": bool(h.get("final")),
                "publisher_seq": h.get("seq"), "stats": frame.stats, "received": frame.received,
                "history": self.seqs(), "count": self.received}


# ── the images the page asks for ───────────────────────────────────────────────

ORDER = 12                                   # the Butterworth order of the display filters


def transfer(q: np.ndarray, hp: float = 0.0, lp: float = 0.0, order: int = ORDER) -> np.ndarray:
    """H(q) of the display filters, the Butterworth high- and low-pass of quantem and of the
    pipeline's own readout (`--lowpass` / `--highpass`): 1/(1 + (q/q_c)^2n) and its complement
    1 − 1/(1 + (q/q_c)^2n), with cutoff periods `hp` / `lp` = 1/q_c (in the units of 1/q, e.g.
    px; 0: off), so H = ½ at the cutoff."""
    h = np.ones_like(q, dtype=np.float64)
    if lp > 0:
        h /= 1.0 + (q * lp) ** (2 * order)
    if hp > 0:
        r = (q * hp) ** (2 * order)
        h *= r / (1.0 + r)
    return h


def _filter(img: np.ndarray, hp: float, lp: float, order: int) -> np.ndarray:
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.rfftfreq(img.shape[1])[None, :]
    h = transfer(np.sqrt(fy**2 + fx**2), hp, lp, order)
    return np.fft.irfft2(np.fft.rfft2(img) * h, s=img.shape).astype(np.float32)


def view(img: np.ndarray, y0: int, x0: int, h: int, w: int, oh: int, ow: int,
         hp: float = 0.0, lp: float = 0.0, order: int = ORDER
         ) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """A region of `img`, block-averaged by the largest integer f that keeps it at least
    oh × ow (f = 1, native pixels, when zoomed in), optionally high-/low-passed (`transfer`:
    Butterworth of `order`, cutoff periods `hp`/`lp` in full-resolution pixels). The filters run
    on the averaged region plus a margin of 3 cutoff periods (so they cost what the screen does,
    not what the snapshot does).
    Returns (data, (y0, x0, h, w) actually covered, f)."""
    H, W = img.shape
    y0, x0 = max(0, min(int(y0), H - 1)), max(0, min(int(x0), W - 1))
    h, w = max(1, min(int(h), H - y0)), max(1, min(int(w), W - x0))
    f = max(1, min(h // max(int(oh), 1), w // max(int(ow), 1)))
    h, w = (h // f) * f, (w // f) * f
    m = int(math.ceil(3 * max(hp, lp) / f)) * f if (hp or lp) else 0
    ya, yb = max(0, y0 - m), min(H, y0 + h + m)
    xa, xb = max(0, x0 - m), min(W, x0 + w + m)
    # widen to whole blocks around the region (the margin may be clipped by the image edge)
    ya, xa = y0 - ((y0 - ya) // f) * f, x0 - ((x0 - xa) // f) * f
    yb, xb = y0 + h + ((yb - y0 - h) // f) * f, x0 + w + ((xb - x0 - w) // f) * f
    sub = img[ya:yb, xa:xb].astype(np.float32, copy=False)
    if f > 1:
        sub = sub.reshape(sub.shape[0] // f, f, sub.shape[1] // f, f).mean(axis=(1, 3))
    if hp > 0 or lp > 0:
        sub = _filter(sub, hp / f, lp / f, order)
    oy, ox = (y0 - ya) // f, (x0 - xa) // f
    out = sub[oy:oy + h // f, ox:ox + w // f]
    return np.ascontiguousarray(out, dtype=np.float32), (y0, x0, h, w), f


def power_spectrum(img: np.ndarray, y0: int, x0: int, h: int, w: int, n: int = 2048,
                   rows: tuple[int, int] | None = None, hp: float = 0.0, lp: float = 0.0,
                   out: int = 1024, order: int = ORDER
                   ) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int], int]:
    """log10 power spectrum (fftshifted, Hann-windowed) of the central n × n native pixels of a
    region (n even, at most the region and 4096), high-/low-passed like /view (`transfer`, cutoff
    periods `hp` / `lp` in full-resolution pixels, H² applied to the spectrum), then averaged in
    b × b blocks to at most `out` pixels a side: the whole q range at about the screen's
    resolution.

    `rows` [r0, r1): the rows that hold a finished image when it is only partly scanned (the
    snapshot meta's `complete_rows`). The square then moves up or down within the region onto
    as many of them as it can, and the window spans only those rows, so the blank part and the
    edge of the scan leave no streak along q_y. Fewer than 8 of them in view: a flat spectrum.
    Returns (spectrum, (y0, x0, n) of the square, (r0, r1) the rows windowed, b). Spectrum pixel
    (i, j) covers frequencies (i·b … i·b + b − 1 − n/2) / n cycles per pixel (and likewise j)."""
    H, W = img.shape
    y0, x0 = max(0, min(int(y0), H - 1)), max(0, min(int(x0), W - 1))
    h, w = max(2, min(int(h), H - y0)), max(2, min(int(w), W - x0))
    n = max(2, (min(int(n), h, w, 4096) // 2) * 2)
    b = max(1, -(-n // max(int(out), 1)))
    n = max(2 * b, (n // (2 * b)) * (2 * b))          # whole blocks, an even number of them
    cy, cx = y0 + (h - n) // 2, x0 + (w - n) // 2
    ra, rb = cy, cy + n
    if rows is not None:
        ra, rb = max(int(rows[0]), y0), min(int(rows[1]), y0 + h)
        if rb - ra >= n:
            cy = ra + (rb - ra - n) // 2
        else:                         # centred on them, within the region (so it contains them)
            cy = min(max(y0, (ra + rb) // 2 - n // 2), y0 + h - n)
        ra, rb = max(ra, cy), min(rb, cy + n)
    if rb - ra < 8:
        return np.zeros((n // b, n // b), np.float32), (cy, cx, n), (ra, max(ra, rb)), b
    sub = img[ra:rb, cx:cx + n].astype(np.float32)
    tile = np.zeros((n, n), np.float32)
    win = np.outer(np.hanning(rb - ra), np.hanning(n)).astype(np.float32)
    tile[ra - cy:rb - cy] = (sub - sub.mean()) * win
    half = np.fft.rfft2(tile)                          # the image is real: P(−q) = P(q)
    ph = half.real ** 2 + half.imag ** 2               # (n, n/2 + 1), q_x ≥ 0
    p = np.empty((n, n), np.float32)
    p[:, :n // 2 + 1] = ph
    p[:, n // 2 + 1:] = ph[(-np.arange(n)) % n][:, n // 2 - 1:0:-1]
    if hp > 0 or lp > 0:
        q = np.sqrt(np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2)
        p *= (transfer(q, hp, lp, order) ** 2).astype(np.float32)
    p = np.fft.fftshift(p)
    if b > 1:
        p = p.reshape(n // b, b, n // b, b).mean(axis=(1, 3))
    floor = p.mean() * 1e-6 + 1e-30
    return np.log10(p + floor).astype(np.float32), (cy, cx, n), (ra, rb), b


def encode(data: np.ndarray) -> tuple[bytes, float, float]:
    """float16 bytes of (data − offset) / scale, so any range keeps float16's 11 bits of
    precision around the image's own level. Returns (bytes, offset, scale)."""
    finite = data[np.isfinite(data)]
    off = float(finite.mean()) if finite.size else 0.0
    scale = float(finite.std()) if finite.size else 1.0
    scale = scale if scale > 0 else 1.0
    z = np.clip((data - off) / scale, -6.0e4, 6.0e4)
    return z.astype("<f2").tobytes(), off, scale


# ── HTTP ───────────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    store: FrameStore = None          # set by make_server
    endpoint: str = ""
    control = None                    # a live control (apps/control.py) when run.py serves the page
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):     # quiet: the page polls /view a lot
        pass

    # helpers
    def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _error(self, code: int, msg: str):
        self._json({"error": msg}, code)

    def _frame_channel(self, q):
        frame = self.store.get(int(q["seq"][0]) if "seq" in q else None)
        if frame is None:
            return None, None, "no snapshot (yet, or no longer in the history)"
        ch = q.get("ch", [frame.names[0]])[0]
        if ch not in frame.names:
            return None, None, f"unknown channel {ch!r}"
        return frame, frame.stack[frame.names.index(ch)], None

    @staticmethod
    def _ints(q, *names, default=None):
        return [int(float(q[n][0])) if n in q else default for n in names]

    # routes
    def do_GET(self):
        url = urlsplit(self.path)
        q = parse_qs(url.query)
        try:
            route = getattr(self, "_get_" + (url.path.strip("/").split("/")[0] or "index"), None)
            if route is None:
                return self._error(404, f"no such path {url.path}")
            route(url, q)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (KeyError, ValueError) as exc:
            self._error(400, f"{type(exc).__name__}: {exc}")

    def _get_index(self, url, q):
        self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")

    def _get_static(self, url, q):
        name = url.path.split("/", 2)[-1]
        f = (STATIC / name).resolve()
        if STATIC not in f.parents or not f.is_file():
            return self._error(404, f"no such file {name}")
        ctype = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}.get(
            f.suffix, "application/octet-stream")
        self._send(200, f.read_bytes(), ctype + "; charset=utf-8")

    def _get_state(self, url, q):
        frame = self.store.get(int(q["seq"][0]) if "seq" in q else None)
        if frame is None:
            return self._json({"seq": -1, "endpoint": self.endpoint, "history": [],
                               "control": self.control is not None})
        self._json({**self.store.state(frame), "endpoint": self.endpoint,
                    "control": self.control is not None})

    def _get_params(self, url, q):
        if self.control is None:
            return self._error(404, "no live control: run.py --viewer-port serves one")
        self._json(self.control.state())

    def do_POST(self):
        url = urlsplit(self.path)
        try:
            if url.path.rstrip("/") != "/params" or self.control is None:
                return self._error(404, f"no such path {url.path}")
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self._json(self.control.update(json.loads(body or b"{}")))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (KeyError, ValueError, TypeError) as exc:
            self._error(400, f"{type(exc).__name__}: {exc}")

    def _get_events(self, url, q):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last = int(q.get("after", ["-1"])[0])
        while True:
            frame = self.store.wait(last, timeout=10.0)
            if frame is None:
                self.wfile.write(b": keep-alive\n\n")
            else:
                last = frame.seq
                body = json.dumps({**self.store.state(frame), "endpoint": self.endpoint,
                                   "control": self.control is not None})
                self.wfile.write(b"data: " + body.encode() + b"\n\n")
            self.wfile.flush()

    def _get_view(self, url, q):
        frame, img, err = self._frame_channel(q)
        if err:
            return self._error(404, err)
        y0, x0, h, w, oh, ow = self._ints(q, "y0", "x0", "h", "w", "oh", "ow", default=0)
        H, W = img.shape
        h, w = h or H, w or W
        hp, lp = (float(q[n][0]) if n in q else 0.0 for n in ("hp", "lp"))
        order = int(q["order"][0]) if "order" in q else ORDER
        data, (ry, rx, rh, rw), f = view(img, y0, x0, h, w, oh or h, ow or w, hp, lp, order)
        body, off, scale = encode(data)
        self._send(200, body, "application/octet-stream", {
            "X-Seq": frame.seq, "X-Shape": f"{data.shape[0]},{data.shape[1]}",
            "X-Region": f"{ry},{rx},{rh},{rw}", "X-Factor": f, "X-Offset": repr(off),
            "X-Scale": repr(scale), "Access-Control-Expose-Headers": "*"})

    def _get_fft(self, url, q):
        frame, img, err = self._frame_channel(q)
        if err:
            return self._error(404, err)
        H, W = img.shape
        y0, x0, h, w, n, out = self._ints(q, "y0", "x0", "h", "w", "n", "out", default=0)
        hp, lp = (float(q[k][0]) if k in q else 0.0 for k in ("hp", "lp"))
        order = int(q["order"][0]) if "order" in q else ORDER
        rows = (frame.header.get("meta") or {}).get("complete_rows")
        spec, (cy, cx, n), (ra, rb), b = power_spectrum(img, y0, x0, h or H, w or W, n or 2048,
                                                        rows, hp, lp, out or 1024, order)
        body, off, scale = encode(spec)
        self._send(200, body, "application/octet-stream", {
            "X-Seq": frame.seq, "X-Shape": f"{spec.shape[0]},{spec.shape[1]}",
            "X-Region": f"{cy},{cx},{n},{n}", "X-Bin": b, "X-Rows": f"{ra},{rb}",
            "X-Offset": repr(off), "X-Scale": repr(scale)})

    def _get_npy(self, url, q):
        frame, img, err = self._frame_channel(q)
        if err:
            return self._error(404, err)
        buf = io.BytesIO()
        np.save(buf, img)
        ch = q.get("ch", [frame.names[0]])[0].replace(" ", "_")
        self._send(200, buf.getvalue(), "application/octet-stream",
                   {"Content-Disposition": f'attachment; filename="snapshot{frame.seq:05d}_{ch}.npy"'})


def make_server(store: FrameStore, host: str = "127.0.0.1", port: int = 8050,
                endpoint: str = "", control=None) -> ThreadingHTTPServer:
    """The viewer's HTTP server over `store`; with a `control` (state() → dict, update(dict) →
    dict) the page also offers the reconstruction's live parameters (/params)."""
    handler = type("Handler", (_Handler,), {"store": store, "endpoint": endpoint,
                                            "control": control})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


class StorePublisher:
    """A publisher (`LivePublisher`'s interface) that puts snapshots straight into a FrameStore:
    the viewer served from inside run.py (--viewer-port) needs no socket."""

    def __init__(self, store: FrameStore):
        self.store, self._seq = store, 0

    def publish(self, arr, *, final: bool = False, channels=None, meta=None) -> int:
        self.store.put(np.asarray(arr), {"channels": channels, "final": final, "meta": meta,
                                         "seq": self._seq})
        self._seq += 1
        return self._seq - 1

    def close(self) -> None:
        pass


def serve(store: FrameStore, host: str, port: int, endpoint: str = "", control=None):
    """Start the viewer's server on a daemon thread; returns the server (its address is
    server.server_address)."""
    server = make_server(store, host, port, endpoint=endpoint, control=control)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True,
                     name="viewer").start()
    return server


def subscribe(endpoint: str, store: FrameStore, stop: threading.Event) -> threading.Thread:
    """Feed `store` from the live stream (newest frame only) until `stop` is set."""
    from realtime_stem.streaming.live_stream import LiveSubscriber

    def run():
        sub = LiveSubscriber(endpoint)
        try:
            while not stop.is_set():
                msg = sub.recv_latest(timeout_ms=200)
                if msg is not None:
                    store.put(*msg)
        finally:
            sub.close()

    thread = threading.Thread(target=run, name="live-subscriber", daemon=True)
    thread.start()
    return thread


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (also rendered into docs/CLI.md)."""
    p = argparse.ArgumentParser(
        prog="python -m realtime_stem.streaming.web",
        description="Browser viewer for the live stream: serves a page on localhost that shows "
                    "the newest snapshot with zoom, contrast, colour maps, filters, a Fourier "
                    "transform and a history.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("endpoint", help="tcp://host:port of the run.py --live-endpoint PUB.")
    p.add_argument("--host", default="127.0.0.1",
                   help="Address to serve the page on (0.0.0.0 for other computers; or keep "
                        "127.0.0.1 and use ssh -L).")
    p.add_argument("--port", type=int, default=8050, help="Port to serve the page on.")
    p.add_argument("--history-mb", type=float, default=512.0, metavar="MB",
                   help="Memory for past snapshots to scrub back through (the newest is always "
                        "kept; 4096² × 2 channels is 128 MB each).")
    p.add_argument("--open", action="store_true", help="Open the page in the default browser.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store = FrameStore(args.history_mb)
    stop = threading.Event()
    subscribe(args.endpoint, store, stop)
    server = make_server(store, args.host, args.port, args.endpoint)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{server.server_port}/"
    print(f"[web] subscribed to {args.endpoint}; open {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        print("[web] stopped", flush=True)
