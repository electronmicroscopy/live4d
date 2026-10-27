"""Bounded, non-synchronizing sink observations and once-per-second host sampling.

Host batch duration measures validation and CPU submission, including existing waits.
It is deliberately NOT reported as GPU execution or end-to-end latency.
"""
from __future__ import annotations

from collections import deque
import ipaddress
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from urllib.parse import urlsplit


def percentile(values, fraction):
    if not values:
        return None
    a = sorted(values)
    x = (len(a) - 1) * fraction
    lo = int(x)
    return a[lo] + (a[min(lo + 1, len(a) - 1)] - a[lo]) * (x - lo)


class Telemetry:
    def __init__(self, endpoint="", target_fps=120000):
        self.endpoint, self.target_fps = endpoint, target_fps
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._history = deque(maxlen=60)
        self._resources = {}
        self._previous = None
        self._samples = deque(maxlen=2048)
        self._scan = 0
        self._frames = self._batches = self._expected = 0
        self._first = self._last = None
        self._last_entry = None
        self._first_frames = 0
        self._status = "waiting"
        self._integrity = None
        self._previous_scan = None

    def begin(self, expected):
        with self._lock:
            if self._frames:
                elapsed = self._last - self._first
                self._previous_scan = dict(scan=self._scan, frames=self._frames,
                    expected=self._expected, status=self._status, integrity=self._integrity,
                    average_fps=(self._frames-self._first_frames)/elapsed if elapsed > 0 else None)
            self._scan += 1
            self._expected = int(expected)
            self._frames = self._batches = self._first_frames = 0
            self._first = self._last = None
            self._last_entry = None
            self._samples.clear()
            self._integrity = None
            self._status = "waiting"

    def record(self, frames, entered, finished):
        with self._lock:
            interval = None if self._last_entry is None else entered - self._last_entry
            self._last_entry = entered
            if self._first is None:
                self._first, self._first_frames = finished, frames
            self._last = finished
            self._frames += frames
            self._batches += 1
            self._status = "receiving"
            self._samples.append((finished, frames, (finished - entered) * 1000,
                                  None if interval is None else interval * 1000))

    def finish(self, covered, repeated=False):
        with self._lock:
            missing = max(0, self._expected - covered)
            duplicates = None if repeated else max(0, self._frames - covered)
            self._integrity = dict(covered=covered, missing=missing, duplicates=duplicates)
            self._status = "complete" if missing == 0 and not duplicates else "incomplete"

    def snapshot(self, now=None):
        now = time.perf_counter() if now is None else now
        with self._lock:
            samples = list(self._samples)
            # Endpoint-difference estimator: exclude the first batch in each window.
            recent = [s for s in samples if now - s[0] <= 1]
            rate = 0.0
            if len(recent) > 1 and recent[-1][0] > recent[0][0]:
                rate = sum(s[1] for s in recent[1:]) / (recent[-1][0] - recent[0][0])
            elapsed = 0 if self._first is None else self._last - self._first
            average = ((self._frames - self._first_frames) / elapsed) if elapsed > 0 else None
            durations = [s[2] for s in samples]
            intervals = [s[3] for s in samples if s[3] is not None]
            return dict(available=True, endpoint=self.endpoint, target_fps=self.target_fps,
                        scan=self._scan, status=self._status, frames=self._frames,
                        expected=self._expected, batches=self._batches, elapsed_s=elapsed,
                        fps=rate, average_fps=average,
                        last_batch_age_s=None if self._last is None else max(0, now-self._last),
                        host_batch_ms={"p50": percentile(durations, .5),
                                       "p95": percentile(durations, .95),
                                       "p99": percentile(durations, .99)},
                        batch_interval_jitter_ms=statistics.pstdev(intervals) if intervals else None,
                        integrity=self._integrity, resources=dict(self._resources),
                        previous_scan=self._previous_scan,
                        resource_age_s=None if not self._resources else
                            max(0, time.time()-self._resources["sampled_at"]),
                        history=list(self._history))

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
            self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            try:
                resources = self._sample_resources()
            except (OSError, ValueError, IndexError):
                resources = {"sampled_at": time.time(), "error": "Host metrics unavailable"}
            with self._lock:
                self._resources = resources
            sample = self.snapshot()
            with self._lock:
                self._history.append(dict(time=time.time(), fps=sample["fps"],
                                          cpu=resources.get("process_cpu_pct"),
                                          gpu=(resources.get("gpus") or [{}])[0].get("util_pct")))
            self._stop.wait(1)

    def _sample_resources(self):
        now = time.perf_counter()
        cpu = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        total, idle = sum(cpu), cpu[3] + cpu[4]
        proc = os.times()
        used = proc.user + proc.system
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v, *_ = line.split()
            mem[k.rstrip(":")] = int(v) * 1024
        rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        nets = {}
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            name, counters = line.split(":", 1)
            v = [int(x) for x in counters.split()]
            nets[name.strip()] = (v[0], v[8], v[2], v[3])
        result = dict(sampled_at=time.time(), process_rss_bytes=rss,
                      host_ram_used_bytes=mem["MemTotal"]-mem["MemAvailable"],
                      host_ram_total_bytes=mem["MemTotal"], cpu_cores=os.cpu_count(),
                      process_cpu_pct=None, host_cpu_pct=None, network=[], gpus=[])
        previous = self._previous
        if previous is not None:
            dt = now-previous[0]
            result["process_cpu_pct"] = max(0, (used-previous[1])/dt*100)
            ticks = total-previous[2]
            result["host_cpu_pct"] = max(0, (1-(idle-previous[3])/ticks)*100) if ticks > 0 else None
            interface = self._interface()
            for name, v in nets.items():
                old = previous[4].get(name)
                if old is not None and name != "lo":
                    result["network"].append(dict(interface=name, route=name == interface,
                        rx_gbps=max(0, v[0]-old[0])*8/dt/1e9,
                        tx_gbps=max(0, v[1]-old[1])*8/dt/1e9,
                        rx_errors_delta=max(0,v[2]-old[2]), rx_drops_delta=max(0,v[3]-old[3])))
        self._previous = (now, used, total, idle, nets)
        try:
            output = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu,power.draw",
                                     "--format=csv,noheader,nounits"], capture_output=True,
                                    text=True, timeout=.8, check=True).stdout
            def number(value):
                try:
                    n = float(value)
                    return n if math.isfinite(n) else None
                except ValueError:
                    return None
            for line in output.splitlines():
                index, name, util, memory, temp, power = [v.strip() for v in line.split(",")]
                result["gpus"].append(dict(index=index, name=name, util_pct=number(util),
                    memory_mib=number(memory), temperature_c=number(temp), power_w=number(power)))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return result

    def _interface(self):
        """Longest-prefix IPv4 route, without spawning a command or sending packets."""
        try:
            address = int(ipaddress.IPv4Address(urlsplit(self.endpoint).hostname))
            routes = []
            for line in Path("/proc/net/route").read_text().splitlines()[1:]:
                row = line.split()
                dest = int.from_bytes(bytes.fromhex(row[1]), "little")
                mask = int.from_bytes(bytes.fromhex(row[7]), "little")
                if address & mask == dest and int(row[3],16) & 1:
                    routes.append((mask.bit_count(), -int(row[6]), row[0]))
            return max(routes)[2] if routes else None
        except (OSError, ValueError, TypeError, IndexError):
            return None
