"""Acquisition lifecycle shared by the live branch and offline replay."""
from pathlib import Path
import argparse

from .algorithm import ParallaxAlgorithm
from .control import LiveControl, QueuedLiveControl
from .ingest.sink import ACC_LOCK, BatchSink, publish_snapshot


class ParallaxRuntime:
    def __init__(self, config, *, device, live=False):
        self.live = live
        self.queued_control = None
        self.config = config
        self.settings = config.parallax
        self.algorithm = ParallaxAlgorithm()
        parser = argparse.ArgumentParser()
        self.algorithm.add_cli_args(parser)
        self.args = parser.parse_args([])
        vars(self.args).update(self.settings)
        self.args.device = device
        self.args.dtype = "uint16"
        self.args.batch_size = config.batch_size
        self.args.out_dir = Path(self.settings["out_dir"])
        self.acc = self.sink = self.publisher = self.telemetry = self.viewer = None
        self._closed = False
        self.control = None
        self.saved_revision = 0
        self.rounds = 0
        self.active = False

    def open(self):
        try:
            if self.settings["live_endpoint"]:
                from .streaming.live_stream import LivePublisher
                self.publisher = LivePublisher(self.settings["live_endpoint"])
            if self.settings["viewer_port"]:
                from .streaming.live_stream import Fanout
                from .streaming.telemetry import Telemetry
                from .streaming.web.server import FrameStore, StorePublisher, serve
                store = FrameStore()
                self.publisher = Fanout(self.publisher, StorePublisher(store))
                endpoint = (self.config.zmq["endpoint"] if self.config.transport == "zmq"
                            else "tcp://" + self.config.rdma["remote_address"])
                self.telemetry = Telemetry(endpoint)
                self.publisher.telemetry = self.telemetry
                self.telemetry.start()
                self.control = LiveControl(self.algorithm, self.args, publisher=self.publisher,
                                           on_new_accumulator=self._hook)
                if self.live:
                    self.queued_control = QueuedLiveControl(self.control)
                self.viewer = serve(store, self.settings["viewer_host"], self.settings["viewer_port"],
                                    endpoint="live4d parallax branch", control=self.queued_control or self.control,
                                    telemetry=self.telemetry)
                print(f"[parallax] viewer on http://{self.settings['viewer_host']}:"
                      f"{self.viewer.server_address[1]}/", flush=True)
        except BaseException:
            self.close()
            raise

    def _hook(self, acc):
        if self.settings["save_passes"]:
            # Synchronous file writes bound pending output memory and propagate disk failures.
            acc.on_pass_complete = lambda k: self.algorithm.save_pass(
                acc, self.args.out_dir, k, self.settings["save_passes_dtype"])

    def start_series(self, shape, scan):
        with ACC_LOCK:
            if self.active:
                raise ValueError("Parallax START arrived before END")
            acc = self.control.acc if self.control is not None else self.acc
            if self.args.accumulate != "scan" and acc is not None:
                if tuple(shape) != tuple(acc.detector_shape) or tuple(scan) != tuple(acc.scan_shape):
                    raise ValueError("Persistent parallax geometry changed between series")
                acc.start_series()
            else:
                self.args.detector = tuple(shape)
                self.args.scan = tuple(scan)
                if self.args.accumulate == "latest" and self.args.batch_size > scan[0] * scan[1]:
                    raise ValueError("parallax latest requires batch_size <= scan positions")
                acc = self.algorithm.build(self.args)
                self._hook(acc)
            self.acc = acc
            self.sink = BatchSink(acc, live_publisher=self.publisher,
                                  live_every=self.settings["live_every"],
                                  live_every_rows=self.settings["live_every_rows"])
            self.rounds += 1
            self.active = True
            if self.control is not None:
                self.control.attach(acc, lambda: self.sink)

    def finish_series(self, *, interrupted=False):
        with ACC_LOCK:
            if not self.active:
                return
            self.acc = self.sink.acc
            if not interrupted and hasattr(self.acc, "end_series"):
                self.acc.end_series()
            print(self.sink.summary(), flush=True)
            print(self.sink.completeness()[1], flush=True)
            if interrupted:
                print("[parallax] interrupted acquisition; saving partial results", flush=True)
            self.algorithm.save(self.acc, self.args.out_dir, tag=f"scan{self.rounds:03d}")
            if self.publisher is not None:
                publish_snapshot(self.publisher, self.acc,
                                 final=interrupted or self.args.accumulate == "scan"
                                 or self.config.execute["mode"] == "finite")
            self.saved_revision = self.control.revision if self.control else 0
            self.active = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.queued_control is not None:
            self.queued_control.closed = True
        # Called after the graph's session has stopped, before transports release their storage.
        try:
            # The session has stopped. Runtime tensors are no longer accessed here. Synchronize
            # accumulated work once before reading partial results on an application-owned stream.
            if self.live and self.acc is not None:
                import torch
                with torch.cuda.device(self.args.device):
                    torch.cuda.synchronize()
            if self.control is not None and self.control.revision != self.saved_revision and not self.active:
                with ACC_LOCK:
                    self.acc = self.control.acc
                    self.algorithm.save(self.acc, self.args.out_dir, tag=f"scan{self.rounds:03d}")
                    print("Tuned flags: " + " ".join(self.algorithm.tuned_flags(self.args)), flush=True)
                    self.saved_revision = self.control.revision
            if self.active:
                with ACC_LOCK:
                    self.finish_series(interrupted=True)
        finally:
            if self.viewer is not None:
                self.viewer.shutdown()
                self.viewer.server_close()
                self.viewer = None
            if self.telemetry is not None:
                self.telemetry.close()
                self.telemetry = None
            if self.publisher is not None:
                self.publisher.close()
                self.publisher = None
