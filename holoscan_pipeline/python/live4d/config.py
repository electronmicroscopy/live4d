"""Validate the common application settings without loading GPU libraries."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass(frozen=True)
class PipelineConfig:
    transport: str = "zmq"
    batch_size: int = 512
    queue_depth: int = 4
    device: int = 0
    scan: dict = field(default_factory=lambda: dict(width=1000, height=1000))
    detector: dict = field(default_factory=lambda: dict(bf_radius=16.0, adf_inner=24.0, adf_outer=48.0))
    execute: dict = field(default_factory=lambda: dict(mode="finite", seconds=0.0))
    zmq: dict = field(default_factory=lambda: dict(endpoint="tcp://127.0.0.1:5555"))
    rdma: dict = field(default_factory=lambda: dict(remote_address="", local_address="", device="",
                                                 ring_memory="gpu", ring_mib=64, max_batch=512,
                                                 timeout=10.0))
    writer: dict = field(default_factory=lambda: dict(enabled=False, output_dir="/data",
                                                   name_pattern="series_$id",
                                                   images_per_file=1000000, staging_batches=4))

    def __post_init__(self):
        defaults = dict(scan=dict(width=1000, height=1000),
                        detector=dict(bf_radius=16.0, adf_inner=24.0, adf_outer=48.0),
                        execute=dict(mode="finite", seconds=0.0),
                        zmq=dict(endpoint="tcp://127.0.0.1:5555"),
                        rdma=dict(remote_address="", local_address="", device="", ring_memory="gpu",
                                  ring_mib=64, max_batch=512, timeout=10.0),
                        writer=dict(enabled=False, output_dir="/data", name_pattern="series_$id",
                                    images_per_file=1000000, staging_batches=4))
        for name, values in defaults.items():
            given = getattr(self, name)
            if not isinstance(given, dict) or set(given) - set(values):
                raise ValueError(f"Invalid {name} settings")
            object.__setattr__(self, name, values | given)
        for name, low, high in (("batch_size", 1, 1000), ("queue_depth", 2, 16), ("device", 0, 255)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")
        if self.transport not in ("zmq", "rdma"):
            raise ValueError("transport must be 'zmq' or 'rdma'")
        for value in self.scan.values():
            if type(value) is not int or not 1 <= value <= 2147483647:
                raise ValueError("scan dimensions must be positive int32 values")
        if self.execute["mode"] not in ("finite", "service") or self.execute["seconds"] < 0:
            raise ValueError("execute requires finite/service mode and nonnegative seconds")
        if not 0 <= self.detector["bf_radius"] <= 136 or not 0 <= self.detector["adf_inner"] <= self.detector["adf_outer"] <= 136:
            raise ValueError("Invalid BF/ADF radii")
        if not self.zmq["endpoint"]:
            raise ValueError("ZMQ endpoint must not be empty")
        rdma = self.rdma
        if rdma["ring_memory"] not in ("gpu", "host_mapped"):
            raise ValueError("rdma.ring_memory must be 'gpu' or 'host_mapped'")
        if type(rdma["max_batch"]) is not int or not 1 <= rdma["max_batch"] <= 1024:
            raise ValueError("rdma.max_batch must be between 1 and 1024")
        if type(rdma["ring_mib"]) is not int or rdma["ring_mib"] < 1 or rdma["timeout"] <= 0:
            raise ValueError("RDMA ring size and timeout must be positive")
        writer = self.writer
        if type(writer["enabled"]) is not bool:
            raise ValueError("writer.enabled must be true or false")
        if not isinstance(writer["output_dir"], str) or not writer["output_dir"]:
            raise ValueError("writer.output_dir must be a non-empty path")
        pattern = writer["name_pattern"]
        if (not isinstance(pattern, str) or not pattern or pattern.startswith("/")
                or pattern.endswith("/") or ".." in pattern.split("/")
                or pattern.split("/")[-1] == "."):
            raise ValueError("writer.name_pattern must be a relative file name stem below output_dir")
        if type(writer["images_per_file"]) is not int or writer["images_per_file"] < 1:
            raise ValueError("writer.images_per_file must be a positive integer")
        if type(writer["staging_batches"]) is not int or not 2 <= writer["staging_batches"] <= 64:
            raise ValueError("writer.staging_batches must be an integer between 2 and 64")
        if self.transport == "rdma":
            host, sep, port = rdma["remote_address"].rpartition(":")
            if not host or not sep or not port.isdigit() or not 1 <= int(port) <= 65535 or not rdma["local_address"]:
                raise ValueError("RDMA requires remote_address host:port and local_address")


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    with path.open(encoding="utf-8") as stream:
        return PipelineConfig(**(yaml.safe_load(stream) or {}))
