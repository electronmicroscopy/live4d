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
    devices: list[int] | None = None
    scan: dict = field(default_factory=lambda: dict(width=1000, height=1000))
    detector: dict = field(default_factory=lambda: dict(bf_radius=16.0, adf_inner=24.0, adf_outer=48.0))
    execute: dict = field(default_factory=lambda: dict(mode="finite", seconds=0.0))
    zmq: dict = field(default_factory=lambda: dict(endpoint="tcp://127.0.0.1:5555"))
    rdma: dict = field(default_factory=lambda: dict(remote_address="", local_address="", device="",
                                                 ring_memory="gpu", ring_mib=64, max_batch=512,
                                                 timeout=10.0, protocol=2, rings=1))
    writer: dict = field(default_factory=lambda: dict(enabled=False, output_dir="/data",
                                                   name_pattern="series_$id",
                                                   format="nxmx", images_per_file=1000000, staging_batches=4))
    # Lossless ANS encoding of every series on the GPU (quantem.gpu); keep: series kept resident (0: all).
    encoder: dict = field(default_factory=lambda: dict(enabled=False, keep=1))

    def __post_init__(self):
        defaults = dict(scan=dict(width=1000, height=1000),
                        detector=dict(bf_radius=16.0, adf_inner=24.0, adf_outer=48.0),
                        execute=dict(mode="finite", seconds=0.0),
                        zmq=dict(endpoint="tcp://127.0.0.1:5555"),
                        rdma=dict(remote_address="", local_address="", device="", ring_memory="gpu",
                                  ring_mib=64, max_batch=512, timeout=10.0, protocol=2, rings=1),
                        writer=dict(enabled=False, output_dir="/data", name_pattern="series_$id",
                                    format="nxmx", images_per_file=1000000, staging_batches=4),
                        encoder=dict(enabled=False, keep=1))
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
        if self.devices is not None:
            if (not isinstance(self.devices, (list, tuple)) or not self.devices
                    or any(type(d) is not int or not 0 <= d <= 255 for d in self.devices)):
                raise ValueError("devices must contain CUDA device ordinals between 0 and 255")
            object.__setattr__(self, "devices", tuple(self.devices))
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
        if rdma["protocol"] not in (2, 3) or type(rdma["protocol"]) is not int:
            raise ValueError("rdma.protocol must be 2 or 3")
        if type(rdma["rings"]) is not int or not 1 <= rdma["rings"] <= 256:
            raise ValueError("rdma.rings must be an integer between 1 and 256")
        if rdma["protocol"] == 2 and rdma["rings"] != 1:
            raise ValueError("rdma.rings above 1 requires rdma.protocol 3")
        if self.devices is not None and self.transport == "rdma" and rdma["protocol"] == 3:
            raise ValueError("rdma.protocol 3 is not supported with `devices` yet")
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
        if writer["format"] not in ("nxmx", "em"):
            raise ValueError("writer.format must be 'nxmx' or 'em'")
        if type(writer["images_per_file"]) is not int or writer["images_per_file"] < 1:
            raise ValueError("writer.images_per_file must be a positive integer")
        if type(writer["staging_batches"]) is not int or not 2 <= writer["staging_batches"] <= 64:
            raise ValueError("writer.staging_batches must be an integer between 2 and 64")
        if type(self.encoder["enabled"]) is not bool:
            raise ValueError("encoder.enabled must be true or false")
        if type(self.encoder["keep"]) is not int or self.encoder["keep"] < 0:
            raise ValueError("encoder.keep must be a nonnegative integer")
        if self.transport == "rdma":
            host, sep, port = rdma["remote_address"].rpartition(":")
            if not host or not sep or not port.isdigit() or not 1 <= int(port) <= 65535 or not rdma["local_address"]:
                raise ValueError("RDMA requires remote_address host:port and local_address")

    @property
    def gpu_devices(self) -> tuple[int, ...]:
        """Ordered routing lanes; repeated CUDA ordinals remain separate entries."""
        return tuple(self.devices) if self.devices is not None else (self.device,)


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    with path.open(encoding="utf-8") as stream:
        return PipelineConfig(**(yaml.safe_load(stream) or {}))
