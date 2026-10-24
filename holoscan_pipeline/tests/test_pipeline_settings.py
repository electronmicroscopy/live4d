"""Configuration tests for transport selection and independent batch settings."""
from pathlib import Path
import pytest
from live4d.config import PipelineConfig, load_config


def test_defaults_and_transport_batch_settings():
    config = PipelineConfig(batch_size=17, rdma=dict(max_batch=3))
    assert config.batch_size == 17 and config.rdma["max_batch"] == 3
    assert config.execute["mode"] == "finite"


def test_detector_radius_accepts_192_pixel_extent():
    PipelineConfig(detector=dict(bf_radius=136, adf_inner=24, adf_outer=136))
    with pytest.raises(ValueError):
        PipelineConfig(detector=dict(bf_radius=137, adf_inner=24, adf_outer=136))


@pytest.mark.parametrize("values", [dict(batch_size=0), dict(batch_size=1001), dict(batch_size=True),
    dict(queue_depth=17), dict(transport="udp"), dict(transport="rdma"),
    dict(rdma=dict(max_batch=0)), dict(rdma=dict(ring_memory="host")),
    dict(scan=dict(width=0)), dict(execute=dict(mode="unknown")), dict(zmq=dict(extra=1))])
def test_rejects_invalid_settings(values):
    with pytest.raises(ValueError):
        PipelineConfig(**values)


def test_shipped_configs():
    root = Path(__file__).resolve().parents[1] / "config"
    assert load_config(root / "basic_ingest.yaml").transport == "zmq"
    assert load_config(root / "rdma.yaml").transport == "rdma"


def test_device_list_and_legacy_scalar_share_normalized_selection():
    rdma = dict(remote_address="sender:31003", local_address="receiver")
    assert PipelineConfig(device=3).gpu_devices == (3,)
    assert PipelineConfig(devices=[3]).gpu_devices == (3,)
    config = PipelineConfig(transport="rdma", devices=[3, 2, 0], rdma=rdma)
    assert config.gpu_devices == (3, 2, 0)
    assert isinstance(config.devices, tuple)
    assert PipelineConfig(transport="rdma", devices=[2], rdma=rdma).gpu_devices == (2,)


@pytest.mark.parametrize("devices", [[], [0, 0], [-1], [256], [True], "0", [0, "1"]])
def test_rejects_invalid_device_lists(devices):
    with pytest.raises(ValueError):
        PipelineConfig(devices=devices)


def test_multiple_devices_require_rdma():
    with pytest.raises(ValueError, match="require RDMA"):
        PipelineConfig(devices=[0, 1])
