"""Configuration tests for transport selection and independent batch settings."""
from pathlib import Path
import pytest
from live4d.config import PipelineConfig, load_config


def test_defaults_and_transport_batch_settings():
    config = PipelineConfig(batch_size=17, rdma=dict(max_batch=3))
    assert config.batch_size == 17 and config.rdma["max_batch"] == 3
    assert config.execute["mode"] == "finite"


def test_writer_defaults_to_disabled_one_million_images_per_file():
    config = PipelineConfig()
    assert config.writer == dict(enabled=False, output_dir="/data", name_pattern="series_$id",
                                 format="nxmx", images_per_file=1000000, staging_batches=4)
    assert PipelineConfig(writer=dict(format="em")).writer["format"] == "em"
    config = PipelineConfig(writer=dict(enabled=True, output_dir="/out", images_per_file=7))
    assert config.writer["enabled"] and config.writer["images_per_file"] == 7
    assert config.writer["name_pattern"] == "series_$id"
    assert PipelineConfig(writer=dict(name_pattern="run42/scan_$id")).writer["name_pattern"] == "run42/scan_$id"


def test_encoder_defaults_to_disabled_keeping_one_series():
    assert PipelineConfig().encoder == dict(enabled=False, keep=1)


def test_rdma_protocol_selects_rings():
    config = PipelineConfig()
    assert (config.rdma["protocol"], config.rdma["rings"]) == (2, 1)
    config = PipelineConfig(rdma=dict(protocol=3, rings=4))
    assert (config.rdma["protocol"], config.rdma["rings"]) == (3, 4)
    assert PipelineConfig(rdma=dict(protocol=3, rings=256)).rdma["rings"] == 256


def test_detector_radius_accepts_192_pixel_extent():
    PipelineConfig(detector=dict(bf_radius=136, adf_inner=24, adf_outer=136))
    with pytest.raises(ValueError):
        PipelineConfig(detector=dict(bf_radius=137, adf_inner=24, adf_outer=136))


@pytest.mark.parametrize("values", [dict(batch_size=0), dict(batch_size=1001), dict(batch_size=True),
    dict(queue_depth=17), dict(transport="udp"), dict(transport="rdma"),
    dict(rdma=dict(max_batch=0)), dict(rdma=dict(ring_memory="host")),
    dict(rdma=dict(protocol=1)), dict(rdma=dict(protocol="3")), dict(rdma=dict(protocol=3, rings=0)),
    dict(rdma=dict(protocol=3, rings=257)), dict(rdma=dict(protocol=3, rings=True)),
    dict(rdma=dict(rings=2)),
    dict(scan=dict(width=0)), dict(execute=dict(mode="unknown")), dict(zmq=dict(extra=1)),
    dict(writer=dict(enabled=1)), dict(writer=dict(output_dir="")),
    dict(writer=dict(name_pattern="/abs/series_$id")), dict(writer=dict(name_pattern="../up_$id")),
    dict(writer=dict(name_pattern="runs/../../up")), dict(writer=dict(name_pattern="runs/")),
    dict(writer=dict(images_per_file=0)), dict(writer=dict(format="EM")),
    dict(writer=dict(images_per_file=True)), dict(writer=dict(staging_batches=1)),
    dict(writer=dict(compression="lz4")), dict(encoder=dict(enabled=1)), dict(encoder=dict(keep=-1)),
    dict(encoder=dict(keep=True)), dict(encoder=dict(chunk=4096))])
def test_rejects_invalid_settings(values):
    with pytest.raises(ValueError):
        PipelineConfig(**values)


def test_shipped_configs():
    root = Path(__file__).resolve().parents[1] / "config"
    assert load_config(root / "basic_ingest.yaml").transport == "zmq"
    assert load_config(root / "rdma.yaml").transport == "rdma"
    buffle = load_config(root / "buffle_rdma.yaml")
    assert buffle.transport == "rdma" and buffle.execute["mode"] == "service"
    assert buffle.scan == dict(width=1000, height=1000)
    rings = load_config(root / "rdma_rings.yaml")
    assert (rings.rdma["protocol"], rings.rdma["rings"], rings.rdma["ring_mib"]) == (3, 4, 64)


def test_device_list_and_legacy_scalar_share_normalized_selection():
    rdma = dict(remote_address="sender:31003", local_address="receiver")
    assert PipelineConfig(device=3).gpu_devices == (3,)
    assert PipelineConfig(devices=[3]).gpu_devices == (3,)
    config = PipelineConfig(transport="rdma", devices=[3, 2, 0], rdma=rdma)
    assert config.gpu_devices == (3, 2, 0)
    assert isinstance(config.devices, tuple)
    assert PipelineConfig(transport="rdma", devices=[2], rdma=rdma).gpu_devices == (2,)


@pytest.mark.parametrize("devices", [[], [-1], [256], [True], "0", [0, "1"]])
def test_rejects_invalid_device_lists(devices):
    with pytest.raises(ValueError):
        PipelineConfig(devices=devices)


def test_multiple_devices_select_the_same_order_for_zmq():
    assert PipelineConfig(devices=[3, 2, 0]).gpu_devices == (3, 2, 0)


@pytest.mark.parametrize("transport", ["zmq", "rdma"])
def test_repeated_devices_preserve_independent_routing_entries(transport):
    rdma = dict(remote_address="sender:31003", local_address="receiver")
    assert PipelineConfig(transport=transport, devices=[0, 0], rdma=rdma).gpu_devices == (0, 0)
    assert PipelineConfig(transport=transport, devices=[2, 3, 2], rdma=rdma).gpu_devices == (2, 3, 2)


def test_protocol_3_with_devices_requires_host_mapped_rings():
    rdma = dict(remote_address="sender:31003", local_address="receiver", protocol=3, rings=2)
    PipelineConfig(transport="rdma", rdma=rdma)
    with pytest.raises(ValueError):
        PipelineConfig(transport="rdma", devices=[0, 0], rdma=rdma)
    PipelineConfig(transport="rdma", devices=[0, 0], rdma=dict(rdma, ring_memory="host_mapped"))
    PipelineConfig(transport="zmq", devices=[0], rdma=dict(protocol=3))
