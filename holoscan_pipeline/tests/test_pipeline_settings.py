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
                                 images_per_file=1000000, staging_batches=4)
    config = PipelineConfig(writer=dict(enabled=True, output_dir="/out", images_per_file=7))
    assert config.writer["enabled"] and config.writer["images_per_file"] == 7
    assert config.writer["name_pattern"] == "series_$id"
    assert PipelineConfig(writer=dict(name_pattern="run42/scan_$id")).writer["name_pattern"] == "run42/scan_$id"


def test_detector_radius_accepts_192_pixel_extent():
    PipelineConfig(detector=dict(bf_radius=136, adf_inner=24, adf_outer=136))
    with pytest.raises(ValueError):
        PipelineConfig(detector=dict(bf_radius=137, adf_inner=24, adf_outer=136))


@pytest.mark.parametrize("values", [dict(batch_size=0), dict(batch_size=1001), dict(batch_size=True),
    dict(queue_depth=17), dict(transport="udp"), dict(transport="rdma"),
    dict(rdma=dict(max_batch=0)), dict(rdma=dict(ring_memory="host")),
    dict(scan=dict(width=0)), dict(execute=dict(mode="unknown")), dict(zmq=dict(extra=1)),
    dict(writer=dict(enabled=1)), dict(writer=dict(output_dir="")),
    dict(writer=dict(name_pattern="/abs/series_$id")), dict(writer=dict(name_pattern="../up_$id")),
    dict(writer=dict(name_pattern="runs/../../up")), dict(writer=dict(name_pattern="runs/")),
    dict(writer=dict(images_per_file=0)),
    dict(writer=dict(images_per_file=True)), dict(writer=dict(staging_batches=1)),
    dict(writer=dict(compression="lz4"))])
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
