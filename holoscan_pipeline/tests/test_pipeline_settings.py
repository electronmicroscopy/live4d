"""Configuration tests for transport selection and independent batch settings."""
from pathlib import Path
import pytest
from live4d.config import PipelineConfig, load_config


def test_defaults_and_transport_batch_settings():
    config = PipelineConfig(batch_size=17, rdma=dict(max_batch=3))
    assert config.batch_size == 17 and config.rdma["max_batch"] == 3
    assert config.execute["mode"] == "finite"


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
