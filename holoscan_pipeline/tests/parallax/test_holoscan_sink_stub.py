"""Runtime lifecycle without a GPU graph; live tensor ownership is tested in test_stream."""
from pathlib import Path
import numpy as np
import pytest
from live4d.config import PipelineConfig
from live4d.parallax.runtime import ParallaxRuntime
from live4d.algorithms.parallax.engine import ParallaxAccumulator


def config(tmp_path, **settings):
    return PipelineConfig(batch_size=3, scan=dict(width=4, height=2), parallax=dict(
        enabled=True, energy=300000, semiangle=4, angular_sampling=1,
        scan_sampling=2, defocus=-100, rotation=0, out_dir=str(tmp_path), **settings))


@pytest.mark.parametrize("shape", [(96, 96), (192, 192)])
def test_runtime_batches_equal_original_accumulator(tmp_path, shape):
    runtime = ParallaxRuntime(config(tmp_path), device="cpu")
    frames = np.random.default_rng(4).integers(0, 5000, size=(8, *shape), dtype=np.uint16)
    runtime.start_series(shape, (2, 4))
    reference = ParallaxAccumulator(runtime.acc.geometry, device="cpu",
                                   readout=runtime.acc.readout_config)
    reference.process(frames, np.arange(8))
    for start in range(0, 8, 3):
        runtime.sink.feed(frames[start:start+3], np.arange(8)[start:start+3])
    runtime.finish_series()
    np.testing.assert_allclose(np.load(tmp_path / "parallax_recon.npy"), reference.result(),
                               rtol=1e-6, atol=.01)
    assert runtime.sink.completeness()[0]
    runtime.close()


def test_consecutive_empty_series_and_geometry_change(tmp_path):
    runtime = ParallaxRuntime(config(tmp_path), device="cpu")
    for shape in ((96, 96), (192, 192), (96, 96)):
        runtime.start_series(shape, (2, 4))
        runtime.finish_series()
    assert runtime.rounds == 3
    assert len(list(tmp_path.glob("parallax_recon_scan*.npy"))) == 3
    runtime.close()


def test_persistent_geometry_rejected_and_interruption_saves(tmp_path):
    runtime = ParallaxRuntime(config(tmp_path, accumulate="latest"), device="cpu")
    runtime.start_series((96, 96), (2, 4))
    runtime.sink.feed(np.ones((2, 96, 96), dtype=np.uint16), np.arange(2))
    runtime.close()
    assert np.load(tmp_path / "parallax_recon.npy").sum() > 0
    with pytest.raises(ValueError, match="geometry changed"):
        runtime.start_series((192, 192), (2, 4))
    runtime.close()


def test_output_failure_propagates_and_resources_close(tmp_path, monkeypatch):
    runtime = ParallaxRuntime(config(tmp_path), device="cpu")
    runtime.start_series((96, 96), (2, 4))
    monkeypatch.setattr(runtime.algorithm, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        runtime.close()
    assert runtime.viewer is None and runtime.publisher is None


def test_offline_uint32_recording_keeps_high_values(tmp_path):
    from live4d.parallax.ingest.codec import encode_start_message, encode_image_message, encode_end_message
    from live4d.parallax.ingest.dstream import write_magic, write_record
    from live4d.parallax.offline import replay
    frames = np.full((2, 96, 96), 2**31 + 17, dtype=np.uint32)
    source = tmp_path / "uint32.dstream"
    with source.open("wb") as stream:
        write_magic(stream)
        write_record(stream, encode_start_message((96, 96), 2, dtype="uint32"))
        for iid, frame in enumerate(frames):
            write_record(stream, encode_image_message(iid, frame))
        write_record(stream, encode_end_message())
    runtime = ParallaxRuntime(config(tmp_path), device="cpu")
    replay(runtime, source)
    reference = ParallaxAccumulator(runtime.acc.geometry, device="cpu")
    reference.process(frames, np.arange(2))
    np.testing.assert_allclose(runtime.acc.result(), reference.result(), rtol=1e-6)
    runtime.close()
