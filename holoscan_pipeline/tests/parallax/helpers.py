from pathlib import Path
import yaml
from live4d.parallax.tools.migrate_flags import convert


def offline_args(tmp_path, flags, source, *, batch_size=40, out_dir=None, **parallax):
    values = convert(flags, batch_size=batch_size, out_dir=out_dir or tmp_path / "out")
    values["parallax"].update(parallax)
    path = tmp_path / "replay.yaml"
    path.write_text(yaml.safe_dump(values))
    return ["--config", str(path), "--source", str(source), "--device", "cpu"]
