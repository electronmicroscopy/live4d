#!/usr/bin/env bash
set -euo pipefail
# Read-only installation checks; no detector commands or acquisition waits.
nvidia-smi
python -c 'import live4d.operators; import live4d.schemas.decoded_batch; import live4d.parallax.offline; from importlib.resources import files; assert (files("live4d.parallax.streaming.web") / "static/index.html").is_file()'
