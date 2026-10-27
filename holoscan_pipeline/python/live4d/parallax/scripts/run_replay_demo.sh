#!/usr/bin/env bash
set -euo pipefail
# Run live4d separately with matching YAML (transport zmq, this endpoint).
python -m live4d.parallax.tools.replay_to_zmq "${1:?capture required}" \
  --endpoint "${ENDPOINT:-tcp://127.0.0.1:5555}"
