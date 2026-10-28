#!/usr/bin/env bash
set -euo pipefail
# Pass the same shared YAML as the live application and a recording or frame array.
python -m live4d.parallax.offline --config "${1:?shared YAML required}" \
  --source "${2:?recording required}" --device "${DEVICE:-cpu}"
