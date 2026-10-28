#!/usr/bin/env bash
# End-to-end demo with NO detector, NO Docker and NO NVIDIA GPU: replay a capture over ZeroMQ into the
# pure-Python ingest (CPU decode, reconstruction on cuda / Apple mps / cpu) and watch the parallax and
# phase-flipped images fill in. Three local processes over localhost:
#
#   tools/replay_to_zmq.py  --(PUSH)-->  run.py --ingest python  --(PUB)-->  examples/viewer.py
#
# The default capture is a simulated 256x256 scan of 96x96 frames (tools/simulate_4dstem.py), made on
# first use (~1 min); its reconstruction flags come from the .flags file written next to it.
#
# Usage:   scripts/run_local_demo.sh [CAPTURE.dstream]
#   RATE=20000 BATCH=256 DEVICE=mps scripts/run_local_demo.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$( [ -x "$REPO/.venv/bin/python" ] && echo "$REPO/.venv/bin/python" || echo python3 )}"  # the repo's .venv if present
CAPTURE="${1:-$REPO/out/sim256.dstream}"
FLAGS_FILE="${CAPTURE%.dstream}.flags"
RATE="${RATE:-10000}"                 # frames/s the feeder sends (a 256x256 scan takes ~6.5 s)
BATCH="${BATCH:-256}"                 # one scan row per batch
DEVICE="${DEVICE:-auto}"              # torch device: auto (cuda, mps, cpu), cuda, mps or cpu
PULL_EP="${PULL_EP:-tcp://127.0.0.1:5555}"   # feeder PUSH  ->  run.py PULL
PUB_EP="${PUB_EP:-tcp://127.0.0.1:5601}"     # run.py PUB   ->  viewer SUB

cd "$REPO"
if [ ! -f "$CAPTURE" ]; then
  if [ "$CAPTURE" = "$REPO/out/sim256.dstream" ]; then
    echo "[demo] simulating $CAPTURE (256x256 scan, 96x96 frames) …"
    "$PY" -m tools.simulate_4dstem --scan 256 256 --format dstream --out "${CAPTURE%.dstream}" || exit 1
  else
    echo "no capture at $CAPTURE"; exit 1
  fi
fi
[ -f "$FLAGS_FILE" ] || { echo "no reconstruction flags at $FLAGS_FILE (pass the geometry flags yourself)"; exit 1; }

PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
mkdir -p "$REPO/out/demo"

echo "[demo] starting the pipeline (python ingest, device $DEVICE, live PUB on $PUB_EP) …"
# shellcheck disable=SC2046
"$PY" -m realtime_stem.apps.run --ingest python --device "$DEVICE" --endpoint "$PULL_EP" \
  --n-connections 1 --batch-size "$BATCH" --live-endpoint "$PUB_EP" --live-every 0.2 \
  --out-dir "$REPO/out/demo" $(cat "$FLAGS_FILE") &
RUN_PID=$!
PIDS+=("$RUN_PID")

echo "[demo] starting the viewer (SUB $PUB_EP) …"
"$PY" examples/viewer.py "$PUB_EP" &
VIEWER_PID=$!
PIDS+=("$VIEWER_PID")
sleep 2

echo "[demo] replaying $CAPTURE into $PULL_EP at $RATE frames/s …"
"$PY" -m tools.replay_to_zmq "$CAPTURE" --endpoint "$PULL_EP" --rate "$RATE"
wait "$RUN_PID"

echo "[demo] done — results in $REPO/out/demo/. Close the viewer window to exit."
wait "$VIEWER_PID" 2>/dev/null || true
