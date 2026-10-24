#!/usr/bin/env bash
# End-to-end demo with NO detector: replay a .dstream into the full pipeline and watch the
# reconstruction wipe in live. Three local processes over localhost:
#
#   host:  tools/replay_to_zmq.py  --(PUSH)-->  run.py (in the container)  --(PUB)-->  viewer.py
#
# The container runs --network host so all three share localhost sockets. Requires a container
# image built from this repository (from the repo root: docker build -t realtime-stem:latest
# -f docker/Dockerfile .) and, for the viewer window, a display.
#
# Usage:   scripts/run_replay_demo.sh [DSTREAM]
#   IMAGE=realtime-stem:latest  GPU=0  SCAN="64 64"  BATCH=256  scripts/run_replay_demo.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DSTREAM="${1:-$REPO/examples/data/sample.dstream}"
IMAGE="${IMAGE:-realtime-stem:latest}"   # the Docker image (docker/Dockerfile)
GPU="${GPU:-0}"                       # which GPU, as nvidia-smi numbers them
SCAN="${SCAN:-32 32}"                 # must match the capture's frame count (Ny*Nx)
BATCH="${BATCH:-128}"                 # frames per batch
PULL_EP="${PULL_EP:-tcp://127.0.0.1:5555}"   # feeder PUSH  ->  run.py PULL
PUB_EP="${PUB_EP:-tcp://127.0.0.1:5601}"     # run.py PUB   ->  viewer SUB
NAME="realtime_stem_demo"
# The bundled sample holds random counts, not a real scan: these acquisition values only exercise
# the pipeline. For a recording of your own, give its real values.
DEMO_FLAGS="${DEMO_FLAGS:---energy 300000 --semiangle 4 --angular-sampling 1.0 --scan-sampling 0.5 --defocus -15000 --rotation 0 --center 48 48}"  # acquisition values

command -v docker >/dev/null || { echo "docker not found"; exit 1; }
[ -f "$DSTREAM" ] || { echo "no capture at $DSTREAM (make one: python -m tools.make_dstream 96 96 uint16 1024 --out $DSTREAM)"; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not found — build it: docker build -t $IMAGE -f docker/Dockerfile ."; exit 1; }

VIEWER_PID=""
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; [ -n "$VIEWER_PID" ] && kill "$VIEWER_PID" 2>/dev/null || true; }
trap cleanup EXIT

mkdir -p "$REPO/out"

# --n-connections 1: one connection is plenty for the small demo (the DCU needs 3 for 120 kHz).
echo "[demo] starting the pipeline in $IMAGE (parallax, scan $SCAN, live PUB on $PUB_EP) …"
docker run --rm --name "$NAME" --network host --gpus "device=$GPU" --ulimit stack=33554432:33554432 \
  -v "$REPO":/work -v "$REPO/out":/out -w /work --entrypoint python3 \
  "$IMAGE" -m realtime_stem.apps.run \
    --algorithm parallax --endpoint "$PULL_EP" --binned --scan $SCAN $DEMO_FLAGS \
    --batch-size "$BATCH" --n-connections 1 --live-endpoint "$PUB_EP" --live-every 0.2 --out-dir /out &

echo "[demo] starting the viewer (SUB $PUB_EP) …"
( cd "$REPO" && python3 examples/viewer.py "$PUB_EP" ) &
VIEWER_PID=$!
sleep 2

echo "[demo] replaying $DSTREAM into $PULL_EP …"
( cd "$REPO" && python3 -m tools.replay_to_zmq "$DSTREAM" --endpoint "$PULL_EP" )

echo "[demo] replay done — reconstruction saved under $REPO/out/. Close the viewer window to exit."
wait "$VIEWER_PID" 2>/dev/null || true
