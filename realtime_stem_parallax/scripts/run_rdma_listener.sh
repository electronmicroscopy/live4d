#!/usr/bin/env bash
# Start the live RDMA listener (Holoscan 5 image) as used at the microscope: a 512x512 binned scan,
# GPU 2, the ARINA's DCU at 10.10.10.2 over RoCE (this machine 10.10.10.1, mlx5_0). Run it from
# anywhere; the results go to this folder's OUT_DIR:
#
#   scripts/run_rdma_listener.sh out/microscope_v9
#
# OUT_DIR must be new or empty, so an earlier run's scans are never overwritten.
# Order for every session: detector disarmed -> this script -> wait for
# "[holoscan] ready (RDMA receiver connected)" -> arm -> scan. The DCU needs ntrigger = 262144,
# compression none, bstream mode cbor-rdma (README, "Before a session").
set -euo pipefail
OUT_DIR="${1:?usage: $0 OUT_DIR   (a new folder, e.g. out/microscope_v9)}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR")" ]; then
  echo "$OUT_DIR already holds results; choose a new folder." >&2
  exit 1
fi

docker run --rm -it --name rts-pipeline --gpus device=2 --network host \
  --device /dev/infiniband --ulimit memlock=-1:-1 --ulimit stack=33554432 \
  --user "$(id -u):$(id -g)" -v "$PWD":/work -w /work --entrypoint python3 \
  realtime-stem:hsdk5 -m realtime_stem.apps.run \
  --transport rdma --rdma-remote 10.10.10.2:31003 --rdma-local 10.10.10.1 --rdma-device mlx5_0 \
  --binned --scan 512 512 \
  --energy 300000 --semiangle 21.4 --angular-sampling 0.9 \
  --scan-sampling 1.48 --defocus=-700 --rotation 0 --upsampling 4 \
  --continuous --viewer-port 18050 --out-dir "$OUT_DIR"
