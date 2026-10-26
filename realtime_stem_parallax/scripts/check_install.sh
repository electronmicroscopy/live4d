#!/usr/bin/env bash
# Check a new machine before the first live scan: the DCU's address, the GPU, the image, the GPU
# copy check, and a replay of a recording through the full pipeline. No detector needed (the DCU
# check only opens a connection). Run from the repository root:
#
#   GPU=0 DCU=<DCU_IP> scripts/check_install.sh                           # the basic checks
#   GPU=0 RECORDING=/data/scan.dstream SCAN="1024 1024" BINNED=1 FLAGS="$(cat site.flags)" \
#     scripts/check_install.sh
#
# Environment:
#   DCU        the DCU's stream address: an IP (port 31001), or tcp://IP:PORT. Checked first, so a
#              wrong address shows up before anything waits for frames from it.
#   GPU        which GPU the pipeline uses, as numbered by nvidia-smi. Required when the machine
#              has more than one GPU.
#   IMAGE      the Docker image (default realtime-stem:latest; realtime-stem:hsdk5 for Holoscan 5)
#   RECORDING + SCAN (+ BINNED=1 for 96x96 frames) + FLAGS (the recording's acquisition values:
#              --energy --semiangle --angular-sampling --scan-sampling --defocus --rotation
#              --center; the
#              rate depends on them): a full-size replay
#   TIMEOUT    seconds a replay may take before it is stopped (default 300)
# Every step prints PASS / FAIL / SKIP.
set -uo pipefail
if [ -n "${RECORDING:-}" ] && [ -d "$(dirname "$RECORDING")" ]; then   # relative to where it was started
  RECORDING="$(cd "$(dirname "$RECORDING")" && pwd)/$(basename "$RECORDING")"
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
IMAGE="${IMAGE:-realtime-stem:latest}"
GPU_SET="${GPU:+yes}"
GPU="${GPU:-0}"
TIMEOUT="${TIMEOUT:-300}"
NAME=realtime_stem_check
FEED=realtime_stem_check_feed
OUT="$REPO/out/check_install"
FAILED=0
DCU_OK=""

pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAILED=$((FAILED + 1)); }
skip() { echo "SKIP  $*"; }
in_image() {   # run python3 in the image on the chosen GPU, the repository mounted at /work
  docker run --rm --gpus "device=$GPU" --ulimit stack=33554432:33554432 --network host \
    -v "$REPO":/work -w /work --entrypoint python3 "$IMAGE" "$@"
}
cleanup() { docker rm -f "$NAME" "$FEED" >/dev/null 2>&1; }
trap cleanup EXIT

echo "== 1. The DCU's stream address"
if [ -n "${DCU:-}" ]; then
  hostport=${DCU#tcp://}
  [[ "$hostport" == *:* ]] || hostport="$hostport:31001"
  host=${hostport%:*}
  port=${hostport##*:}
  if timeout 3 bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null; then
    pass "the DCU at $host:$port accepts connections"
    DCU_OK=yes
  else
    fail "nothing accepts connections at $host:$port"
    if ping -c 1 -W 2 "$host" >/dev/null 2>&1; then
      echo "      $host answers ping, but port $port is closed: is the DCU's stream interface"
      echo "      enabled, and is $port its stream port?"
    else
      echo "      $host does not answer ping (some networks block it). Check the address: the"
      echo "      stream comes from the DCU's data interface on the network to this machine,"
      echo "      which can be a different address from the one used for the DCU's web page."
    fi
    echo "      Route from this machine: $(ip route get "$host" 2>/dev/null | head -1)"
    echo "      Fix this first: the pipeline and the recorder wait for frames from this address."
  fi
  # The link towards the DCU: 120 kHz of binned beam-on frames is about 3 Gbit/s of data.
  dev=$(ip route get "$host" 2>/dev/null | grep -o "dev [^ ]*" | head -1 | cut -d" " -f2)
  speed=$(cat "/sys/class/net/$dev/speed" 2>/dev/null)
  if [ -z "$dev" ] || [ "$dev" = lo ]; then
    skip "link speed (no route to $host, or a local address)"
  elif ! [[ "$speed" =~ ^[0-9]+$ ]] || [ "$speed" = 0 ]; then
    skip "link speed of $dev unknown (virtual or bonded interface?)"
  elif [ "$speed" -ge 10000 ]; then
    pass "link to the DCU: $dev at $((speed / 1000)) Gbit/s"
  else
    fail "link to the DCU: $dev at $speed Mbit/s; 120 kHz needs 10 Gbit/s or more"
  fi
  echo "      (make sure nothing else on the network reads this stream during the scans)"
else
  skip "set DCU=<DCU_IP> (the DCU's stream address; ask whoever set up the detector)"
fi

echo "== 2. GPU and driver"
if gpus=$(nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,mig.mode.current \
            --format=csv,noheader); then
  echo "$gpus" | sed 's/^/      /'
  pass "nvidia-smi"
  n_gpus=$(echo "$gpus" | grep -c .)
  if ! echo "$gpus" | cut -d, -f1 | grep -qx "$GPU"; then
    fail "there is no GPU $GPU on this machine (GPU=<index> from the list above)"
    echo "Stopping: choose the GPU first."; exit 1
  fi
  if [ "$n_gpus" -gt 1 ] && [ -z "$GPU_SET" ]; then
    fail "this machine has $n_gpus GPUs: choose the one for the pipeline with GPU=<index>"
    echo "Stopping: choose the GPU first."; exit 1
  fi
  echo "      GPU $GPU will be used: $(echo "$gpus" | awk -F', ' -v g="$GPU" '$1 == g {print $2}')"
  others=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
             --format=csv,noheader 2>/dev/null)
  if [ -n "$others" ]; then
    echo "      other programs already running on GPU $GPU (they share it with the pipeline):"
    echo "$others" | sed 's/^/        /'
  fi
else
  fail "nvidia-smi (is the NVIDIA driver installed?)"
fi

echo "== 3. Docker can use the GPU, and the image exists"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  pass "image $IMAGE"
else
  fail "image $IMAGE not found: build it (README, Install step 3) or docker load it"
  built=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep realtime-stem)
  [ -n "$built" ] && echo "      images on this machine: $(echo $built); choose one with IMAGE=<name:tag>"
  echo "Stopping: the remaining checks need the image."; exit 1
fi
if seen=$(in_image -c "$(cat <<'PY'
import importlib.util, torch, holoscan
from realtime_stem import __version__
p = torch.cuda.get_device_properties(0)
if importlib.util.find_spec("live4d"):   # the Holoscan 5 image: the shared ZMQ / RDMA receivers
    ops = "Holoscan 5 ingest"
else:
    import dectris_holoscan_ops
    ops = "operators " + getattr(dectris_holoscan_ops, "__version__", "before 2.0")
print(f"      {p.name}, compute capability {p.major}.{p.minor}, {p.total_memory / 2**30:.0f} GiB")
print(f"      torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"holoscan {getattr(holoscan, '__version__', '?')}")
print(f"      realtime-stem {__version__}, {ops}")
torch.ones(1, device="cuda").add_(1)
bus = getattr(p, "pci_bus_id", None)
print(f"BUS={bus:02X}" if isinstance(bus, int) else "BUS=")
PY
)"); then
  echo "$seen" | grep -v "^BUS="
  pass "the image runs on GPU $GPU"
  bus=$(echo "$seen" | sed -n 's/^BUS=//p')
  host_bus=$(nvidia-smi -i "$GPU" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null \
               | cut -d: -f2)
  if [ -n "$bus" ] && [ -n "$host_bus" ] && [ "$bus" != "$host_bus" ]; then
    fail "the container uses the GPU on PCI bus $bus, not GPU $GPU (bus $host_bus)"
  fi
else
  fail "the image does not run on GPU $GPU (NVIDIA container toolkit installed? docker run --gpus works?)"
fi

echo "== 4. Copies between pinned memory and the GPU"
if in_image -m tools.check_gpu_copies; then pass "GPU copies"; else fail "GPU copies"; fi

replay() {   # replay DSTREAM through the pipeline: replay LABEL DSTREAM SCAN EXTRA_ARGS...
  local label=$1 dstream=$2 scan=$3; shift 3
  local log="$OUT/$label.log" feed_log="$OUT/$label.feeder.log"
  mkdir -p "$OUT/$label"
  cleanup
  [ -f "$dstream" ] || { fail "$label: no file at $dstream"; return; }
  if (exec 3<>/dev/tcp/127.0.0.1/5599) 2>/dev/null; then
    fail "$label: port 5599 is in use (a replay or tuning container still running? docker ps)"
    return
  fi
  docker run -d --name "$NAME" --gpus "device=$GPU" --network host \
    --ulimit stack=33554432:33554432 -v "$REPO":/work -v "$OUT/$label":/out -w /work \
    --entrypoint python3 "$IMAGE" -m realtime_stem.apps.run --algorithm parallax \
    --endpoint tcp://127.0.0.1:5599 --scan $scan --n-connections 3 --out-dir /out "$@" \
    >/dev/null || { fail "$label: pipeline did not start"; return; }
  echo "      replaying $(basename "$dstream") (stopped after $TIMEOUT s at the latest) …"
  sleep 8
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]; then
    docker logs "$NAME" > "$log" 2>&1
    tail -n 5 "$log" | sed 's/^/      /'
    fail "$label: the pipeline stopped before the replay began (log: $log)"
    return
  fi
  docker run -d --name "$FEED" --network host -v "$REPO":/work \
    -v "$(dirname "$dstream")":/rec:ro -w /work --entrypoint python3 "$IMAGE" \
    -m tools.replay_to_zmq "/rec/$(basename "$dstream")" --endpoint tcp://127.0.0.1:5599 --fast \
    >/dev/null || { fail "$label: the replay feeder did not start"; return; }
  if ! timeout "$TIMEOUT" docker wait "$NAME" >/dev/null; then
    echo "      no end of scan after $TIMEOUT s: stopping the pipeline"
    docker kill -s INT "$NAME" >/dev/null 2>&1; sleep 5
  fi
  docker logs "$NAME" > "$log" 2>&1
  docker logs "$FEED" > "$feed_log" 2>&1
  # A feeder with frames left (the pipeline stopped early, or SCAN is smaller than the recording)
  # would wait for a reader forever: remove it.
  docker rm -f "$FEED" >/dev/null 2>&1
  grep -h "TOTAL frames\|COMPLETE" "$log" | sed 's/^/      /'
  if grep -q "\[sink\] COMPLETE" "$log"; then
    pass "$label"
  else
    tail -n 5 "$log" | sed 's/^/      /'
    fail "$label (logs: $log, $feed_log)"
  fi
}

echo "== 5. Replay of the bundled sample through the full pipeline"
# The sample holds random counts, not a real scan: these values only exercise the pipeline.
replay sample "$REPO/examples/data/sample.dstream" "32 32" --binned --batch-size 128 \
  --energy 300000 --semiangle 4 --angular-sampling 1.0 --scan-sampling 0.5 --defocus -15000 \
  --rotation 0 --center 48 48

echo "== 6. Full-size replay of a real recording"
if [ -n "${RECORDING:-}" ] && [ -n "${SCAN:-}" ] && [ -z "${FLAGS:-}" ]; then
  fail "set FLAGS to the recording's acquisition values (e.g. FLAGS=\"\$(cat site.flags)\"): the rate depends on them"
elif [ -n "${RECORDING:-}" ] && [ -n "${SCAN:-}" ]; then
  ARGS=(); [ "${BINNED:-0}" = 1 ] && ARGS+=(--binned)
  # shellcheck disable=SC2206
  ARGS+=($FLAGS)
  replay recording "$RECORDING" "$SCAN" "${ARGS[@]}"
  echo "      (the rate above should be well over the detector's frame rate, e.g. > 120 kHz)"
else
  skip "set RECORDING=<file.dstream>, SCAN=\"Ny Nx\" and FLAGS=\"<acquisition values>\" (and BINNED=1 for 96x96 frames)"
fi

echo
if [ "$FAILED" = 0 ]; then echo "All checks passed."; else echo "$FAILED check(s) FAILED."; fi
if [ "$FAILED" = 0 ] || [ -n "$DCU_OK" ]; then
  echo "For the run commands: --gpus \"device=$GPU\" (and IMAGE $IMAGE)"
  [ -n "$DCU_OK" ] && echo "                      --endpoint tcp://$host:$port"
fi
exit $(( FAILED > 0 ))
