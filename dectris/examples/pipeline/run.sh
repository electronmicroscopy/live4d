#!/bin/sh
# Build and run the pipeline example container with what RDMA and GPUDirect need: the RDMA
# devices, the GPU, the host network (RoCE interface and imagemux's handshake address) and
# unlimited locked memory for the memory registration. The settings are in pipeline.yaml (baked
# into the image); --config <file> uses a file from the host instead (mounted into the container).
# Other arguments go to app.py.
#
#   examples/pipeline/run.sh --config my.yaml
#
# IMAGE=<tag> selects another image; BUILD=0 skips the build.
set -eu

IMAGE=${IMAGE:-imx-pipeline-example}
if [ "${BUILD:-1}" != 0 ]; then
    docker build -q -t "$IMAGE" "$(dirname "$0")" >/dev/null
fi

# --config <host file>: mount it and hand the container's path to app.py.
CONFIG_MOUNT=
if [ "${1:-}" = --config ] && [ $# -ge 2 ]; then
    CONFIG_MOUNT="-v $(realpath "$2"):/config/pipeline.yaml:ro"
    shift 2
    set -- --config /config/pipeline.yaml "$@"
fi

exec docker run --rm --privileged \
    --device /dev/infiniband \
    --runtime=nvidia \
    --net host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=33554432 \
    $CONFIG_MOUNT \
    ${DOCKER_ARGS:-} \
    "$IMAGE" "$@"
