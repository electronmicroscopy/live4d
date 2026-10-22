#!/bin/sh
# Build and run the pipeline example container with what RDMA and GPUDirect need: the RDMA
# devices, the GPU, the host network (RoCE interface and imagemux's handshake address) and
# unlimited locked memory for the memory registration. Arguments go to pipeline.py, e.g.
#
#   examples/pipeline/run.sh 10.42.41.10:31003 --local-addr 10.42.41.11 --seconds 0
#
# IMAGE=<tag> selects another image; BUILD=0 skips the build.
set -eu

IMAGE=${IMAGE:-imx-pipeline-example}
if [ "${BUILD:-1}" != 0 ]; then
    docker build -q -t "$IMAGE" "$(dirname "$0")" >/dev/null
fi

exec docker run --rm --privileged \
    --device /dev/infiniband \
    --runtime=nvidia \
    --net host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=33554432 \
    ${DOCKER_ARGS:-} \
    "$IMAGE" "$@"
