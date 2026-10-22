#!/bin/sh
# Build the imx-receiver base image (Dockerfile target imx-base): holoscan + rdma-core + the
# imx-receiver wheel, installed, with the wheel itself in $IMX_WHEELS. Other images use it with
#   FROM imx-receiver-base:<version>
#
#   dectris/tools/build_base_image.sh                      # local tags imx-receiver-base:<version>, :latest
#   REGISTRY=registry.example.com/group dectris/tools/build_base_image.sh   # also tag and push there
set -eu

cd "$(dirname "$0")/../.."
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' dectris/pyproject.toml)
IMAGE=imx-receiver-base

docker build --target imx-base --build-arg IMX_VERSION="$VERSION" \
    -f dectris/Dockerfile -t "$IMAGE:$VERSION" -t "$IMAGE:latest" .
echo "built $IMAGE:$VERSION"

if [ -n "${REGISTRY:-}" ]; then
    for tag in "$VERSION" latest; do
        docker tag "$IMAGE:$tag" "$REGISTRY/$IMAGE:$tag"
        docker push "$REGISTRY/$IMAGE:$tag"
    done
fi
