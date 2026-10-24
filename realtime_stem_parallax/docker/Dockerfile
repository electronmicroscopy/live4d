# The complete runtime image: NVIDIA's public Holoscan base (Holoscan 4.4, a CUDA build of
# PyTorch, nvcomp) + this repository's C++/CUDA operators (operators/) + pyzmq for the live
# stream-out. Build from the repository root:
#
#   docker build -t realtime-stem:latest -f docker/Dockerfile .
#
# CUDA_ARCHITECTURES lists the GPU generations the kernels are compiled for (see the README);
# add yours if it is missing, e.g. --build-arg CUDA_ARCHITECTURES="75;80;86;89;90".
#
# The base image and the pip packages are pinned to the versions this release was tested with, so
# a later build gives the same image. The digest names the multi-architecture image (x86_64 and
# aarch64); Docker picks the variant of the build machine.
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v4.4.0-cuda12-dgpu@sha256:f0ed6528892d508d97e3fe282c3ea80b6d4836ae3942ac71dbcf6d3f9aa1bcaf

ARG CUDA_ARCHITECTURES=80;86;89;90

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_PREFIX_PATH=/opt/nvidia/holoscan

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libzmq3-dev \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    pybind11==2.13.6 \
    PyYAML==6.0.3 \
    pyzmq==27.2.0

WORKDIR /opt/realtime-stem

COPY operators ./operators

RUN cmake -S operators -B build/operators -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
  && cmake --build build/operators

ENV PYTHONPATH=/opt/realtime-stem/build/operators/python:/opt/nvidia/holoscan/python/lib:/opt/nvidia/hololink/python
# The repository is bind-mounted at run time: don't leave root-owned __pycache__ folders in it.
ENV PYTHONDONTWRITEBYTECODE=1
ENV LD_LIBRARY_PATH=/opt/nvidia/holoscan/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
