ARG CUDA_ARCHITECTURES=86;89
ARG BASE_SDK_VERSION=v4.4.0
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:${BASE_SDK_VERSION}-cuda13

FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_PREFIX_PATH=/opt/nvidia/holoscan


FROM base AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake build-essential pkg-config python3-dev libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

COPY dectris/pyproject.toml dectris/README.md /src/
COPY dectris/native /src/native
COPY dectris/src /src/src
RUN python3 -m pip wheel --no-cache-dir --no-deps /src -w /dist \
    && cmake -S /src/native -B /tmp/loopback -DIMXRX_BUILD_PYTHON=OFF \
    && cmake --build /tmp/loopback -j"$(nproc)" \
    && install -D -m 755 /tmp/loopback/imxrx_loopback /dist/bin/imxrx_loopback


FROM base AS imx-base

ARG IMX_VERSION=0.1.0
LABEL org.opencontainers.image.title="imx-receiver-base" \
      org.opencontainers.image.version="${IMX_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        rdma-core ibverbs-providers ibverbs-utils \
    && rm -rf /var/lib/apt/lists/*

ENV IMX_WHEELS=/opt/dectris/stanford-dectris-nvidia-4d-stem/wheels
COPY --from=build /dist/*.whl ${IMX_WHEELS}/
COPY --from=build /dist/bin/imxrx_loopback /usr/local/bin/

RUN python3 -m pip install --no-cache-dir ${IMX_WHEELS}/imx_receiver-*.whl

ENV PYTHONDONTWRITEBYTECODE=1



FROM imx-base AS receiver

RUN python3 -m pip install --no-cache-dir pyyaml

WORKDIR /opt/dectris/stanford-dectris-nvidia-4d-stem
COPY examples ./examples

CMD ["python3", "examples/pipeline/app.py", "--help"]
