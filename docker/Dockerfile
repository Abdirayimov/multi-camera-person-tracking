# syntax=docker/dockerfile:1.6
ARG DS_VERSION=8.0
ARG DS_BASE=nvcr.io/nvidia/deepstream:${DS_VERSION}-gc-triton-devel

# ----- builder -----
FROM ${DS_BASE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build pkg-config \
        libeigen3-dev libspdlog-dev libyaml-cpp-dev \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake/ ./cmake/
COPY include/ ./include/
COPY src/ ./src/
COPY tools/ ./tools/

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TOOLS=ON \
        -DBUILD_TESTS=OFF \
    && cmake --build build --parallel \
    && cmake --install build --prefix /opt/mc_tracking

# ----- runtime -----
FROM ${DS_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libspdlog1 libyaml-cpp0.7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/mc_tracking /opt/mc_tracking
COPY configs/ /app/configs/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PATH="/opt/mc_tracking/bin:${PATH}"
WORKDIR /app

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--config", "/app/configs/system_config.yaml", \
     "--cameras", "/app/configs/cameras.yaml", \
     "--output-dir", "/app/data/out"]
