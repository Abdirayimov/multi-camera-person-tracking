#!/usr/bin/env bash
set -euo pipefail

ENGINES_DIR="${ENGINES_DIR:-/app/models/engines}"
ONNX_DIR="${ONNX_DIR:-/app/models/onnx}"

build_engine() {
    local name="$1"
    local extra="${2:-}"
    if [[ ! -f "${ENGINES_DIR}/${name}_fp16.engine" && -f "${ONNX_DIR}/${name}.onnx" ]]; then
        echo "[entrypoint] building ${name} engine..."
        mkdir -p "${ENGINES_DIR}"
        # shellcheck disable=SC2086
        trtexec \
            --onnx="${ONNX_DIR}/${name}.onnx" \
            --saveEngine="${ENGINES_DIR}/${name}_fp16.engine" \
            --fp16 \
            --workspace=4096 \
            ${extra}
    fi
}

build_engine yolov8s_person
build_engine osnet_x0_25 "--minShapes=input:1x3x256x128 --optShapes=input:8x3x256x128 --maxShapes=input:16x3x256x128"

mkdir -p "$(dirname "${OUTPUT_DIR:-/app/data/out}")" 2>/dev/null || true
exec mc_tracking_video "$@"
