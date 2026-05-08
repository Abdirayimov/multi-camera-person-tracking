#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_DIR="${ROOT}/models/onnx"
ENG_DIR="${ROOT}/models/engines"
mkdir -p "${ENG_DIR}"

if [[ -f "${ONNX_DIR}/yolov8s.onnx" ]]; then
    echo "building YOLOv8s engine..."
    trtexec --onnx="${ONNX_DIR}/yolov8s.onnx" \
        --saveEngine="${ENG_DIR}/yolov8s_person_fp16.engine" \
        --fp16 --workspace=4096
fi

if [[ -f "${ONNX_DIR}/osnet_x0_25.onnx" ]]; then
    echo "building OSNet engine..."
    trtexec --onnx="${ONNX_DIR}/osnet_x0_25.onnx" \
        --saveEngine="${ENG_DIR}/osnet_x0_25_fp16.engine" \
        --fp16 \
        --minShapes=input:1x3x256x128 \
        --optShapes=input:8x3x256x128 \
        --maxShapes=input:16x3x256x128 \
        --workspace=4096
fi

ls -lh "${ENG_DIR}/"
