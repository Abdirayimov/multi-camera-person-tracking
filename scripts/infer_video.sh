#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:?usage: infer_video.sh INPUT_VIDEO OUTPUT_VIDEO}"
OUTPUT="${2:?usage: infer_video.sh INPUT_VIDEO OUTPUT_VIDEO}"
"${ROOT}/build/mc_tracking_video" \
    --config "${ROOT}/configs/system_config.yaml" \
    --input "${INPUT}" --output "${OUTPUT}"
