// Custom gst-nvinfer bbox parser for the Ultralytics detection head.
//
// YOLOv8 / YOLOv9 / YOLO11 export a single `1 x 84 x N` tensor: rows 0-3
// are cx, cy, w, h in network-input pixels, rows 4-83 are per-class
// scores. DeepStream ships no parser for this layout, so the pgie config
// points `custom-lib-path` at this library and `parse-bbox-func-name` at
// the function below. Only the person class (0) is emitted; nvinfer's
// cluster-mode=2 does the NMS afterwards.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "nvdsinfer_custom_impl.h"

// The project builds with -fvisibility=hidden; gst-nvinfer finds this
// entry point with dlsym, so it must be exported explicitly.
#define MC_EXPORT __attribute__((visibility("default")))

extern "C" MC_EXPORT bool NvDsInferParseCustomYoloUltralytics(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

namespace {

constexpr unsigned int kPersonClass = 0;
constexpr unsigned int kBboxRows = 4;

float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

}  // namespace

extern "C" MC_EXPORT bool NvDsInferParseCustomYoloUltralytics(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList) {
    if (outputLayersInfo.empty()) {
        return false;
    }
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];

    // Accept [84, N] or [1, 84, N]: TensorRT may or may not keep the
    // leading batch-1 dimension in the per-buffer dims.
    const NvDsInferDims& dims = layer.inferDims;
    unsigned int rows = 0;
    unsigned int n = 0;
    if (dims.numDims == 2) {
        rows = dims.d[0];
        n = dims.d[1];
    } else if (dims.numDims == 3 && dims.d[0] == 1) {
        rows = dims.d[1];
        n = dims.d[2];
    } else {
        return false;
    }
    if (rows <= kBboxRows || n == 0 || layer.buffer == nullptr) {
        return false;
    }

    const float threshold = detectionParams.perClassPreclusterThreshold.empty()
                                ? 0.25F
                                : detectionParams.perClassPreclusterThreshold[kPersonClass];
    const auto* data = static_cast<const float*>(layer.buffer);
    const auto net_w = static_cast<float>(networkInfo.width);
    const auto net_h = static_cast<float>(networkInfo.height);

    for (unsigned int i = 0; i < n; ++i) {
        const float score = data[(kBboxRows + kPersonClass) * n + i];
        if (score < threshold) {
            continue;
        }
        const float cx = data[0 * n + i];
        const float cy = data[1 * n + i];
        const float w = data[2 * n + i];
        const float h = data[3 * n + i];

        NvDsInferParseObjectInfo obj{};
        obj.classId = kPersonClass;
        obj.detectionConfidence = score;
        obj.left = clampf(cx - w / 2.0F, 0.0F, net_w);
        obj.top = clampf(cy - h / 2.0F, 0.0F, net_h);
        obj.width = clampf(w, 0.0F, net_w - obj.left);
        obj.height = clampf(h, 0.0F, net_h - obj.top);
        if (obj.width <= 0.0F || obj.height <= 0.0F) {
            continue;
        }
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloUltralytics);
