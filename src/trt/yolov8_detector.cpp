#include "mc_tracking/trt/yolov8_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

#include "mc_tracking/trt/trt_engine.hpp"
#include "mc_tracking/utils/cuda_helpers.hpp"

namespace mc_tracking::trt {

namespace {

struct LetterboxResult {
    cv::Mat image;
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

LetterboxResult letterbox(const cv::Mat& src, int target_w, int target_h) {
    const float r = std::min(static_cast<float>(target_w) / static_cast<float>(src.cols),
                             static_cast<float>(target_h) / static_cast<float>(src.rows));
    const int new_w = static_cast<int>(std::round(static_cast<float>(src.cols) * r));
    const int new_h = static_cast<int>(std::round(static_cast<float>(src.rows) * r));
    const int pad_x = (target_w - new_w) / 2;
    const int pad_y = (target_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(target_h, target_w, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_x, pad_y, new_w, new_h)));
    return {out, r, pad_x, pad_y};
}

void hwc_bgr_to_chw_rgb_norm(const cv::Mat& src, float* dst) {
    const int H = src.rows;
    const int W = src.cols;
    const int channel_stride = H * W;
    for (int y = 0; y < H; ++y) {
        const auto* row = src.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            const auto& px = row[x];
            const int idx = y * W + x;
            dst[0 * channel_stride + idx] = static_cast<float>(px[2]) / 255.0f;
            dst[1 * channel_stride + idx] = static_cast<float>(px[1]) / 255.0f;
            dst[2 * channel_stride + idx] = static_cast<float>(px[0]) / 255.0f;
        }
    }
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float xx1 = std::max(a.x, b.x);
    const float yy1 = std::max(a.y, b.y);
    const float xx2 = std::min(a.x + a.width, b.x + b.width);
    const float yy2 = std::min(a.y + a.height, b.y + b.height);
    const float w = std::max(0.0f, xx2 - xx1);
    const float h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;
    const float u = a.area() + b.area() - inter;
    return (u > 0.0f) ? inter / u : 0.0f;
}

std::vector<tracker::Detection> nms(std::vector<tracker::Detection> dets, float thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const tracker::Detection& a, const tracker::Detection& b) {
                  return a.score > b.score;
              });
    std::vector<tracker::Detection> kept;
    std::vector<bool> sup(dets.size(), false);
    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (sup[i]) continue;
        kept.push_back(dets[i]);
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (sup[j]) continue;
            if (iou(dets[i].bbox, dets[j].bbox) > thresh) sup[j] = true;
        }
    }
    return kept;
}

}  // namespace

YOLOv8Detector::YOLOv8Detector(const config::DetectionConfig& cfg)
    : cfg_(cfg), engine_(std::make_unique<TrtEngine>(cfg.engine_path)) {
    input_scratch_.resize(static_cast<std::size_t>(3) * cfg.input_height * cfg.input_width);
}

YOLOv8Detector::~YOLOv8Detector() = default;

std::vector<tracker::Detection> YOLOv8Detector::detect(const cv::Mat& image) {
    const auto lb = letterbox(image, static_cast<int>(cfg_.input_width),
                              static_cast<int>(cfg_.input_height));
    hwc_bgr_to_chw_rgb_norm(lb.image, input_scratch_.data());

    const std::string input_name = engine_->bindings().front().name;
    utils::CudaStream stream;
    engine_->copy_input(input_name, input_scratch_.data(),
                        input_scratch_.size() * sizeof(float), stream.get());
    engine_->infer(stream.get());

    std::string out_name;
    for (const auto& b : engine_->bindings()) {
        if (!b.is_input) {
            out_name = b.name;
            break;
        }
    }
    const auto& ob = engine_->binding(out_name);
    std::vector<float> out(ob.volume);
    engine_->copy_output(out_name, out.data(), out.size() * sizeof(float), stream.get());
    stream.synchronize();

    constexpr int kClassCount = 80;
    const int n_anchors = static_cast<int>(ob.volume / (kClassCount + 4));
    std::vector<tracker::Detection> raw;
    for (int i = 0; i < n_anchors; ++i) {
        const float score = out[(4 + cfg_.person_class_id) * n_anchors + i];
        if (score < cfg_.confidence_threshold) continue;

        const float cx = out[0 * n_anchors + i];
        const float cy = out[1 * n_anchors + i];
        const float w = out[2 * n_anchors + i];
        const float h = out[3 * n_anchors + i];
        const float x1 = cx - w * 0.5f;
        const float y1 = cy - h * 0.5f;

        tracker::Detection d;
        d.score = score;
        const float ox = (x1 - static_cast<float>(lb.pad_x)) / lb.scale;
        const float oy = (y1 - static_cast<float>(lb.pad_y)) / lb.scale;
        d.bbox = cv::Rect2f(ox, oy, w / lb.scale, h / lb.scale);
        raw.push_back(d);
    }
    return nms(std::move(raw), cfg_.nms_iou_threshold);
}

}  // namespace mc_tracking::trt
