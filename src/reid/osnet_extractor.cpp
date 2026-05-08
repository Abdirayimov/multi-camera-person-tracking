#include "mc_tracking/reid/osnet_extractor.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "mc_tracking/trt/trt_engine.hpp"
#include "mc_tracking/utils/cuda_helpers.hpp"

namespace mc_tracking::reid {

namespace {

constexpr float kImageMean[3] = {123.675f, 116.28f, 103.53f};
constexpr float kImageStd[3] = {58.395f, 57.12f, 57.375f};

void preprocess(const cv::Mat& crop, int out_w, int out_h, float* dst) {
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
    const int channel_stride = out_w * out_h;
    for (int y = 0; y < out_h; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < out_w; ++x) {
            const auto& px = row[x];
            const int idx = y * out_w + x;
            const float r = static_cast<float>(px[2]);
            const float g = static_cast<float>(px[1]);
            const float b = static_cast<float>(px[0]);
            dst[0 * channel_stride + idx] = (r - kImageMean[0]) / kImageStd[0];
            dst[1 * channel_stride + idx] = (g - kImageMean[1]) / kImageStd[1];
            dst[2 * channel_stride + idx] = (b - kImageMean[2]) / kImageStd[2];
        }
    }
}

void l2_normalize(Embedding& v) {
    const float n = v.norm();
    if (n > 1e-12f) v /= n;
}

}  // namespace

OSNetExtractor::OSNetExtractor(const config::ReidConfig& cfg)
    : cfg_(cfg), engine_(std::make_unique<trt::TrtEngine>(cfg.engine_path)) {
    const std::size_t per_img = static_cast<std::size_t>(3) * cfg.input_height * cfg.input_width;
    input_scratch_.resize(cfg.batch_size * per_img);
    output_scratch_.resize(static_cast<std::size_t>(cfg.batch_size) * cfg.embedding_dim);
}

OSNetExtractor::~OSNetExtractor() = default;

std::vector<Embedding> OSNetExtractor::extract(const std::vector<cv::Mat>& crops) {
    std::vector<Embedding> out;
    out.reserve(crops.size());
    if (crops.empty()) return out;
    const std::size_t bsz = cfg_.batch_size;
    for (std::size_t i = 0; i < crops.size(); i += bsz) {
        const std::size_t end = std::min(i + bsz, crops.size());
        std::vector<cv::Mat> chunk(crops.begin() + static_cast<std::ptrdiff_t>(i),
                                   crops.begin() + static_cast<std::ptrdiff_t>(end));
        run_chunk_(chunk, out);
    }
    return out;
}

void OSNetExtractor::run_chunk_(const std::vector<cv::Mat>& chunk,
                                std::vector<Embedding>& out) {
    if (chunk.empty()) return;
    const std::int64_t bsz = static_cast<std::int64_t>(chunk.size());
    const std::int64_t in_w = static_cast<std::int64_t>(cfg_.input_width);
    const std::int64_t in_h = static_cast<std::int64_t>(cfg_.input_height);
    const std::int64_t emb = static_cast<std::int64_t>(cfg_.embedding_dim);

    const std::string input_name = engine_->bindings().front().name;
    std::string out_name;
    for (const auto& b : engine_->bindings()) {
        if (!b.is_input) {
            out_name = b.name;
            break;
        }
    }
    if (out_name.empty()) throw std::runtime_error("OSNet engine has no output binding");

    engine_->set_input_shape(input_name, {bsz, 3, in_h, in_w});

    const std::size_t per_img = static_cast<std::size_t>(3 * in_w * in_h);
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        if (chunk[i].empty()) {
            std::memset(input_scratch_.data() + i * per_img, 0, per_img * sizeof(float));
            continue;
        }
        preprocess(chunk[i], static_cast<int>(in_w), static_cast<int>(in_h),
                   input_scratch_.data() + i * per_img);
    }

    utils::CudaStream stream;
    engine_->copy_input(input_name, input_scratch_.data(),
                        chunk.size() * per_img * sizeof(float), stream.get());
    engine_->infer(stream.get());
    engine_->copy_output(out_name, output_scratch_.data(),
                         chunk.size() * static_cast<std::size_t>(emb) * sizeof(float),
                         stream.get());
    stream.synchronize();

    for (std::size_t i = 0; i < chunk.size(); ++i) {
        Embedding v(emb);
        std::memcpy(v.data(), output_scratch_.data() + i * static_cast<std::size_t>(emb),
                    static_cast<std::size_t>(emb) * sizeof(float));
        l2_normalize(v);
        out.push_back(std::move(v));
    }
}

std::vector<cv::Mat> crop_persons(const cv::Mat& frame,
                                  const std::vector<cv::Rect2f>& bboxes) {
    std::vector<cv::Mat> out(bboxes.size());
    for (std::size_t i = 0; i < bboxes.size(); ++i) {
        const auto& b = bboxes[i];
        const int x1 = std::max(0, static_cast<int>(std::floor(b.x)));
        const int y1 = std::max(0, static_cast<int>(std::floor(b.y)));
        const int x2 = std::min(frame.cols, static_cast<int>(std::ceil(b.x + b.width)));
        const int y2 = std::min(frame.rows, static_cast<int>(std::ceil(b.y + b.height)));
        if (x2 <= x1 || y2 <= y1) continue;
        out[i] = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    }
    return out;
}

}  // namespace mc_tracking::reid
