#include "mc_tracking/overlay/visualizer.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdio>
#include <string>

namespace mc_tracking::overlay {

namespace {

cv::Scalar color_for(std::uint64_t id) {
    static const std::array<cv::Scalar, 8> palette{
        cv::Scalar(58, 184, 255),   cv::Scalar(99, 211, 142),
        cv::Scalar(244, 173, 66),   cv::Scalar(120, 105, 245),
        cv::Scalar(72, 207, 235),   cv::Scalar(255, 95, 128),
        cv::Scalar(180, 220, 100),  cv::Scalar(96, 180, 200),
    };
    return palette[id % palette.size()];
}

}  // namespace

void Visualizer::render(cv::Mat& frame, const pipeline::CameraFrameResult& result) const {
    for (const auto& t : result.tracks) {
        const std::uint64_t id_for_color = t.global_id.value_or(t.local_id);
        const cv::Scalar color = color_for(id_for_color);

        cv::rectangle(frame, t.bbox, color, 2);

        std::string label;
        if (t.global_id.has_value()) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "G%lu (L%lu)",
                          static_cast<unsigned long>(*t.global_id),
                          static_cast<unsigned long>(t.local_id));
            label = buf;
        } else {
            label = "L" + std::to_string(t.local_id);
        }
        char conf[16];
        std::snprintf(conf, sizeof(conf), "  %.2f", static_cast<double>(t.confidence));
        label += conf;

        const cv::Point origin(static_cast<int>(t.bbox.x),
                               std::max(0, static_cast<int>(t.bbox.y) - 8));
        cv::putText(frame, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

}  // namespace mc_tracking::overlay
