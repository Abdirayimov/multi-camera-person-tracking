#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/tracker/track.hpp"

namespace mc_tracking::trt {

class TrtEngine;

/// YOLOv8 person detector. Returns mc_tracking::tracker::Detection
/// objects directly (filtered to the configured person class), so it
/// drops straight into the tracker pipeline.
class YOLOv8Detector {
public:
    explicit YOLOv8Detector(const config::DetectionConfig& cfg);
    ~YOLOv8Detector();

    YOLOv8Detector(const YOLOv8Detector&) = delete;
    YOLOv8Detector& operator=(const YOLOv8Detector&) = delete;

    std::vector<tracker::Detection> detect(const cv::Mat& image);

    const config::DetectionConfig& config() const noexcept { return cfg_; }

private:
    config::DetectionConfig cfg_;
    std::unique_ptr<TrtEngine> engine_;
    std::vector<float> input_scratch_;
};

}  // namespace mc_tracking::trt
