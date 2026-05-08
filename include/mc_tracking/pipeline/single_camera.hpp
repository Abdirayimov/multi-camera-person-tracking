#pragma once

#include <memory>
#include <string>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/pipeline/frame_meta.hpp"
#include "mc_tracking/reid/osnet_extractor.hpp"
#include "mc_tracking/reid/reid_gallery.hpp"
#include "mc_tracking/tracker/tracker_iface.hpp"
#include "mc_tracking/trt/yolov8_detector.hpp"

namespace mc_tracking::pipeline {

/// Detector + tracker + (optional) ReID per camera, sequential
/// per-frame execution.
class SingleCameraPipeline {
public:
    SingleCameraPipeline(std::string camera_id, std::string zone, const config::SystemConfig& cfg);

    CameraFrameResult process_frame(std::uint64_t frame_number, TimePoint pts,
                                    const cv::Mat& frame);

    const std::string& camera_id() const noexcept { return camera_id_; }
    const std::string& zone() const noexcept { return zone_; }
    const reid::ReidGallery* gallery() const noexcept { return gallery_.get(); }

private:
    std::string camera_id_;
    std::string zone_;
    config::SystemConfig cfg_;

    std::unique_ptr<trt::YOLOv8Detector> detector_;
    std::unique_ptr<tracker::ITracker> tracker_;
    std::unique_ptr<reid::OSNetExtractor> reid_;
    std::unique_ptr<reid::ReidGallery> gallery_;
};

}  // namespace mc_tracking::pipeline
