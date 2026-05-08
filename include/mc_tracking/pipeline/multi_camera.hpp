#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/crosscam/identity_matcher.hpp"
#include "mc_tracking/pipeline/single_camera.hpp"

namespace mc_tracking::pipeline {

/// Synchronously drives every camera one frame at a time, collects
/// each camera's confirmed tracks, runs the cross-camera matcher,
/// and stamps the resulting global ids back onto the per-camera
/// results. The DeepStream variant runs each camera on its own
/// GLib loop instead; this class is the OpenCV-only path.
class MultiCameraOrchestrator {
public:
    MultiCameraOrchestrator(const config::SystemConfig& cfg,
                            const config::CamerasConfig& cameras);

    /// Add a camera; returns the SingleCameraPipeline so callers can
    /// feed frames into it directly.
    SingleCameraPipeline& add_camera(const std::string& id, const std::string& zone);

    /// Stamp global ids on `frames` based on the cross-camera matcher.
    /// `frames` is mutated in place (each Track::global_id is filled).
    void stamp_global_ids(std::vector<CameraFrameResult>& frames);

    std::size_t num_cameras() const noexcept { return cameras_.size(); }

private:
    config::SystemConfig cfg_;
    config::CamerasConfig cameras_cfg_;
    std::vector<std::unique_ptr<SingleCameraPipeline>> cameras_;
    std::unique_ptr<crosscam::IdentityMatcher> matcher_;
};

}  // namespace mc_tracking::pipeline
