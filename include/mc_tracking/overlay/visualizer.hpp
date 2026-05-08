#pragma once

#include <opencv2/core.hpp>

#include "mc_tracking/pipeline/frame_meta.hpp"

namespace mc_tracking::overlay {

/// Draw bounding boxes + track labels on a frame.
///
/// Box color follows the global id when present (so the same physical
/// person gets a consistent color across cameras); falls back to
/// local_id otherwise.
class Visualizer {
public:
    void render(cv::Mat& frame, const pipeline::CameraFrameResult& result) const;
};

}  // namespace mc_tracking::overlay
