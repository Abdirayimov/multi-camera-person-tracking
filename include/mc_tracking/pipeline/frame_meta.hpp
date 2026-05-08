#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "mc_tracking/tracker/track.hpp"

namespace mc_tracking::pipeline {

using TimePoint = std::chrono::steady_clock::time_point;

/// Output of a single camera's pipeline at one frame.
struct CameraFrameResult {
    std::string camera_id;
    std::uint64_t frame_number = 0;
    TimePoint pts;
    std::vector<tracker::Track> tracks;  ///< local_id assigned; global_id filled by orchestrator
};

}  // namespace mc_tracking::pipeline
