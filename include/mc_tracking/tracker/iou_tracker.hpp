#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/tracker/tracker_iface.hpp"

namespace mc_tracking::tracker {

/// Greedy-IoU tracker: for each frame, match the highest-IoU
/// detection to each existing track (above `iou_thresh`); unmatched
/// detections seed new tracks; tracks unseen for `max_age` frames are
/// dropped.
///
/// Useful as a baseline against BYTETrack: minimal and easy to reason
/// about, but no motion prediction so IoU drops in fast scenes.
class IouTracker final : public ITracker {
public:
    explicit IouTracker(const config::IouParams& params);

    std::vector<Track> update(const std::vector<Detection>& detections) override;
    void reset() override;

private:
    config::IouParams params_;
    std::unordered_map<std::uint64_t, Track> tracks_;
    std::uint64_t next_id_ = 1;
};

}  // namespace mc_tracking::tracker
