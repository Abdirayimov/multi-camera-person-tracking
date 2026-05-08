#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/tracker/kalman_filter.hpp"
#include "mc_tracking/tracker/tracker_iface.hpp"

namespace mc_tracking::tracker {

/// One Kalman-filtered track inside the BYTETrack state machine.
struct ByteTrackState {
    std::uint64_t local_id = 0;
    cv::Rect2f bbox;
    float confidence = 0.0f;
    TrackState state = TrackState::Tentative;

    KalmanFilter kalman;

    std::uint32_t hit_streak = 0;
    std::uint32_t time_since_update = 0;
    std::uint32_t age = 0;
    std::uint32_t start_frame = 0;
};

/// BYTETrack (Zhang et al., ECCV 2022).
///
/// The interesting idea is the two-stage association cascade: high-
/// confidence detections are matched to active tracks first, then the
/// *low*-confidence detections are matched against the still-unmatched
/// tracks. This recovers a lot of the IDs that single-stage trackers
/// drop under partial occlusion.
class ByteTrack final : public ITracker {
public:
    explicit ByteTrack(const config::ByteTrackParams& params);

    std::vector<Track> update(const std::vector<Detection>& detections) override;
    void reset() override;

private:
    config::ByteTrackParams params_;
    std::vector<std::unique_ptr<ByteTrackState>> tracked_;       ///< Confirmed active tracks
    std::vector<std::unique_ptr<ByteTrackState>> lost_;          ///< Awaiting reacquisition
    std::vector<std::unique_ptr<ByteTrackState>> removed_;       ///< Past max_age, kept for one frame
    std::uint64_t next_id_ = 1;
    std::uint32_t frame_id_ = 0;

    void predict_all_();
    void prune_aspect_ratio_(std::vector<Detection>& detections) const;
};

}  // namespace mc_tracking::tracker
