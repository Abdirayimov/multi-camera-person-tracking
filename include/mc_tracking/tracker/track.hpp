#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <cstdint>
#include <deque>
#include <optional>
#include <string>

namespace mc_tracking::tracker {

enum class TrackState {
    Tentative,   ///< First few observations; not yet reported to consumers
    Confirmed,   ///< Active and visible
    Lost,        ///< Detection missed for >= 1 frame; awaiting reacquisition
    Removed,     ///< Past max_age; ready for eviction
};

inline const char* track_state_name(TrackState s) noexcept {
    switch (s) {
        case TrackState::Tentative: return "tentative";
        case TrackState::Confirmed: return "confirmed";
        case TrackState::Lost:      return "lost";
        case TrackState::Removed:   return "removed";
    }
    return "?";
}

/// One detection observation that the tracker considers for association.
struct Detection {
    cv::Rect2f bbox;
    float score = 0.0f;
};

/// A single tracked entity.
///
/// `local_id` is unique within a single camera's tracker instance.
/// `global_id` is assigned by the cross-camera matcher and is shared
/// across cameras for the same physical person; it is std::nullopt
/// until the matcher links the track.
struct Track {
    std::uint64_t local_id = 0;
    std::optional<std::uint64_t> global_id;
    std::string camera_id;

    cv::Rect2f bbox;
    float confidence = 0.0f;
    TrackState state = TrackState::Tentative;

    std::uint32_t hit_streak = 0;     ///< Consecutive frames with a matching detection
    std::uint32_t time_since_update = 0;
    std::uint32_t age = 0;            ///< Total frames since track was created

    /// Most recent appearance feature plus a small rolling bank used by
    /// cross-camera matching. The newest is at the back.
    std::deque<Eigen::VectorXf> appearance_gallery;
};

}  // namespace mc_tracking::tracker
