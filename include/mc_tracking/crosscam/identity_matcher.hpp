#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/reid/osnet_extractor.hpp"
#include "mc_tracking/reid/reid_gallery.hpp"
#include "mc_tracking/tracker/track.hpp"

namespace mc_tracking::crosscam {

/// One snapshot of a track on a particular camera at a particular
/// instant. Fed into the matcher every frame.
struct CameraTrackObservation {
    std::string camera_id;
    std::string zone;
    std::uint64_t local_id = 0;
    std::chrono::steady_clock::time_point pts;
    cv::Rect2f bbox;
    /// Most recent appearance feature; the matcher also queries the
    /// per-track ReidGallery for its best historical match.
    reid::Embedding embedding;
};

/// Cross-camera identity matcher.
///
/// On each call to `update`, the matcher receives every camera's
/// confirmed tracks for the current frame. It produces a stable
/// global id per physical person:
///
///   1. Tracks already mapped to a global id keep that id.
///   2. New tracks are scored against active global ids using
///      ReID cosine similarity, gated by:
///        - The cameras-zone transition table (if declared)
///        - A spatial-overlap window (a person can't be in two
///          places that are far apart at the same instant unless
///          the cameras observe overlapping fields of view)
///   3. Hungarian assignment picks the optimal pairing under the
///      hungarian_cost_cap; unassigned new tracks get fresh ids.
class IdentityMatcher {
public:
    IdentityMatcher(const config::CrossCamConfig& cfg, const config::CamerasConfig& cameras);

    /// Inject the per-track ReidGallery for a camera so the matcher
    /// can query historical embeddings instead of just the latest.
    /// Optional; if not provided, only the embedding on the
    /// observation is used.
    void register_gallery(const std::string& camera_id, const reid::ReidGallery* gallery);

    /// Returns the global id assigned to each observation, in the
    /// same order. The matcher updates its internal state so
    /// subsequent calls are stable.
    std::vector<std::uint64_t> update(const std::vector<CameraTrackObservation>& observations);

    /// Total number of unique global ids that have ever been assigned.
    std::uint64_t total_global_ids() const noexcept { return next_global_id_ - 1; }

private:
    config::CrossCamConfig cfg_;
    config::CamerasConfig cameras_;

    struct GlobalRecord {
        std::uint64_t global_id = 0;
        std::string last_zone;
        std::chrono::steady_clock::time_point last_seen;
        reid::Embedding canonical_embedding;
    };

    std::unordered_map<std::uint64_t, GlobalRecord> globals_;

    /// (camera_id, local_id) -> global_id; established assignments.
    std::unordered_map<std::string, std::unordered_map<std::uint64_t, std::uint64_t>>
        camera_to_global_;

    std::unordered_map<std::string, const reid::ReidGallery*> galleries_;

    std::uint64_t next_global_id_ = 1;

    bool transition_allowed_(const std::string& from_zone, const std::string& to_zone) const;
};

}  // namespace mc_tracking::crosscam
