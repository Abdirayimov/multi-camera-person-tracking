#pragma once

#include <memory>
#include <vector>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/tracker/track.hpp"

namespace mc_tracking::tracker {

/// Common interface every single-camera tracker exposes.
///
/// `update()` is called once per frame with the latest detection set
/// and returns the snapshot of currently active tracks (state ==
/// Confirmed). Implementations own their own internal state, including
/// any motion model and association logic.
class ITracker {
public:
    virtual ~ITracker() = default;

    /// Process one frame. The returned vector references tracks owned
    /// by the implementation; do not mutate them after the call.
    virtual std::vector<Track> update(const std::vector<Detection>& detections) = 0;

    /// Reset all internal state (drop every track). Called when the
    /// upstream pipeline restarts or seeks.
    virtual void reset() = 0;
};

/// Construct the tracker requested by `cfg`. NvDCF is only available
/// when the project was built with BUILD_DEEPSTREAM=ON; otherwise
/// requesting it throws std::runtime_error.
std::unique_ptr<ITracker> make_tracker(const config::TrackerConfig& cfg);

}  // namespace mc_tracking::tracker
