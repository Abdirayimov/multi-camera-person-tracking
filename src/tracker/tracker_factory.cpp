#include <stdexcept>

#include "mc_tracking/tracker/bytetrack.hpp"
#include "mc_tracking/tracker/iou_tracker.hpp"
#include "mc_tracking/tracker/tracker_iface.hpp"

namespace mc_tracking::tracker {

std::unique_ptr<ITracker> make_tracker(const config::TrackerConfig& cfg) {
    switch (cfg.type) {
        case config::TrackerType::ByteTrack:
            return std::make_unique<ByteTrack>(cfg.bytetrack);
        case config::TrackerType::Iou:
            return std::make_unique<IouTracker>(cfg.iou);
        case config::TrackerType::NvDcf:
            // NvDCF lives inside DeepStream's nvtracker element; it cannot be
            // hosted by this in-process driver. The DeepStream pipeline that
            // runs it is mc_tracking_ds (BUILD_DEEPSTREAM=ON), which delivers
            // NvDCF tracks through its probe callback instead of ITracker.
            throw std::runtime_error(
                "the NvDCF tracker runs inside DeepStream; use the mc_tracking_ds binary "
                "(BUILD_DEEPSTREAM=ON), or bytetrack/iou for this in-process driver");
    }
    throw std::runtime_error("unhandled tracker type");
}

}  // namespace mc_tracking::tracker
