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
            // NvDCF is not a tracker this project implements: it lives inside
            // DeepStream's libnvds_nvmultiobjecttracker.so and only produces
            // tracks when a DeepStream pipeline forwards them from a src-pad
            // probe. That pipeline is on the roadmap, not in the tree, so the
            // value is rejected in every build configuration rather than
            // returning a backend that would silently track nobody.
            throw std::runtime_error(
                "the NvDCF tracker requires a DeepStream pipeline that is not implemented in "
                "this reference; use bytetrack or iou");
    }
    throw std::runtime_error("unhandled tracker type");
}

}  // namespace mc_tracking::tracker
