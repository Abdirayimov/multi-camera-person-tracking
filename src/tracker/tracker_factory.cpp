#include <stdexcept>

#include "mc_tracking/tracker/bytetrack.hpp"
#include "mc_tracking/tracker/iou_tracker.hpp"
#include "mc_tracking/tracker/tracker_iface.hpp"

#ifdef MCT_HAVE_DEEPSTREAM
#include "mc_tracking/tracker/nvdcf_tracker.hpp"
#endif

namespace mc_tracking::tracker {

std::unique_ptr<ITracker> make_tracker(const config::TrackerConfig& cfg) {
    switch (cfg.type) {
        case config::TrackerType::ByteTrack:
            return std::make_unique<ByteTrack>(cfg.bytetrack);
        case config::TrackerType::Iou:
            return std::make_unique<IouTracker>(cfg.iou);
        case config::TrackerType::NvDcf:
#ifdef MCT_HAVE_DEEPSTREAM
            return std::make_unique<NvDcfTracker>(cfg.nvdcf);
#else
            throw std::runtime_error(
                "NvDCF tracker requested but project was built without DeepStream support");
#endif
    }
    throw std::runtime_error("unhandled tracker type");
}

}  // namespace mc_tracking::tracker
