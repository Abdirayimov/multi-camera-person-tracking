#include "mc_tracking/tracker/nvdcf_tracker.hpp"

#include <utility>

#include "mc_tracking/utils/logger.hpp"

namespace mc_tracking::tracker {

struct NvDcfTracker::Impl {
    config::NvDcfParams params;
    std::vector<Track> latest;  ///< most recent ingest from DeepStream
};

NvDcfTracker::NvDcfTracker(const config::NvDcfParams& params)
    : impl_(std::make_unique<Impl>()) {
    impl_->params = params;
    MCT_LOG_INFO("NvDcfTracker: this backend is a thin wrapper over DeepStream's "
                 "libnvds_nvmultiobjecttracker.so; use the DeepStream pipeline path "
                 "for actual tracker behaviour. config_path={}",
                 params.config_path);
}

NvDcfTracker::~NvDcfTracker() = default;

void NvDcfTracker::ingest_tracker_results(std::vector<Track> tracks) {
    impl_->latest = std::move(tracks);
}

std::vector<Track> NvDcfTracker::update(const std::vector<Detection>& /*detections*/) {
    // The DeepStream pipeline owns the actual NvDCF instance; this
    // method just returns whatever the probe most recently ingested.
    return impl_->latest;
}

void NvDcfTracker::reset() { impl_->latest.clear(); }

}  // namespace mc_tracking::tracker
