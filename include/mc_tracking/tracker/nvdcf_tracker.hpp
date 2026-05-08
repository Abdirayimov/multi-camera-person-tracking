#pragma once

#include <vector>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/tracker/tracker_iface.hpp"

namespace mc_tracking::tracker {

/// NvDCF wrapper. Honours the same interface as the pure-C++ trackers
/// but delegates to NVIDIA's libnvds_nvmultiobjecttracker.so under the
/// hood. Useful when running through the DeepStream pipeline path
/// where the tracker is already attached as a GStreamer element; this
/// class is a no-op shell that forwards detections it has been fed by
/// the pipeline's src-pad probe.
///
/// Available only when the project is built with BUILD_DEEPSTREAM=ON.
class NvDcfTracker final : public ITracker {
public:
    explicit NvDcfTracker(const config::NvDcfParams& params);
    ~NvDcfTracker() override;

    std::vector<Track> update(const std::vector<Detection>& detections) override;
    void reset() override;

    /// Manual override used by the DeepStream probe: it has already
    /// observed track ids assigned by NvDCF and feeds them in.
    void ingest_tracker_results(std::vector<Track> tracks);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mc_tracking::tracker
