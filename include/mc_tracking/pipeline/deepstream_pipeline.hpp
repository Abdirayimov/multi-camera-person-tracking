#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/pipeline/frame_meta.hpp"

namespace mc_tracking::pipeline {

/// Fired from the tracker's src-pad probe with one camera's NvDCF tracks
/// for one frame. Runs on a GStreamer streaming thread owned by the
/// pipeline: copy out what you need and return quickly — blocking here
/// throttles decode upstream.
using TrackBatchCallback = std::function<void(const CameraFrameResult&)>;

/// Multi-source DeepStream pipeline with NvDCF tracking:
///
///     nvurisrcbin (xN) -> nvstreammux -> nvinfer (YOLO person)
///                      -> nvtracker (NvDCF) -> fakesink
///
/// A buffer probe on nvtracker's src pad walks the batch metadata and
/// converts each frame's `NvDsObjectMeta` into `tracker::Track` values —
/// `object_id` is NvDCF's per-stream stable track id — then hands each
/// source's per-frame batch to the registered callback.
///
/// Scope, stated plainly: this driver gives you *per-camera* NvDCF tracks.
/// It does not run cross-camera identity matching, because the matcher
/// scores ReID embeddings and nothing in this pipeline extracts them yet
/// (an OSNet secondary GIE is the roadmap entry). Consumers that want
/// global ids can feed these tracks plus their own embeddings into
/// `crosscam::IdentityMatcher`.
class DeepStreamPipeline {
public:
    /// Muxer geometry and batching come from `cfg.pipeline`; the NvDCF
    /// element is configured from `cfg.tracker.nvdcf` (config_path,
    /// tracker_width, tracker_height).
    ///
    /// @throws std::runtime_error when an element cannot be created or linked.
    DeepStreamPipeline(const config::SystemConfig& cfg, const std::string& pgie_config_path);
    ~DeepStreamPipeline();

    DeepStreamPipeline(const DeepStreamPipeline&) = delete;
    DeepStreamPipeline& operator=(const DeepStreamPipeline&) = delete;

    /// Add a source before `start()`. Returns false (and logs why) on a
    /// duplicate id or a source the pipeline cannot wire up.
    bool add_source(const std::string& camera_id, const std::string& uri);

    /// Register the consumer of per-frame track batches. Call before
    /// `start()`; the probe reads it without further synchronisation.
    void set_track_callback(TrackBatchCallback cb);

    void start();
    void wait();  ///< Block until EOS on every source or `stop()`.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    config::SystemConfig cfg_;
    std::atomic<bool> running_{false};

    std::mutex sources_mutex_;
    std::unordered_map<std::string, int> sources_;  ///< camera_id -> muxer pad index
};

}  // namespace mc_tracking::pipeline
