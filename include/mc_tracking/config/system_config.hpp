#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mc_tracking::config {

struct PipelineConfig {
    std::uint32_t muxer_width = 1280;
    std::uint32_t muxer_height = 720;
    std::uint32_t batch_size = 1;
    bool emit_overlay = true;
};

struct DetectionConfig {
    std::string engine_path;
    std::uint32_t input_width = 640;
    std::uint32_t input_height = 640;
    float confidence_threshold = 0.4f;
    float nms_iou_threshold = 0.5f;
    std::int32_t person_class_id = 0;
};

enum class TrackerType {
    ByteTrack,
    Iou,
    NvDcf,
};

struct ByteTrackParams {
    float high_thresh = 0.5f;
    float low_thresh = 0.1f;
    float new_track_thresh = 0.6f;
    std::uint32_t track_buffer = 30;
    float match_thresh = 0.8f;
    float aspect_ratio_thresh = 1.6f;
};

struct IouParams {
    float iou_thresh = 0.3f;
    std::uint32_t max_age = 30;
    std::uint32_t min_hits = 3;
};

struct NvDcfParams {
    std::string config_path;
    std::uint32_t tracker_width = 640;
    std::uint32_t tracker_height = 384;
};

struct TrackerConfig {
    TrackerType type = TrackerType::ByteTrack;
    ByteTrackParams bytetrack;
    IouParams iou;
    NvDcfParams nvdcf;
};

struct ReidConfig {
    bool enabled = true;
    std::string engine_path;
    std::uint32_t input_width = 128;
    std::uint32_t input_height = 256;
    std::uint32_t embedding_dim = 256;
    std::uint32_t batch_size = 16;
    std::uint32_t gallery_size_per_track = 8;
};

struct CrossCamConfig {
    bool enabled = true;
    float reid_threshold = 0.7f;
    std::uint32_t spatial_overlap_window_ms = 5000;
    float hungarian_cost_cap = 0.4f;
};

struct LoggingConfig {
    std::string level = "info";
    bool json = true;
};

struct SystemConfig {
    PipelineConfig pipeline;
    DetectionConfig detection;
    TrackerConfig tracker;
    ReidConfig reid;
    CrossCamConfig crosscam;
    LoggingConfig logging;

    static SystemConfig load(const std::string& yaml_path);
};

struct CameraEntry {
    std::string id;
    std::string uri;
    std::string zone;
};

struct ZoneTransition {
    std::string from;
    std::string to;
};

struct CamerasConfig {
    std::vector<CameraEntry> cameras;
    std::vector<ZoneTransition> transitions;

    /// True when no transitions are declared, i.e. any-to-any pairing is
    /// allowed by the matcher.
    bool any_transition_allowed() const noexcept { return transitions.empty(); }

    /// True when a person is allowed to cross from `from_zone` into
    /// `to_zone`, given the topology declared in cameras.yaml.
    bool transition_allowed(const std::string& from_zone, const std::string& to_zone) const;

    static CamerasConfig load(const std::string& yaml_path);
};

}  // namespace mc_tracking::config
