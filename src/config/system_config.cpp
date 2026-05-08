#include "mc_tracking/config/system_config.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>

namespace mc_tracking::config {

namespace {

template <typename T>
T require(const YAML::Node& node, const std::string& key) {
    if (!node[key]) throw std::runtime_error("missing required config key: " + key);
    return node[key].as<T>();
}

template <typename T>
T optional(const YAML::Node& node, const std::string& key, T fallback) {
    return node[key] ? node[key].as<T>() : fallback;
}

TrackerType parse_tracker(const std::string& s) {
    if (s == "bytetrack") return TrackerType::ByteTrack;
    if (s == "iou") return TrackerType::Iou;
    if (s == "nvdcf") return TrackerType::NvDcf;
    throw std::runtime_error("unknown tracker.type: " + s);
}

}  // namespace

SystemConfig SystemConfig::load(const std::string& yaml_path) {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    SystemConfig out;

    if (const auto p = root["pipeline"]; p) {
        out.pipeline.muxer_width = optional<std::uint32_t>(p, "muxer_width", 1280);
        out.pipeline.muxer_height = optional<std::uint32_t>(p, "muxer_height", 720);
        out.pipeline.batch_size = optional<std::uint32_t>(p, "batch_size", 1);
        out.pipeline.emit_overlay = optional<bool>(p, "emit_overlay", true);
    }

    if (const auto d = root["detection"]; d) {
        out.detection.engine_path = require<std::string>(d, "engine_path");
        out.detection.input_width = optional<std::uint32_t>(d, "input_width", 640);
        out.detection.input_height = optional<std::uint32_t>(d, "input_height", 640);
        out.detection.confidence_threshold = optional<float>(d, "confidence_threshold", 0.4f);
        out.detection.nms_iou_threshold = optional<float>(d, "nms_iou_threshold", 0.5f);
        out.detection.person_class_id = optional<std::int32_t>(d, "person_class_id", 0);
    } else {
        throw std::runtime_error("missing 'detection' section in config");
    }

    if (const auto t = root["tracker"]; t) {
        out.tracker.type = parse_tracker(optional<std::string>(t, "type", "bytetrack"));
        if (const auto bt = t["bytetrack"]; bt) {
            out.tracker.bytetrack.high_thresh = optional<float>(bt, "high_thresh", 0.5f);
            out.tracker.bytetrack.low_thresh = optional<float>(bt, "low_thresh", 0.1f);
            out.tracker.bytetrack.new_track_thresh =
                optional<float>(bt, "new_track_thresh", 0.6f);
            out.tracker.bytetrack.track_buffer =
                optional<std::uint32_t>(bt, "track_buffer", 30);
            out.tracker.bytetrack.match_thresh = optional<float>(bt, "match_thresh", 0.8f);
            out.tracker.bytetrack.aspect_ratio_thresh =
                optional<float>(bt, "aspect_ratio_thresh", 1.6f);
        }
        if (const auto io = t["iou"]; io) {
            out.tracker.iou.iou_thresh = optional<float>(io, "iou_thresh", 0.3f);
            out.tracker.iou.max_age = optional<std::uint32_t>(io, "max_age", 30);
            out.tracker.iou.min_hits = optional<std::uint32_t>(io, "min_hits", 3);
        }
        if (const auto nv = t["nvdcf"]; nv) {
            out.tracker.nvdcf.config_path = optional<std::string>(nv, "config_path", "");
            out.tracker.nvdcf.tracker_width =
                optional<std::uint32_t>(nv, "tracker_width", 640);
            out.tracker.nvdcf.tracker_height =
                optional<std::uint32_t>(nv, "tracker_height", 384);
        }
    }

    if (const auto r = root["reid"]; r) {
        out.reid.enabled = optional<bool>(r, "enabled", true);
        out.reid.engine_path = optional<std::string>(r, "engine_path", "");
        out.reid.input_width = optional<std::uint32_t>(r, "input_width", 128);
        out.reid.input_height = optional<std::uint32_t>(r, "input_height", 256);
        out.reid.embedding_dim = optional<std::uint32_t>(r, "embedding_dim", 256);
        out.reid.batch_size = optional<std::uint32_t>(r, "batch_size", 16);
        out.reid.gallery_size_per_track =
            optional<std::uint32_t>(r, "gallery_size_per_track", 8);
    }

    if (const auto c = root["crosscam"]; c) {
        out.crosscam.enabled = optional<bool>(c, "enabled", true);
        out.crosscam.reid_threshold = optional<float>(c, "reid_threshold", 0.7f);
        out.crosscam.spatial_overlap_window_ms =
            optional<std::uint32_t>(c, "spatial_overlap_window_ms", 5000);
        out.crosscam.hungarian_cost_cap = optional<float>(c, "hungarian_cost_cap", 0.4f);
    }

    if (const auto l = root["logging"]; l) {
        out.logging.level = optional<std::string>(l, "level", "info");
        out.logging.json = optional<bool>(l, "json", true);
    }
    return out;
}

bool CamerasConfig::transition_allowed(const std::string& from_zone,
                                       const std::string& to_zone) const {
    if (any_transition_allowed()) return true;
    for (const auto& tr : transitions) {
        if (tr.from == from_zone && tr.to == to_zone) return true;
    }
    return false;
}

CamerasConfig CamerasConfig::load(const std::string& yaml_path) {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    CamerasConfig out;
    if (const auto cams = root["cameras"]; cams && cams.IsSequence()) {
        for (const auto& c : cams) {
            CameraEntry e;
            e.id = require<std::string>(c, "id");
            e.uri = require<std::string>(c, "uri");
            e.zone = optional<std::string>(c, "zone", "");
            out.cameras.push_back(std::move(e));
        }
    }
    if (const auto trs = root["transitions"]; trs && trs.IsSequence()) {
        for (const auto& t : trs) {
            ZoneTransition z;
            z.from = require<std::string>(t, "from");
            z.to = require<std::string>(t, "to");
            out.transitions.push_back(std::move(z));
        }
    }
    return out;
}

}  // namespace mc_tracking::config
