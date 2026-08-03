#include "mc_tracking/config/system_config.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>

namespace mc_tracking::config {

namespace {

void require_positive(std::uint32_t value, const char* key) {
    if (value == 0) {
        throw std::runtime_error(std::string("config: ") + key + " must be greater than zero");
    }
}

void require_in_range(float value, float lo, float hi, const char* key) {
    if (!(value >= lo && value <= hi)) {
        throw std::runtime_error(std::string("config: ") + key + " must be in [" +
                                 std::to_string(lo) + ", " + std::to_string(hi) + "], got " +
                                 std::to_string(value));
    }
}

/// Both loaders report a missing file by path rather than letting a bare
/// yaml-cpp BadFile escape.
void require_readable(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        throw std::runtime_error("config file not found: " + path);
    }
}

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

void SystemConfig::validate() const {
    require_positive(pipeline.muxer_width, "pipeline.muxer_width");
    require_positive(pipeline.muxer_height, "pipeline.muxer_height");
    require_positive(pipeline.batch_size, "pipeline.batch_size");

    if (detection.engine_path.empty()) {
        throw std::runtime_error("config: detection.engine_path must not be empty");
    }
    require_positive(detection.input_width, "detection.input_width");
    require_positive(detection.input_height, "detection.input_height");
    require_in_range(detection.confidence_threshold, 0.0f, 1.0f, "detection.confidence_threshold");
    require_in_range(detection.nms_iou_threshold, 0.0f, 1.0f, "detection.nms_iou_threshold");
    if (detection.person_class_id < 0) {
        throw std::runtime_error("config: detection.person_class_id must not be negative");
    }

    require_in_range(tracker.bytetrack.high_thresh, 0.0f, 1.0f, "tracker.bytetrack.high_thresh");
    require_in_range(tracker.bytetrack.low_thresh, 0.0f, 1.0f, "tracker.bytetrack.low_thresh");
    if (tracker.bytetrack.low_thresh > tracker.bytetrack.high_thresh) {
        throw std::runtime_error(
            "config: tracker.bytetrack.low_thresh must not exceed high_thresh — the two-stage "
            "cascade would have no low-confidence band to work with");
    }
    require_in_range(tracker.bytetrack.new_track_thresh, 0.0f, 1.0f,
                     "tracker.bytetrack.new_track_thresh");
    require_in_range(tracker.bytetrack.match_thresh, 0.0f, 1.0f, "tracker.bytetrack.match_thresh");
    require_positive(tracker.bytetrack.track_buffer, "tracker.bytetrack.track_buffer");
    require_in_range(tracker.iou.iou_thresh, 0.0f, 1.0f, "tracker.iou.iou_thresh");
    require_positive(tracker.iou.max_age, "tracker.iou.max_age");
    require_positive(tracker.iou.min_hits, "tracker.iou.min_hits");

    if (reid.enabled) {
        if (reid.engine_path.empty()) {
            throw std::runtime_error("config: reid.engine_path must not be empty when reid is on");
        }
        require_positive(reid.input_width, "reid.input_width");
        require_positive(reid.input_height, "reid.input_height");
        require_positive(reid.embedding_dim, "reid.embedding_dim");
        require_positive(reid.batch_size, "reid.batch_size");
        require_positive(reid.gallery_size_per_track, "reid.gallery_size_per_track");
    }

    // Cosine similarity of unit-norm embeddings is bounded by [-1, 1], and
    // the Hungarian cost is 1 - similarity.
    require_in_range(crosscam.reid_threshold, -1.0f, 1.0f, "crosscam.reid_threshold");
    require_in_range(crosscam.hungarian_cost_cap, 0.0f, 2.0f, "crosscam.hungarian_cost_cap");
}

SystemConfig SystemConfig::load(const std::string& yaml_path) {
    require_readable(yaml_path);
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
            out.tracker.bytetrack.new_track_thresh = optional<float>(bt, "new_track_thresh", 0.6f);
            out.tracker.bytetrack.track_buffer = optional<std::uint32_t>(bt, "track_buffer", 30);
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
            out.tracker.nvdcf.tracker_width = optional<std::uint32_t>(nv, "tracker_width", 640);
            out.tracker.nvdcf.tracker_height = optional<std::uint32_t>(nv, "tracker_height", 384);
        }
    }

    if (const auto r = root["reid"]; r) {
        out.reid.enabled = optional<bool>(r, "enabled", true);
        out.reid.engine_path = optional<std::string>(r, "engine_path", "");
        out.reid.input_width = optional<std::uint32_t>(r, "input_width", 128);
        out.reid.input_height = optional<std::uint32_t>(r, "input_height", 256);
        out.reid.embedding_dim = optional<std::uint32_t>(r, "embedding_dim", 256);
        out.reid.batch_size = optional<std::uint32_t>(r, "batch_size", 16);
        out.reid.gallery_size_per_track = optional<std::uint32_t>(r, "gallery_size_per_track", 8);
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
    out.validate();
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

void CamerasConfig::validate() const {
    std::set<std::string> ids;
    std::set<std::string> zones;
    for (const auto& c : cameras) {
        if (c.id.empty()) {
            throw std::runtime_error("cameras: every entry needs a non-empty id");
        }
        if (c.uri.empty()) {
            throw std::runtime_error("cameras: camera '" + c.id + "' has no uri");
        }
        if (!ids.insert(c.id).second) {
            throw std::runtime_error("cameras: duplicate camera id '" + c.id + "'");
        }
        if (!c.zone.empty()) zones.insert(c.zone);
    }

    // A transition naming a zone no camera sits in is dead topology: it can
    // never fire, and it usually means a typo that silently blocks a real
    // hand-off instead.
    for (const auto& tr : transitions) {
        if (tr.from.empty() || tr.to.empty()) {
            throw std::runtime_error("cameras: a transition needs both 'from' and 'to'");
        }
        if (zones.count(tr.from) == 0) {
            throw std::runtime_error("cameras: transition from unknown zone '" + tr.from + "'");
        }
        if (zones.count(tr.to) == 0) {
            throw std::runtime_error("cameras: transition to unknown zone '" + tr.to + "'");
        }
    }
}

CamerasConfig CamerasConfig::load(const std::string& yaml_path) {
    require_readable(yaml_path);
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
    out.validate();
    return out;
}

}  // namespace mc_tracking::config
