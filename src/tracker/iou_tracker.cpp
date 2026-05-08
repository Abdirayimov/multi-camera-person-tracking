#include "mc_tracking/tracker/iou_tracker.hpp"

#include <algorithm>
#include <unordered_set>

namespace mc_tracking::tracker {

namespace {

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float xx1 = std::max(a.x, b.x);
    const float yy1 = std::max(a.y, b.y);
    const float xx2 = std::min(a.x + a.width, b.x + b.width);
    const float yy2 = std::min(a.y + a.height, b.y + b.height);
    const float w = std::max(0.0f, xx2 - xx1);
    const float h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;
    const float u = a.area() + b.area() - inter;
    return (u > 0.0f) ? inter / u : 0.0f;
}

}  // namespace

IouTracker::IouTracker(const config::IouParams& params) : params_(params) {}

void IouTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}

std::vector<Track> IouTracker::update(const std::vector<Detection>& detections) {
    // Greedy IoU matching: for each existing track, pick the
    // highest-IoU unmatched detection above the threshold.
    std::unordered_set<std::size_t> matched_dets;
    std::unordered_set<std::uint64_t> matched_tracks;

    // Sort track ids for stable iteration order.
    std::vector<std::uint64_t> ids;
    ids.reserve(tracks_.size());
    for (const auto& [id, _] : tracks_) ids.push_back(id);
    std::sort(ids.begin(), ids.end());

    for (std::uint64_t id : ids) {
        auto& track = tracks_[id];
        float best_iou = params_.iou_thresh;
        int best_idx = -1;
        for (std::size_t i = 0; i < detections.size(); ++i) {
            if (matched_dets.count(i)) continue;
            const float v = iou(track.bbox, detections[i].bbox);
            if (v > best_iou) {
                best_iou = v;
                best_idx = static_cast<int>(i);
            }
        }
        if (best_idx >= 0) {
            track.bbox = detections[static_cast<std::size_t>(best_idx)].bbox;
            track.confidence = detections[static_cast<std::size_t>(best_idx)].score;
            track.hit_streak += 1;
            track.time_since_update = 0;
            track.age += 1;
            if (track.state == TrackState::Tentative && track.hit_streak >= params_.min_hits) {
                track.state = TrackState::Confirmed;
            } else if (track.state == TrackState::Lost) {
                track.state = TrackState::Confirmed;
            }
            matched_dets.insert(static_cast<std::size_t>(best_idx));
            matched_tracks.insert(id);
        }
    }

    // Age unmatched tracks; evict if too old.
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (matched_tracks.count(it->first) == 0) {
            it->second.time_since_update += 1;
            it->second.age += 1;
            it->second.hit_streak = 0;
            if (it->second.state == TrackState::Confirmed) {
                it->second.state = TrackState::Lost;
            }
            if (it->second.time_since_update > params_.max_age) {
                it = tracks_.erase(it);
                continue;
            }
        }
        ++it;
    }

    // Spawn tracks for unmatched detections.
    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (matched_dets.count(i)) continue;
        Track t;
        t.local_id = next_id_++;
        t.bbox = detections[i].bbox;
        t.confidence = detections[i].score;
        t.state = TrackState::Tentative;
        t.hit_streak = 1;
        t.age = 1;
        t.time_since_update = 0;
        tracks_.emplace(t.local_id, std::move(t));
    }

    // Snapshot confirmed tracks.
    std::vector<Track> out;
    out.reserve(tracks_.size());
    for (auto& [id, track] : tracks_) {
        if (track.state == TrackState::Confirmed) {
            out.push_back(track);
        }
    }
    return out;
}

}  // namespace mc_tracking::tracker
