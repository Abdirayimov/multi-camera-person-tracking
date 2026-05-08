#include "mc_tracking/tracker/bytetrack.hpp"

#include <algorithm>
#include <unordered_set>

#include "mc_tracking/crosscam/hungarian.hpp"

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

/// Build a (1 - IoU) cost matrix; cells above `match_thresh` are
/// flagged as infeasible so the assignment solver can leave them out.
std::vector<float> build_iou_cost(const std::vector<ByteTrackState*>& tracks,
                                  const std::vector<Detection>& detections,
                                  float match_thresh) {
    std::vector<float> cost(tracks.size() * detections.size(),
                            crosscam::INFEASIBLE_COST);
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        for (std::size_t j = 0; j < detections.size(); ++j) {
            const float v = iou(tracks[i]->bbox, detections[j].bbox);
            const float c = 1.0f - v;
            if (c <= match_thresh) {
                cost[i * detections.size() + j] = c;
            }
        }
    }
    return cost;
}

}  // namespace

ByteTrack::ByteTrack(const config::ByteTrackParams& params) : params_(params) {}

void ByteTrack::reset() {
    tracked_.clear();
    lost_.clear();
    removed_.clear();
    next_id_ = 1;
    frame_id_ = 0;
}

void ByteTrack::predict_all_() {
    for (auto& t : tracked_) t->kalman.predict();
    for (auto& t : lost_) t->kalman.predict();
}

void ByteTrack::prune_aspect_ratio_(std::vector<Detection>& detections) const {
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
                       [&](const Detection& d) {
                           if (d.bbox.height <= 0.0f) return true;
                           const float aspect = d.bbox.width / d.bbox.height;
                           return aspect > params_.aspect_ratio_thresh;
                       }),
        detections.end());
}

std::vector<Track> ByteTrack::update(const std::vector<Detection>& detections_in) {
    ++frame_id_;

    auto detections = detections_in;
    prune_aspect_ratio_(detections);

    // Split detections by confidence into high / low buckets per the
    // BYTETrack two-stage cascade.
    std::vector<Detection> high;
    std::vector<Detection> low;
    high.reserve(detections.size());
    low.reserve(detections.size());
    for (const auto& d : detections) {
        if (d.score >= params_.high_thresh) {
            high.push_back(d);
        } else if (d.score >= params_.low_thresh) {
            low.push_back(d);
        }
    }

    // Predict all active and lost tracks one step forward.
    predict_all_();

    // First-stage association against tracked + lost using high
    // confidence detections.
    std::vector<ByteTrackState*> stage1_pool;
    stage1_pool.reserve(tracked_.size() + lost_.size());
    for (auto& t : tracked_) stage1_pool.push_back(t.get());
    for (auto& t : lost_) stage1_pool.push_back(t.get());

    std::vector<int> match1;
    if (!stage1_pool.empty() && !high.empty()) {
        const auto cost = build_iou_cost(stage1_pool, high, params_.match_thresh);
        match1 = crosscam::solve_assignment(cost, stage1_pool.size(), high.size());
    } else {
        match1.assign(stage1_pool.size(), -1);
    }

    std::unordered_set<std::size_t> matched_high;
    std::unordered_set<std::size_t> matched_pool;
    for (std::size_t i = 0; i < match1.size(); ++i) {
        if (match1[i] < 0) continue;
        auto* track = stage1_pool[i];
        const auto& det = high[static_cast<std::size_t>(match1[i])];
        track->kalman.update(bbox_to_measurement(det.bbox));
        track->bbox = track->kalman.to_xywh();
        track->confidence = det.score;
        track->time_since_update = 0;
        track->hit_streak += 1;
        track->age += 1;
        if (track->state == TrackState::Tentative) {
            track->state = TrackState::Confirmed;
        } else if (track->state == TrackState::Lost) {
            track->state = TrackState::Confirmed;
        }
        matched_high.insert(static_cast<std::size_t>(match1[i]));
        matched_pool.insert(i);
    }

    // Second-stage association: remaining tracked tracks vs low
    // confidence detections.
    std::vector<ByteTrackState*> stage2_pool;
    for (std::size_t i = 0; i < tracked_.size(); ++i) {
        // tracked tracks live at indices [0, tracked_.size()) of stage1_pool
        if (matched_pool.count(i) == 0) stage2_pool.push_back(tracked_[i].get());
    }
    if (!stage2_pool.empty() && !low.empty()) {
        const auto cost = build_iou_cost(stage2_pool, low, 0.5f);
        const auto match2 = crosscam::solve_assignment(cost, stage2_pool.size(), low.size());
        for (std::size_t i = 0; i < match2.size(); ++i) {
            if (match2[i] < 0) continue;
            auto* track = stage2_pool[i];
            const auto& det = low[static_cast<std::size_t>(match2[i])];
            track->kalman.update(bbox_to_measurement(det.bbox));
            track->bbox = track->kalman.to_xywh();
            track->confidence = det.score;
            track->time_since_update = 0;
            track->hit_streak += 1;
            track->age += 1;
        }
    }

    // Promote unmatched-but-fresh-enough high detections into new
    // tentative tracks.
    for (std::size_t j = 0; j < high.size(); ++j) {
        if (matched_high.count(j)) continue;
        if (high[j].score < params_.new_track_thresh) continue;
        auto state = std::make_unique<ByteTrackState>();
        state->local_id = next_id_++;
        state->bbox = high[j].bbox;
        state->confidence = high[j].score;
        state->state = TrackState::Tentative;
        state->hit_streak = 1;
        state->age = 1;
        state->start_frame = frame_id_;
        state->kalman.initiate(bbox_to_measurement(high[j].bbox));
        tracked_.push_back(std::move(state));
    }

    // Move tracks that did not get an update this frame to the lost
    // pool; remove tracks that have been lost for too long.
    auto move_to_lost = [&](std::vector<std::unique_ptr<ByteTrackState>>& src) {
        for (auto it = src.begin(); it != src.end();) {
            if ((*it)->time_since_update > 0) {
                ++(*it)->time_since_update;
            } else {
                ++it;
                continue;
            }
            if ((*it)->time_since_update > params_.track_buffer) {
                removed_.push_back(std::move(*it));
                it = src.erase(it);
            } else {
                (*it)->state = TrackState::Lost;
                lost_.push_back(std::move(*it));
                it = src.erase(it);
            }
        }
    };
    // Tag all unmatched tracks with time_since_update bump first.
    for (auto& t : tracked_) {
        if (t->time_since_update == 0 && matched_pool.count(0) == 0) {
            // No-op marker: time_since_update bump is handled below.
        }
    }
    // For simplicity we bump time_since_update on any unmatched track.
    for (std::size_t i = 0; i < tracked_.size(); ++i) {
        // Track is unmatched if its state was not refreshed by either
        // association stage. We detect that by checking the timestamp.
        if (tracked_[i]->time_since_update == 0 && matched_pool.count(i) == 0) {
            tracked_[i]->time_since_update = 1;
            tracked_[i]->state = TrackState::Lost;
        }
    }
    move_to_lost(tracked_);

    // Lost pool ages out the same way.
    for (auto it = lost_.begin(); it != lost_.end();) {
        ++(*it)->time_since_update;
        ++(*it)->age;
        if ((*it)->time_since_update > params_.track_buffer) {
            removed_.push_back(std::move(*it));
            it = lost_.erase(it);
        } else {
            ++it;
        }
    }

    // Emit confirmed tracks.
    std::vector<Track> out;
    out.reserve(tracked_.size());
    for (const auto& t : tracked_) {
        if (t->state != TrackState::Confirmed) continue;
        Track snap;
        snap.local_id = t->local_id;
        snap.bbox = t->bbox;
        snap.confidence = t->confidence;
        snap.state = t->state;
        snap.hit_streak = t->hit_streak;
        snap.time_since_update = t->time_since_update;
        snap.age = t->age;
        out.push_back(std::move(snap));
    }
    return out;
}

}  // namespace mc_tracking::tracker
