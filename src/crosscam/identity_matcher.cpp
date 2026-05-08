#include "mc_tracking/crosscam/identity_matcher.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "mc_tracking/crosscam/hungarian.hpp"
#include "mc_tracking/utils/logger.hpp"

namespace mc_tracking::crosscam {

namespace {

float cosine_similarity(const reid::Embedding& a, const reid::Embedding& b) {
    if (a.size() == 0 || a.size() != b.size()) return -1.0f;
    return a.dot(b);  // both L2-normalized
}

}  // namespace

IdentityMatcher::IdentityMatcher(const config::CrossCamConfig& cfg,
                                 const config::CamerasConfig& cameras)
    : cfg_(cfg), cameras_(cameras) {}

void IdentityMatcher::register_gallery(const std::string& camera_id,
                                        const reid::ReidGallery* gallery) {
    galleries_[camera_id] = gallery;
}

bool IdentityMatcher::transition_allowed_(const std::string& from_zone,
                                          const std::string& to_zone) const {
    if (from_zone.empty() || to_zone.empty()) return true;
    return cameras_.transition_allowed(from_zone, to_zone);
}

std::vector<std::uint64_t> IdentityMatcher::update(
    const std::vector<CameraTrackObservation>& observations) {
    std::vector<std::uint64_t> result(observations.size(), 0);

    // Step 1: carry over already-assigned (camera_id, local_id) pairs.
    std::vector<std::size_t> unassigned;
    unassigned.reserve(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
        const auto& obs = observations[i];
        const auto cam_it = camera_to_global_.find(obs.camera_id);
        if (cam_it != camera_to_global_.end()) {
            const auto local_it = cam_it->second.find(obs.local_id);
            if (local_it != cam_it->second.end()) {
                const std::uint64_t gid = local_it->second;
                result[i] = gid;
                auto& rec = globals_[gid];
                rec.last_zone = obs.zone;
                rec.last_seen = obs.pts;
                rec.canonical_embedding = obs.embedding;
                continue;
            }
        }
        unassigned.push_back(i);
    }
    if (unassigned.empty()) return result;

    // Step 2: build the cost matrix between unassigned observations
    // and currently-active global ids. INFEASIBLE entries:
    //   * transition not allowed by zone topology
    //   * spatial overlap window exceeded
    //   * cosine similarity below threshold
    std::vector<std::uint64_t> active_global_ids;
    active_global_ids.reserve(globals_.size());
    for (const auto& [gid, _] : globals_) active_global_ids.push_back(gid);

    if (!active_global_ids.empty()) {
        const std::size_t rows = unassigned.size();
        const std::size_t cols = active_global_ids.size();
        std::vector<float> cost(rows * cols, INFEASIBLE_COST);

        for (std::size_t i = 0; i < rows; ++i) {
            const auto& obs = observations[unassigned[i]];
            for (std::size_t j = 0; j < cols; ++j) {
                const auto& rec = globals_[active_global_ids[j]];

                if (!transition_allowed_(rec.last_zone, obs.zone)) continue;

                const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       obs.pts - rec.last_seen)
                                       .count();
                if (dt_ms < 0) continue;
                if (static_cast<std::uint32_t>(dt_ms) > cfg_.spatial_overlap_window_ms) continue;

                float sim = cosine_similarity(obs.embedding, rec.canonical_embedding);

                // If the same camera also has a gallery for this
                // local id, query historical observations and keep
                // the best match.
                const auto gal_it = galleries_.find(obs.camera_id);
                if (gal_it != galleries_.end() && gal_it->second != nullptr) {
                    const float gallery_sim =
                        gal_it->second->best_similarity(obs.local_id, rec.canonical_embedding);
                    if (gallery_sim > sim) sim = gallery_sim;
                }

                if (sim < cfg_.reid_threshold) continue;
                cost[i * cols + j] = 1.0f - sim;
            }
        }

        const auto match = solve_assignment(cost, rows, cols);
        for (std::size_t i = 0; i < rows; ++i) {
            const int j = match[i];
            if (j < 0) continue;
            const float c = cost[i * cols + static_cast<std::size_t>(j)];
            if (c > cfg_.hungarian_cost_cap) continue;

            const std::uint64_t gid = active_global_ids[static_cast<std::size_t>(j)];
            const auto& obs = observations[unassigned[i]];
            result[unassigned[i]] = gid;
            camera_to_global_[obs.camera_id][obs.local_id] = gid;

            auto& rec = globals_[gid];
            rec.last_zone = obs.zone;
            rec.last_seen = obs.pts;
            rec.canonical_embedding = obs.embedding;

            unassigned[i] = std::numeric_limits<std::size_t>::max();  // mark consumed
        }
    }

    // Step 3: anything left needs a fresh global id.
    for (std::size_t idx : unassigned) {
        if (idx == std::numeric_limits<std::size_t>::max()) continue;
        const auto& obs = observations[idx];
        GlobalRecord rec;
        rec.global_id = next_global_id_++;
        rec.last_zone = obs.zone;
        rec.last_seen = obs.pts;
        rec.canonical_embedding = obs.embedding;
        const std::uint64_t gid = rec.global_id;
        globals_[gid] = std::move(rec);
        camera_to_global_[obs.camera_id][obs.local_id] = gid;
        result[idx] = gid;
        MCT_LOG_DEBUG("crosscam: new global id {} created for cam={} local={}", gid,
                      obs.camera_id, obs.local_id);
    }

    return result;
}

}  // namespace mc_tracking::crosscam
