#include "mc_tracking/pipeline/multi_camera.hpp"

#include <utility>

namespace mc_tracking::pipeline {

MultiCameraOrchestrator::MultiCameraOrchestrator(const config::SystemConfig& cfg,
                                                  const config::CamerasConfig& cameras)
    : cfg_(cfg), cameras_cfg_(cameras) {
    matcher_ = std::make_unique<crosscam::IdentityMatcher>(cfg_.crosscam, cameras_cfg_);
}

SingleCameraPipeline& MultiCameraOrchestrator::add_camera(const std::string& id,
                                                           const std::string& zone) {
    auto pipeline = std::make_unique<SingleCameraPipeline>(id, zone, cfg_);
    if (pipeline->gallery() != nullptr) {
        matcher_->register_gallery(id, pipeline->gallery());
    }
    cameras_.push_back(std::move(pipeline));
    return *cameras_.back();
}

void MultiCameraOrchestrator::stamp_global_ids(std::vector<CameraFrameResult>& frames) {
    if (!cfg_.crosscam.enabled) return;

    std::vector<crosscam::CameraTrackObservation> observations;
    std::vector<std::pair<std::size_t, std::size_t>> back_refs;

    // Collect every track that has at least one appearance embedding.
    for (std::size_t f = 0; f < frames.size(); ++f) {
        for (std::size_t t = 0; t < frames[f].tracks.size(); ++t) {
            const auto& track = frames[f].tracks[t];
            if (track.appearance_gallery.empty()) continue;

            crosscam::CameraTrackObservation obs;
            obs.camera_id = track.camera_id;
            // Resolve zone from cameras_cfg_.
            for (const auto& cam : cameras_cfg_.cameras) {
                if (cam.id == track.camera_id) {
                    obs.zone = cam.zone;
                    break;
                }
            }
            obs.local_id = track.local_id;
            obs.pts = frames[f].pts;
            obs.bbox = track.bbox;
            obs.embedding = track.appearance_gallery.back();
            observations.push_back(std::move(obs));
            back_refs.emplace_back(f, t);
        }
    }

    if (observations.empty()) return;
    const auto global_ids = matcher_->update(observations);
    for (std::size_t i = 0; i < global_ids.size(); ++i) {
        const auto [f, t] = back_refs[i];
        frames[f].tracks[t].global_id = global_ids[i];
    }
}

}  // namespace mc_tracking::pipeline
