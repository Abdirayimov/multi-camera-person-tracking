#include "mc_tracking/pipeline/single_camera.hpp"

#include <utility>

namespace mc_tracking::pipeline {

SingleCameraPipeline::SingleCameraPipeline(std::string camera_id, std::string zone,
                                            const config::SystemConfig& cfg)
    : camera_id_(std::move(camera_id)), zone_(std::move(zone)), cfg_(cfg) {
    detector_ = std::make_unique<trt::YOLOv8Detector>(cfg_.detection);
    tracker_ = tracker::make_tracker(cfg_.tracker);
    if (cfg_.reid.enabled) {
        reid_ = std::make_unique<reid::OSNetExtractor>(cfg_.reid);
        gallery_ = std::make_unique<reid::ReidGallery>(cfg_.reid.gallery_size_per_track);
    }
}

CameraFrameResult SingleCameraPipeline::process_frame(std::uint64_t frame_number, TimePoint pts,
                                                       const cv::Mat& frame) {
    CameraFrameResult result;
    result.camera_id = camera_id_;
    result.frame_number = frame_number;
    result.pts = pts;

    const auto detections = detector_->detect(frame);
    auto tracks = tracker_->update(detections);

    if (reid_ && !tracks.empty()) {
        std::vector<cv::Rect2f> bboxes;
        bboxes.reserve(tracks.size());
        for (const auto& t : tracks) bboxes.push_back(t.bbox);

        const auto crops = reid::crop_persons(frame, bboxes);
        const auto embeddings = reid_->extract(crops);

        for (std::size_t i = 0; i < tracks.size(); ++i) {
            if (i < embeddings.size() && embeddings[i].size() > 0) {
                tracks[i].appearance_gallery.push_back(embeddings[i]);
                gallery_->push(tracks[i].local_id, embeddings[i]);
            }
        }
    }

    for (auto& t : tracks) t.camera_id = camera_id_;
    result.tracks = std::move(tracks);
    return result;
}

}  // namespace mc_tracking::pipeline
