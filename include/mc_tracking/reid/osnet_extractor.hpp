#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <memory>
#include <vector>

#include "mc_tracking/config/system_config.hpp"

namespace mc_tracking::trt {
class TrtEngine;
}

namespace mc_tracking::reid {

using Embedding = Eigen::Matrix<float, Eigen::Dynamic, 1>;

/// OSNet (Zhou et al., ICCV 2019) appearance feature extractor.
///
/// Takes person crops (bounding boxes from the detector), runs them
/// through the TRT engine in batches of `cfg.batch_size`, and returns
/// L2-normalized embeddings. Used both inside the tracker (to reduce
/// ID switches) and across cameras (as the primary signal for
/// matching).
class OSNetExtractor {
public:
    explicit OSNetExtractor(const config::ReidConfig& cfg);
    ~OSNetExtractor();

    OSNetExtractor(const OSNetExtractor&) = delete;
    OSNetExtractor& operator=(const OSNetExtractor&) = delete;

    /// Extract one embedding per cropped person. Crops are expected
    /// in BGR; preprocessing (resize + ImageNet normalization) happens
    /// internally.
    std::vector<Embedding> extract(const std::vector<cv::Mat>& person_crops);

    const config::ReidConfig& config() const noexcept { return cfg_; }

private:
    config::ReidConfig cfg_;
    std::unique_ptr<trt::TrtEngine> engine_;
    std::vector<float> input_scratch_;
    std::vector<float> output_scratch_;

    void run_chunk_(const std::vector<cv::Mat>& chunk, std::vector<Embedding>& out);
};

/// Pull person crops out of a frame at the given track bboxes.
/// Bboxes are clipped to the image bounds; degenerate (zero-area)
/// crops are silently skipped and the corresponding output entry is
/// left default-constructed.
std::vector<cv::Mat> crop_persons(const cv::Mat& frame,
                                  const std::vector<cv::Rect2f>& bboxes);

}  // namespace mc_tracking::reid
