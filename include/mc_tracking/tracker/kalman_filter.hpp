#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core.hpp>

namespace mc_tracking::tracker {

/// 8-state constant-velocity Kalman filter on (x, y, a, h) where:
///   x, y = bbox centre
///   a    = aspect ratio (w / h)
///   h    = bbox height
///
/// State vector: [x, y, a, h, vx, vy, va, vh]
/// Measurement vector: [x, y, a, h]
///
/// This is the classical formulation used by SORT and BYTETrack.
class KalmanFilter {
public:
    static constexpr int kStateDim = 8;
    static constexpr int kMeasDim = 4;

    using StateVec = Eigen::Matrix<float, kStateDim, 1>;
    using StateMat = Eigen::Matrix<float, kStateDim, kStateDim>;
    using MeasVec = Eigen::Matrix<float, kMeasDim, 1>;
    using MeasMat = Eigen::Matrix<float, kMeasDim, kMeasDim>;

    KalmanFilter();

    /// Initialize from a measurement (centre x, centre y, aspect, height).
    /// The velocity components are zeroed.
    void initiate(const MeasVec& measurement);

    /// Propagate one frame forward.
    void predict();

    /// Apply a measurement update.
    void update(const MeasVec& measurement);

    /// Estimated bbox in (x1, y1, w, h) image coordinates.
    cv::Rect2f to_xywh() const noexcept;

    const StateVec& state() const noexcept { return mean_; }

private:
    StateVec mean_ = StateVec::Zero();
    StateMat covariance_ = StateMat::Identity();
    StateMat transition_matrix_;
    Eigen::Matrix<float, kMeasDim, kStateDim> projection_matrix_;

    // Hyper-parameters; tuned to match the SORT defaults.
    float std_weight_position_ = 1.0f / 20.0f;
    float std_weight_velocity_ = 1.0f / 160.0f;
};

/// Convert a (x, y, w, h) bbox to the Kalman measurement form
/// (cx, cy, aspect, h). Standalone helper used by trackers.
inline KalmanFilter::MeasVec bbox_to_measurement(const cv::Rect2f& b) {
    KalmanFilter::MeasVec m;
    m[0] = b.x + b.width * 0.5f;
    m[1] = b.y + b.height * 0.5f;
    m[2] = (b.height > 0.0f) ? b.width / b.height : 0.0f;
    m[3] = b.height;
    return m;
}

}  // namespace mc_tracking::tracker
