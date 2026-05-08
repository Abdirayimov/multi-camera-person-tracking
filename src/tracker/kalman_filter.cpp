#include "mc_tracking/tracker/kalman_filter.hpp"

namespace mc_tracking::tracker {

KalmanFilter::KalmanFilter() {
    // Constant-velocity transition matrix:
    //   [I  I]   <- position += velocity
    //   [0  I]   <- velocity unchanged
    transition_matrix_ = StateMat::Identity();
    for (int i = 0; i < kMeasDim; ++i) {
        transition_matrix_(i, kMeasDim + i) = 1.0f;
    }

    // Measurement matrix selects the position part of the state.
    projection_matrix_.setZero();
    for (int i = 0; i < kMeasDim; ++i) {
        projection_matrix_(i, i) = 1.0f;
    }
}

void KalmanFilter::initiate(const MeasVec& measurement) {
    mean_.setZero();
    mean_.head<kMeasDim>() = measurement;

    const float h = measurement[3];
    const float pos_std = 2.0f * std_weight_position_ * h;
    const float vel_std = 10.0f * std_weight_velocity_ * h;

    covariance_.setZero();
    covariance_(0, 0) = pos_std * pos_std;
    covariance_(1, 1) = pos_std * pos_std;
    covariance_(2, 2) = 1e-2f;
    covariance_(3, 3) = pos_std * pos_std;
    covariance_(4, 4) = vel_std * vel_std;
    covariance_(5, 5) = vel_std * vel_std;
    covariance_(6, 6) = 1e-5f;
    covariance_(7, 7) = vel_std * vel_std;
}

void KalmanFilter::predict() {
    const float h = mean_[3];
    const float pos_std = std_weight_position_ * h;
    const float vel_std = std_weight_velocity_ * h;

    StateMat motion_cov = StateMat::Zero();
    motion_cov(0, 0) = pos_std * pos_std;
    motion_cov(1, 1) = pos_std * pos_std;
    motion_cov(2, 2) = 1e-2f;
    motion_cov(3, 3) = pos_std * pos_std;
    motion_cov(4, 4) = vel_std * vel_std;
    motion_cov(5, 5) = vel_std * vel_std;
    motion_cov(6, 6) = 1e-5f;
    motion_cov(7, 7) = vel_std * vel_std;

    mean_ = transition_matrix_ * mean_;
    covariance_ = transition_matrix_ * covariance_ * transition_matrix_.transpose() + motion_cov;
}

void KalmanFilter::update(const MeasVec& measurement) {
    const float h = mean_[3];
    const float meas_std = std_weight_position_ * h;

    MeasMat innovation_cov = MeasMat::Zero();
    innovation_cov(0, 0) = meas_std * meas_std;
    innovation_cov(1, 1) = meas_std * meas_std;
    innovation_cov(2, 2) = 1e-1f;
    innovation_cov(3, 3) = meas_std * meas_std;

    const auto projected_cov =
        projection_matrix_ * covariance_ * projection_matrix_.transpose() + innovation_cov;
    const Eigen::Matrix<float, kStateDim, kMeasDim> kalman_gain =
        covariance_ * projection_matrix_.transpose() * projected_cov.inverse();

    const MeasVec innovation = measurement - projection_matrix_ * mean_;
    mean_ = mean_ + kalman_gain * innovation;
    covariance_ = covariance_ - kalman_gain * projection_matrix_ * covariance_;
}

cv::Rect2f KalmanFilter::to_xywh() const noexcept {
    const float aspect = mean_[2];
    const float h = mean_[3];
    const float w = aspect * h;
    return cv::Rect2f(mean_[0] - w * 0.5f, mean_[1] - h * 0.5f, w, h);
}

}  // namespace mc_tracking::tracker
