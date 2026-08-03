#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "mc_tracking/tracker/kalman_filter.hpp"

namespace {

using mc_tracking::tracker::bbox_to_measurement;
using mc_tracking::tracker::KalmanFilter;

/// The filter carries float state through several matrix products, so the
/// bbox round-trip is exact only to within a few ULPs of the height.
constexpr float kTol = 1e-3f;

void expect_rect_near(const cv::Rect2f& got, const cv::Rect2f& want, float tol) {
    EXPECT_NEAR(got.x, want.x, tol);
    EXPECT_NEAR(got.y, want.y, tol);
    EXPECT_NEAR(got.width, want.width, tol);
    EXPECT_NEAR(got.height, want.height, tol);
}

// ------------------------------------------------------- the measurement

TEST(BboxToMeasurement, ConvertsCornerFormToCentreAspectHeight) {
    const cv::Rect2f box(100.0f, 200.0f, 40.0f, 80.0f);

    const auto m = bbox_to_measurement(box);

    EXPECT_FLOAT_EQ(m[0], 120.0f);  // centre x
    EXPECT_FLOAT_EQ(m[1], 240.0f);  // centre y
    EXPECT_FLOAT_EQ(m[2], 0.5f);    // aspect = w / h
    EXPECT_FLOAT_EQ(m[3], 80.0f);   // height
}

TEST(BboxToMeasurement, ReportsAZeroAspectForADegenerateBox) {
    // A zero-height box has no defined aspect ratio; the helper must not
    // divide by zero.
    const auto m = bbox_to_measurement(cv::Rect2f(0.0f, 0.0f, 40.0f, 0.0f));

    EXPECT_FLOAT_EQ(m[2], 0.0f);
    EXPECT_FLOAT_EQ(m[3], 0.0f);
}

// -------------------------------------------------------------- the filter

TEST(KalmanFilter, InitiateRoundTripsTheBoundingBox) {
    const cv::Rect2f box(100.0f, 200.0f, 40.0f, 80.0f);
    KalmanFilter kf;

    kf.initiate(bbox_to_measurement(box));

    expect_rect_near(kf.to_xywh(), box, kTol);
}

TEST(KalmanFilter, InitiateZeroesTheVelocityComponents) {
    KalmanFilter kf;

    kf.initiate(bbox_to_measurement(cv::Rect2f(100.0f, 200.0f, 40.0f, 80.0f)));

    const auto& s = kf.state();
    EXPECT_FLOAT_EQ(s[4], 0.0f);
    EXPECT_FLOAT_EQ(s[5], 0.0f);
    EXPECT_FLOAT_EQ(s[6], 0.0f);
    EXPECT_FLOAT_EQ(s[7], 0.0f);
}

TEST(KalmanFilter, PredictLeavesAStationaryTrackWhereItIs) {
    // Velocity starts at zero, so a single predict must not move the box.
    const cv::Rect2f box(100.0f, 200.0f, 40.0f, 80.0f);
    KalmanFilter kf;
    kf.initiate(bbox_to_measurement(box));

    kf.predict();

    expect_rect_near(kf.to_xywh(), box, kTol);
}

TEST(KalmanFilter, RepeatedPredictsStillDoNotMoveAStationaryTrack) {
    const cv::Rect2f box(100.0f, 200.0f, 40.0f, 80.0f);
    KalmanFilter kf;
    kf.initiate(bbox_to_measurement(box));

    for (int i = 0; i < 10; ++i) kf.predict();

    expect_rect_near(kf.to_xywh(), box, kTol);
}

TEST(KalmanFilter, UpdatePullsTheStateTowardTheMeasurement) {
    KalmanFilter kf;
    kf.initiate(bbox_to_measurement(cv::Rect2f(100.0f, 200.0f, 40.0f, 80.0f)));
    kf.predict();

    // A box 50 px to the right.
    const cv::Rect2f observed(150.0f, 200.0f, 40.0f, 80.0f);
    kf.update(bbox_to_measurement(observed));

    const float cx = kf.state()[0];
    EXPECT_GT(cx, 120.0f) << "the estimate should move toward the observation";
    EXPECT_LE(cx, 170.0f) << "and must not overshoot past it";
}

TEST(KalmanFilter, LearnsAConstantVelocityFromRepeatedObservations) {
    // Feed a box moving a steady +10 px per frame in x.
    KalmanFilter kf;
    kf.initiate(bbox_to_measurement(cv::Rect2f(100.0f, 200.0f, 40.0f, 80.0f)));

    for (int i = 1; i <= 20; ++i) {
        kf.predict();
        const float x = 100.0f + 10.0f * static_cast<float>(i);
        kf.update(bbox_to_measurement(cv::Rect2f(x, 200.0f, 40.0f, 80.0f)));
    }

    EXPECT_NEAR(kf.state()[4], 10.0f, 1.5f) << "vx should converge on the true velocity";
    EXPECT_NEAR(kf.state()[5], 0.0f, 1.5f) << "vy should stay near zero";
}

TEST(KalmanFilter, ExtrapolatesForwardOnceItHasLearnedAVelocity) {
    KalmanFilter kf;
    kf.initiate(bbox_to_measurement(cv::Rect2f(100.0f, 200.0f, 40.0f, 80.0f)));
    for (int i = 1; i <= 20; ++i) {
        kf.predict();
        const float x = 100.0f + 10.0f * static_cast<float>(i);
        kf.update(bbox_to_measurement(cv::Rect2f(x, 200.0f, 40.0f, 80.0f)));
    }

    const float before = kf.state()[0];
    kf.predict();  // no observation this frame

    // A coasting track should keep moving in the direction it was going.
    EXPECT_GT(kf.state()[0], before);
    EXPECT_NEAR(kf.state()[0] - before, 10.0f, 2.0f);
}

TEST(KalmanFilter, KeepsTheHeightStableUnderAConstantObservation) {
    KalmanFilter kf;
    kf.initiate(bbox_to_measurement(cv::Rect2f(100.0f, 200.0f, 40.0f, 80.0f)));

    for (int i = 0; i < 15; ++i) {
        kf.predict();
        kf.update(bbox_to_measurement(cv::Rect2f(100.0f, 200.0f, 40.0f, 80.0f)));
    }

    expect_rect_near(kf.to_xywh(), cv::Rect2f(100.0f, 200.0f, 40.0f, 80.0f), 0.5f);
}

TEST(KalmanFilter, TracksAGrowingBoxAsTheSubjectApproaches) {
    KalmanFilter kf;
    kf.initiate(bbox_to_measurement(cv::Rect2f(100.0f, 200.0f, 40.0f, 80.0f)));

    // Height grows 4 px per frame, aspect held constant.
    for (int i = 1; i <= 15; ++i) {
        kf.predict();
        const float h = 80.0f + 4.0f * static_cast<float>(i);
        kf.update(bbox_to_measurement(cv::Rect2f(100.0f, 200.0f, h * 0.5f, h)));
    }

    EXPECT_NEAR(kf.to_xywh().height, 140.0f, 6.0f);
    EXPECT_GT(kf.state()[7], 0.0f) << "vh should be positive while the box grows";
}

}  // namespace
