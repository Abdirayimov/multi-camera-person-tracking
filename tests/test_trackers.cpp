#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

#include "mc_tracking/tracker/bytetrack.hpp"
#include "mc_tracking/tracker/iou_tracker.hpp"
#include "mc_tracking/tracker/tracker_iface.hpp"

namespace {

using mc_tracking::config::ByteTrackParams;
using mc_tracking::config::IouParams;
using mc_tracking::config::TrackerConfig;
using mc_tracking::config::TrackerType;
using mc_tracking::tracker::ByteTrack;
using mc_tracking::tracker::Detection;
using mc_tracking::tracker::IouTracker;
using mc_tracking::tracker::make_tracker;
using mc_tracking::tracker::Track;
using mc_tracking::tracker::TrackState;

Detection det(float x, float y, float w, float h, float score) {
    Detection d;
    d.bbox = cv::Rect2f(x, y, w, h);
    d.score = score;
    return d;
}

/// A person-shaped box: taller than it is wide, so BYTETrack's aspect
/// filter never discards it.
Detection person(float x, float score = 0.9f) {
    return det(x, 100.0f, 40.0f, 100.0f, score);
}

/// Feed the same detection `frames` times and return the last output.
std::vector<Track> run(mc_tracking::tracker::ITracker& t, const std::vector<Detection>& dets,
                       int frames) {
    std::vector<Track> out;
    for (int i = 0; i < frames; ++i) out = t.update(dets);
    return out;
}

// ------------------------------------------------------------- IouTracker

TEST(IouTracker, EmptyInputProducesNoTracks) {
    IouTracker t{IouParams{}};
    EXPECT_TRUE(t.update({}).empty());
}

TEST(IouTracker, WithholdsATrackUntilMinHitsIsReached) {
    IouParams params;
    params.min_hits = 3;
    IouTracker t{params};
    const std::vector<Detection> dets{person(100.0f, 0.9f)};

    // Frames 1 and 2 leave the track tentative.
    EXPECT_TRUE(t.update(dets).empty());
    EXPECT_TRUE(t.update(dets).empty());
    EXPECT_EQ(t.update(dets).size(), 1u) << "third hit should confirm the track";
}

TEST(IouTracker, KeepsTheSameLocalIdWhileTheBoxOverlaps) {
    IouTracker t{IouParams{}};
    const std::vector<Detection> dets{person(100.0f)};

    const auto first = run(t, dets, 3);
    ASSERT_EQ(first.size(), 1u);
    const auto id = first.front().local_id;

    // Shift by 10 px: still a large IoU, so the same track continues.
    const auto second = t.update({person(110.0f)});
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.front().local_id, id);
}

TEST(IouTracker, StartsANewTrackWhenTheBoxJumpsBeyondTheIouThreshold) {
    IouTracker t{IouParams{}};

    const auto first = run(t, {person(100.0f)}, 3);
    ASSERT_EQ(first.size(), 1u);
    const auto id = first.front().local_id;

    // Far enough that IoU with the old box is zero.
    const auto after = run(t, {person(900.0f)}, 3);
    ASSERT_FALSE(after.empty());
    bool saw_new_id = false;
    for (const auto& tr : after) {
        if (tr.local_id != id) saw_new_id = true;
    }
    EXPECT_TRUE(saw_new_id);
}

TEST(IouTracker, TracksTwoWellSeparatedPeopleIndependently) {
    IouTracker t{IouParams{}};
    const std::vector<Detection> dets{person(100.0f), person(600.0f)};

    const auto out = run(t, dets, 3);

    ASSERT_EQ(out.size(), 2u);
    EXPECT_NE(out[0].local_id, out[1].local_id);
}

TEST(IouTracker, MarksATrackLostAsSoonAsItIsMissed) {
    IouTracker t{IouParams{}};
    run(t, {person(100.0f)}, 3);

    // No detections this frame: the track drops out of the confirmed set.
    EXPECT_TRUE(t.update({}).empty());
}

TEST(IouTracker, ReacquiresALostTrackWithTheSameId) {
    IouTracker t{IouParams{}};
    const auto first = run(t, {person(100.0f)}, 3);
    ASSERT_EQ(first.size(), 1u);
    const auto id = first.front().local_id;

    t.update({});  // one missed frame

    const auto back = t.update({person(100.0f)});
    ASSERT_EQ(back.size(), 1u);
    EXPECT_EQ(back.front().local_id, id) << "a brief gap must not burn a new id";
}

TEST(IouTracker, EvictsATrackThatStaysMissingPastMaxAge) {
    IouParams params;
    params.max_age = 3;
    IouTracker t{params};

    const auto first = run(t, {person(100.0f)}, 3);
    ASSERT_EQ(first.size(), 1u);
    const auto id = first.front().local_id;

    for (int i = 0; i < 6; ++i) t.update({});

    // The old track is gone, so the same box has to become a new id.
    const auto revived = run(t, {person(100.0f)}, 3);
    ASSERT_EQ(revived.size(), 1u);
    EXPECT_NE(revived.front().local_id, id);
}

TEST(IouTracker, ResetDropsEveryTrackAndRestartsIds) {
    IouTracker t{IouParams{}};
    run(t, {person(100.0f)}, 3);

    t.reset();

    EXPECT_TRUE(t.update({}).empty());
    const auto after = run(t, {person(100.0f)}, 3);
    ASSERT_EQ(after.size(), 1u);
    EXPECT_EQ(after.front().local_id, 1u) << "ids restart from 1 after reset";
}

// --------------------------------------------------------------- ByteTrack

TEST(ByteTrack, EmptyInputProducesNoTracks) {
    ByteTrack t{ByteTrackParams{}};
    EXPECT_TRUE(t.update({}).empty());
}

TEST(ByteTrack, ConfirmsATrackOnItsSecondObservation) {
    ByteTrack t{ByteTrackParams{}};
    const std::vector<Detection> dets{person(100.0f)};

    EXPECT_TRUE(t.update(dets).empty()) << "the first frame only seeds a tentative track";
    EXPECT_EQ(t.update(dets).size(), 1u);
}

TEST(ByteTrack, IgnoresDetectionsBelowTheNewTrackThreshold) {
    ByteTrackParams params;
    params.new_track_thresh = 0.6f;
    ByteTrack t{params};

    // Above high_thresh (0.5) so it is not discarded outright, but below
    // new_track_thresh, so it must not seed a track on its own.
    run(t, {person(100.0f, 0.55f)}, 4);
    EXPECT_TRUE(t.update({person(100.0f, 0.55f)}).empty());
}

TEST(ByteTrack, DiscardsBoxesWiderThanTheAspectRatioThreshold) {
    ByteTrackParams params;
    params.aspect_ratio_thresh = 1.6f;
    ByteTrack t{params};

    // 200x40 has aspect 5.0 — nothing person-shaped.
    const std::vector<Detection> wide{det(100.0f, 100.0f, 200.0f, 40.0f, 0.95f)};

    EXPECT_TRUE(run(t, wide, 4).empty());
}

TEST(ByteTrack, DiscardsAZeroHeightBoxWithoutDividingByZero) {
    ByteTrack t{ByteTrackParams{}};
    const std::vector<Detection> degenerate{det(100.0f, 100.0f, 40.0f, 0.0f, 0.95f)};

    EXPECT_NO_THROW(run(t, degenerate, 3));
    EXPECT_TRUE(t.update(degenerate).empty());
}

TEST(ByteTrack, KeepsTheSameLocalIdAcrossSmoothMotion) {
    ByteTrack t{ByteTrackParams{}};

    t.update({person(100.0f)});
    const auto confirmed = t.update({person(105.0f)});
    ASSERT_EQ(confirmed.size(), 1u);
    const auto id = confirmed.front().local_id;

    for (int i = 2; i <= 8; ++i) {
        const auto out = t.update({person(100.0f + 5.0f * static_cast<float>(i))});
        ASSERT_EQ(out.size(), 1u) << "lost the track at step " << i;
        EXPECT_EQ(out.front().local_id, id) << "id switched at step " << i;
    }
}

TEST(ByteTrack, RecoversATrackFromALowConfidenceDetection) {
    // This is the whole point of the two-stage cascade: a track that is
    // briefly occluded still gets associated with its weak detection
    // instead of being dropped.
    ByteTrackParams params;
    params.high_thresh = 0.5f;
    params.low_thresh = 0.1f;
    ByteTrack t{params};

    t.update({person(100.0f, 0.9f)});
    const auto confirmed = t.update({person(100.0f, 0.9f)});
    ASSERT_EQ(confirmed.size(), 1u);
    const auto id = confirmed.front().local_id;

    // A weak detection at the same place keeps the track alive and
    // emitted, because a confirmed track stays confirmed.
    const auto out = t.update({person(100.0f, 0.2f)});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out.front().local_id, id);
}

TEST(ByteTrack, TracksTwoWellSeparatedPeopleIndependently) {
    ByteTrack t{ByteTrackParams{}};
    const std::vector<Detection> dets{person(100.0f), person(600.0f)};

    t.update(dets);
    const auto out = t.update(dets);

    ASSERT_EQ(out.size(), 2u);
    EXPECT_NE(out[0].local_id, out[1].local_id);
}

TEST(ByteTrack, DropsATrackFromTheOutputOnceItIsMissed) {
    ByteTrack t{ByteTrackParams{}};
    t.update({person(100.0f)});
    ASSERT_EQ(t.update({person(100.0f)}).size(), 1u);

    EXPECT_TRUE(t.update({}).empty());
}

TEST(ByteTrack, ReacquiresALostTrackWithTheSameId) {
    ByteTrack t{ByteTrackParams{}};
    t.update({person(100.0f)});
    const auto confirmed = t.update({person(100.0f)});
    ASSERT_EQ(confirmed.size(), 1u);
    const auto id = confirmed.front().local_id;

    t.update({});  // one occluded frame

    const auto back = t.update({person(100.0f)});
    ASSERT_EQ(back.size(), 1u);
    EXPECT_EQ(back.front().local_id, id);
}

TEST(ByteTrack, EvictsATrackThatStaysLostPastTheTrackBuffer) {
    ByteTrackParams params;
    params.track_buffer = 3;
    ByteTrack t{params};

    t.update({person(100.0f)});
    const auto confirmed = t.update({person(100.0f)});
    ASSERT_EQ(confirmed.size(), 1u);
    const auto id = confirmed.front().local_id;

    for (int i = 0; i < 8; ++i) t.update({});

    t.update({person(100.0f)});
    const auto revived = t.update({person(100.0f)});
    ASSERT_EQ(revived.size(), 1u);
    EXPECT_NE(revived.front().local_id, id);
}

TEST(ByteTrack, ReportsConfirmedStateOnEveryEmittedTrack) {
    ByteTrack t{ByteTrackParams{}};
    t.update({person(100.0f)});

    const auto out = t.update({person(100.0f)});

    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out.front().state, TrackState::Confirmed);
    EXPECT_EQ(out.front().time_since_update, 0u);
    EXPECT_GE(out.front().hit_streak, 2u);
}

TEST(ByteTrack, ResetDropsEveryTrackAndRestartsIds) {
    ByteTrack t{ByteTrackParams{}};
    t.update({person(100.0f)});
    ASSERT_EQ(t.update({person(100.0f)}).size(), 1u);

    t.reset();

    EXPECT_TRUE(t.update({}).empty());
    t.update({person(100.0f)});
    const auto after = t.update({person(100.0f)});
    ASSERT_EQ(after.size(), 1u);
    EXPECT_EQ(after.front().local_id, 1u);
}

// ------------------------------------------------------------ the factory

TEST(TrackerFactory, BuildsTheRequestedCpuTrackers) {
    TrackerConfig cfg;

    cfg.type = TrackerType::ByteTrack;
    EXPECT_NE(make_tracker(cfg), nullptr);

    cfg.type = TrackerType::Iou;
    EXPECT_NE(make_tracker(cfg), nullptr);
}

TEST(TrackerFactory, AlwaysRefusesNvDcf) {
    TrackerConfig cfg;
    cfg.type = TrackerType::NvDcf;

    // Unconditional, in every build configuration: NvDCF needs a DeepStream
    // pipeline this repo does not implement. The rejected-at-construction
    // path used to be conditional, and the other branch handed back a
    // wrapper whose ingest method nothing ever called - a tracker that
    // returned zero tracks for every frame, forever, without an error.
    EXPECT_THROW(make_tracker(cfg), std::runtime_error);
}

}  // namespace
