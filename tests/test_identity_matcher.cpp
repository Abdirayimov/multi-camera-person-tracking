#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "mc_tracking/crosscam/identity_matcher.hpp"

namespace {

using mc_tracking::config::CameraEntry;
using mc_tracking::config::CamerasConfig;
using mc_tracking::config::CrossCamConfig;
using mc_tracking::config::ZoneTransition;
using mc_tracking::crosscam::CameraTrackObservation;
using mc_tracking::crosscam::IdentityMatcher;
using mc_tracking::reid::Embedding;
using mc_tracking::reid::ReidGallery;

constexpr int kDim = 4;

Embedding basis(int axis) {
    Embedding e = Embedding::Zero(kDim);
    e[axis] = 1.0f;
    return e;
}

/// A unit-norm embedding whose cosine similarity against `basis(0)` is
/// exactly `cosine`. Stating the similarity directly keeps the threshold
/// tests readable instead of hiding the value behind a normalisation.
Embedding at_similarity(float cosine) {
    Embedding e = Embedding::Zero(kDim);
    e[0] = cosine;
    e[1] = std::sqrt(1.0f - cosine * cosine);
    return e;
}

/// A fixed instant plus an offset, so tests control the spatial-overlap
/// window exactly rather than depending on wall-clock timing.
std::chrono::steady_clock::time_point at_ms(long ms) {
    static const auto epoch = std::chrono::steady_clock::now();
    return epoch + std::chrono::milliseconds(ms);
}

CameraTrackObservation obs(const std::string& camera, std::uint64_t local_id, const Embedding& emb,
                           long ms = 0, const std::string& zone = "") {
    CameraTrackObservation o;
    o.camera_id = camera;
    o.zone = zone;
    o.local_id = local_id;
    o.pts = at_ms(ms);
    o.bbox = cv::Rect2f(0.0f, 0.0f, 40.0f, 100.0f);
    o.embedding = emb;
    return o;
}

CamerasConfig two_cameras(const std::string& zone_a, const std::string& zone_b) {
    CamerasConfig c;
    c.cameras.push_back(CameraEntry{"cam_a", "file://a.mp4", zone_a});
    c.cameras.push_back(CameraEntry{"cam_b", "file://b.mp4", zone_b});
    return c;
}

// ------------------------------------------------------------ fresh ids

TEST(IdentityMatcher, AssignsAGlobalIdToTheFirstObservation) {
    IdentityMatcher m{CrossCamConfig{}, CamerasConfig{}};

    const auto ids = m.update({obs("cam_a", 1, basis(0))});

    ASSERT_EQ(ids.size(), 1u);
    EXPECT_GT(ids[0], 0u);
    EXPECT_EQ(m.total_global_ids(), 1u);
}

TEST(IdentityMatcher, HandlesAnEmptyFrame) {
    IdentityMatcher m{CrossCamConfig{}, CamerasConfig{}};

    EXPECT_TRUE(m.update({}).empty());
    EXPECT_EQ(m.total_global_ids(), 0u);
}

TEST(IdentityMatcher, GivesDissimilarPeopleDistinctGlobalIds) {
    IdentityMatcher m{CrossCamConfig{}, CamerasConfig{}};

    const auto ids = m.update({obs("cam_a", 1, basis(0)), obs("cam_a", 2, basis(1))});

    ASSERT_EQ(ids.size(), 2u);
    EXPECT_NE(ids[0], ids[1]);
    EXPECT_EQ(m.total_global_ids(), 2u);
}

// ----------------------------------------------------------- stability

TEST(IdentityMatcher, KeepsTheGlobalIdStableForTheSameCameraAndLocalId) {
    IdentityMatcher m{CrossCamConfig{}, CamerasConfig{}};

    const auto first = m.update({obs("cam_a", 1, basis(0), 0)});
    const auto second = m.update({obs("cam_a", 1, basis(0), 40)});
    const auto third = m.update({obs("cam_a", 1, basis(0), 80)});

    EXPECT_EQ(first[0], second[0]);
    EXPECT_EQ(second[0], third[0]);
    EXPECT_EQ(m.total_global_ids(), 1u) << "a carried-over track must not mint a new id";
}

TEST(IdentityMatcher, KeepsTheCarriedOverIdEvenWhenTheAppearanceDrifts) {
    // Once (camera, local_id) is bound to a global id, the single-camera
    // tracker owns that identity — the matcher must not re-litigate it
    // because the person turned around.
    IdentityMatcher m{CrossCamConfig{}, CamerasConfig{}};

    const auto first = m.update({obs("cam_a", 1, basis(0), 0)});
    const auto later = m.update({obs("cam_a", 1, basis(1), 40)});

    EXPECT_EQ(first[0], later[0]);
    EXPECT_EQ(m.total_global_ids(), 1u);
}

// -------------------------------------------------------- cross-camera

TEST(IdentityMatcher, LinksTheSamePersonAcrossTwoCameras) {
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    cfg.hungarian_cost_cap = 0.4f;
    IdentityMatcher m{cfg, two_cameras("hall", "lobby")};

    const auto a = m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    const auto b = m.update({obs("cam_b", 7, basis(0), 500, "lobby")});

    ASSERT_EQ(b.size(), 1u);
    EXPECT_EQ(b[0], a[0]) << "an identical embedding should reuse the global id";
    EXPECT_EQ(m.total_global_ids(), 1u);
}

TEST(IdentityMatcher, RefusesToLinkWhenTheAppearanceIsBelowTheThreshold) {
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    IdentityMatcher m{cfg, two_cameras("hall", "lobby")};

    m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    // Orthogonal: cosine 0, far under the 0.7 threshold.
    m.update({obs("cam_b", 7, basis(1), 500, "lobby")});

    EXPECT_EQ(m.total_global_ids(), 2u);
}

TEST(IdentityMatcher, RefusesToLinkWhenTheCostExceedsTheHungarianCap) {
    // Similarity 0.75 clears reid_threshold but leaves a cost of 0.25,
    // which the tighter cap rejects.
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    cfg.hungarian_cost_cap = 0.1f;
    IdentityMatcher m{cfg, two_cameras("hall", "lobby")};

    m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    m.update({obs("cam_b", 7, at_similarity(0.75f), 500, "lobby")});

    EXPECT_EQ(m.total_global_ids(), 2u);
}

TEST(IdentityMatcher, RefusesToLinkOutsideTheSpatialOverlapWindow) {
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    cfg.spatial_overlap_window_ms = 1000;
    IdentityMatcher m{cfg, two_cameras("hall", "lobby")};

    m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    // Same person, but five seconds later — too long to be the same
    // physical hand-off.
    m.update({obs("cam_b", 7, basis(0), 5000, "lobby")});

    EXPECT_EQ(m.total_global_ids(), 2u);
}

TEST(IdentityMatcher, LinksInsideTheSpatialOverlapWindow) {
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    cfg.spatial_overlap_window_ms = 5000;
    IdentityMatcher m{cfg, two_cameras("hall", "lobby")};

    const auto a = m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    const auto b = m.update({obs("cam_b", 7, basis(0), 4000, "lobby")});

    EXPECT_EQ(b[0], a[0]);
}

// ----------------------------------------------------- zone topology

TEST(IdentityMatcher, LinksAcrossAnyZonePairWhenNoTopologyIsDeclared) {
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    IdentityMatcher m{cfg, two_cameras("hall", "roof")};

    const auto a = m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    const auto b = m.update({obs("cam_b", 7, basis(0), 500, "roof")});

    EXPECT_EQ(b[0], a[0]) << "an empty transition table means any-to-any";
}

TEST(IdentityMatcher, BlocksAHandOffTheZoneTopologyForbids) {
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    CamerasConfig cameras = two_cameras("hall", "roof");
    // Only hall -> lobby is declared, so hall -> roof is impossible.
    cameras.cameras.push_back(CameraEntry{"cam_c", "file://c.mp4", "lobby"});
    cameras.transitions.push_back(ZoneTransition{"hall", "lobby"});
    IdentityMatcher m{cfg, cameras};

    m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    m.update({obs("cam_b", 7, basis(0), 500, "roof")});

    EXPECT_EQ(m.total_global_ids(), 2u);
}

TEST(IdentityMatcher, AllowsAHandOffTheZoneTopologyDeclares) {
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    CamerasConfig cameras = two_cameras("hall", "lobby");
    cameras.transitions.push_back(ZoneTransition{"hall", "lobby"});
    IdentityMatcher m{cfg, cameras};

    const auto a = m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    const auto b = m.update({obs("cam_b", 7, basis(0), 500, "lobby")});

    EXPECT_EQ(b[0], a[0]);
}

TEST(IdentityMatcher, TreatsAnUnlabelledZoneAsUnconstrained) {
    // A camera with no zone declared should not be silently cut off from
    // every hand-off just because a topology exists.
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    CamerasConfig cameras = two_cameras("hall", "lobby");
    cameras.transitions.push_back(ZoneTransition{"hall", "lobby"});
    IdentityMatcher m{cfg, cameras};

    const auto a = m.update({obs("cam_a", 1, basis(0), 0, "")});
    const auto b = m.update({obs("cam_b", 7, basis(0), 500, "")});

    EXPECT_EQ(b[0], a[0]);
}

// --------------------------------------------------------- the gallery

TEST(IdentityMatcher, UsesTheGalleryToRecoverAMatchTheLatestFrameWouldMiss) {
    // cam_b's newest embedding looks nothing like the person cam_a saw,
    // but the gallery still holds a face-on observation that does.
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    cfg.hungarian_cost_cap = 0.4f;
    IdentityMatcher m{cfg, two_cameras("hall", "lobby")};

    ReidGallery gallery{8};
    gallery.push(7, basis(0));  // historical, matches cam_a
    m.register_gallery("cam_b", &gallery);

    const auto a = m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    const auto b = m.update({obs("cam_b", 7, basis(1), 500, "lobby")});

    EXPECT_EQ(b[0], a[0]);
    EXPECT_EQ(m.total_global_ids(), 1u);
}

TEST(IdentityMatcher, ToleratesAGalleryRegisteredAsNull) {
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    IdentityMatcher m{cfg, two_cameras("hall", "lobby")};
    m.register_gallery("cam_b", nullptr);

    const auto a = m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    std::vector<std::uint64_t> b;
    ASSERT_NO_THROW(b = m.update({obs("cam_b", 7, basis(0), 500, "lobby")}));

    EXPECT_EQ(b[0], a[0]);
}

// ---------------------------------------------------------- contention

TEST(IdentityMatcher, NeverHandsOneGlobalIdToTwoTracksInTheSameFrame) {
    // Two people on cam_b both look like the one person cam_a saw. Only
    // one of them may inherit that global id.
    CrossCamConfig cfg;
    cfg.reid_threshold = 0.7f;
    cfg.hungarian_cost_cap = 0.4f;
    IdentityMatcher m{cfg, two_cameras("hall", "lobby")};

    const auto a = m.update({obs("cam_a", 1, basis(0), 0, "hall")});
    const auto b = m.update({
        obs("cam_b", 7, basis(0), 500, "lobby"),
        obs("cam_b", 8, basis(0), 500, "lobby"),
    });

    ASSERT_EQ(b.size(), 2u);
    EXPECT_NE(b[0], b[1]) << "the assignment must be one-to-one";
    EXPECT_TRUE(b[0] == a[0] || b[1] == a[0]);
    EXPECT_EQ(m.total_global_ids(), 2u);
}

TEST(IdentityMatcher, ReturnsOneIdPerObservationInInputOrder) {
    IdentityMatcher m{CrossCamConfig{}, CamerasConfig{}};

    const auto ids = m.update({
        obs("cam_a", 1, basis(0)),
        obs("cam_a", 2, basis(1)),
        obs("cam_a", 3, basis(2)),
    });

    ASSERT_EQ(ids.size(), 3u);
    for (const auto id : ids) EXPECT_GT(id, 0u);
}

}  // namespace
