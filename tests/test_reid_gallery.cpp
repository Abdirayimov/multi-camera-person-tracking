#include <gtest/gtest.h>

#include <cmath>

#include "mc_tracking/reid/reid_gallery.hpp"

namespace {

using mc_tracking::reid::Embedding;
using mc_tracking::reid::ReidGallery;

constexpr int kDim = 4;

/// A unit-norm embedding with all of its mass on one axis. Two such
/// vectors on different axes are orthogonal, which makes the expected
/// cosine similarities exact rather than approximate.
Embedding basis(int axis) {
    Embedding e = Embedding::Zero(kDim);
    e[axis] = 1.0f;
    return e;
}

/// A unit-norm blend of two axes, `t` of the way from `a` to `b`.
Embedding blend(int a, int b, float t) {
    Embedding e = Embedding::Zero(kDim);
    e[a] = 1.0f - t;
    e[b] = t;
    e /= e.norm();
    return e;
}

TEST(ReidGallery, ReportsNoSimilarityForAnUnknownTrack) {
    const ReidGallery g{8};
    EXPECT_FLOAT_EQ(g.best_similarity(42, basis(0)), -1.0f);
    EXPECT_FALSE(g.has(42));
}

TEST(ReidGallery, ReportsNoMeanForAnUnknownTrack) {
    const ReidGallery g{8};
    EXPECT_FALSE(g.mean_embedding(42).has_value());
}

TEST(ReidGallery, RemembersATrackOnceSomethingIsPushed) {
    ReidGallery g{8};
    g.push(1, basis(0));

    EXPECT_TRUE(g.has(1));
    EXPECT_FLOAT_EQ(g.best_similarity(1, basis(0)), 1.0f);
}

TEST(ReidGallery, ScoresOrthogonalEmbeddingsAtZero) {
    ReidGallery g{8};
    g.push(1, basis(0));

    EXPECT_NEAR(g.best_similarity(1, basis(1)), 0.0f, 1e-6f);
}

TEST(ReidGallery, ReturnsTheBestMatchRatherThanTheAverage) {
    // This is the reason the gallery keeps a bank instead of collapsing to
    // one vector: a track seen from two viewpoints must still match either.
    ReidGallery g{8};
    g.push(1, basis(0));  // face-on
    g.push(1, basis(1));  // turned away

    // A query identical to the second observation scores 1.0. An average
    // of the two would have scored about 0.707.
    EXPECT_NEAR(g.best_similarity(1, basis(1)), 1.0f, 1e-6f);
}

TEST(ReidGallery, KeepsTracksIndependent) {
    ReidGallery g{8};
    g.push(1, basis(0));
    g.push(2, basis(1));

    EXPECT_NEAR(g.best_similarity(1, basis(0)), 1.0f, 1e-6f);
    EXPECT_NEAR(g.best_similarity(2, basis(0)), 0.0f, 1e-6f);
}

TEST(ReidGallery, EvictsTheOldestObservationOnceCapacityIsReached) {
    ReidGallery g{2};
    g.push(1, basis(0));
    g.push(1, basis(1));
    g.push(1, basis(2));  // pushes basis(0) out

    EXPECT_NEAR(g.best_similarity(1, basis(1)), 1.0f, 1e-6f);
    EXPECT_NEAR(g.best_similarity(1, basis(2)), 1.0f, 1e-6f);
    EXPECT_NEAR(g.best_similarity(1, basis(0)), 0.0f, 1e-6f)
        << "the oldest observation should no longer be reachable";
}

TEST(ReidGallery, HoldsOnlyTheNewestObservationAtCapacityOne) {
    ReidGallery g{1};
    g.push(1, basis(0));
    g.push(1, basis(1));

    EXPECT_NEAR(g.best_similarity(1, basis(1)), 1.0f, 1e-6f);
    EXPECT_NEAR(g.best_similarity(1, basis(0)), 0.0f, 1e-6f);
}

TEST(ReidGallery, AveragesAndRenormalizesTheStoredEmbeddings) {
    ReidGallery g{8};
    g.push(1, basis(0));
    g.push(1, basis(1));

    const auto mean = g.mean_embedding(1);

    ASSERT_TRUE(mean.has_value());
    EXPECT_NEAR(mean->norm(), 1.0f, 1e-6f) << "the mean must come back unit-norm";
    const float expected = 1.0f / std::sqrt(2.0f);
    EXPECT_NEAR((*mean)[0], expected, 1e-6f);
    EXPECT_NEAR((*mean)[1], expected, 1e-6f);
}

TEST(ReidGallery, MeanOfASingleObservationIsThatObservation) {
    ReidGallery g{8};
    g.push(1, basis(2));

    const auto mean = g.mean_embedding(1);

    ASSERT_TRUE(mean.has_value());
    EXPECT_NEAR((*mean)[2], 1.0f, 1e-6f);
}

TEST(ReidGallery, MeanOfOpposingEmbeddingsStaysFiniteRatherThanDividingByZero) {
    // The two cancel exactly, so the pre-normalization sum is the zero
    // vector. The guard must return it as-is instead of producing NaNs.
    ReidGallery g{8};
    Embedding pos = basis(0);
    Embedding neg = -basis(0);
    g.push(1, pos);
    g.push(1, neg);

    const auto mean = g.mean_embedding(1);

    ASSERT_TRUE(mean.has_value());
    for (int i = 0; i < kDim; ++i) {
        EXPECT_TRUE(std::isfinite((*mean)[i])) << "component " << i;
    }
}

TEST(ReidGallery, ScoresAPartialBlendBetweenZeroAndOne) {
    ReidGallery g{8};
    g.push(1, basis(0));

    const float sim = g.best_similarity(1, blend(0, 1, 0.5f));

    EXPECT_GT(sim, 0.0f);
    EXPECT_LT(sim, 1.0f);
    EXPECT_NEAR(sim, 1.0f / std::sqrt(2.0f), 1e-6f);
}

TEST(ReidGallery, EraseForgetsOneTrackAndLeavesTheRest) {
    ReidGallery g{8};
    g.push(1, basis(0));
    g.push(2, basis(1));

    g.erase(1);

    EXPECT_FALSE(g.has(1));
    EXPECT_FLOAT_EQ(g.best_similarity(1, basis(0)), -1.0f);
    EXPECT_TRUE(g.has(2));
}

TEST(ReidGallery, ErasingAnUnknownTrackIsHarmless) {
    ReidGallery g{8};
    g.push(1, basis(0));

    EXPECT_NO_THROW(g.erase(999));
    EXPECT_TRUE(g.has(1));
}

TEST(ReidGallery, ClearForgetsEverything) {
    ReidGallery g{8};
    g.push(1, basis(0));
    g.push(2, basis(1));

    g.clear();

    EXPECT_FALSE(g.has(1));
    EXPECT_FALSE(g.has(2));
}

TEST(ReidGallery, ReportsNoMatchForAMismatchedEmbeddingDimension) {
    // A caller that mixes embedding sizes has a bug upstream, but the
    // gallery must surface it as "no match" rather than running a dot
    // product off the end of the shorter vector.
    ReidGallery g{8};
    g.push(1, basis(0));

    const Embedding wrong_size = Embedding::Ones(kDim + 3).normalized();
    EXPECT_FLOAT_EQ(g.best_similarity(1, wrong_size), -1.0f);
}

}  // namespace
