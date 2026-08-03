#include <gtest/gtest.h>

#include <vector>

#include "mc_tracking/crosscam/hungarian.hpp"

namespace {

using mc_tracking::crosscam::INFEASIBLE_COST;
using mc_tracking::crosscam::solve_assignment;

/// Every row is matched to a distinct column, or to -1.
void expect_valid_assignment(const std::vector<int>& match, std::size_t cols) {
    std::vector<bool> used(cols, false);
    for (const int j : match) {
        if (j < 0) continue;
        ASSERT_LT(static_cast<std::size_t>(j), cols);
        EXPECT_FALSE(used[static_cast<std::size_t>(j)]) << "column " << j << " assigned twice";
        used[static_cast<std::size_t>(j)] = true;
    }
}

TEST(Assignment, EmptyCostMatrixYieldsNoMatches) {
    EXPECT_TRUE(solve_assignment({}, 0, 0).empty());
}

TEST(Assignment, ZeroColumnsLeavesEveryRowUnmatched) {
    const auto match = solve_assignment({}, 3, 0);

    ASSERT_EQ(match.size(), 3u);
    for (const int j : match) EXPECT_EQ(j, -1);
}

TEST(Assignment, SingleCellMatchesItself) {
    const auto match = solve_assignment({0.5f}, 1, 1);

    ASSERT_EQ(match.size(), 1u);
    EXPECT_EQ(match[0], 0);
}

TEST(Assignment, PicksTheDiagonalWhenItIsClearlyCheapest) {
    // Row i is cheap only on column i.
    const std::vector<float> cost{
        0.1f, 0.9f, 0.9f,  //
        0.9f, 0.1f, 0.9f,  //
        0.9f, 0.9f, 0.1f,  //
    };

    const auto match = solve_assignment(cost, 3, 3);

    ASSERT_EQ(match.size(), 3u);
    EXPECT_EQ(match[0], 0);
    EXPECT_EQ(match[1], 1);
    EXPECT_EQ(match[2], 2);
}

TEST(Assignment, PicksTheAntiDiagonalWhenThatIsCheapest) {
    const std::vector<float> cost{
        0.9f, 0.9f, 0.1f,  //
        0.9f, 0.1f, 0.9f,  //
        0.1f, 0.9f, 0.9f,  //
    };

    const auto match = solve_assignment(cost, 3, 3);

    ASSERT_EQ(match.size(), 3u);
    EXPECT_EQ(match[0], 2);
    EXPECT_EQ(match[1], 1);
    EXPECT_EQ(match[2], 0);
}

TEST(Assignment, NeverAssignsTheSameColumnTwice) {
    // Both rows would prefer column 0; exactly one of them can have it.
    const std::vector<float> cost{
        0.1f, 0.2f,  //
        0.1f, 0.3f,  //
    };

    const auto match = solve_assignment(cost, 2, 2);

    ASSERT_EQ(match.size(), 2u);
    expect_valid_assignment(match, 2);
    EXPECT_NE(match[0], match[1]);
}

TEST(Assignment, LeavesSurplusRowsUnmatchedWhenRowsExceedColumns) {
    const std::vector<float> cost{
        0.1f,  //
        0.2f,  //
        0.3f,  //
    };

    const auto match = solve_assignment(cost, 3, 1);

    ASSERT_EQ(match.size(), 3u);
    expect_valid_assignment(match, 1);

    int assigned = 0;
    for (const int j : match) {
        if (j >= 0) ++assigned;
    }
    EXPECT_EQ(assigned, 1) << "only one row can take the single column";
}

TEST(Assignment, MatchesEveryRowWhenColumnsExceedRows) {
    const std::vector<float> cost{
        0.1f, 0.5f, 0.9f,  //
        0.9f, 0.1f, 0.5f,  //
    };

    const auto match = solve_assignment(cost, 2, 3);

    ASSERT_EQ(match.size(), 2u);
    EXPECT_GE(match[0], 0);
    EXPECT_GE(match[1], 0);
    expect_valid_assignment(match, 3);
}

TEST(Assignment, NeverSelectsAnInfeasibleCell) {
    // Row 0 may only take column 1; row 1 may only take column 0.
    const std::vector<float> cost{
        INFEASIBLE_COST, 0.2f,  //
        0.3f, INFEASIBLE_COST,  //
    };

    const auto match = solve_assignment(cost, 2, 2);

    ASSERT_EQ(match.size(), 2u);
    EXPECT_EQ(match[0], 1);
    EXPECT_EQ(match[1], 0);
}

TEST(Assignment, LeavesARowUnmatchedWhenEveryCellIsInfeasible) {
    const std::vector<float> cost{
        INFEASIBLE_COST, INFEASIBLE_COST,  //
        0.3f, 0.4f,                        //
    };

    const auto match = solve_assignment(cost, 2, 2);

    ASSERT_EQ(match.size(), 2u);
    EXPECT_EQ(match[0], -1) << "row 0 has no feasible column";
    EXPECT_GE(match[1], 0);
}

TEST(Assignment, HandlesAFullyInfeasibleMatrix) {
    const std::vector<float> cost(4, INFEASIBLE_COST);

    const auto match = solve_assignment(cost, 2, 2);

    ASSERT_EQ(match.size(), 2u);
    EXPECT_EQ(match[0], -1);
    EXPECT_EQ(match[1], -1);
}

TEST(Assignment, IsUnaffectedByAConstantOffsetOnEveryCell) {
    // Adding the same constant everywhere cannot change which pairing is
    // optimal, so the solver must return the same answer.
    const std::vector<float> base{
        0.1f, 0.9f, 0.4f,  //
        0.7f, 0.2f, 0.8f,  //
        0.6f, 0.5f, 0.3f,  //
    };
    std::vector<float> shifted = base;
    for (float& c : shifted) c += 10.0f;

    EXPECT_EQ(solve_assignment(base, 3, 3), solve_assignment(shifted, 3, 3));
}

TEST(Assignment, IsUnaffectedByAConstantRowOffset) {
    // Munkres subtracts the row minimum first, so a per-row offset must
    // also leave the result unchanged.
    const std::vector<float> base{
        0.1f, 0.9f,  //
        0.7f, 0.2f,  //
    };
    const std::vector<float> shifted{
        0.1f + 5.0f, 0.9f + 5.0f,  //
        0.7f, 0.2f,                //
    };

    EXPECT_EQ(solve_assignment(base, 2, 2), solve_assignment(shifted, 2, 2));
}

}  // namespace
