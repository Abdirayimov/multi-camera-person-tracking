#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace mc_tracking::crosscam {

/// Munkres / Hungarian assignment for rectangular cost matrices.
///
/// Input is a row-major `n x m` cost matrix. Output is a vector of
/// length `n` whose i-th entry is the column matched to row i, or
/// -1 when row i is left unmatched (which only happens for n > m).
///
/// Cells with cost >= INFEASIBLE_COST are treated as forbidden and
/// will never be selected; use this to encode "this pair is not
/// allowed" (e.g. cross-camera matches that violate spatial-temporal
/// constraints).
constexpr float INFEASIBLE_COST = std::numeric_limits<float>::infinity();

std::vector<int> solve_assignment(const std::vector<float>& cost_matrix,
                                  std::size_t rows, std::size_t cols);

}  // namespace mc_tracking::crosscam
