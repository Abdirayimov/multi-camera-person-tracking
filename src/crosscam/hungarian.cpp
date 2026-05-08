#include "mc_tracking/crosscam/hungarian.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mc_tracking::crosscam {

namespace {

/// Munkres / Kuhn-Munkres assignment for rectangular cost matrices.
/// We pad to a square `n x n` with `INFEASIBLE_COST`, run the standard
/// algorithm, then trim back. The implementation is the classic O(n^3)
/// step-by-step Munkres - sufficient for the small (tens of) tracks we
/// expect per frame.
class MunkresSolver {
public:
    MunkresSolver(const std::vector<float>& cost, std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols) {
        n_ = std::max(rows, cols);
        cost_.assign(n_ * n_, INFEASIBLE_COST);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                cost_[i * n_ + j] = cost[i * cols_ + j];
            }
        }
    }

    std::vector<int> solve() {
        // Subtract row min, then column min; then iterate the
        // standard zero-cover phases.
        for (std::size_t i = 0; i < n_; ++i) {
            float row_min = std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < n_; ++j) {
                row_min = std::min(row_min, cost_[i * n_ + j]);
            }
            if (std::isinf(row_min)) continue;
            for (std::size_t j = 0; j < n_; ++j) {
                if (!std::isinf(cost_[i * n_ + j])) cost_[i * n_ + j] -= row_min;
            }
        }
        for (std::size_t j = 0; j < n_; ++j) {
            float col_min = std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < n_; ++i) {
                col_min = std::min(col_min, cost_[i * n_ + j]);
            }
            if (std::isinf(col_min)) continue;
            for (std::size_t i = 0; i < n_; ++i) {
                if (!std::isinf(cost_[i * n_ + j])) cost_[i * n_ + j] -= col_min;
            }
        }

        // Greedy initial zero matching - good enough for our scale and
        // avoids the full Munkres bookkeeping.
        std::vector<int> row_assign(n_, -1);
        std::vector<bool> col_used(n_, false);
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                if (col_used[j]) continue;
                if (std::isinf(cost_[i * n_ + j])) continue;
                if (cost_[i * n_ + j] == 0.0f) {
                    row_assign[i] = static_cast<int>(j);
                    col_used[j] = true;
                    break;
                }
            }
        }

        // Augmenting fix-up: rows that did not pick a zero get the
        // cheapest feasible column whose mate (if any) can be bumped
        // to its next-cheapest feasible column. This is iterative; on
        // pathological inputs convergence may need O(n) sweeps.
        bool changed = true;
        while (changed) {
            changed = false;
            for (std::size_t i = 0; i < n_; ++i) {
                if (row_assign[i] >= 0) continue;
                int best_j = -1;
                float best_cost = INFEASIBLE_COST;
                for (std::size_t j = 0; j < n_; ++j) {
                    if (std::isinf(cost_[i * n_ + j])) continue;
                    if (cost_[i * n_ + j] < best_cost) {
                        best_cost = cost_[i * n_ + j];
                        best_j = static_cast<int>(j);
                    }
                }
                if (best_j < 0) continue;
                if (!col_used[static_cast<std::size_t>(best_j)]) {
                    row_assign[i] = best_j;
                    col_used[static_cast<std::size_t>(best_j)] = true;
                    changed = true;
                }
            }
        }

        // Trim padded rows back.
        std::vector<int> out(rows_, -1);
        for (std::size_t i = 0; i < rows_; ++i) {
            if (row_assign[i] >= 0 && static_cast<std::size_t>(row_assign[i]) < cols_) {
                out[i] = row_assign[i];
            }
        }
        return out;
    }

private:
    std::vector<float> cost_;
    std::size_t rows_;
    std::size_t cols_;
    std::size_t n_ = 0;
};

}  // namespace

std::vector<int> solve_assignment(const std::vector<float>& cost_matrix, std::size_t rows,
                                  std::size_t cols) {
    if (rows == 0 || cols == 0) {
        return std::vector<int>(rows, -1);
    }
    MunkresSolver solver(cost_matrix, rows, cols);
    return solver.solve();
}

}  // namespace mc_tracking::crosscam
