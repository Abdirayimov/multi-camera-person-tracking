#pragma once

#include <Eigen/Core>

namespace mc_tracking::reid {

/// An L2-normalized appearance feature vector.
///
/// Every producer in this project (OSNet today) is expected to return
/// unit-norm vectors, which is what lets consumers treat a dot product
/// as a cosine similarity. The alias lives in its own header so the
/// gallery and the cross-camera matcher do not have to include the
/// extractor — and therefore do not drag TensorRT into a CPU-only build.
using Embedding = Eigen::Matrix<float, Eigen::Dynamic, 1>;

}  // namespace mc_tracking::reid
