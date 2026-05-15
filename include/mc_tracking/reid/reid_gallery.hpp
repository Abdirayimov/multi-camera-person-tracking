#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>

#include "mc_tracking/reid/osnet_extractor.hpp"

namespace mc_tracking::reid {

/// Per-track rolling appearance bank.
///
/// We do not collapse to a single average vector because a track that
/// briefly faces away from the camera produces an embedding that is
/// genuinely different from a face-on observation; cosine averages
/// would dilute both. Storing the most recent K embeddings and
/// matching against the *best* one preserves both viewpoints.
class ReidGallery {
public:
    explicit ReidGallery(std::uint32_t capacity_per_track) : capacity_(capacity_per_track) {}

    void push(std::uint64_t local_id, const Embedding& emb);
    void erase(std::uint64_t local_id);
    void clear();

    /// Best (highest) cosine similarity between `query` and any
    /// embedding stored under `local_id`. Returns -1.0f when the
    /// track has no embeddings.
    float best_similarity(std::uint64_t local_id, const Embedding& query) const;

    /// Average embedding (after L2-renorm). Returns std::nullopt when
    /// the track has no embeddings.
    std::optional<Embedding> mean_embedding(std::uint64_t local_id) const;

    bool has(std::uint64_t local_id) const noexcept {
        return galleries_.find(local_id) != galleries_.end();
    }

private:
    std::uint32_t capacity_;
    std::unordered_map<std::uint64_t, std::deque<Embedding>> galleries_;
};

}  // namespace mc_tracking::reid
