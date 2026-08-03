#include "mc_tracking/reid/reid_gallery.hpp"

namespace mc_tracking::reid {

void ReidGallery::push(std::uint64_t local_id, const Embedding& emb) {
    auto& d = galleries_[local_id];
    d.push_back(emb);
    while (d.size() > capacity_) d.pop_front();
}

void ReidGallery::erase(std::uint64_t local_id) {
    galleries_.erase(local_id);
}
void ReidGallery::clear() {
    galleries_.clear();
}

float ReidGallery::best_similarity(std::uint64_t local_id, const Embedding& query) const {
    const auto it = galleries_.find(local_id);
    if (it == galleries_.end() || it->second.empty()) return -1.0f;
    float best = -1.0f;
    for (const auto& e : it->second) {
        // A caller that mixes embedding dimensions has a bug upstream, but
        // dot() on mismatched sizes is undefined behaviour once Eigen's
        // asserts are compiled out. Report "no match" instead.
        if (e.size() != query.size()) continue;
        const float sim = e.dot(query);  // both L2-normalized -> cosine
        if (sim > best) best = sim;
    }
    return best;
}

std::optional<Embedding> ReidGallery::mean_embedding(std::uint64_t local_id) const {
    const auto it = galleries_.find(local_id);
    if (it == galleries_.end() || it->second.empty()) return std::nullopt;
    Embedding sum = Embedding::Zero(it->second.front().size());
    for (const auto& e : it->second) sum += e;
    sum /= static_cast<float>(it->second.size());
    const float n = sum.norm();
    if (n > 1e-12f) sum /= n;
    return sum;
}

}  // namespace mc_tracking::reid
