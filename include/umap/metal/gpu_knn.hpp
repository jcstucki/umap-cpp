// Exact brute-force kNN on the GPU (Apple Metal) — Phase 3, branch metal-gpu.
//
// One GPU thread per query row computes distances to all data rows and keeps a
// per-thread insertion-sorted top-k (k <= 64). Distance formulas match
// src/distances.cpp (umap/distances.py) but are evaluated in float32 on
// device (the CPU reference evaluates in float64), so distances agree to
// ~1e-5 relative and neighbour indices are identical up to genuine ties.
//
// Useful as an exact-kNN path for small/medium N and to validate NN-descent
// recall. Only available when built with UMAP_ENABLE_METAL; check
// umap::metal_available() first.
#pragma once

#include <cstdint>

namespace umap {
namespace metal {

enum class KnnMetric {
    euclidean,  // sqrt(sum (x-y)^2)
    cosine,     // 1 - <x,y>/(|x||y|), zero-norm special cases as in distances.py
};

// data:    n x dim row-major float32
// queries: nq x dim row-major float32 (may be `data` itself)
// out_idx: nq x k int32 (ascending by distance; -1 padding when k > n)
// out_dists: nq x k float32
// Returns false (outputs untouched) if no Metal device / library is available.
// Throws std::invalid_argument for k < 1 or k > 64.
bool gpu_brute_force_knn(const float* data, int n, int dim,
                         const float* queries, int nq, int k, KnnMetric metric,
                         int32_t* out_idx, float* out_dists);

}  // namespace metal
}  // namespace umap
