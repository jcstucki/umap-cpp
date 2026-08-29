// Exact brute-force kNN on the GPU (Apple Metal) — Phase 3, branch metal-gpu;
// extended to the full tiled engine in the gpu-knn pass (2026-08-29).
//
// One GPU thread per query row streams data tiles through threadgroup memory
// (any n, any n_queries) and keeps a per-thread top-k: an insertion-sorted
// register list for k <= 64, a (dist, index)-lexicographic heap for
// 64 < k <= 256 — both emit the identical ascending (dist, index) order.
// Distance formulas match src/distances.cpp (umap/distances.py) but are
// evaluated in float32 on device (the CPU reference evaluates in float64), so
// distances agree to ~1e-5 relative and neighbour indices are identical up to
// genuine ties. Dense metrics are function-constant pipeline specializations
// (one pipeline per metric, lazily created); sparse CSR input is covered for
// the norm-trick family (euclidean / sqeuclidean / cosine).
//
// Useful as the exact-kNN path of device=GPU fits/transforms and to validate
// NN-descent recall. Only available when built with UMAP_ENABLE_METAL; check
// umap::metal_available() first.
#pragma once

#include <cstdint>
#include <vector>

namespace umap {
namespace metal {

// Order must match the metric ids in src/metal/umap_kernels.metal.
enum class KnnMetric {
    euclidean,    // sqrt(sum (x-y)^2)
    cosine,       // 1 - <x,y>/(|x||y|), zero-norm special cases as distances.py
    sqeuclidean,  // sum (x-y)^2
    manhattan,    // sum |x-y|
    chebyshev,    // max |x-y|
    minkowski,    // (sum |x-y|^p)^(1/p), p from KnnMetricParams
    correlation,  // 1 - centred cosine, zero cases as distances.py
    hamming,      // mean(x_i != y_i)
    canberra,     // sum |x-y| / (|x|+|y|), zero-denominator terms skipped
    braycurtis,   // sum|x-y| / sum|x+y|, 0 when the denominator is 0
    seuclidean,   // sqrt(sum (x-y)^2 / v_i)          (aux: v)
    wminkowski,   // (sum w_i |x-y|^p)^(1/p)          (aux: v as w, p)
    mahalanobis,  // sqrt((x-y)^T Vinv (x-y))         (aux: vinv)
};

// Parameters for the parameterised metrics; mirrors umap::MetricKwds
// semantics (empty v -> ones, empty vinv -> identity).
struct KnnMetricParams {
    float p = 2.0f;           // minkowski / wminkowski exponent
    std::vector<float> v;     // seuclidean sigma / wminkowski w (dim floats)
    std::vector<float> vinv;  // mahalanobis V^-1, row-major dim x dim
};

// data:    n x dim row-major float32
// queries: nq x dim row-major float32 (may be `data` itself)
// out_idx: nq x k int32 (ascending by (distance, index); -1 padding if k > n)
// out_dists: nq x k float32
// metric_params: required content only for minkowski (p), seuclidean (v),
//   wminkowski (v, p), mahalanobis (vinv); nullptr uses the defaults.
// Returns false (outputs untouched) if no Metal device / library is
// available or a buffer would exceed the device's maximum length.
// Throws std::invalid_argument for k < 1 or k > 256, or mis-sized aux arrays.
bool gpu_brute_force_knn(const float* data, int n, int dim,
                         const float* queries, int nq, int k, KnnMetric metric,
                         int32_t* out_idx, float* out_dists,
                         const KnnMetricParams* metric_params = nullptr);

// CSR sparse variant (norm-trick family only: euclidean / sqeuclidean /
// cosine; other KnnMetric values throw std::invalid_argument). Column
// indices must be sorted ascending within each row (the library-wide CSR
// invariant). The query triple may alias the data triple for self-kNN.
// Same output conventions / return semantics as the dense engine.
bool gpu_sparse_brute_force_knn(const int32_t* d_indptr,
                                const int32_t* d_indices, const float* d_data,
                                int n, const int32_t* q_indptr,
                                const int32_t* q_indices, const float* q_data,
                                int nq, int k, KnnMetric metric,
                                int32_t* out_idx, float* out_dists);

}  // namespace metal
}  // namespace umap
