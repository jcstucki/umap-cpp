// Faithful C++ port of smooth_knn_dist, compute_membership_strengths and
// fuzzy_simplicial_set from reference/umap/umap/umap_.py (lmcinnes/umap,
// BSD 3 clause). Arithmetic mirrors the numba-jitted float32 code paths.
#pragma once

#include <cstdint>
#include <vector>

#include "umap/sparse_matrix.hpp"

namespace umap {

constexpr double SMOOTH_K_TOLERANCE = 1e-5;
constexpr double MIN_K_DIST_SCALE = 1e-3;

struct SmoothKnnResult {
    std::vector<float> sigmas;  // "result" in the reference
    std::vector<float> rhos;
};

// Compute a continuous version of the distance to the kth nearest neighbor.
// `distances` is row-major (n_samples x n_neighbors), each row sorted ascending.
SmoothKnnResult smooth_knn_dist(const float* distances, int n_samples,
                                int n_neighbors, double k, int n_iter = 64,
                                double local_connectivity = 1.0,
                                double bandwidth = 1.0);

struct MembershipStrengths {
    std::vector<int32_t> rows;
    std::vector<int32_t> cols;
    std::vector<float> vals;
    std::vector<float> dists;  // filled only when return_dists=true
};

// Construct the membership strength data for the 1-skeleton of each local
// fuzzy simplicial set (COO triplets, n_samples * n_neighbors entries).
MembershipStrengths compute_membership_strengths(
    const int32_t* knn_indices, const float* knn_dists, const float* sigmas,
    const float* rhos, int n_samples, int n_neighbors,
    bool return_dists = false, bool bipartite = false);

struct FuzzySimplicialSetResult {
    SparseMatrix graph;  // symmetrised fuzzy simplicial set
    std::vector<float> sigmas;
    std::vector<float> rhos;
    bool has_dists = false;
    SparseMatrix dists;  // max(D, D^T) of edge distances, if return_dists
};

// Build the (symmetrised) fuzzy simplicial set from a precomputed kNN graph.
// knn_indices: row-major (n_samples x n_neighbors) int32 (-1 marks missing);
// knn_dists: row-major (n_samples x n_neighbors) float32, rows sorted ascending.
FuzzySimplicialSetResult fuzzy_simplicial_set(
    int n_samples, const int32_t* knn_indices, const float* knn_dists,
    int n_neighbors, double set_op_mix_ratio = 1.0,
    double local_connectivity = 1.0, bool apply_set_operations = true,
    bool return_dists = false);

}  // namespace umap
