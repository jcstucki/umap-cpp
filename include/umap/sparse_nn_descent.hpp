// Faithful C++ port of the approximate kNN path UMAP uses for *sparse* (CSR)
// input data: pynndescent's sparse RP-tree forest initialisation
// (rp_trees.py::make_sparse_tree + the sparse random projection splits) and
// the sparse NN-descent loop (sparse_nndescent.py), as invoked by
// umap_.py::nearest_neighbors -> NNDescent(csr_matrix, ...)
// (reference/pynndescent, BSD 2 clause; reference/umap, BSD 3 clause).
//
// Scope: CSR float32 data with sorted, duplicate-free column indices per row
// (scipy sort_indices() semantics; NNDescent calls sort_indices() itself when
// needed). The default UMAP fit path is covered: tree_init=True,
// init_graph=None, no metric kwds. Metrics are the sparse registry in
// umap/sparse_distances.hpp (euclidean/l2, sqeuclidean, manhattan/l1/taxicab,
// cosine); anything else throws std::invalid_argument from
// spdist::resolve_sparse_metric.
//
// Structure mirrors the dense port (umap/nn_descent.hpp) and *reuses* its
// building blocks unchanged -- NNHeap, checked_flagged_heap_push,
// new_build_candidates, deheap_sort, LeafArray, NumpyRandomState, pymod --
// so dense behaviour is untouched. Only the distance kernels, the RP-tree
// splits, and the CSR plumbing differ, exactly as in Python.
//
// Faithfulness notes (vs the numba reference):
//  * sparse_nndescent.py's generate_leaf_updates partitions leaves per virtual
//    thread with ceil division ((n + t - 1) // t), unlike the dense
//    pynndescent_.py version's (n // t) + 1. Both forms are replicated in
//    their respective ports (the partition affects update ordering and hence
//    heap tie-breaking).
//  * the sparse euclidean split draws its two anchor points with
//    abs(tau_rand_int) % m (the dense euclidean split has no abs); the
//    angular split uses tau_rand_int % m with Python modulo. Coin flips
//    likewise: abs(...) % 2 (euclidean) vs ... % 2 (angular).
//  * numba type promotion is replicated: hyperplane/offset values stay
//    float32 (sparse_diff/sparse_sum/sparse_mul), margins and offsets
//    accumulate in float64, the angular split's point norms are promoted to
//    float64 by the `norm = 1.0` fallback unification while the hyperplane
//    norm stays float32 (declared numba local).
//
// Determinism / seeding: identical to the dense port -- NNDescent.__init__
// draws rng_state then search_rng_state, then make_forest draws one
// (n_trees, 3) block of per-tree states, all from np.random.RandomState;
// nnd::NumpyRandomState reproduces those draws bit-for-bit. Threading is
// single-threaded with the same "virtual thread" semantics as the dense port
// (init_rp_tree n_threads=8, NN-descent loop n_threads=1).
//
// ---------------------------------------------------------------------------
// Integration surface for the UMAP class (owned by another agent; do NOT
// reach into umap.hpp/umap.cpp from here). What umap_.py does for sparse X:
//
//  fit(X) with scipy.sparse.issparse(X):
//    X = X.tocsr(); X.sort_indices()   [check_array(accept_sparse="csr")]
//    if n < 4096 and not force_approximation_algorithm:
//        dmat = sklearn pairwise_distances(X, metric)  (float64 dense)
//        fuzzy_simplicial_set(dmat, "precomputed")     -> stable-argsort kNN
//        == call sparse_brute_force_knn() below, then feed
//           fuzzy_simplicial_set exactly as in the dense small-data path.
//    else:
//        nn_metric = metric if metric in sparse_named_distances else callable
//        knn = nearest_neighbors(X, ...) -> NNDescent sparse branch
//        == call nearest_neighbors_sparse() below (routes identically,
//           including the <4096 cutoff when force_approximation is false).
//    Everything downstream (smooth_knn_dist, membership strengths, spectral
//    init, SGD) consumes only (knn_indices, knn_dists) and is shared with
//    the dense path.
//
//  transform(X_new) for sparse training data uses the pynndescent *sparse*
//  prepared-search path (sparse hub trees + sparse graph search), which is
//  NOT ported yet; a UMAP class should throw for transform-on-sparse until
//  then (fit/fit_transform are fully covered).
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "umap/nn_descent.hpp"
#include "umap/sparse_distances.hpp"

namespace umap {
namespace spnnd {

// Borrowed dense-port building blocks (see nn_descent.hpp).
using nnd::LeafArray;
using nnd::NNHeap;
using nnd::NumpyRandomState;

// A non-owning view of a CSR matrix (scipy layout): row i's column indices
// are indices[indptr[i] : indptr[i+1]] (sorted ascending, no duplicates) with
// values data[indptr[i] : indptr[i+1]].
struct CsrView {
    const int32_t* indptr = nullptr;   // length n_rows + 1
    const int32_t* indices = nullptr;  // length indptr[n_rows]
    const float* data = nullptr;       // length indptr[n_rows]
    int64_t n_rows = 0;
    int64_t n_cols = 0;  // informational only (no ported metric needs it)

    const int32_t* row_ind(int64_t i) const { return indices + indptr[i]; }
    const float* row_data(int64_t i) const { return data + indptr[i]; }
    int64_t row_nnz(int64_t i) const {
        return static_cast<int64_t>(indptr[i + 1]) - indptr[i];
    }
};

// ------------------------------------------------------------------ RP trees

// Ports of rp_trees.py::sparse_{euclidean,angular}_random_projection_split.
// `indices` are the data-point ids in the current tree node; `left`/`right`
// receive the two sides in input order. The (sparse) hyperplane and offset
// are only needed by search trees and are not returned.
void sparse_euclidean_random_projection_split(
    const CsrView& X, const std::vector<int32_t>& indices,
    int64_t rng_state[3], std::vector<int32_t>& left,
    std::vector<int32_t>& right);
void sparse_angular_random_projection_split(
    const CsrView& X, const std::vector<int32_t>& indices,
    int64_t rng_state[3], std::vector<int32_t>& left,
    std::vector<int32_t>& right);

// Builds one sparse tree (make_sparse_tree) but records only its leaves, in
// recursion-emission order (== get_leaves_from_tree order, which skips the
// [-1] sentinel rows of internal nodes). Returns the tree's max leaf size
// (>= leaf_size if an unsplittable node exceeded it).
int64_t make_sparse_tree_leaves(const CsrView& X, int64_t rng_state[3],
                                int64_t leaf_size, bool angular,
                                int64_t max_depth,
                                std::vector<std::vector<int32_t>>& leaves_out);

// make_forest (sparse branch) + rptree_leaf_array: per-tree rng states are
// drawn from `numpy_rs` exactly as make_forest does (one (n_trees, 3) randint
// block, row-major). leaf_size < 0 selects max(60, min(256, 5 * n_neighbors)).
LeafArray make_sparse_forest_leaf_array(const CsrView& X, int64_t n_neighbors,
                                        int64_t n_trees, int64_t leaf_size,
                                        bool angular, int64_t max_depth,
                                        NumpyRandomState& numpy_rs);

// ---------------------------------------------------------------- NN-descent

using SparseDistFn = spdist::SparseDistFn;

// sparse_nndescent.py::init_rp_tree (virtual n_threads=8 partitioning, ceil
// leaves-per-thread split -- see header notes) and ::init_random.
void sparse_init_rp_tree(const CsrView& X, SparseDistFn dist, NNHeap& graph,
                         const LeafArray& leaf_array);
void sparse_init_random(int64_t n_neighbors, const CsrView& X, NNHeap& heap,
                        SparseDistFn dist, int64_t rng_state[3]);

// Full sparse_nndescent.py::nn_descent for the EMPTY init graph path:
// rp-tree init (if leaf_array non-empty), random fill, iterated candidate
// processing (array-update path, n_threads=1), deheap_sort. Result heap rows
// are sorted ascending by (internal, uncorrected) distance on return.
NNHeap sparse_nn_descent(const CsrView& X, int64_t n_neighbors,
                         int64_t rng_state[3], int64_t max_candidates,
                         SparseDistFn dist, int64_t n_iters, double delta,
                         const LeafArray& leaf_array, bool verbose = false);

// ------------------------------------------------------------------ toplevel

// Exact kNN by full pairwise scan of the *true* sparse metric, ties broken by
// ascending index like np.argsort(kind="stable") (fast_knn_indices). This is
// what UMAP's small-data (< 4096) path reduces to; upstream computes the
// distance matrix with sklearn pairwise_distances in float64 whereas this
// uses the float32-faithful sparse kernels, so distances can differ in the
// last ulps (indices agree except on sub-float32 ties). Each point's own
// index appears first at distance 0. Outputs are (n, n_neighbors) row-major.
void sparse_brute_force_knn(const CsrView& X, int64_t n_neighbors,
                            const std::string& metric, int32_t* knn_indices,
                            float* knn_dists);

// Port of the umap_.py routing for sparse X, mirroring the dense
// nnd::nearest_neighbors signature. Replicates UMAP.fit's dispatch:
//   n < 4096 && !force_approximation  -> exact sparse brute-force kNN
//   otherwise                         -> NNDescent sparse branch with UMAP's
//     hacked values (n_trees = min(64, 5 + round(sqrt(n)/20)),
//     n_iters = max(5, round(log2 n)), max_candidates = 60) and the fast
//     alternative metric + final correction.
// `random_seed` corresponds to np.random.RandomState(random_seed) as UMAP's
// random_state. `angular` mirrors umap's angular_rp_forest parameter; as in
// pynndescent it only matters for callable metrics, so tree flavour is
// decided by the metric name. Outputs are (n, n_neighbors) row-major;
// unfilled slots (if any) are index -1 / distance +inf.
void nearest_neighbors_sparse(const int32_t* indptr, const int32_t* indices,
                              const float* data, int64_t n, int64_t dim,
                              int64_t n_neighbors, const std::string& metric,
                              bool angular, uint32_t random_seed,
                              int32_t* knn_indices, float* knn_dists,
                              bool force_approximation = false,
                              bool verbose = false);

// Same, but always the NN-descent path with explicit tau-prng states (for
// tests / callers that extracted states from Python): `rng_state` is
// NNDescent.rng_state and `tree_rng_states` the per-tree states drawn in
// make_forest.
void nearest_neighbors_sparse_with_state(
    const int32_t* indptr, const int32_t* indices, const float* data,
    int64_t n, int64_t dim, int64_t n_neighbors, const std::string& metric,
    bool angular, const int64_t rng_state[3],
    const std::vector<std::array<int64_t, 3>>& tree_rng_states,
    int32_t* knn_indices, float* knn_dists, bool verbose = false);

}  // namespace spnnd
}  // namespace umap
