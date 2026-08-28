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
// init_graph=None. Metrics are the full sparse registry in
// umap/sparse_distances.hpp (pynndescent's sparse_named_distances incl. fast
// alternatives/corrections and n_features binding, plus umap-only
// ll_dirichlet); names absent upstream or unported (kantorovich family,
// jensen-shannon, symmetric-kl, dot, yule, ...) throw std::invalid_argument
// from spdist::resolve_sparse_metric. minkowski's p travels via the optional
// SparseMetricKwds parameters below.
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
//  prepared-search path (sparse hub trees + sparse graph search), ported
//  below as SparsePreparedIndex (mirroring the dense nnd::PreparedIndex).
//  Wiring for the UMAP class (owned by another agent):
//    * fit: when n >= 4096 (or force_approximation_algorithm), build the
//      index via SparsePreparedIndex::build(...) continuing the fit's
//      NumpyRandomState (draw order identical to NNDescent.__init__; prepare
//      draws nothing from the RandomState) and consume knn_indices()/
//      knn_dists() exactly like the dense path keeps its PreparedIndex.
//    * transform: X_new as CSR (sorted indices, float32);
//      epsilon = 0.24 if angular trees (cosine) else 0.12; k = n_neighbors:
//        index.query(q_indptr, q_indices, q_data, n_queries, n_neighbors,
//                    epsilon, out_ind, out_dist);
//      downstream (smooth_knn_dist with adjusted local_connectivity,
//      bipartite membership, init_graph_transform, SGD) is shared with dense.
//    * small-data fits (n < 4096) never build an index upstream; transform
//      then brute-forces kNN of X_new vs the training data
//      (sparse_brute_force_knn generalises; upstream uses sklearn
//      pairwise_distances in float64).
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
    int64_t n_cols = 0;  // bound as n_features for the metrics that need it

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
// distance matrix with sklearn pairwise_distances in float64 (falling back to
// pairwise_special_metric on densified data for names sklearn rejects)
// whereas this uses the float32-faithful sparse kernels, so distances can
// differ in the last ulps (indices agree except on sub-float32 ties). Each
// point's own index appears first at distance 0. Outputs are
// (n, n_neighbors) row-major. n_features-requiring metrics bind X.n_cols;
// `kwds` carries minkowski's p.
void sparse_brute_force_knn(
    const CsrView& X, int64_t n_neighbors, const std::string& metric,
    int32_t* knn_indices, float* knn_dists,
    const spdist::SparseMetricKwds& kwds = spdist::SparseMetricKwds());

// Port of the umap_.py routing for sparse X, mirroring the dense
// nnd::nearest_neighbors signature. Replicates UMAP.fit's dispatch:
//   n < 4096 && !force_approximation  -> exact sparse brute-force kNN
//   otherwise                         -> NNDescent sparse branch with UMAP's
//     hacked values (n_trees = min(64, 5 + round(sqrt(n)/20)),
//     n_iters = max(5, round(log2 n)), max_candidates = 60) and the fast
//     alternative metric + final correction.
// `random_seed` corresponds to np.random.RandomState(random_seed) as UMAP's
// random_state. `angular` mirrors umap's angular_rp_forest parameter; as in
// pynndescent it only matters for callable metrics -- here that is
// "ll_dirichlet", which upstream NN-descent receives as a callable (all other
// tree flavours are decided by the metric name). `kwds` carries minkowski's
// p; n_features-requiring metrics bind `dim`. Outputs are (n, n_neighbors)
// row-major; unfilled slots (if any) are index -1 / distance +inf.
void nearest_neighbors_sparse(
    const int32_t* indptr, const int32_t* indices, const float* data,
    int64_t n, int64_t dim, int64_t n_neighbors, const std::string& metric,
    bool angular, uint32_t random_seed, int32_t* knn_indices, float* knn_dists,
    bool force_approximation = false, bool verbose = false,
    const spdist::SparseMetricKwds& kwds = spdist::SparseMetricKwds());

// Same, but always the NN-descent path with explicit tau-prng states (for
// tests / callers that extracted states from Python): `rng_state` is
// NNDescent.rng_state and `tree_rng_states` the per-tree states drawn in
// make_forest.
void nearest_neighbors_sparse_with_state(
    const int32_t* indptr, const int32_t* indices, const float* data,
    int64_t n, int64_t dim, int64_t n_neighbors, const std::string& metric,
    bool angular, const int64_t rng_state[3],
    const std::vector<std::array<int64_t, 3>>& tree_rng_states,
    int32_t* knn_indices, float* knn_dists, bool verbose = false,
    const spdist::SparseMetricKwds& kwds = spdist::SparseMetricKwds());

// --------------------------------------------------- query path (transform)
// Faithful port of NNDescent.prepare() + query() for the *sparse* path
// UMAP's transform() hits when the training data was sparse: graph-informed
// sparse hub search tree (rp_trees.py::make_sparse_hub_tree +
// convert_tree_format's sparse branch), occlusion-pruned search graph
// (sparse.py::diversify / diversify_csr + the shared degree_prune /
// symmetric-max machinery), and the sparse epsilon-bounded graph search
// (pynndescent_.py::_init_sparse_search_function's search_closure).
//
// Faithfulness notes (vs the numba reference):
//  * the flat sparse tree stores each internal node's hyperplane as a padded
//    (2, n_samples) float32 block whose row 0 holds the CSR column indices as
//    floats; sparse_select_side re-derives the true length by trimming
//    trailing negative entries. The port stores the trimmed (int32, float32)
//    pairs directly -- an exact representation of the same information
//    (indices < 2^24 round-trip through float32 losslessly; degenerate
//    fallback splits store the sentinel [-1] hyperplane, which trims empty).
//  * sparse.py::diversify_csr indexes its occluder correctly via order[k]
//    (unlike the dense diversify_csr's raw-position quirk, which the dense
//    port replicates separately); both behaviours are kept as upstream.
//  * the sparse search closure's epsilon bound is
//    float32(1 + epsilon) * heap_priorities[0] (min_distance is NOT used,
//    unlike the dense closure), and query vectors are never normalised: the
//    reference's `dist == alternative_dot/alternative_cosine` check compares
//    against the *dense* kernels, which a sparse _distance_func never equals.
//    Both quirks are replicated (cosine is scale-invariant, so only float
//    rounding is affected).

// Converted flat sparse search tree (convert_tree_format sparse layout, with
// hyperplanes held trimmed per node -- see notes above). Internal nodes:
// children[2i] = left node, children[2i+1] = right node (> 0). Leaves:
// children[2i] = -leaf_start, children[2i+1] = -leaf_end into `indices`.
struct SparseFlatTree {
    int64_t n_nodes = 0;
    std::vector<std::vector<int32_t>> hyperplane_inds;  // per node (trimmed)
    std::vector<std::vector<float>> hyperplane_data;    // per node (trimmed)
    std::vector<float> offsets;                         // leaves: -inf
    std::vector<int32_t> children;                      // n_nodes * 2
    std::vector<int32_t> indices;                       // concatenated leaves
};

// rp_trees.py::make_sparse_hub_tree followed by convert_tree_format. The
// neighbor graph (`knn_indices`, (n, n_neighbors) row-major, -1 padded)
// drives the top-3-hub edge-cut-minimising hyperplane selection; `rng_state`
// is only consumed by the degenerate-split random fallback. UMAP defaults:
// leaf_size=30, max_depth=200.
SparseFlatTree make_sparse_hub_tree(const CsrView& X,
                                    const int32_t* knn_indices,
                                    int64_t n_neighbors, int64_t rng_state[3],
                                    int64_t leaf_size, bool angular,
                                    int64_t max_depth);

// Port of the graph half of NNDescent._init_search_graph for sparse data:
// forward sparse.diversify on a copy of the kNN graph, CSR assembly (zero
// distances -> FLOAT32_EPS, -1 entries dropped), sparse.diversify_csr
// applied to the forward rows (scipy's zero-copy transpose quirk, exactly as
// in the dense port), symmetric maximum combine, diagonal removal,
// degree_prune to round(pruning_degree_multiplier * n_neighbors), binarise.
// `knn_dists` are the *internal* (uncorrected) metric distances; `dist` the
// matching internal metric. `rng_state` threads through both diversify
// passes exactly as in Python.
nnd::SearchGraph build_sparse_search_graph(
    const CsrView& X, const int32_t* knn_indices, const float* knn_dists,
    int64_t n_neighbors, SparseDistFn dist, int64_t rng_state[3],
    double diversify_prob, double pruning_degree_multiplier);

// A fitted + prepared index over a copy of the (CSR) training data. Mirrors
// what UMAP keeps as `_knn_search_index` for sparse input: fit-time kNN
// graph (for fuzzy_simplicial_set) plus everything query() needs.
class SparsePreparedIndex {
 public:
    // NNDescent(csr_matrix, n_neighbors, metric, random_state=seed,
    //           n_trees/n_iters = umap "hacked values", max_candidates=60,
    //           low_memory=True, compressed=False) followed by .prepare().
    // `kwds` carries minkowski's p; n_features-requiring metrics bind `dim`.
    // NB "ll_dirichlet" builds euclidean trees here (== upstream's
    // angular_rp_forest default False for the callable-metric branch).
    static SparsePreparedIndex build(
        const int32_t* indptr, const int32_t* indices, const float* data,
        int64_t n, int64_t dim, int64_t n_neighbors, const std::string& metric,
        uint32_t random_seed, bool verbose = false,
        const spdist::SparseMetricKwds& kwds = spdist::SparseMetricKwds());

    // Overload continuing an existing numpy RandomState stream (for the UMAP
    // class, so init/SGD seeding after NN search stays stream-parallel with
    // Python). Draw order and count are identical to NNDescent.__init__
    // (prepare() draws nothing from the RandomState).
    static SparsePreparedIndex build(
        const int32_t* indptr, const int32_t* indices, const float* data,
        int64_t n, int64_t dim, int64_t n_neighbors, const std::string& metric,
        NumpyRandomState& numpy_rs, bool verbose = false,
        const spdist::SparseMetricKwds& kwds = spdist::SparseMetricKwds());

    // NNDescent.query(query_data, k, epsilon) for CSR query rows (sorted,
    // duplicate-free column indices; float32 values). Outputs are
    // (n_queries, k) row-major; indices refer to the original data order.
    // UMAP's transform calls this with k = n_neighbors and epsilon = 0.12
    // (0.24 for angular metrics).
    void query(const int32_t* q_indptr, const int32_t* q_indices,
               const float* q_data, int64_t n_queries, int64_t k,
               double epsilon, int32_t* out_indices, float* out_dists) const;

    // Fit-time kNN graph in original data order (NNDescent.neighbor_graph:
    // corrected distances), (n, n_neighbors) row-major.
    const std::vector<int32_t>& knn_indices() const { return knn_indices_; }
    const std::vector<float>& knn_dists() const { return knn_dists_; }

    // Introspection (parity tests against the Python reference).
    const std::vector<int32_t>& vertex_order() const { return vertex_order_; }
    const nnd::SearchGraph& search_graph() const { return search_graph_; }
    const SparseFlatTree& search_tree() const { return tree_; }
    const std::array<int64_t, 3>& search_rng_state() const {
        return search_rng_state_;
    }
    float min_distance() const { return search_graph_.min_distance; }
    int64_t n_points() const { return n_; }
    int64_t dim() const { return dim_; }
    int64_t n_neighbors() const { return n_neighbors_; }

 private:
    int64_t n_ = 0;
    int64_t dim_ = 0;
    int64_t n_neighbors_ = 0;
    spdist::SparseMetricInfo metric_;
    // training data re-ordered by vertex_order_ (owned CSR)
    std::vector<int32_t> data_indptr_;
    std::vector<int32_t> data_indices_;
    std::vector<float> data_data_;
    std::vector<int32_t> vertex_order_;  // new label -> original label
    nnd::SearchGraph search_graph_;      // re-ordered, rows column-sorted
    SparseFlatTree tree_;                // resorted (indices == arange)
    std::array<int64_t, 3> search_rng_state_ = {0, 0, 0};  // post warm-up
    std::vector<int32_t> knn_indices_;
    std::vector<float> knn_dists_;
};

}  // namespace spnnd
}  // namespace umap
