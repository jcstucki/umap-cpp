// Faithful C++ port of the approximate kNN path UMAP uses:
// pynndescent's RP-tree forest initialisation + NN-descent, as invoked by
// umap_.py::nearest_neighbors (reference/pynndescent, BSD 2 clause;
// reference/umap, BSD 3 clause).
//
// Scope: dense float32 input, the default path UMAP hits
// (tree_init=True, low_memory=True, init_graph=None, no metric kwds).
// UMAP only consumes NNDescent._neighbor_graph (the kNN graph); the search
// graph / prepare() / diversify machinery is not needed for fit() and is not
// ported. Querying new points (initialized_nnd_search / search trees), needed
// by transform(), is likewise not ported yet.
//
// Determinism / seeding:
//   pynndescent seeds its tau-prng states from a numpy RandomState:
//     rng_state        = random_state.randint(INT32_MIN+1, INT32_MAX-1, 3)
//     search_rng_state = random_state.randint(INT32_MIN+1, INT32_MAX-1, 3)
//     per-tree states  = random_state.randint(..., size=(n_trees, 3))  (in make_forest)
//   NumpyRandomState below reproduces numpy's legacy RandomState.randint for
//   this exact range bit-for-bit (MT19937 + masked rejection on single 32-bit
//   draws, since the range fits in 32 bits), so `nearest_neighbors(seed)`
//   matches Python given np.random.RandomState(seed).
//   All tau_rand_int consumption mirrors the Python state threading, including
//   the quirk that new_build_candidates only ever advances a *copy*
//   (local_rng_state = rng_state + thread_id) of the state.
//
// Threading: single-threaded execution, but "virtual thread" partitioning
// semantics are preserved where they affect results (init_rp_tree uses
// n_threads=8 blocks exactly like the numba default argument; the NN-descent
// loop and candidate building use n_threads=1, which is what UMAP forces when
// a random_state is supplied).
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "umap/distances.hpp"  // MetricKwds / make_metric_f32 (generic hook)

namespace umap {
namespace nnd {

// INT32_MIN/INT32_MAX as pynndescent defines them (note the +1/-1).
constexpr int64_t PYNN_INT32_MIN = -2147483647LL;       // iinfo(int32).min + 1
constexpr int64_t PYNN_INT32_MAX = 2147483646LL;        // iinfo(int32).max - 1

// ------------------------------------------------------------------ RNG glue

// Bit-exact replication of numpy legacy RandomState.randint(low, high) for
// dtype int64 when high-low fits in 32 bits (which PYNN range does).
class NumpyRandomState {
 public:
    explicit NumpyRandomState(uint32_t seed) : mt_(seed) {}

    // random_state.randint(PYNN_INT32_MIN, PYNN_INT32_MAX) -- one value.
    int64_t randint_pynn() {
        // rng = (high - 1) - low, mask = smallest (2^k - 1) >= rng
        constexpr uint64_t rng = static_cast<uint64_t>(PYNN_INT32_MAX - 1 - PYNN_INT32_MIN);
        constexpr uint64_t mask = 0xFFFFFFFFULL;
        uint64_t val;
        do {
            val = static_cast<uint64_t>(mt_()) & mask;
        } while (val > rng);
        return PYNN_INT32_MIN + static_cast<int64_t>(val);
    }

    std::array<int64_t, 3> rng_state3() {
        return {randint_pynn(), randint_pynn(), randint_pynn()};
    }

    // ---- umap-class additions: numpy legacy continuous draws ----
    // rk_double: 53-bit uniform double in [0, 1).
    double random_sample() {
        uint64_t a = mt_() >> 5;  // 27 bits
        uint64_t b = mt_() >> 6;  // 26 bits
        return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) /
               9007199254740992.0;
    }

    // RandomState.uniform(low, high) for scalars (rk_uniform).
    double uniform(double low, double high) {
        return low + (high - low) * random_sample();
    }

    // rk_gauss: legacy polar (Marsaglia) method with the cached second value,
    // exactly as numpy's RandomState.standard_normal / normal.
    double gauss() {
        if (has_gauss_) {
            has_gauss_ = false;
            return gauss_;
        }
        double x1, x2, r2;
        do {
            x1 = 2.0 * random_sample() - 1.0;
            x2 = 2.0 * random_sample() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        const double f = std::sqrt(-2.0 * std::log(r2) / r2);
        gauss_ = f * x1;
        has_gauss_ = true;
        return f * x2;
    }

    // RandomState.normal(loc, scale) for scalars (rk_normal).
    double normal(double loc, double scale) { return loc + scale * gauss(); }

 private:
    std::mt19937 mt_;
    bool has_gauss_ = false;
    double gauss_ = 0.0;
};

// Python-semantics modulo (result sign follows the divisor; here divisor > 0).
inline int64_t pymod(int64_t a, int64_t m) {
    int64_t r = a % m;
    return r < 0 ? r + m : r;
}

// ---------------------------------------------------------------- heap utils
// Port of pynndescent/utils.py heap functions. A "heap" is the triple
// (indices int32, distances float32, flags uint8), each (n_points, size),
// max-heap per row keyed on distance.

struct NNHeap {
    int64_t n_points = 0;
    int64_t size = 0;
    std::vector<int32_t> ind;   // init -1
    std::vector<float> dist;    // init +inf
    std::vector<uint8_t> flag;  // init 0

    static NNHeap make(int64_t n_points, int64_t size);

    int32_t* ind_row(int64_t i) { return ind.data() + i * size; }
    float* dist_row(int64_t i) { return dist.data() + i * size; }
    uint8_t* flag_row(int64_t i) { return flag.data() + i * size; }
    const int32_t* ind_row(int64_t i) const { return ind.data() + i * size; }
    const float* dist_row(int64_t i) const { return dist.data() + i * size; }
};

int simple_heap_push(float* priorities, int32_t* indices, int64_t size,
                     float p, int32_t n);
int checked_heap_push(float* priorities, int32_t* indices, int64_t size,
                      float p, int32_t n);
int checked_flagged_heap_push(float* priorities, int32_t* indices,
                              uint8_t* flags, int64_t size, float p, int32_t n,
                              uint8_t f);
void siftdown(float* heap1, int32_t* heap2, int64_t size, int64_t elt);
// In-place ascending sort of every heap row (second half of heapsort).
void deheap_sort(int32_t* indices, float* distances, int64_t n_points,
                 int64_t size);

// ------------------------------------------------------------------- metrics
// float32-faithful ports of the pynndescent distance kernels actually used by
// the dense NN-descent path for UMAP-relevant metrics.

float squared_euclidean(const float* x, const float* y, int64_t dim);   // internal for "euclidean"
float manhattan(const float* x, const float* y, int64_t dim);
float alternative_cosine(const float* x, const float* y, int64_t dim);  // internal for "cosine"
float true_euclidean(const float* x, const float* y, int64_t dim);
float true_cosine(const float* x, const float* y, int64_t dim);

// Distance callable used throughout NN-descent / prepare / query. A
// std::function so parameterised metrics (metric_kwds) can be bound as
// closures like pynndescent's `_partial_dist_func`; plain function pointers
// convert implicitly for the fast built-in metrics.
using DistFn = std::function<float(const float*, const float*, int64_t)>;

struct MetricInfo {
    // distance used inside NN-descent (fast alternative when one exists)
    DistFn dist;
    // elementwise correction applied to final distances (nullptr = identity)
    float (*correction)(float) = nullptr;
    // the true metric (for brute-force kNN)
    DistFn true_dist;
    bool angular_trees = false;
};

// Supported: "euclidean"/"l2", "sqeuclidean", "manhattan"/"taxicab"/"l1",
// "cosine", and (tests-port additions, pynndescent fast_distance_alternatives
// ports) "jaccard" + "hellinger". Throws std::invalid_argument otherwise. The
// angular-trees decision replicates pynndescent's metric list (umap's
// `angular` flag only matters for callable metrics, which we do not support).
MetricInfo resolve_metric(const std::string& name);

// --- densmap additions ------------------------------------------------------
// Kwds-aware overload: the fast built-in metrics above resolve as before
// (their kwds are empty upstream); every other metric in umap's named
// registry takes the *generic-metric hook*, replicating pynndescent's
// no-fast-alternative branch: NN-descent runs directly on the metric
// (correction = identity), with metric_kwds bound positionally like
// `_partial_dist_func`. The float32 rows are evaluated via make_metric_f32
// (f32-faithful for minkowski/seuclidean/manhattan, float64-widened adapter
// otherwise); numba pushes the float64 kernel result onto the float32 heap,
// so the rounding point matches. angular_trees mirrors pynndescent's angular
// metric list (cosine/dot/correlation/dice/jaccard/hellinger/hamming/bit_*).
// Deviation: pynndescent's fast alternatives for hellinger/jaccard are not
// ported; those metrics run on the true kernel (same final distances,
// candidate ordering may differ from Python's alternative-metric run).
MetricInfo resolve_metric(const std::string& name, const MetricKwds& kwds);
// ----------------------------------------------------------------------------

// ------------------------------------------------------------------ RP trees

struct LeafArray {
    int64_t n_rows = 0;
    int64_t row_len = 0;
    std::vector<int32_t> idx;  // n_rows * row_len, padded with -1
    const int32_t* row(int64_t i) const { return idx.data() + i * row_len; }
};

// Ports of {euclidean,angular}_random_projection_split (rp_trees.py).
void euclidean_random_projection_split(
    const float* data, int64_t dim, const std::vector<int32_t>& indices,
    int64_t rng_state[3], std::vector<int32_t>& left,
    std::vector<int32_t>& right, std::vector<float>& hyperplane,
    float& offset);
void angular_random_projection_split(
    const float* data, int64_t dim, const std::vector<int32_t>& indices,
    int64_t rng_state[3], std::vector<int32_t>& left,
    std::vector<int32_t>& right, std::vector<float>& hyperplane,
    float& offset);

// Builds one dense tree (make_dense_tree) but records only its leaves, in the
// same order the recursion emits them. Returns the tree's max_leaf_size
// (>= leaf_size if an unsplittable node exceeded it).
int64_t make_dense_tree_leaves(const float* data, int64_t n, int64_t dim,
                               int64_t rng_state[3], int64_t leaf_size,
                               bool angular, int64_t max_depth,
                               std::vector<std::vector<int32_t>>& leaves_out);

// make_forest + rptree_leaf_array: per-tree rng states are drawn from
// `numpy_rs` exactly as make_forest does. leaf_size < 0 selects the default
// max(60, min(256, 5 * n_neighbors)).
LeafArray make_forest_leaf_array(const float* data, int64_t n, int64_t dim,
                                 int64_t n_neighbors, int64_t n_trees,
                                 int64_t leaf_size, bool angular,
                                 int64_t max_depth, NumpyRandomState& numpy_rs);

// ---------------------------------------------------------------- NN-descent

void init_rp_tree(const float* data, int64_t dim, DistFn dist, NNHeap& graph,
                  const LeafArray& leaf_array);
void init_random(int64_t n_neighbors, const float* data, int64_t n,
                 int64_t dim, NNHeap& heap, DistFn dist, int64_t rng_state[3]);
// Port of utils.new_build_candidates (n_threads virtual-thread semantics kept;
// UMAP's reproducible path uses n_threads=1). Outputs are
// (n_vertices, max_candidates) int32 arrays; also clears "new" flags.
void new_build_candidates(NNHeap& current_graph, int64_t max_candidates,
                          const int64_t rng_state[3], int64_t n_threads,
                          std::vector<int32_t>& new_candidates,
                          std::vector<int32_t>& old_candidates);

// Full nn_descent (EMPTY init graph path): rp-tree init (if leaf_array
// non-empty), random fill, iterated candidate processing, deheap_sort.
// Result heap rows are sorted ascending by distance on return.
NNHeap nn_descent(const float* data, int64_t n, int64_t dim,
                  int64_t n_neighbors, int64_t rng_state[3],
                  int64_t max_candidates, DistFn dist, int64_t n_iters,
                  double delta, const LeafArray& leaf_array,
                  bool verbose = false);

// --- inverse-update additions ------------------------------------------------
// pynndescent_.py::init_from_neighbor_graph: flag-0 checked pushes of an
// existing kNN graph (indices/distances, (n_rows, n_cols) row-major; the
// distances are the *internal* metric values) into a heap.
void init_from_neighbor_graph(NNHeap& heap, const int32_t* indices,
                              const float* distances, int64_t n_rows,
                              int64_t n_cols);

// The nn_descent flow NNDescent.update() runs: `current_graph` was already
// initialised by the caller (init_from_neighbor_graph + init_rp_tree over the
// fresh forest; rp_tree_init=False and no init_random, unlike the empty-graph
// path), so only the iterated candidate processing + deheap_sort remain.
void nn_descent_with_init_graph(NNHeap& current_graph, const float* data,
                                int64_t n, int64_t dim, int64_t n_neighbors,
                                int64_t rng_state[3], int64_t max_candidates,
                                DistFn dist, int64_t n_iters, double delta,
                                bool verbose = false);
// -----------------------------------------------------------------------------

// ------------------------------------------------------------------ toplevel

// Exact kNN by full pairwise scan of the true metric (each point's own index
// appears first at distance 0), ties broken by ascending index like
// np.argsort(kind="stable"). Outputs are (n, n_neighbors) row-major.
void brute_force_knn(const float* data, int64_t n, int64_t dim,
                     int64_t n_neighbors, const std::string& metric,
                     int32_t* knn_indices, float* knn_dists);

// Port of umap_.py::nearest_neighbors for the NNDescent branch:
//   n_trees = min(64, 5 + round(sqrt(n) / 20))
//   n_iters = max(5, round(log2(n)))
//   max_candidates = 60, low_memory=True
// `random_seed` corresponds to np.random.RandomState(random_seed) being passed
// as UMAP's random_state. `angular` mirrors umap's parameter; it is accepted
// for signature parity but the tree flavour is decided by the metric, exactly
// as pynndescent does for named metrics. Outputs are (n, n_neighbors)
// row-major; unfilled slots (if any) are index -1 / distance +inf.
void nearest_neighbors(const float* data, int64_t n, int64_t dim,
                       int64_t n_neighbors, const std::string& metric,
                       bool angular, uint32_t random_seed,
                       int32_t* knn_indices, float* knn_dists,
                       bool verbose = false);

// Overload continuing an existing numpy RandomState stream (the situation in
// UMAP.fit, where the same random_state object is consumed by NNDescent and
// then by the embedding initialisation). Draw order matches
// NNDescent.__init__: rng_state (3), search_rng_state (3), per-tree states.
void nearest_neighbors(const float* data, int64_t n, int64_t dim,
                       int64_t n_neighbors, const std::string& metric,
                       bool angular, NumpyRandomState& numpy_rs,
                       int32_t* knn_indices, float* knn_dists,
                       bool verbose = false);

// --- densmap additions ------------------------------------------------------
// metric_kwds-aware overload (NNDescent(metric_kwds=...) routing; see the
// kwds-aware resolve_metric above).
void nearest_neighbors(const float* data, int64_t n, int64_t dim,
                       int64_t n_neighbors, const std::string& metric,
                       const MetricKwds& metric_kwds, bool angular,
                       NumpyRandomState& numpy_rs, int32_t* knn_indices,
                       float* knn_dists, bool verbose = false);
// ----------------------------------------------------------------------------

// Same, but with explicit tau-prng states (for tests / callers that extracted
// states from Python): `rng_state` is NNDescent.rng_state and
// `tree_rng_states` the per-tree states drawn in make_forest.
void nearest_neighbors_with_state(
    const float* data, int64_t n, int64_t dim, int64_t n_neighbors,
    const std::string& metric, bool angular, const int64_t rng_state[3],
    const std::vector<std::array<int64_t, 3>>& tree_rng_states,
    int32_t* knn_indices, float* knn_dists, bool verbose = false);

// --------------------------------------------------- query path (transform)
// Faithful port of NNDescent.prepare() + query() for the dense path UMAP's
// transform() hits: graph-informed hub search tree (rp_trees.py::make_hub_tree
// + convert_tree_format), occlusion-pruned search graph (diversify /
// diversify_csr / degree_prune with UMAP-default diversify_prob=1.0 and
// pruning_degree_multiplier=1.5), and the epsilon-bounded graph search
// (pynndescent_.py::search_closure).

// Converted flat search tree (rp_trees.py::convert_tree_format layout).
// Internal nodes: children[2i] = left node, children[2i+1] = right node (>0).
// Leaves: children[2i] = -leaf_start, children[2i+1] = -leaf_end into
// `indices`.
struct FlatTree {
    int64_t n_nodes = 0;
    int64_t dim = 0;
    std::vector<float> hyperplanes;  // n_nodes * dim (leaves: zero rows)
    std::vector<float> offsets;      // n_nodes (leaves: -inf)
    std::vector<int32_t> children;   // n_nodes * 2
    std::vector<int32_t> indices;    // data_size (concatenated leaves)
};

// rp_trees.py::make_hub_tree followed by convert_tree_format. The neighbor
// graph (`knn_indices`, (n, n_neighbors) row-major, -1 padded) drives the
// hub-based hyperplane selection; `rng_state` is only consumed by the
// degenerate-split random fallback. UMAP defaults: leaf_size=30 (search
// leaf size when NNDescent.leaf_size is None), max_depth=200.
FlatTree make_hub_tree(const float* data, int64_t n, int64_t dim,
                       const int32_t* knn_indices, int64_t n_neighbors,
                       int64_t rng_state[3], int64_t leaf_size, bool angular,
                       int64_t max_depth);

// Binarised CSR search graph (structure only) + the minimum forward-graph
// distance (NNDescent._min_distance, used in the epsilon distance bound).
struct SearchGraph {
    std::vector<int32_t> indptr;   // n + 1
    std::vector<int32_t> indices;  // column-sorted within each row
    float min_distance = 0.0f;
};

// Port of the graph half of NNDescent._init_search_graph (dense,
// diversify_method="standard"): forward diversify on a copy of the kNN graph,
// CSR assembly (zero distances -> FLOAT32_EPS, -1 entries dropped),
// diversify_csr applied to the forward rows (scipy's zero-copy transpose
// makes the "reverse" pass operate on the forward CSR arrays), symmetric
// maximum combine, diagonal removal, degree_prune to
// round(pruning_degree_multiplier * n_neighbors), binarise. `knn_dists` are
// the *internal* (uncorrected) metric distances. `rng_state` threads through
// both diversify passes exactly as in Python.
SearchGraph build_search_graph(const float* data, int64_t n, int64_t dim,
                               const int32_t* knn_indices,
                               const float* knn_dists, int64_t n_neighbors,
                               DistFn dist, int64_t rng_state[3],
                               double diversify_prob,
                               double pruning_degree_multiplier);

// A fitted + prepared index over a copy of the training data. Mirrors what
// UMAP keeps as `_knn_search_index`: fit-time kNN graph (for
// fuzzy_simplicial_set) plus everything query() needs.
class PreparedIndex {
 public:
    // NNDescent(data, n_neighbors, metric, random_state=seed,
    //           n_trees/n_iters = umap "hacked values", max_candidates=60,
    //           low_memory=True, compressed=False) followed by .prepare().
    static PreparedIndex build(const float* data, int64_t n, int64_t dim,
                               int64_t n_neighbors, const std::string& metric,
                               uint32_t random_seed, bool verbose = false);

    // Overload continuing an existing numpy RandomState stream (used by the
    // UMAP class so init/SGD seeding after NN search stays stream-parallel
    // with Python). Numpy draw order and count are identical to
    // NNDescent.__init__ (prepare() draws nothing from the RandomState).
    static PreparedIndex build(const float* data, int64_t n, int64_t dim,
                               int64_t n_neighbors, const std::string& metric,
                               NumpyRandomState& numpy_rs,
                               bool verbose = false);

    // --- densmap additions --------------------------------------------------
    // metric_kwds-aware overload (kwds bound into the distance closure like
    // pynndescent's _partial_dist_func; see resolve_metric(name, kwds)).
    static PreparedIndex build(const float* data, int64_t n, int64_t dim,
                               int64_t n_neighbors, const std::string& metric,
                               const MetricKwds& metric_kwds,
                               NumpyRandomState& numpy_rs,
                               bool verbose = false);
    // ------------------------------------------------------------------------

    // NNDescent.query(query_data, k, epsilon). Outputs are (n_queries, k)
    // row-major; indices refer to the original data order. UMAP's transform
    // calls this with k = n_neighbors and epsilon = 0.12 (0.24 for angular
    // metrics).
    void query(const float* query_data, int64_t n_queries, int64_t k,
               double epsilon, int32_t* out_indices, float* out_dists) const;

    // --- inverse-update additions --------------------------------------------
    // pynndescent_.py::NNDescent.update(xs_fresh=new_data) followed by the
    // prepare() re-run UMAP.update() triggers (the search graph existed, so
    // pynndescent deletes it and prepares again). Steps, faithfully:
    //   * restore the raw data to original order (raw[np.argsort(
    //     _vertex_order)]) and append the fresh rows;
    //   * numpy draws: randint(3) (the `rng_state` local pynndescent never
    //     uses for dense data) then make_forest per-tree states, both from
    //     `numpy_rs` -- which must be the same RandomState stream that seeded
    //     build() (NNDescent keeps the instance; UMAP::update passes the
    //     stream persisted after fit);
    //   * self.n_trees = n_trees_after_update = max(2, round(n_trees / 3))
    //     (computed once at build from the build-time n_trees);
    //   * make_heap + init_from_neighbor_graph (internal distances) +
    //     init_rp_tree over the fresh forest, then the candidate loop with
    //     rp_tree_init=False, max_candidates = 60 (UMAP's hacked value),
    //     n_iters unchanged from build, continuing the *persisted* tau
    //     rng_state exactly where build()'s prepare left it;
    //   * re-prepare: hub tree + diversify/degree-prune/reorder on the new
    //     graph, continuing the same tau state. search_rng_state is *not*
    //     re-warmed (the 10x warm-up lives in NNDescent.__init__ only).
    // After the call n_points() includes the new rows; knn_indices()/
    // knn_dists() are (n_total, n_neighbors) in original data order.
    void update(const float* new_data, int64_t n_new, NumpyRandomState& numpy_rs,
                bool verbose = false);

    // The training data as the index stores it after prepare() -- reordered
    // by vertex_order (row i is original row vertex_order()[i]). UMAP.update
    // reads `self._knn_search_index._raw_data` at exactly this state.
    const std::vector<float>& stored_data() const { return data_; }
    // -------------------------------------------------------------------------

    // Fit-time kNN graph in original data order (NNDescent.neighbor_graph:
    // corrected distances), (n, n_neighbors) row-major.
    const std::vector<int32_t>& knn_indices() const { return knn_indices_; }
    const std::vector<float>& knn_dists() const { return knn_dists_; }

    // Introspection (parity tests against the Python reference).
    const std::vector<int32_t>& vertex_order() const { return vertex_order_; }
    const SearchGraph& search_graph() const { return search_graph_; }
    const FlatTree& search_tree() const { return tree_; }
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
    MetricInfo metric_;
    bool normalize_query_ = false;  // pynndescent: metric in ("cosine", "dot")
    std::vector<float> data_;       // training data re-ordered by vertex_order_
    std::vector<int32_t> vertex_order_;  // new label -> original label
    SearchGraph search_graph_;           // re-ordered, rows column-sorted
    FlatTree tree_;                      // resorted (indices == arange)
    std::array<int64_t, 3> search_rng_state_ = {0, 0, 0};  // post warm-up
    std::vector<int32_t> knn_indices_;
    std::vector<float> knn_dists_;
    // --- inverse-update additions --------------------------------------------
    // State NNDescent.update() consumes: the tau rng_state array as left by
    // build()'s nn_descent + prepare (self.rng_state, mutated in place), the
    // *internal* (uncorrected) kNN distances (self._neighbor_graph[1]), and
    // the build-time tree bookkeeping.
    std::array<int64_t, 3> rng_state_ = {0, 0, 0};
    std::vector<float> knn_internal_dists_;
    int64_t n_trees_after_update_ = 2;  // max(2, round(n_trees / 3))
    int64_t n_iters_ = 5;               // unchanged across updates
    // -------------------------------------------------------------------------
};

}  // namespace nnd
}  // namespace umap
