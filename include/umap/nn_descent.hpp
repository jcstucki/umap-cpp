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
#include <cstdint>
#include <random>
#include <string>
#include <vector>

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

 private:
    std::mt19937 mt_;
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

struct MetricInfo {
    // distance used inside NN-descent (fast alternative when one exists)
    float (*dist)(const float*, const float*, int64_t) = nullptr;
    // elementwise correction applied to final distances (nullptr = identity)
    float (*correction)(float) = nullptr;
    // the true metric (for brute-force kNN)
    float (*true_dist)(const float*, const float*, int64_t) = nullptr;
    bool angular_trees = false;
};

// Supported: "euclidean"/"l2", "sqeuclidean", "manhattan"/"taxicab"/"l1",
// "cosine". Throws std::invalid_argument otherwise. The angular-trees decision
// replicates pynndescent's metric list (umap's `angular` flag only matters for
// callable metrics, which we do not support).
MetricInfo resolve_metric(const std::string& name);

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

using DistFn = float (*)(const float*, const float*, int64_t);

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

// Same, but with explicit tau-prng states (for tests / callers that extracted
// states from Python): `rng_state` is NNDescent.rng_state and
// `tree_rng_states` the per-tree states drawn in make_forest.
void nearest_neighbors_with_state(
    const float* data, int64_t n, int64_t dim, int64_t n_neighbors,
    const std::string& metric, bool angular, const int64_t rng_state[3],
    const std::vector<std::array<int64_t, 3>>& tree_rng_states,
    int32_t* knn_indices, float* knn_dists, bool verbose = false);

}  // namespace nnd
}  // namespace umap
