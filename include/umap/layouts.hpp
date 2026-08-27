// Faithful C++ port of the embedding-layout SGD from
// reference/umap/umap/layouts.py plus the non-init glue of
// simplicial_set_embedding / make_epochs_per_sample from
// reference/umap/umap/umap_.py (lmcinnes/umap, BSD 3 clause).
//
// Arithmetic mirrors the numba-jitted code paths:
//  - embeddings are float32; gradient coefficients / alpha / epoch bookkeeping
//    are float64 (numba promotes f4 operands into the f8 expressions and the
//    result of `current[d] += grad_d * alpha` is rounded back to f4 on store);
//  - rdist is the f4(f4[::1], f4[::1]) fastmath kernel (float32 accumulation);
//  - negative sampling uses tau_rand_int on *per-head-vertex* rng states
//    (rng_state_per_sample[j] = rng_state + int64-bitcast(double(head_embedding[j, 0])),
//    computed once from the embedding passed to optimize_layout_euclidean),
//    with Python modulo semantics (result takes the sign of n_vertices).
#pragma once

#include <cstdint>
#include <vector>

#include "umap/sparse_matrix.hpp"

namespace umap {

// Standard clamping of a value into the fixed range [-4.0, 4.0]
// (layouts.py::clip; the euclidean SGD path instantiates it on float64).
inline double clip(double val) {
    if (val > 4.0) {
        return 4.0;
    } else if (val < -4.0) {
        return -4.0;
    } else {
        return val;
    }
}

// Reduced (squared) euclidean distance (layouts.py::rdist).
// Matches the numba signature f4(f4[::1], f4[::1]): float32 accumulation.
inline float rdist(const float* x, const float* y, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }
    return result;
}

// Given a set of weights and number of epochs generate the number of epochs
// per sample for each weight (umap_.py::make_epochs_per_sample).
// weights are float32; n_samples = f4(n_epochs) * (w / w.max()) is computed in
// float32, the result is float64 (entries -1.0 where n_samples <= 0).
std::vector<double> make_epochs_per_sample(const float* weights,
                                           std::size_t n_weights, int n_epochs);

// Improve an embedding using SGD to minimise the fuzzy set cross entropy
// between the 1-skeletons of the high- and low-dimensional fuzzy simplicial
// sets (layouts.py::optimize_layout_euclidean). Operates on head_embedding in
// place (and tail_embedding too when move_other is true; head_embedding and
// tail_embedding may alias, as in the standard fit path).
//
//  head_embedding: n_head_vertices x dim, row-major float32 (modified in place)
//  tail_embedding: n_tail_vertices x dim (== head_embedding for standard fits)
//  head/tail:      edge endpoint indices (1-simplices with non-zero membership)
//  epochs_per_sample: float64, one entry per edge (make_epochs_per_sample)
//  rng_state:      3-word tau_prng seed; per-head-vertex states are derived
//                  from it exactly as in the reference (not modified)
//  parallel:       "hogwild" std::thread mirror of numba prange(parallel=True):
//                  racy embedding/rng updates, non-deterministic (numba
//                  semantics). parallel=false is the sequential numba path and
//                  is deterministic.
//  densmap:        TODO: densMAP objective not yet ported; must be false
//                  (parameters accepted for signature parity only).
void optimize_layout_euclidean(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    int n_epochs, int n_vertices, const std::vector<double>& epochs_per_sample,
    double a, double b, const int64_t rng_state[3], double gamma = 1.0,
    double initial_alpha = 1.0, double negative_sample_rate = 5.0,
    bool parallel = false, bool densmap = false, bool move_other = false);

// The non-initialisation glue of umap_.py::simplicial_set_embedding for the
// euclidean output metric, densmap off:
//   - n_epochs default: 500 if n_samples <= 10000 else 200 (n_epochs < 0 means
//     "None"/use default);
//   - weights below graph.data.max() / n_epochs (or / default_epochs when
//     n_epochs <= 10) are zeroed and eliminated;
//   - head/tail come from the COO form of the filtered graph;
//   - epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs);
//   - the initial embedding is rescaled per-component to [0, 10] in float32,
//     exactly `10.0 * (embedding - min) / (max - min)` (this applies to *all*
//     init paths in the reference, including user-supplied arrays);
//   - optimize_layout_euclidean(..., move_other=true).
//
// Initialisation itself (spectral / pca / random + noisy_scale_coords) is a
// separate concern: callers pass the raw initial coordinates through
// `init_embedding` (n_samples x n_components, row-major float32) and this
// function applies the [0, 10] rescale before optimisation. rng_state stands
// in for the reference's random_state.randint(INT32_MIN, INT32_MAX, 3).
std::vector<float> simplicial_set_embedding_from_init(
    const SparseMatrix& graph, const float* init_embedding, int n_components,
    double a, double b, const int64_t rng_state[3], int n_epochs = -1,
    double gamma = 1.0, double initial_alpha = 1.0,
    double negative_sample_rate = 5.0, bool parallel = false,
    bool graph_prefiltered = false);

// --- umap-class additions ---------------------------------------------------
// The n_epochs-resolution + weight-filtering prologue of
// umap_.py::simplicial_set_embedding, split out because UMAP.fit needs the
// *filtered* graph before this point (spectral initialisation runs on it):
//   - default_epochs = 500 if n_samples <= 10000 else 200; n_epochs < 0 means
//     "None" and resolves to the default;
//   - graph.data[graph.data < graph.data.max() / n_epochs] = 0.0 (the divisor
//     is default_epochs when n_epochs <= 10), then eliminate_zeros().
// Returns the resolved n_epochs; `graph` is modified in place.
// simplicial_set_embedding_from_init applies this internally unless
// graph_prefiltered is true (in which case the caller must pass the resolved
// n_epochs and an already-filtered graph).
int resolve_epochs_and_filter_graph(SparseMatrix& graph, int n_epochs);
// ----------------------------------------------------------------------------

}  // namespace umap
