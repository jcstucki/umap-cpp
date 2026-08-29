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

#include "umap/device.hpp"           // Device (phase4 GPU dispatch)
#include "umap/layouts_generic.hpp"  // OutputMetricFn (generic-output SGD)
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

// --- inverse-update additions -----------------------------------------------
// Float64-weight overload: UMAP.inverse_transform builds graph.data in float64
// (`weights = 1 / (1 + a * distances ** (2 * b))` over np.array triplets), so
// numba instantiates make_epochs_per_sample with f8 weights and n_samples is
// computed in float64.
std::vector<double> make_epochs_per_sample(const double* weights,
                                           std::size_t n_weights, int n_epochs);
// ----------------------------------------------------------------------------

// --- composite additions ------------------------------------------------------
// np.int64-epochs flavour: the composite operators (__mul__/__add__/__sub__)
// pass n_epochs = np.max(result.n_epochs), an np.int64 -- and int64 * float32
// promotes to FLOAT64 in numpy, so n_samples = n_epochs * (w / w.max()) is
// computed as f8(f4(w / w.max())) * n_epochs instead of the all-float32
// product of the normal fit path (python-int n_epochs).
std::vector<double> make_epochs_per_sample_i64(const float* weights,
                                               std::size_t n_weights,
                                               int n_epochs);
// ------------------------------------------------------------------------------

// --- densmap additions ------------------------------------------------------
// The densMAP auxiliary data optimize_layout_euclidean consumes (upstream's
// densmap_kwds dict after simplicial_set_embedding fills in mu / mu_sum / R).
// Scalar types mirror what the numba epoch kernel receives from the reference
// environment (numpy 2.x / NEP 50): lambda / frac / var_shift are Python
// floats (float64); mu (graph.data), mu_sum and R are float32 arrays.
struct DensmapParams {
    double lambda = 0.0;     // densmap_kwds["lambda"]
    double frac = 0.0;       // densmap_kwds["frac"]
    double var_shift = 0.1;  // densmap_kwds["var_shift"]
    std::vector<float> mu;      // per-edge membership (filtered graph.data)
    std::vector<float> mu_sum;  // per-vertex membership sums
    std::vector<float> R;       // per-vertex standardised log radii
};
// ----------------------------------------------------------------------------

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
//  densmap:        enable the density-augmented densMAP objective; when true,
//                  densmap_kwds must point at a populated DensmapParams.
//                  Faithful port of _optimize_layout_euclidean_densmap_epoch_init
//                  (mu_tot = f4 np.sum(mu_sum)/2, per-epoch re_sum/phi_sum
//                  accumulation, f4 np.var/np.mean and the np.dot(re_sum, R)
//                  BLAS sdot for re_std/re_mean/re_cov) plus the
//                  grad_cor_coeff correction inside the epoch loop, gated by
//                  lambda > 0 and (n+1)/n_epochs > 1 - frac exactly as
//                  upstream. On Apple platforms np.dot is reproduced
//                  bit-exactly via Accelerate's cblas_sdot (the BLAS numpy's
//                  macOS wheels link); elsewhere a sequential f32 dot is used
//                  (documented deviation).
//  epochs_list / embedding_list: the "n_epochs is a list" variant. When
//                  epochs_list is non-null, n_epochs is overridden with
//                  max(*epochs_list) (as upstream does) and a copy of
//                  head_embedding is appended to *embedding_list after every
//                  epoch n with n in *epochs_list (checked after the alpha
//                  decay, exactly where the reference snapshots), plus one
//                  final copy after the last epoch. The reference then returns
//                  the list instead of the embedding; here head_embedding
//                  still holds the final state (== embedding_list->back()).
//  n_threads:      hardening additions -- thread cap for the parallel path
//                  (upstream: numba.set_num_threads(n_jobs) before the
//                  prange loop). <= 0 means "all cores"
//                  (hardware_concurrency); ignored when parallel is false.
//                  parallel && n_threads == 1 runs the sequential loop, like
//                  numba's parallel build on a 1-thread pool.
void optimize_layout_euclidean(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    int n_epochs, int n_vertices, const std::vector<double>& epochs_per_sample,
    double a, double b, const int64_t rng_state[3], double gamma = 1.0,
    double initial_alpha = 1.0, double negative_sample_rate = 5.0,
    bool parallel = false, bool densmap = false, bool move_other = false,
    const std::vector<int>* epochs_list = nullptr,
    std::vector<std::vector<float>>* embedding_list = nullptr,
    const DensmapParams* densmap_kwds = nullptr, int n_threads = 0);

// The non-initialisation glue of umap_.py::simplicial_set_embedding:
//   - n_epochs default: 500 if n_samples <= 10000 else 200, +200 when densmap
//     (n_epochs < 0 means "None"/use default);
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
// epochs_list / embedding_list mirror optimize_layout_euclidean's list
// variant; when epochs_list is set, n_epochs must be max(*epochs_list) (the
// value the reference's `n_epochs_max` takes for filtering and
// epochs-per-sample) and the snapshots land in *embedding_list.
//
// --- densmap additions ---
// densmap_kwds non-null enables the densMAP objective (upstream densmap=True):
// the euclidean SGD runs with the density correction; when the graph is not
// prefiltered the epochs default gains upstream's +200 densMAP boost. mu /
// mu_sum / R must already be populated (UMAP.fit computes them from
// graph_dists_; see _densmap_original_densities).
// output_metric non-null selects upstream's euclidean_output=False branch:
// optimize_layout_generic(..., move_other=True) after the identical [0,10]
// rescale and with the identical rng_state. Incompatible with densmap_kwds
// (upstream validates "Non-Euclidean output metric not supported for
// densMAP.") and with epochs_list (upstream's optimize_layout_generic cannot
// take a list of epochs -- numba typing error; we throw std::invalid_argument
// instead).
// numpy_int64_epochs selects the composite operators' epochs-per-sample
// flavour (make_epochs_per_sample_i64 above): they pass an np.int64 n_epochs
// whose numpy promotion computes n_samples in float64.
//
// --- phase4 additions ---
// device selects the SGD backend for the euclidean branch: Device::GPU routes
// the epoch loop to metal::gpu_optimize_layout_euclidean (quality parity, not
// bitwise; see include/umap/metal/gpu_layout.hpp) when the build has Metal
// and a device exists, falling back to the CPU loop with a stderr note
// otherwise. densmap / epochs_list / non-euclidean output requests always run
// on the CPU regardless of device (the kernel does not cover them; UMAP::fit
// warns before asking).
// n_threads (hardening additions): thread cap for the parallel euclidean SGD
// (n_jobs semantics; <= 0 = all cores; only meaningful when parallel=true).
std::vector<float> simplicial_set_embedding_from_init(
    const SparseMatrix& graph, const float* init_embedding, int n_components,
    double a, double b, const int64_t rng_state[3], int n_epochs = -1,
    double gamma = 1.0, double initial_alpha = 1.0,
    double negative_sample_rate = 5.0, bool parallel = false,
    bool graph_prefiltered = false,
    const std::vector<int>* epochs_list = nullptr,
    std::vector<std::vector<float>>* embedding_list = nullptr,
    const DensmapParams* densmap_kwds = nullptr,
    const OutputMetricFn* output_metric = nullptr,
    bool numpy_int64_epochs = false, Device device = Device::CPU,
    int n_threads = 0);

// --- umap-class additions ---------------------------------------------------
// The n_epochs-resolution + weight-filtering prologue of
// umap_.py::simplicial_set_embedding, split out because UMAP.fit needs the
// *filtered* graph before this point (spectral initialisation runs on it):
//   - default_epochs = 500 if n_samples <= 10000 else 200, plus
//     default_epochs_boost (upstream adds 200 for densMAP); n_epochs < 0 means
//     "None" and resolves to the default;
//   - graph.data[graph.data < graph.data.max() / n_epochs] = 0.0 (the divisor
//     is default_epochs when n_epochs <= 10), then eliminate_zeros().
// Returns the resolved n_epochs; `graph` is modified in place.
// simplicial_set_embedding_from_init applies this internally unless
// graph_prefiltered is true (in which case the caller must pass the resolved
// n_epochs and an already-filtered graph).
//
// write_through (optional): a second matrix whose data receives the same
// sub-threshold zeroing WITHOUT eliminate_zeros. This replicates a scipy
// sharing quirk upstream: simplicial_set_embedding's `graph = graph.tocoo()`
// shares the .data array with self.graph_ (coo.tocoo() returns self;
// csr.tocoo() reuses the data buffer), so the epoch-based weight filter
// writes explicit zeros back into the model's stored graph_ -- its structure
// (indptr/indices) is untouched because eliminate_zeros() rebuilds only the
// local copy. UMAP::fit passes its graph_ here; `write_through` must hold
// identical data to `graph` on entry.
int resolve_epochs_and_filter_graph(SparseMatrix& graph, int n_epochs,
                                    int default_epochs_boost = 0,
                                    SparseMatrix* write_through = nullptr);
// ----------------------------------------------------------------------------

}  // namespace umap
