// Faithful C++ port of layouts.py::optimize_layout_generic and
// layouts.py::optimize_layout_inverse (lmcinnes/umap, BSD 3 clause): the SGD
// loops used for non-euclidean output metrics and for inverse_transform.
//
// Arithmetic mirrors the sequential numba-jitted code paths (both epoch
// kernels are compiled with parallel=False upstream, so there is no hogwild
// variant here):
//  - embeddings are float32; the output metric is called on the float32 rows
//    and returns a float64 distance plus a per-dimension gradient;
//  - w_l = pow(1 + a * dist^{2b}, -1) (numba lowers the integer -1 exponent
//    to a reciprocal), grad_coeff and the clip/step arithmetic are float64,
//    and `current[d] += grad_d * alpha` rounds back to float32 on store;
//  - optimize_layout_generic derives *per-head-vertex* rng states exactly as
//    optimize_layout_euclidean does (rng_state + int64-bitcast of
//    double(head_embedding[j, 0]) at entry);
//  - optimize_layout_inverse uses a single shared rng_state (upstream passes
//    the raw 3-word state, which is mutated across the whole run).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "umap/distances.hpp"

namespace umap {

// Output metric on float32 embedding rows: returns the distance and writes
// d(dist)/dx into grad (length dim, float64 like the numba f8 gradient the
// f4-instantiated metrics return). x/y are non-const because two upstream
// metrics mutate the embedding rows in place (symmetric_kl, gaussian_energy).
using OutputMetricFn =
    std::function<double(float* x, float* y, int dim, double* grad)>;

// f32-faithful kernels for the two metrics we guarantee bit-exactness for:
// only the element reads are float32, all other arithmetic is float64 --
// exactly what numba emits for the f4 instantiations under NEP 50 promotion.
double haversine_output_grad(const float* x, const float* y, double* grad);
double euclidean_output_grad(const float* x, const float* y, int dim,
                             double* grad);

// Resolve a named output metric (+ kwds) into an OutputMetricFn, mirroring
// upstream's `named_distances_with_gradients[output_metric]` +
// tuple(output_metric_kwds.values()). "haversine" and "euclidean"/"l2" use
// the exact f32 kernels above; all other metrics go through a documented
// adapter that copies the float32 rows into float64 buffers and calls the
// double-precision gradient (near-exact; in-place mutations by
// symmetric_kl/gaussian_energy hit the copy, not the embedding -- deviation).
OutputMetricFn make_output_metric(const std::string& name,
                                  const MetricKwds& kwds = {});

// layouts.py::optimize_layout_generic (sequential numba path). Operates on
// head_embedding in place (and tail_embedding when move_other is true;
// head_embedding and tail_embedding may alias, as in the standard fit path).
void optimize_layout_generic(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    int n_epochs, int n_vertices, const std::vector<double>& epochs_per_sample,
    double a, double b, const int64_t rng_state[3],
    const OutputMetricFn& output_metric, double gamma = 1.0,
    double initial_alpha = 1.0, double negative_sample_rate = 5.0,
    bool move_other = false);

// layouts.py::optimize_layout_inverse (used by UMAP.inverse_transform: the
// head embedding is the set of new points in *data* space, tail_embedding is
// the training data, and the output metric is the gradient variant of the
// *input* metric). weight is the per-edge membership strength (graph.data),
// sigmas/rhos are the smooth-knn parameters of the tail vertices (float32,
// as stored on the fitted model). rng_state is mutated (shared state).
// a and b are accepted for signature parity but unused, as upstream.
void optimize_layout_inverse(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    const float* weight, const float* sigmas, const float* rhos, int n_epochs,
    int n_vertices, const std::vector<double>& epochs_per_sample, double a,
    double b, int64_t rng_state[3], const OutputMetricFn& output_metric,
    double gamma = 1.0, double initial_alpha = 1.0,
    double negative_sample_rate = 5.0, bool move_other = false);

}  // namespace umap
