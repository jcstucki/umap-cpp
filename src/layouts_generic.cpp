// Faithful C++ port of layouts.py::optimize_layout_generic and
// optimize_layout_inverse (lmcinnes/umap, BSD 3 clause). See
// include/umap/layouts_generic.hpp for the semantics notes.
#include "umap/layouts_generic.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "umap/layouts.hpp"  // clip
#include "umap/rng.hpp"

namespace umap {

double haversine_output_grad(const float* x, const float* y, double* grad) {
    // numba f4 instantiation of distances.py::haversine_grad: the element
    // subtractions x[i] - y[i] happen in float32, everything after is f64.
    const double sin_lat =
        std::sin(0.5 * static_cast<double>(x[0] - y[0]));
    const double cos_lat =
        std::cos(0.5 * static_cast<double>(x[0] - y[0]));
    const double sin_long =
        std::sin(0.5 * static_cast<double>(x[1] - y[1]));
    const double cos_long =
        std::cos(0.5 * static_cast<double>(x[1] - y[1]));

    const double a_0 = std::cos(static_cast<double>(x[0]) + M_PI / 2.0) *
                       std::cos(static_cast<double>(y[0]) + M_PI / 2.0) *
                       sin_long * sin_long;
    const double a_1 = a_0 + sin_lat * sin_lat;

    const double d = 2.0 * std::asin(std::sqrt(
                               std::min(std::max(std::fabs(a_1), 0.0), 1.0)));
    const double denom =
        std::sqrt(std::fabs(a_1 - 1.0)) * std::sqrt(std::fabs(a_1));
    grad[0] = (sin_lat * cos_lat -
               std::sin(static_cast<double>(x[0]) + M_PI / 2.0) *
                   std::cos(static_cast<double>(y[0]) + M_PI / 2.0) *
                   sin_long * sin_long) /
              (denom + 1e-6);
    grad[1] = (std::cos(static_cast<double>(x[0]) + M_PI / 2.0) *
               std::cos(static_cast<double>(y[0]) + M_PI / 2.0) * sin_long *
               cos_long) /
              (denom + 1e-6);
    return d;
}

double euclidean_output_grad(const float* x, const float* y, int dim,
                             double* grad) {
    // numba f4 instantiation of distances.py::euclidean_grad: the diffs and
    // their squares are float32, the accumulator is float64 (result = 0.0),
    // and grad = (x - y) / (1e-6 + d) promotes to float64 (NEP 50).
    double result = 0.0;
    for (int i = 0; i < dim; ++i) {
        const float diff = x[i] - y[i];
        result += static_cast<double>(diff * diff);
    }
    const double d = std::sqrt(result);
    for (int i = 0; i < dim; ++i) {
        grad[i] = static_cast<double>(x[i] - y[i]) / (1e-6 + d);
    }
    return d;
}

OutputMetricFn make_output_metric(const std::string& name,
                                  const MetricKwds& kwds) {
    if (name == "haversine") {
        return [](float* x, float* y, int dim, double* grad) {
            if (dim != 2) {
                throw std::invalid_argument(
                    "haversine is only defined for 2 dimensional data");
            }
            return haversine_output_grad(x, y, grad);
        };
    }
    if (name == "euclidean" || name == "l2") {
        return [](float* x, float* y, int dim, double* grad) {
            return euclidean_output_grad(x, y, dim, grad);
        };
    }
    // Generic adapter: float32 rows widened to float64 (documented deviation:
    // in-place mutations by symmetric_kl / gaussian_energy modify the copy).
    GradientFn g = make_metric_grad(name, kwds);
    return [g](float* x, float* y, int dim, double* grad) {
        std::vector<double> xd(x, x + dim);
        std::vector<double> yd(y, y + dim);
        // diagonal_gaussian_energy needs a 6-entry gradient buffer.
        std::vector<double> gbuf(dim < 6 ? 6 : dim, 0.0);
        const double d = g(xd.data(), yd.data(), static_cast<std::size_t>(dim),
                           gbuf.data());
        for (int i = 0; i < dim; ++i) grad[i] = gbuf[i];
        return d;
    };
}

namespace {

// Derive the per-head-vertex rng states exactly as layouts.py does:
// rng_state_per_sample = np.full((n_head, 3), rng_state, dtype=np.int64)
//     + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)
std::vector<int64_t> derive_rng_state_per_sample(const float* head_embedding,
                                                 int n_head_vertices, int dim,
                                                 const int64_t rng_state[3]) {
    std::vector<int64_t> states(static_cast<std::size_t>(n_head_vertices) * 3);
    for (int j = 0; j < n_head_vertices; ++j) {
        const double v0 = static_cast<double>(
            head_embedding[static_cast<std::size_t>(j) * dim]);
        int64_t offset;
        static_assert(sizeof(offset) == sizeof(v0), "bitcast size");
        std::memcpy(&offset, &v0, sizeof(offset));
        for (int c = 0; c < 3; ++c) {
            states[static_cast<std::size_t>(j) * 3 + c] =
                static_cast<int64_t>(static_cast<uint64_t>(rng_state[c]) +
                                     static_cast<uint64_t>(offset));
        }
    }
    return states;
}

}  // namespace

void optimize_layout_generic(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    int n_epochs, int n_vertices, const std::vector<double>& epochs_per_sample,
    double a, double b, const int64_t rng_state[3],
    const OutputMetricFn& output_metric, double gamma, double initial_alpha,
    double negative_sample_rate, bool move_other) {
    double alpha = initial_alpha;
    const std::size_t n_edges = epochs_per_sample.size();

    std::vector<double> epochs_per_negative_sample(n_edges);
    for (std::size_t i = 0; i < n_edges; ++i) {
        epochs_per_negative_sample[i] =
            epochs_per_sample[i] / negative_sample_rate;
    }
    std::vector<double> epoch_of_next_negative_sample =
        epochs_per_negative_sample;
    std::vector<double> epoch_of_next_sample = epochs_per_sample;

    std::vector<int64_t> rng_state_per_sample = derive_rng_state_per_sample(
        head_embedding, n_head_vertices, dim, rng_state);

    std::vector<double> grad_dist_output(dim);
    std::vector<double> rev_grad_dist_output(dim);

    for (int n = 0; n < n_epochs; ++n) {
        for (std::size_t i = 0; i < n_edges; ++i) {
            if (epoch_of_next_sample[i] <= static_cast<double>(n)) {
                const int32_t j = head[i];
                int32_t k = tail[i];

                float* current =
                    head_embedding + static_cast<std::size_t>(j) * dim;
                float* other =
                    tail_embedding + static_cast<std::size_t>(k) * dim;

                double dist_output =
                    output_metric(current, other, dim, grad_dist_output.data());
                // _, rev_grad_dist_output = output_metric(other, current, ...)
                output_metric(other, current, dim,
                              rev_grad_dist_output.data());

                double w_l;
                if (dist_output > 0.0) {
                    // pow(1 + a * dist^{2b}, -1): numba lowers the integer -1
                    // exponent to a reciprocal.
                    w_l = 1.0 / (1.0 + a * std::pow(dist_output, 2.0 * b));
                } else {
                    w_l = 1.0;
                }
                double grad_coeff = 2.0 * b * (w_l - 1.0) /
                                    (dist_output + 1e-6);

                for (int d = 0; d < dim; ++d) {
                    double grad_d = clip(grad_coeff * grad_dist_output[d]);
                    current[d] =
                        static_cast<float>(current[d] + grad_d * alpha);
                    if (move_other) {
                        grad_d = clip(grad_coeff * rev_grad_dist_output[d]);
                        other[d] =
                            static_cast<float>(other[d] + grad_d * alpha);
                    }
                }

                epoch_of_next_sample[i] += epochs_per_sample[i];

                // int() truncates toward zero, as does the C++ cast.
                const long long n_neg_samples = static_cast<long long>(
                    (static_cast<double>(n) -
                     epoch_of_next_negative_sample[i]) /
                    epochs_per_negative_sample[i]);

                int64_t* state =
                    rng_state_per_sample.data() +
                    3 * static_cast<std::size_t>(j);

                for (long long p = 0; p < n_neg_samples; ++p) {
                    // numba `%` follows Python semantics (sign of divisor).
                    long long r =
                        static_cast<long long>(tau_rand_int(state)) %
                        static_cast<long long>(n_vertices);
                    if (r < 0) r += n_vertices;
                    k = static_cast<int32_t>(r);

                    other = tail_embedding + static_cast<std::size_t>(k) * dim;

                    dist_output = output_metric(current, other, dim,
                                                grad_dist_output.data());

                    if (dist_output > 0.0) {
                        w_l = 1.0 /
                              (1.0 + a * std::pow(dist_output, 2.0 * b));
                    } else if (j == k) {
                        continue;
                    } else {
                        w_l = 1.0;
                    }

                    grad_coeff =
                        gamma * 2.0 * b * w_l / (dist_output + 1e-6);

                    for (int d = 0; d < dim; ++d) {
                        const double grad_d =
                            clip(grad_coeff * grad_dist_output[d]);
                        current[d] =
                            static_cast<float>(current[d] + grad_d * alpha);
                    }
                }

                epoch_of_next_negative_sample[i] +=
                    static_cast<double>(n_neg_samples) *
                    epochs_per_negative_sample[i];
            }
        }

        alpha = initial_alpha *
                (1.0 - (static_cast<double>(n) / static_cast<double>(n_epochs)));
    }
}

void optimize_layout_inverse(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    const float* weight, const float* sigmas, const float* rhos, int n_epochs,
    int n_vertices, const std::vector<double>& epochs_per_sample, double a,
    double b, int64_t rng_state[3], const OutputMetricFn& output_metric,
    double gamma, double initial_alpha, double negative_sample_rate,
    bool move_other) {
    (void)a;
    (void)b;  // unused upstream too (accepted for signature parity)
    (void)n_head_vertices;

    double alpha = initial_alpha;
    const std::size_t n_edges = epochs_per_sample.size();

    std::vector<double> epochs_per_negative_sample(n_edges);
    for (std::size_t i = 0; i < n_edges; ++i) {
        epochs_per_negative_sample[i] =
            epochs_per_sample[i] / negative_sample_rate;
    }
    std::vector<double> epoch_of_next_negative_sample =
        epochs_per_negative_sample;
    std::vector<double> epoch_of_next_sample = epochs_per_sample;

    std::vector<double> grad_dist_output(dim);

    for (int n = 0; n < n_epochs; ++n) {
        for (std::size_t i = 0; i < n_edges; ++i) {
            if (epoch_of_next_sample[i] <= static_cast<double>(n)) {
                const int32_t j = head[i];
                int32_t k = tail[i];

                float* current =
                    head_embedding + static_cast<std::size_t>(j) * dim;
                float* other =
                    tail_embedding + static_cast<std::size_t>(k) * dim;

                double dist_output =
                    output_metric(current, other, dim, grad_dist_output.data());

                // w_l = weight[i]; grad_coeff = -(1 / (w_l * sigmas[k] +
                // 1e-6)). numba types literals as float64, so only the
                // w_l * sigmas[k] product is float32; the + 1e-6 and the
                // reciprocal are float64 (verified against numba 0.67).
                const float w_l = weight[i];
                double grad_coeff =
                    -(1.0 /
                      (static_cast<double>(w_l * sigmas[k]) + 1e-6));

                for (int d = 0; d < dim; ++d) {
                    const double grad_d =
                        clip(grad_coeff * grad_dist_output[d]);
                    current[d] =
                        static_cast<float>(current[d] + grad_d * alpha);
                    if (move_other) {
                        other[d] =
                            static_cast<float>(other[d] + (-grad_d) * alpha);
                    }
                }

                epoch_of_next_sample[i] += epochs_per_sample[i];

                const long long n_neg_samples = static_cast<long long>(
                    (static_cast<double>(n) -
                     epoch_of_next_negative_sample[i]) /
                    epochs_per_negative_sample[i]);

                for (long long p = 0; p < n_neg_samples; ++p) {
                    long long r =
                        static_cast<long long>(tau_rand_int(rng_state)) %
                        static_cast<long long>(n_vertices);
                    if (r < 0) r += n_vertices;
                    k = static_cast<int32_t>(r);

                    other = tail_embedding + static_cast<std::size_t>(k) * dim;

                    dist_output = output_metric(current, other, dim,
                                                grad_dist_output.data());

                    // w_h = np.exp(-max(dist - rhos[k], 1e-6) /
                    //              (sigmas[k] + 1e-6)) -- all float64 (the
                    // f4 reads promote; literals are f8 in numba).
                    const double w_h = std::exp(
                        -std::max(dist_output -
                                      static_cast<double>(rhos[k]),
                                  1e-6) /
                        (static_cast<double>(sigmas[k]) + 1e-6));
                    grad_coeff =
                        -gamma *
                        ((0.0 - w_h) /
                         ((1.0 - w_h) * static_cast<double>(sigmas[k]) +
                          1e-6));

                    for (int d = 0; d < dim; ++d) {
                        const double grad_d =
                            clip(grad_coeff * grad_dist_output[d]);
                        current[d] =
                            static_cast<float>(current[d] + grad_d * alpha);
                    }
                }

                epoch_of_next_negative_sample[i] +=
                    static_cast<double>(n_neg_samples) *
                    epochs_per_negative_sample[i];
            }
        }

        alpha = initial_alpha *
                (1.0 - (static_cast<double>(n) / static_cast<double>(n_epochs)));
    }
}

}  // namespace umap
