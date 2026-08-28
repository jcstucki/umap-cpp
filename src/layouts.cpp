// Faithful C++ port of layouts.py::optimize_layout_euclidean and the
// simplicial_set_embedding glue from umap_.py (lmcinnes/umap, BSD 3 clause).
// See include/umap/layouts.hpp for the semantics notes.
#include "umap/layouts.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>

#include "umap/rng.hpp"

namespace umap {

std::vector<double> make_epochs_per_sample(const float* weights,
                                           std::size_t n_weights,
                                           int n_epochs) {
    // result = np.full(n, -1.0, dtype=np.float64)
    // n_samples = n_epochs * (weights / weights.max())   # float32
    // result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[...])
    std::vector<double> result(n_weights, -1.0);
    if (n_weights == 0) return result;
    float w_max = weights[0];
    for (std::size_t i = 1; i < n_weights; ++i) {
        if (weights[i] > w_max) w_max = weights[i];
    }
    const float n_epochs_f = static_cast<float>(n_epochs);
    for (std::size_t i = 0; i < n_weights; ++i) {
        float n_samples = n_epochs_f * (weights[i] / w_max);
        if (n_samples > 0.0f) {
            result[i] =
                static_cast<double>(n_epochs) / static_cast<double>(n_samples);
        }
    }
    return result;
}

// --- densmap additions ------------------------------------------------------
#if defined(__APPLE__)
// numpy's macOS wheels (>= 2.0) link Accelerate; np.dot(f32, f32) is
// cblas_sdot, whose SIMD accumulation order differs from a sequential loop.
// Calling the same library reproduces np.dot bit for bit.
extern "C" float cblas_sdot(int n, const float* x, int incx, const float* y,
                            int incy);
#endif

namespace {

// numpy FLOAT_pairwise_sum (see src/umap.cpp for the float64/float32 twins in
// the umap-class translation unit): 8 accumulators over blocks of 8, reduced
// ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)), recursive split above 128.
float numpy_pairwise_sum_f32(const float* a, int64_t n) {
    if (n < 8) {
        float res = -0.0f;
        for (int64_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        float r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        int64_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (int k = 0; k < 8; ++k) r[k] += a[i + k];
        }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) +
                    ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return numpy_pairwise_sum_f32(a, n2) +
           numpy_pairwise_sum_f32(a + n2, n - n2);
}

// np.sum(float32 array): plain pairwise summation over the whole array
// (verified against np.sum in the reference environment; the
// "first-element-seeds-the-accumulator" form only applies to
// np.add.reduceat, see src/umap.cpp::numpy_add_reduce_segment).
float numpy_sum_f32(const float* a, int64_t n) {
    if (n <= 0) return 0.0f;
    return numpy_pairwise_sum_f32(a, n);
}

// np.mean(float32 array): umr_sum / n, all float32.
float numpy_mean_f32(const float* a, int64_t n) {
    return numpy_sum_f32(a, n) / static_cast<float>(n);
}

// np.var(float32 array) (numpy _methods._var): arrmean = umr_sum/n, then
// x = arr - arrmean, ret = umr_sum(x * x) / n, all float32.
float numpy_var_f32(const float* a, int64_t n) {
    const float mean = numpy_mean_f32(a, n);
    std::vector<float> x(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const float d = a[i] - mean;
        x[static_cast<std::size_t>(i)] = d * d;
    }
    return numpy_sum_f32(x.data(), n) / static_cast<float>(n);
}

// np.dot(float32, float32) -> float32 scalar.
float numpy_dot_f32(const float* a, const float* b, int64_t n) {
#if defined(__APPLE__)
    return cblas_sdot(static_cast<int>(n), a, 1, b, 1);
#else
    // Documented deviation off Apple platforms: sequential f32 accumulation
    // (BLAS vendors block/vectorise differently).
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
#endif
}

// Per-epoch densMAP state handed into the edge loop (mirrors the scalars the
// numba epoch kernel receives; f4/f8 typing preserved).
struct DensmapEpochState {
    bool flag = false;          // densmap_flag for this epoch
    const float* phi_sum = nullptr;  // dens_phi_sum (f4[n_vertices])
    const float* re_sum = nullptr;   // dens_re_sum (f4[n_vertices])
    float re_cov = 0.0f;        // np.dot(re_sum, R) / (n_vertices - 1)  (f4)
    float re_std = 0.0f;        // np.sqrt(np.var(re_sum) + var_shift)   (f4)
    float re_mean = 0.0f;       // np.mean(re_sum)                       (f4)
    double lambda = 0.0;        // densmap_kwds["lambda"]                (f8)
    const float* R = nullptr;   // dens_R (f4[n_vertices])
    const float* mu = nullptr;  // dens_mu (f4[n_edges])
    float mu_tot = 0.0f;        // np.sum(mu_sum) / 2                    (f4)
};

// layouts.py::_optimize_layout_euclidean_densmap_epoch_init (sequential
// instantiation; the prange is a plain range for the seeded path).
// re_sum / phi_sum are float32; phi and the accumulating expressions are
// float64 (literals promote), rounded to float32 on store. The final log runs
// on the float64 `epsilon + (f4 / f4)` sum, rounded to float32 on store.
void densmap_epoch_init(const float* head_embedding,
                        const float* tail_embedding, const int32_t* head,
                        const int32_t* tail, std::size_t n_edges, double a,
                        double b, int dim, std::vector<float>& re_sum,
                        std::vector<float>& phi_sum) {
    std::fill(re_sum.begin(), re_sum.end(), 0.0f);
    std::fill(phi_sum.begin(), phi_sum.end(), 0.0f);

    for (std::size_t i = 0; i < n_edges; ++i) {
        const int32_t j = head[i];
        const int32_t k = tail[i];
        const float* current =
            head_embedding + static_cast<std::size_t>(j) * dim;
        const float* other = tail_embedding + static_cast<std::size_t>(k) * dim;
        const float dist_squared = rdist(current, other, dim);

        const double phi =
            1.0 / (1.0 + a * std::pow(static_cast<double>(dist_squared), b));

        re_sum[static_cast<std::size_t>(j)] = static_cast<float>(
            re_sum[static_cast<std::size_t>(j)] +
            phi * static_cast<double>(dist_squared));
        re_sum[static_cast<std::size_t>(k)] = static_cast<float>(
            re_sum[static_cast<std::size_t>(k)] +
            phi * static_cast<double>(dist_squared));
        phi_sum[static_cast<std::size_t>(j)] =
            static_cast<float>(phi_sum[static_cast<std::size_t>(j)] + phi);
        phi_sum[static_cast<std::size_t>(k)] =
            static_cast<float>(phi_sum[static_cast<std::size_t>(k)] + phi);
    }

    const double epsilon = 1e-8;
    for (std::size_t i = 0; i < re_sum.size(); ++i) {
        const float ratio = re_sum[i] / phi_sum[i];  // f4 / f4 stays f4
        re_sum[i] =
            static_cast<float>(std::log(epsilon + static_cast<double>(ratio)));
    }
}
// ----------------------------------------------------------------------------

// One epoch over edges [i_begin, i_end) — the numba
// _optimize_layout_euclidean_single_epoch body (including the densmap_flag
// branch; `dens` carries the per-epoch densMAP scalars/arrays).
void optimize_euclidean_epoch_range(
    std::size_t i_begin, std::size_t i_end, float* head_embedding,
    float* tail_embedding, const int32_t* head, const int32_t* tail,
    int n_vertices, const double* epochs_per_sample, double a, double b,
    int64_t* rng_state_per_sample, double gamma, int dim, bool move_other,
    double alpha, const double* epochs_per_negative_sample,
    double* epoch_of_next_negative_sample, double* epoch_of_next_sample,
    int n, const DensmapEpochState& dens) {
    for (std::size_t i = i_begin; i < i_end; ++i) {
        if (epoch_of_next_sample[i] <= static_cast<double>(n)) {
            const int32_t j = head[i];
            int32_t k = tail[i];

            float* current = head_embedding + static_cast<std::size_t>(j) * dim;
            float* other = tail_embedding + static_cast<std::size_t>(k) * dim;

            float dist_squared = rdist(current, other, dim);

            // --- densmap additions ------------------------------------------
            // grad_cor_coeff: float64 except where two float32 operands meet
            // (re_std_sq, weight_j/k, mu[i] * re_std), exactly as numba types
            // the reference kernel. np.exp(f4) is float32 expf.
            double grad_cor_coeff = 0.0;
            if (dens.flag) {
                const double dsq = static_cast<double>(dist_squared);
                const double phi = 1.0 / (1.0 + a * std::pow(dsq, b));
                const double dphi_term = a * b * std::pow(dsq, b - 1.0) /
                                         (1.0 + a * std::pow(dsq, b));

                const double q_jk =
                    phi / static_cast<double>(
                              dens.phi_sum[static_cast<std::size_t>(k)]);
                const double q_kj =
                    phi / static_cast<double>(
                              dens.phi_sum[static_cast<std::size_t>(j)]);

                const double drk =
                    q_jk *
                    ((1.0 - b * (1.0 - phi)) /
                         static_cast<double>(expf(
                             dens.re_sum[static_cast<std::size_t>(k)])) +
                     dphi_term);
                const double drj =
                    q_kj *
                    ((1.0 - b * (1.0 - phi)) /
                         static_cast<double>(expf(
                             dens.re_sum[static_cast<std::size_t>(j)])) +
                     dphi_term);

                const float re_std_sq = dens.re_std * dens.re_std;  // f4
                const float weight_k =                              // all f4
                    dens.R[static_cast<std::size_t>(k)] -
                    dens.re_cov *
                        (dens.re_sum[static_cast<std::size_t>(k)] -
                         dens.re_mean) /
                        re_std_sq;
                const float weight_j =
                    dens.R[static_cast<std::size_t>(j)] -
                    dens.re_cov *
                        (dens.re_sum[static_cast<std::size_t>(j)] -
                         dens.re_mean) /
                        re_std_sq;

                grad_cor_coeff =
                    dens.lambda * static_cast<double>(dens.mu_tot) *
                    (static_cast<double>(weight_k) * drk +
                     static_cast<double>(weight_j) * drj) /
                    static_cast<double>(dens.mu[i] * dens.re_std) /
                    static_cast<double>(n_vertices);
            }
            // ----------------------------------------------------------------

            double grad_coeff;
            if (dist_squared > 0.0f) {
                grad_coeff = -2.0 * a * b *
                             std::pow(static_cast<double>(dist_squared), b - 1.0);
                grad_coeff /=
                    a * std::pow(static_cast<double>(dist_squared), b) + 1.0;
            } else {
                grad_coeff = 0.0;
            }

            for (int d = 0; d < dim; ++d) {
                double grad_d = clip(
                    grad_coeff * static_cast<double>(current[d] - other[d]));

                // --- densmap additions --------------------------------------
                if (dens.flag) {
                    grad_d += clip(2.0 * grad_cor_coeff *
                                   static_cast<double>(current[d] - other[d]));
                }
                // ------------------------------------------------------------

                current[d] =
                    static_cast<float>(current[d] + grad_d * alpha);
                if (move_other) {
                    other[d] =
                        static_cast<float>(other[d] + (-grad_d) * alpha);
                }
            }

            epoch_of_next_sample[i] += epochs_per_sample[i];

            // int() in numba truncates toward zero, as does the C++ cast.
            const long long n_neg_samples = static_cast<long long>(
                (static_cast<double>(n) - epoch_of_next_negative_sample[i]) /
                epochs_per_negative_sample[i]);

            int64_t* rng_state = rng_state_per_sample + 3 * static_cast<std::size_t>(j);

            for (long long p = 0; p < n_neg_samples; ++p) {
                // numba `%` follows Python semantics: result has the sign of
                // the (positive) divisor.
                long long r = static_cast<long long>(tau_rand_int(rng_state)) %
                              static_cast<long long>(n_vertices);
                if (r < 0) r += n_vertices;
                k = static_cast<int32_t>(r);

                other = tail_embedding + static_cast<std::size_t>(k) * dim;

                dist_squared = rdist(current, other, dim);

                if (dist_squared > 0.0f) {
                    grad_coeff = 2.0 * gamma * b;
                    grad_coeff /=
                        (0.001 + static_cast<double>(dist_squared)) *
                        (a * std::pow(static_cast<double>(dist_squared), b) +
                         1.0);
                } else if (j == k) {
                    continue;
                } else {
                    grad_coeff = 0.0;
                }

                for (int d = 0; d < dim; ++d) {
                    double grad_d;
                    if (grad_coeff > 0.0) {
                        grad_d = clip(grad_coeff *
                                      static_cast<double>(current[d] - other[d]));
                    } else {
                        grad_d = 0.0;
                    }
                    current[d] =
                        static_cast<float>(current[d] + grad_d * alpha);
                }
            }

            epoch_of_next_negative_sample[i] +=
                static_cast<double>(n_neg_samples) *
                epochs_per_negative_sample[i];
        }
    }
}

}  // namespace

void optimize_layout_euclidean(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    int n_epochs, int n_vertices, const std::vector<double>& epochs_per_sample,
    double a, double b, const int64_t rng_state[3], double gamma,
    double initial_alpha, double negative_sample_rate, bool parallel,
    bool densmap, bool move_other, const std::vector<int>* epochs_list,
    std::vector<std::vector<float>>* embedding_list,
    const DensmapParams* densmap_kwds) {
    // layouts.py: if isinstance(n_epochs, list): epochs_list = n_epochs;
    //             n_epochs = max(epochs_list)
    if (epochs_list != nullptr) {
        if (epochs_list->empty() || embedding_list == nullptr) {
            throw std::invalid_argument(
                "optimize_layout_euclidean: epochs_list requires a non-empty "
                "list and an embedding_list output");
        }
        n_epochs = *std::max_element(epochs_list->begin(), epochs_list->end());
        embedding_list->clear();
    }
    // --- densmap additions --------------------------------------------------
    if (densmap && densmap_kwds == nullptr) {
        throw std::invalid_argument(
            "optimize_layout_euclidean: densmap=true requires densmap_kwds");
    }
    // if densmap: dens_mu_tot = np.sum(densmap_kwds["mu_sum"]) / 2 (float32),
    // dens_phi_sum / dens_re_sum = np.zeros(n_vertices, dtype=np.float32).
    DensmapEpochState dens;
    std::vector<float> dens_phi_sum;
    std::vector<float> dens_re_sum;
    if (densmap) {
        dens.mu_tot =
            numpy_sum_f32(densmap_kwds->mu_sum.data(),
                          static_cast<int64_t>(densmap_kwds->mu_sum.size())) /
            2.0f;
        dens.lambda = densmap_kwds->lambda;
        dens.R = densmap_kwds->R.data();
        dens.mu = densmap_kwds->mu.data();
        dens_phi_sum.assign(static_cast<std::size_t>(n_vertices), 0.0f);
        dens_re_sum.assign(static_cast<std::size_t>(n_vertices), 0.0f);
        dens.phi_sum = dens_phi_sum.data();
        dens.re_sum = dens_re_sum.data();
    }
    // ------------------------------------------------------------------------

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

    // rng_state_per_sample = np.full((n_head, 3), rng_state, dtype=np.int64)
    //     + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)
    // i.e. each head vertex gets its own state, offset by the int64 bit pattern
    // of the double-precision value of its first coordinate at entry (int64
    // addition wraps, as numpy's does).
    std::vector<int64_t> rng_state_per_sample(
        static_cast<std::size_t>(n_head_vertices) * 3);
    for (int j = 0; j < n_head_vertices; ++j) {
        const double v0 = static_cast<double>(
            head_embedding[static_cast<std::size_t>(j) * dim]);
        int64_t offset;
        static_assert(sizeof(offset) == sizeof(v0), "bitcast size");
        std::memcpy(&offset, &v0, sizeof(offset));
        for (int c = 0; c < 3; ++c) {
            rng_state_per_sample[static_cast<std::size_t>(j) * 3 + c] =
                static_cast<int64_t>(static_cast<uint64_t>(rng_state[c]) +
                                     static_cast<uint64_t>(offset));
        }
    }

    unsigned n_threads = 1;
    if (parallel) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 4;
        if (static_cast<std::size_t>(n_threads) > n_edges && n_edges > 0) {
            n_threads = static_cast<unsigned>(n_edges);
        }
    }

    for (int n = 0; n < n_epochs; ++n) {
        // --- densmap additions ----------------------------------------------
        // densmap_flag = densmap and lambda > 0
        //                and (n + 1) / n_epochs > 1 - frac
        dens.flag = densmap && (densmap_kwds->lambda > 0.0) &&
                    ((static_cast<double>(n + 1) /
                      static_cast<double>(n_epochs)) >
                     (1.0 - densmap_kwds->frac));
        if (dens.flag) {
            densmap_epoch_init(head_embedding, tail_embedding, head.data(),
                               tail.data(), n_edges, a, b, dim, dens_re_sum,
                               dens_phi_sum);
            // dens_re_std = np.sqrt(np.var(re_sum) + var_shift): float32
            // (NEP 50: the Python-float var_shift is weak). Likewise mean/cov.
            const int64_t nv = static_cast<int64_t>(dens_re_sum.size());
            const float var = numpy_var_f32(dens_re_sum.data(), nv);
            dens.re_std = std::sqrt(
                var + static_cast<float>(densmap_kwds->var_shift));
            dens.re_mean = numpy_mean_f32(dens_re_sum.data(), nv);
            dens.re_cov =
                numpy_dot_f32(dens_re_sum.data(), densmap_kwds->R.data(), nv) /
                static_cast<float>(n_vertices - 1);
        } else {
            dens.re_std = 0.0f;
            dens.re_mean = 0.0f;
            dens.re_cov = 0.0f;
        }
        // --------------------------------------------------------------------
        if (!parallel || n_threads <= 1) {
            // Sequential numba path (parallel=False): edges in order.
            optimize_euclidean_epoch_range(
                0, n_edges, head_embedding, tail_embedding, head.data(),
                tail.data(), n_vertices, epochs_per_sample.data(), a, b,
                rng_state_per_sample.data(), gamma, dim, move_other, alpha,
                epochs_per_negative_sample.data(),
                epoch_of_next_negative_sample.data(),
                epoch_of_next_sample.data(), n, dens);
        } else {
            // "Hogwild" mirror of numba prange(parallel=True): contiguous
            // static chunks, racy float32 embedding / rng-state updates
            // (that is numba's semantics too; non-deterministic).
            std::vector<std::thread> workers;
            workers.reserve(n_threads);
            const std::size_t chunk = (n_edges + n_threads - 1) / n_threads;
            for (unsigned t = 0; t < n_threads; ++t) {
                const std::size_t lo = static_cast<std::size_t>(t) * chunk;
                const std::size_t hi = std::min(n_edges, lo + chunk);
                if (lo >= hi) break;
                workers.emplace_back(
                    optimize_euclidean_epoch_range, lo, hi, head_embedding,
                    tail_embedding, head.data(), tail.data(), n_vertices,
                    epochs_per_sample.data(), a, b,
                    rng_state_per_sample.data(), gamma, dim, move_other, alpha,
                    epochs_per_negative_sample.data(),
                    epoch_of_next_negative_sample.data(),
                    epoch_of_next_sample.data(), n, std::cref(dens));
            }
            for (auto& w : workers) w.join();
        }

        alpha = initial_alpha *
                (1.0 - (static_cast<double>(n) / static_cast<double>(n_epochs)));

        // if epochs_list is not None and n in epochs_list:
        //     embedding_list.append(head_embedding.copy())
        if (epochs_list != nullptr &&
            std::find(epochs_list->begin(), epochs_list->end(), n) !=
                epochs_list->end()) {
            embedding_list->emplace_back(
                head_embedding,
                head_embedding +
                    static_cast<std::size_t>(n_head_vertices) * dim);
        }
    }

    // Add the last embedding to the list as well.
    if (epochs_list != nullptr) {
        embedding_list->emplace_back(
            head_embedding,
            head_embedding + static_cast<std::size_t>(n_head_vertices) * dim);
    }
}

// --- umap-class additions ---------------------------------------------------
int resolve_epochs_and_filter_graph(SparseMatrix& graph, int n_epochs,
                                    int default_epochs_boost) {
    // For smaller datasets we can use more epochs (+200 for densMAP, which
    // callers pass through default_epochs_boost).
    const int default_epochs =
        ((graph.n_rows <= 10000) ? 500 : 200) + default_epochs_boost;
    if (n_epochs < 0) n_epochs = default_epochs;  // n_epochs is None
    // For the list-of-epochs variant callers pass max(list) here (upstream's
    // n_epochs_max).
    const int n_epochs_max = n_epochs;

    // graph.data[graph.data < graph.data.max() / n_epochs_max] = 0.0
    // (threshold computed in float32, like np.float32.max() / float(n)).
    if (!graph.data.empty()) {
        float data_max = graph.data[0];
        for (float v : graph.data) {
            if (v > data_max) data_max = v;
        }
        const float threshold =
            data_max / static_cast<float>(n_epochs_max > 10 ? n_epochs_max
                                                            : default_epochs);
        for (float& v : graph.data) {
            if (v < threshold) v = 0.0f;
        }
    }
    graph.eliminate_zeros();
    return n_epochs;
}
// ----------------------------------------------------------------------------

std::vector<float> simplicial_set_embedding_from_init(
    const SparseMatrix& graph_in, const float* init_embedding, int n_components,
    double a, double b, const int64_t rng_state[3], int n_epochs, double gamma,
    double initial_alpha, double negative_sample_rate, bool parallel,
    bool graph_prefiltered, const std::vector<int>* epochs_list,
    std::vector<std::vector<float>>* embedding_list,
    const DensmapParams* densmap_kwds, const OutputMetricFn* output_metric) {
    // --- densmap additions --------------------------------------------------
    if (densmap_kwds != nullptr && output_metric != nullptr) {
        throw std::invalid_argument(
            "Non-Euclidean output metric not supported for densMAP.");
    }
    if (output_metric != nullptr && epochs_list != nullptr) {
        throw std::invalid_argument(
            "simplicial_set_embedding_from_init: n_epochs as a list is not "
            "supported for non-euclidean output metrics (the reference fails "
            "with a numba typing error)");
    }
    // ------------------------------------------------------------------------
    // graph = graph.tocoo(); graph.sum_duplicates() — identity for our
    // canonical CSR (row-major, per-row sorted, no duplicates).
    SparseMatrix graph = graph_in;
    const int n_samples = graph.n_rows;
    const int n_vertices = graph.n_cols;

    if (!graph_prefiltered) {
        // Use more epochs for densMAP (upstream: default_epochs += 200).
        n_epochs = resolve_epochs_and_filter_graph(
            graph, n_epochs, densmap_kwds != nullptr ? 200 : 0);
    } else if (n_epochs < 0) {
        throw std::invalid_argument(
            "simplicial_set_embedding_from_init: graph_prefiltered requires "
            "the resolved n_epochs");
    }
    const int n_epochs_max = n_epochs;

    // head = graph.row; tail = graph.col (COO of the canonical CSR).
    std::vector<int32_t> head;
    std::vector<int32_t> tail;
    head.reserve(graph.nnz());
    tail.reserve(graph.nnz());
    for (int32_t row = 0; row < graph.n_rows; ++row) {
        for (int32_t idx = graph.indptr[row]; idx < graph.indptr[row + 1];
             ++idx) {
            head.push_back(row);
            tail.push_back(graph.indices[idx]);
        }
    }

    std::vector<double> epochs_per_sample =
        make_epochs_per_sample(graph.data.data(), graph.data.size(),
                               n_epochs_max);

    // embedding = (10.0 * (embedding - min0) / (max0 - min0)).astype(f4, "C")
    // per-component, float32 arithmetic, multiplication before division.
    std::vector<float> embedding(
        init_embedding,
        init_embedding + static_cast<std::size_t>(n_samples) * n_components);
    for (int d = 0; d < n_components; ++d) {
        float mn = embedding[d];
        float mx = embedding[d];
        for (int i = 1; i < n_samples; ++i) {
            const float v = embedding[static_cast<std::size_t>(i) * n_components + d];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        const float span = mx - mn;
        for (int i = 0; i < n_samples; ++i) {
            float& v = embedding[static_cast<std::size_t>(i) * n_components + d];
            v = (10.0f * (v - mn)) / span;
        }
    }

    // Spectral/pca/random initialisation (+ noisy_scale_coords) happens
    // before this function; the densMAP aux (mu / mu_sum / R) and the
    // output_dens radii are the caller's concern (UMAP.fit) because they need
    // graph_dists_ / the numpy RandomState.
    if (output_metric != nullptr) {
        // euclidean_output=False: optimize_layout_generic (sequential numba
        // path; there is no parallel instantiation upstream either).
        optimize_layout_generic(embedding.data(), n_samples, embedding.data(),
                                n_components, head, tail, n_epochs, n_vertices,
                                epochs_per_sample, a, b, rng_state,
                                *output_metric, gamma, initial_alpha,
                                negative_sample_rate, /*move_other=*/true);
    } else {
        optimize_layout_euclidean(
            embedding.data(), n_samples, embedding.data(), n_components, head,
            tail, n_epochs, n_vertices, epochs_per_sample, a, b, rng_state,
            gamma, initial_alpha, negative_sample_rate, parallel,
            /*densmap=*/densmap_kwds != nullptr, /*move_other=*/true,
            epochs_list, embedding_list, densmap_kwds);
    }
    return embedding;
}

}  // namespace umap
