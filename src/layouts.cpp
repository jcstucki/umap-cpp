// Faithful C++ port of layouts.py::optimize_layout_euclidean and the
// simplicial_set_embedding glue from umap_.py (lmcinnes/umap, BSD 3 clause).
// See include/umap/layouts.hpp for the semantics notes.
#include "umap/layouts.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
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

namespace {

// One epoch over edges [i_begin, i_end) — the numba
// _optimize_layout_euclidean_single_epoch body (densmap_flag == False path).
void optimize_euclidean_epoch_range(
    std::size_t i_begin, std::size_t i_end, float* head_embedding,
    float* tail_embedding, const int32_t* head, const int32_t* tail,
    int n_vertices, const double* epochs_per_sample, double a, double b,
    int64_t* rng_state_per_sample, double gamma, int dim, bool move_other,
    double alpha, const double* epochs_per_negative_sample,
    double* epoch_of_next_negative_sample, double* epoch_of_next_sample,
    int n) {
    for (std::size_t i = i_begin; i < i_end; ++i) {
        if (epoch_of_next_sample[i] <= static_cast<double>(n)) {
            const int32_t j = head[i];
            int32_t k = tail[i];

            float* current = head_embedding + static_cast<std::size_t>(j) * dim;
            float* other = tail_embedding + static_cast<std::size_t>(k) * dim;

            float dist_squared = rdist(current, other, dim);

            // TODO(densmap): densmap_flag branch (grad_cor_coeff) not ported.

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
    std::vector<std::vector<float>>* embedding_list) {
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
    if (densmap) {
        // TODO(densmap): port _optimize_layout_euclidean_densmap_epoch_init and
        // the densmap_flag gradient correction. Accepted for signature parity.
        throw std::invalid_argument(
            "optimize_layout_euclidean: densmap=true is not implemented yet");
    }

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
        if (!parallel || n_threads <= 1) {
            // Sequential numba path (parallel=False): edges in order.
            optimize_euclidean_epoch_range(
                0, n_edges, head_embedding, tail_embedding, head.data(),
                tail.data(), n_vertices, epochs_per_sample.data(), a, b,
                rng_state_per_sample.data(), gamma, dim, move_other, alpha,
                epochs_per_negative_sample.data(),
                epoch_of_next_negative_sample.data(),
                epoch_of_next_sample.data(), n);
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
                    epoch_of_next_sample.data(), n);
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
int resolve_epochs_and_filter_graph(SparseMatrix& graph, int n_epochs) {
    // For smaller datasets we can use more epochs.
    const int default_epochs = (graph.n_rows <= 10000) ? 500 : 200;
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
    std::vector<std::vector<float>>* embedding_list) {
    // graph = graph.tocoo(); graph.sum_duplicates() — identity for our
    // canonical CSR (row-major, per-row sorted, no duplicates).
    SparseMatrix graph = graph_in;
    const int n_samples = graph.n_rows;
    const int n_vertices = graph.n_cols;

    if (!graph_prefiltered) {
        n_epochs = resolve_epochs_and_filter_graph(graph, n_epochs);
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

    // Spectral/pca/random initialisation (+ noisy_scale_coords) happens before
    // this function; densmap / output_dens aux computation is TODO.
    optimize_layout_euclidean(embedding.data(), n_samples, embedding.data(),
                              n_components, head, tail, n_epochs, n_vertices,
                              epochs_per_sample, a, b, rng_state, gamma,
                              initial_alpha, negative_sample_rate, parallel,
                              /*densmap=*/false, /*move_other=*/true,
                              epochs_list, embedding_list);
    return embedding;
}

}  // namespace umap
