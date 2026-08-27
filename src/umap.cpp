// Faithful C++ port of the UMAP class from reference/umap/umap/umap_.py
// (lmcinnes/umap, BSD 3 clause). See include/umap/umap.hpp for scope notes.
#include "umap/umap.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "umap/distances.hpp"
#include "umap/find_ab_params.hpp"
#include "umap/fuzzy_simplicial_set.hpp"
#include "umap/layouts.hpp"
#include "umap/spectral.hpp"

namespace umap {

// ---------------------------------------------------------------- pairwise

namespace {

// umap.distances.euclidean under numba on float32 rows: the squared diffs are
// float32, the accumulator is float64 (`result = 0.0`), sqrt in float64,
// rounded to float32 on store into the float32 output matrix. (Verified
// element-exact against pairwise_distances(X, metric=named_distances
// ["euclidean"]) from the reference environment.)
inline float numba_euclidean_f32(const float* x, const float* y, int64_t dim) {
    double result = 0.0;
    for (int64_t k = 0; k < dim; ++k) {
        const float d = x[k] - y[k];
        const float sq = d * d;
        result += static_cast<double>(sq);
    }
    return static_cast<float>(std::sqrt(result));
}

}  // namespace

std::vector<float> pairwise_distances_sklearn(const float* X, int64_t nx,
                                              const float* Y, int64_t ny,
                                              int64_t dim,
                                              const std::string& metric,
                                              bool x_is_y) {
    // Upstream's small-data branch calls sklearn's pairwise_distances with
    // metric = self._input_distance_func, a *callable* (umap's numba kernel),
    // which routes to sklearn's _pairwise_callable: a float32 output matrix
    // filled per pair (upper triangle + transpose + diagonal when X is Y).
    // The symmetrisation `out + out.T` adds exact zeros, and the diagonal is
    // metric(x, x); both are reproduced by direct evaluation below.
    std::vector<float> out(static_cast<std::size_t>(nx) * ny);
    if (metric == "euclidean" || metric == "l2") {
        for (int64_t i = 0; i < nx; ++i) {
            const float* xi = X + i * dim;
            const int64_t j0 = x_is_y ? i : 0;
            for (int64_t j = j0; j < ny; ++j) {
                const float d = (x_is_y && i == j)
                                    ? 0.0f
                                    : numba_euclidean_f32(xi, Y + j * dim, dim);
                out[static_cast<std::size_t>(i) * ny + j] = d;
                if (x_is_y) out[static_cast<std::size_t>(j) * ny + i] = d;
            }
        }
        return out;
    }

    // Other named metrics: umap's distance kernels via the double-precision
    // registry in distances.hpp (documented: matches the numba float32 kernels
    // only to float32 rounding of the double result).
    const auto& registry = named_distances();
    auto it = registry.find(metric);
    if (it == registry.end()) {
        throw std::invalid_argument(
            "pairwise_distances_sklearn: unsupported metric '" + metric + "'");
    }
    const DistanceFn& fn = it->second;
    std::vector<double> xbuf(dim), ybuf(dim);
    for (int64_t i = 0; i < nx; ++i) {
        for (int64_t k = 0; k < dim; ++k)
            xbuf[k] = static_cast<double>(X[i * dim + k]);
        for (int64_t j = 0; j < ny; ++j) {
            for (int64_t k = 0; k < dim; ++k)
                ybuf[k] = static_cast<double>(Y[j * dim + k]);
            out[static_cast<std::size_t>(i) * ny + j] = static_cast<float>(
                fn(xbuf.data(), ybuf.data(), static_cast<std::size_t>(dim)));
        }
    }
    return out;
}

void fast_knn_from_dmat(const std::vector<float>& dmat, int64_t n, int64_t m,
                        int64_t n_neighbors, int32_t* knn_indices,
                        float* knn_dists) {
    std::vector<int32_t> order(m);
    for (int64_t row = 0; row < n; ++row) {
        const float* d = dmat.data() + row * m;
        std::iota(order.begin(), order.end(), 0);
        // argsort(kind="mergesort"): stable, so ties keep ascending index.
        std::stable_sort(order.begin(), order.end(),
                         [d](int32_t a, int32_t b) { return d[a] < d[b]; });
        for (int64_t j = 0; j < n_neighbors; ++j) {
            knn_indices[row * n_neighbors + j] = order[j];
            knn_dists[row * n_neighbors + j] = d[order[j]];
        }
    }
}

// ------------------------------------------------------ init_graph_transform

namespace {

// numpy's FLOAT_pairwise_sum (loops_utils.h.src): 8 scalar accumulators over
// blocks of 8, reduced ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)), sequential tail,
// recursive split above PW_BLOCKSIZE = 128.
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
    return numpy_pairwise_sum_f32(a, n2) + numpy_pairwise_sum_f32(a + n2, n - n2);
}

// np.add.reduceat semantics per segment: the first element seeds the
// accumulator, the remainder goes through pairwise summation. This is what
// scipy's csr.sum(axis=1) uses (_minor_reduce).
float numpy_add_reduce_segment(const float* a, int64_t n) {
    if (n <= 0) return 0.0f;
    return a[0] + numpy_pairwise_sum_f32(a + 1, n - 1);
}

}  // namespace

std::vector<float> init_graph_transform(const SparseMatrix& graph,
                                        const float* embedding, int dim) {
    const int64_t n_new = graph.n_rows;
    std::vector<float> result(static_cast<std::size_t>(n_new) * dim, 0.0f);
    for (int64_t i = 0; i < n_new; ++i) {
        const int32_t start = graph.indptr[i];
        const int32_t end = graph.indptr[i + 1];
        float* out = result.data() + static_cast<std::size_t>(i) * dim;
        if (start == end) {
            for (int d = 0; d < dim; ++d)
                out[d] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        // First strength == 1.0 in CSR order copies that training point.
        int32_t exact_col = -1;
        for (int32_t k = start; k < end; ++k) {
            if (graph.data[k] == 1.0f) {
                exact_col = graph.indices[k];
                break;
            }
        }
        if (exact_col >= 0) {
            const float* src =
                embedding + static_cast<std::size_t>(exact_col) * dim;
            for (int d = 0; d < dim; ++d) out[d] = src[d];
            continue;
        }
        // Weighted mean, replicating scipy float32 arithmetic:
        // row_sums = csr.sum(axis=1) -> np.add.reduceat per row (numpy
        // pairwise summation seeded with the first element); normalized =
        // diags(1.0f / row_sums) @ graph -- scipy's SMMP csr_matmat emits
        // each result row in *reversed* column order -- then
        // normalized @ embedding runs csr_matvecs in that stored order, whose
        // axpy is compiled with fp contraction (fmaf) in the scipy wheels.
        const float row_sum =
            numpy_add_reduce_segment(graph.data.data() + start, end - start);
        const float inv = 1.0f / row_sum;
        for (int32_t k = end - 1; k >= start; --k) {
            const float w = graph.data[k] * inv;
            const float* src =
                embedding + static_cast<std::size_t>(graph.indices[k]) * dim;
            for (int d = 0; d < dim; ++d) out[d] = std::fma(w, src[d], out[d]);
        }
    }
    return result;
}

// ------------------------------------------------------------------- UMAP

UMAP::UMAP(UMAPParams params) : params_(std::move(params)) {}

void UMAP::validate_parameters() const {
    const UMAPParams& p = params_;
    if (p.set_op_mix_ratio < 0.0 || p.set_op_mix_ratio > 1.0)
        throw std::invalid_argument(
            "set_op_mix_ratio must be between 0.0 and 1.0");
    if (p.repulsion_strength < 0.0)
        throw std::invalid_argument("repulsion_strength cannot be negative");
    if (p.min_dist > p.spread)
        throw std::invalid_argument(
            "min_dist must be less than or equal to spread");
    if (p.min_dist < 0.0)
        throw std::invalid_argument("min_dist cannot be negative");
    if (p.init != "pca" && p.init != "spectral" && p.init != "random" &&
        p.init != "tswspectral")
        throw std::invalid_argument(
            "string init values must be one of: \"pca\", \"tswspectral\","
            " \"spectral\" or \"random\"");
    if (p.negative_sample_rate < 0)
        throw std::invalid_argument("negative sample rate must be positive");
    if (p.learning_rate < 0.0)
        throw std::invalid_argument("learning_rate must be positive");
    if (p.n_neighbors < 2)
        throw std::invalid_argument("n_neighbors must be greater than 1");
    if (p.n_components < 1)
        throw std::invalid_argument("n_components must be greater than 0");
    if (p.n_epochs.has_value() && *p.n_epochs < 0)
        throw std::invalid_argument(
            "n_epochs must be a nonnegative integer "
            "or a list of nonnegative integers");
}

std::vector<float> UMAP::make_fit_init(const SparseMatrix& filtered_graph,
                                       nnd::NumpyRandomState& rs) const {
    const int dim = params_.n_components;
    const int64_t n = filtered_graph.n_rows;
    const std::size_t total = static_cast<std::size_t>(n) * dim;

    if (params_.init == "random") {
        // random_state.uniform(low=-10, high=10, size=(n, dim)).astype(f32)
        std::vector<float> out(total);
        for (auto& v : out)
            v = static_cast<float>(rs.uniform(-10.0, 10.0));
        return out;
    }
    if (params_.init != "spectral") {
        throw std::runtime_error("UMAP: init='" + params_.init +
                                 "' is not implemented yet");
    }

    // init == "spectral": spectral_layout followed by noisy_scale_coords.
    std::vector<double> emb64(total);
    std::vector<int32_t> labels;
    const int32_t n_comp = connected_components(filtered_graph, labels);
    if (n_comp > 1) {
        // Documented deviation: the multi-component path uses the tau-prng
        // implementation in spectral.hpp (sklearn SpectralEmbedding /
        // per-component solves); numpy stream parity is not maintained.
        auto s = rs.rng_state3();
        int64_t tau_state[3] = {s[0], s[1], s[2]};
        const std::vector<float> multi = multi_component_layout(
            raw_data_.empty() ? nullptr : raw_data_.data(), dim_,
            filtered_graph, n_comp, labels, dim, tau_state);
        for (std::size_t i = 0; i < total; ++i)
            emb64[i] = static_cast<double>(multi[i]);
    } else {
        // _spectral_layout draws X = gen.normal(size=(n, dim + 1)) as an
        // eigsh warm start it never uses (v0=ones); consume for stream parity.
        const int k = dim + 1;
        for (int64_t i = 0; i < n * k; ++i) (void)rs.gauss();
        std::vector<double> eigenvalues, eigenvectors;
        if (normalized_laplacian_eigs(filtered_graph, k, eigenvalues,
                                      eigenvectors)) {
            // order = np.argsort(eigenvalues)[1:k] (already ascending).
            for (int64_t i = 0; i < n; ++i)
                for (int j = 0; j < dim; ++j)
                    emb64[static_cast<std::size_t>(i) * dim + j] =
                        eigenvectors[static_cast<std::size_t>(i) * k + j + 1];
        } else {
            // ArpackError fallback: gen.uniform(-10, 10, size=(n, dim)).
            std::fprintf(stderr,
                         "Spectral initialisation failed! Falling back to "
                         "random initialisation!\n");
            for (auto& v : emb64) v = rs.uniform(-10.0, 10.0);
        }
    }

    // noisy_scale_coords(embedding, random_state, max_coord=10, noise=0.0001):
    // expansion computed on the float64 spectral coordinates, scaled values
    // rounded to float32, plus float32 N(0, 1e-4) noise (numpy draw order).
    double max_abs = 0.0;
    for (double v : emb64) max_abs = std::max(max_abs, std::abs(v));
    const double expansion = 10.0 / max_abs;
    std::vector<float> out(total);
    for (std::size_t i = 0; i < total; ++i) {
        out[i] = static_cast<float>(emb64[i] * expansion) +
                 static_cast<float>(rs.normal(0.0, 0.0001));
    }
    return out;
}

UMAP& UMAP::fit(const float* X, int64_t n_samples, int64_t n_features) {
    validate_parameters();
    n_ = n_samples;
    dim_ = n_features;
    raw_data_.assign(X, X + static_cast<std::size_t>(n_samples) * n_features);

    // Handle all the optional arguments, setting default.
    if (params_.a.has_value() && params_.b.has_value()) {
        a_ = *params_.a;
        b_ = *params_.b;
    } else {
        const ABParams ab = find_ab_params(params_.spread, params_.min_dist);
        a_ = ab.a;
        b_ = ab.b;
    }
    initial_alpha_ = params_.learning_rate;

    if (n_samples <= params_.n_neighbors) {
        if (n_samples == 1) {
            embedding_.assign(params_.n_components, 0.0f);
            fitted_ = true;
            return *this;
        }
        std::fprintf(stderr,
                     "n_neighbors is larger than the dataset size; truncating "
                     "to X.shape[0] - 1\n");
        n_neighbors_ = static_cast<int>(n_samples - 1);
    } else {
        n_neighbors_ = params_.n_neighbors;
    }

    // random_state = check_random_state(self.random_state)
    nnd::NumpyRandomState rs(params_.random_state.has_value()
                                 ? *params_.random_state
                                 : std::random_device{}());

    FuzzySimplicialSetResult fuzzy;
    if (n_samples < 4096 && !params_.force_approximation_algorithm) {
        // Handle small cases efficiently by computing all distances.
        small_data_ = true;
        knn_index_.reset();
        const std::vector<float> dmat = pairwise_distances_sklearn(
            raw_data_.data(), n_samples, raw_data_.data(), n_samples,
            n_features, params_.metric, /*x_is_y=*/true);
        // fuzzy_simplicial_set(dmat, "precomputed") -> nearest_neighbors
        // precomputed branch: fast_knn_indices + row gather, inf -> -1.
        knn_indices_.assign(static_cast<std::size_t>(n_samples) * n_neighbors_,
                            -1);
        knn_dists_.assign(static_cast<std::size_t>(n_samples) * n_neighbors_,
                          0.0f);
        fast_knn_from_dmat(dmat, n_samples, n_samples, n_neighbors_,
                           knn_indices_.data(), knn_dists_.data());
        for (std::size_t i = 0; i < knn_dists_.size(); ++i) {
            if (std::isinf(knn_dists_[i])) knn_indices_[i] = -1;
        }
        fuzzy = fuzzy_simplicial_set(
            static_cast<int>(n_samples), knn_indices_.data(),
            knn_dists_.data(), n_neighbors_, params_.set_op_mix_ratio,
            params_.local_connectivity, /*apply_set_operations=*/true,
            /*return_dists=*/false);
    } else {
        // Standard case: NN-descent (continuing the numpy RandomState stream)
        // with the prepared search index kept for transform().
        small_data_ = false;
        knn_index_ = std::make_unique<nnd::PreparedIndex>(
            nnd::PreparedIndex::build(raw_data_.data(), n_samples, n_features,
                                      n_neighbors_, params_.metric, rs,
                                      params_.verbose));
        knn_indices_ = knn_index_->knn_indices();
        knn_dists_ = knn_index_->knn_dists();
        // Disconnect vertices past _disconnection_distance (default inf:
        // only unfilled heap slots at +inf are masked).
        for (std::size_t i = 0; i < knn_dists_.size(); ++i) {
            if (std::isinf(knn_dists_[i])) knn_indices_[i] = -1;
        }
        fuzzy = fuzzy_simplicial_set(
            static_cast<int>(n_samples), knn_indices_.data(),
            knn_dists_.data(), n_neighbors_, params_.set_op_mix_ratio,
            params_.local_connectivity, /*apply_set_operations=*/true,
            /*return_dists=*/false);
    }
    graph_ = fuzzy.graph;
    sigmas_ = std::move(fuzzy.sigmas);
    rhos_ = std::move(fuzzy.rhos);

    // _fit_embed_data -> simplicial_set_embedding. The weight filtering runs
    // *before* initialisation upstream (spectral init sees the filtered
    // graph), so resolve/filter first, then init, then hand the prefiltered
    // graph to the optimisation glue.
    SparseMatrix g = graph_;
    const int n_epochs =
        resolve_epochs_and_filter_graph(g, params_.n_epochs.value_or(-1));
    fit_init_ = make_fit_init(g, rs);
    const std::vector<float>& init_emb = fit_init_;
    const auto r3 = rs.rng_state3();
    const int64_t rng_state[3] = {r3[0], r3[1], r3[2]};
    embedding_ = simplicial_set_embedding_from_init(
        g, init_emb.data(), params_.n_components, a_, b_, rng_state, n_epochs,
        params_.repulsion_strength, initial_alpha_,
        static_cast<double>(params_.negative_sample_rate),
        /*parallel=*/!params_.random_state.has_value(),
        /*graph_prefiltered=*/true);

    // Fully-disconnected vertices embed as NaN (mirrors fit()).
    for (int32_t i = 0; i < graph_.n_rows; ++i) {
        float row_sum = 0.0f;
        for (int32_t k = graph_.indptr[i]; k < graph_.indptr[i + 1]; ++k)
            row_sum += graph_.data[k];
        if (row_sum == 0.0f) {
            for (int d = 0; d < params_.n_components; ++d)
                embedding_[static_cast<std::size_t>(i) * params_.n_components +
                           d] = std::numeric_limits<float>::quiet_NaN();
        }
    }

    fitted_ = true;
    return *this;
}

std::vector<float> UMAP::fit_transform(const float* X, int64_t n_samples,
                                       int64_t n_features) {
    fit(X, n_samples, n_features);
    return embedding_;
}

std::vector<float> UMAP::transform(const float* X_new,
                                   int64_t n_queries) const {
    if (!fitted_)
        throw std::runtime_error("UMAP: transform() called before fit()");
    if (n_ == 1)
        throw std::invalid_argument(
            "Transform unavailable when model was fit with only a single "
            "data sample.");
    const int dim_out = params_.n_components;

    // Input-hash shortcut: transforming the training data returns embedding_.
    if (n_queries == n_ &&
        std::memcmp(X_new, raw_data_.data(),
                    raw_data_.size() * sizeof(float)) == 0) {
        return embedding_;
    }

    nnd::NumpyRandomState rs(params_.transform_seed);
    const auto r3 = rs.rng_state3();
    const int64_t rng_state[3] = {r3[0], r3[1], r3[2]};

    // kNN of the new points against the training data.
    int64_t cols;
    std::vector<int32_t> indices;
    std::vector<float> dists;
    if (small_data_) {
        cols = n_neighbors_;
        const std::vector<float> dmat = pairwise_distances_sklearn(
            X_new, n_queries, raw_data_.data(), n_, dim_, params_.metric,
            /*x_is_y=*/false);
        indices.resize(static_cast<std::size_t>(n_queries) * cols);
        dists.resize(static_cast<std::size_t>(n_queries) * cols);
        // argpartition + argsort: k smallest per row, sorted by distance
        // (ties by ascending index).
        fast_knn_from_dmat(dmat, n_queries, n_, cols, indices.data(),
                           dists.data());
    } else {
        cols = params_.n_neighbors;
        const double epsilon =
            nnd::resolve_metric(params_.metric).angular_trees ? 0.24 : 0.12;
        indices.resize(static_cast<std::size_t>(n_queries) * cols);
        dists.resize(static_cast<std::size_t>(n_queries) * cols);
        knn_index_->query(X_new, n_queries, cols, epsilon, indices.data(),
                          dists.data());
    }

    // smooth_knn_dist with local_connectivity reduced by 1 for transform.
    const double adjusted_local_connectivity =
        std::max(0.0, params_.local_connectivity - 1.0);
    const SmoothKnnResult sk = smooth_knn_dist(
        dists.data(), static_cast<int>(n_queries), static_cast<int>(cols),
        static_cast<double>(n_neighbors_), 64, adjusted_local_connectivity);
    const MembershipStrengths ms = compute_membership_strengths(
        indices.data(), dists.data(), sk.sigmas.data(), sk.rhos.data(),
        static_cast<int>(n_queries), static_cast<int>(cols),
        /*return_dists=*/false, /*bipartite=*/true);

    // Weighted-mean initialisation uses the *unfiltered* graph in canonical
    // CSR form (graph.tocsr(); eliminate_zeros()).
    SparseMatrix csr = SparseMatrix::from_coo(
        static_cast<int32_t>(n_queries), static_cast<int32_t>(n_), ms.rows,
        ms.cols, ms.vals);
    csr.eliminate_zeros();
    std::vector<float> embedding =
        init_graph_transform(csr, embedding_.data(), dim_out);

    // Transform n_epochs rule.
    int n_epochs;
    if (!params_.n_epochs.has_value()) {
        n_epochs = (n_queries <= 10000) ? 100 : 30;
    } else {
        n_epochs = static_cast<int>(*params_.n_epochs / 3);  // n_epochs // 3.0
    }

    // Transform-specific weight filtering happens on the COO arrays in
    // generation order (row-major by query, neighbour-rank order), unlike the
    // fit path which is column-sorted.
    float vmax = 0.0f;
    for (float v : ms.vals) vmax = std::max(vmax, v);
    const float threshold = vmax / static_cast<float>(n_epochs);
    std::vector<int32_t> head, tail;
    std::vector<float> weights;
    head.reserve(ms.vals.size());
    tail.reserve(ms.vals.size());
    weights.reserve(ms.vals.size());
    for (std::size_t e = 0; e < ms.vals.size(); ++e) {
        float v = ms.vals[e];
        if (v < threshold) v = 0.0f;
        if (v != 0.0f) {
            head.push_back(ms.rows[e]);
            tail.push_back(ms.cols[e]);
            weights.push_back(v);
        }
    }
    const std::vector<double> epochs_per_sample =
        make_epochs_per_sample(weights.data(), weights.size(), n_epochs);

    // optimize_layout_euclidean(head=new points, tail=training embedding
    // (float32 copy, fixes #179 & #217), move_other=False, alpha/4).
    std::vector<float> tail_embedding = embedding_;
    optimize_layout_euclidean(
        embedding.data(), static_cast<int>(n_queries), tail_embedding.data(),
        dim_out, head, tail, n_epochs, static_cast<int>(n_),
        epochs_per_sample, a_, b_, rng_state, params_.repulsion_strength,
        initial_alpha_ / 4.0,
        static_cast<double>(params_.negative_sample_rate),
        /*parallel=*/!params_.random_state.has_value(), /*densmap=*/false,
        /*move_other=*/false);
    return embedding;
}

}  // namespace umap
