// Shared helpers for the Phase 2 port of the upstream test suite
// (reference/umap/umap/tests/*.py -> doctest). Ports of the sklearn/scipy
// utilities those tests lean on: KDTree-style exact kNN accuracy scoring
// (test_umap_nn.py::knn), KMeans(k-means++/Lloyd) + adjusted_rand_score
// (test_umap_ops.py / test_umap_on_iris.py), disconnected_vertices
// (umap/utils.py), and small dataset conversion utilities for the baked
// float64 conftest datasets in upstream_data.hpp.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "umap/sparse_matrix.hpp"

namespace testutil {

// ------------------------------------------------------------- conversions

inline std::vector<float> to_f32(const double* a, std::size_t n) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(a[i]);
    return out;
}

inline std::vector<float> to_f32(const std::uint8_t* a, std::size_t n) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(a[i]);
    return out;
}

inline std::vector<double> to_f64(const std::uint8_t* a, std::size_t n) {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<double>(a[i]);
    return out;
}

// Row subset X[mask] / X[~mask] of a row-major matrix.
template <typename T>
std::vector<T> select_rows(const T* X, int64_t n, int64_t dim,
                           const std::uint8_t* mask, bool keep) {
    std::vector<T> out;
    for (int64_t i = 0; i < n; ++i)
        if ((mask[i] != 0) == keep)
            out.insert(out.end(), X + i * dim, X + (i + 1) * dim);
    return out;
}

// Dense (row-major) -> CSR arrays (explicit zeros dropped, like
// scipy.sparse.csr_matrix of a dense array).
struct Csr {
    std::vector<int32_t> indptr, indices;
    std::vector<float> data;
};

inline Csr dense_to_csr(const float* X, int64_t n, int64_t dim) {
    Csr c;
    c.indptr.assign(1, 0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            const float v = X[i * dim + j];
            if (v != 0.0f) {
                c.indices.push_back(static_cast<int32_t>(j));
                c.data.push_back(v);
            }
        }
        c.indptr.push_back(static_cast<int32_t>(c.indices.size()));
    }
    return c;
}

// Full euclidean distance matrix in float64 (scipy pdist/cdist semantics).
inline std::vector<double> pairwise_euclidean_f64(const double* X, int64_t nx,
                                                  const double* Y, int64_t ny,
                                                  int64_t dim) {
    std::vector<double> out(static_cast<std::size_t>(nx) * ny);
    for (int64_t i = 0; i < nx; ++i)
        for (int64_t j = 0; j < ny; ++j) {
            double s = 0.0;
            for (int64_t d = 0; d < dim; ++d) {
                const double diff = X[i * dim + d] - Y[j * dim + d];
                s += diff * diff;
            }
            out[i * ny + j] = std::sqrt(s);
        }
    return out;
}

// --------------------------------------------- kNN accuracy (test_umap_nn)

// test_umap_nn.py::knn -- fraction of the true k_true=10 nearest neighbours
// (exact, KDTree equivalent: euclidean, self included, ties by index)
// recovered in `indices` (row width k). Data in float64 like upstream; the
// upstream helper always scores against the true 10-NN regardless of k.
inline double knn_accuracy(const int32_t* indices, int64_t k,
                           const double* data, int64_t n, int64_t dim,
                           int64_t k_true = 10) {
    std::vector<double> drow(n);
    std::vector<int32_t> order(n);
    double num_correct = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t d = 0; d < dim; ++d) {
                const double diff = data[i * dim + d] - data[j * dim + d];
                s += diff * diff;
            }
            drow[j] = s;
        }
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return drow[a] < drow[b];
        });
        for (int64_t t = 0; t < k_true; ++t) {
            const int32_t true_idx = order[t];
            for (int64_t s = 0; s < k; ++s)
                if (indices[i * k + s] == true_idx) {
                    num_correct += 1.0;
                    break;
                }
        }
    }
    return num_correct / (static_cast<double>(n) * k_true);
}

// sklearn.preprocessing.normalize(X, norm="l2") in float64.
inline std::vector<double> l2_normalize_rows(const double* X, int64_t n,
                                             int64_t dim) {
    std::vector<double> out(static_cast<std::size_t>(n) * dim);
    for (int64_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t d = 0; d < dim; ++d) s += X[i * dim + d] * X[i * dim + d];
        const double norm = s > 0.0 ? std::sqrt(s) : 1.0;
        for (int64_t d = 0; d < dim; ++d) out[i * dim + d] = X[i * dim + d] / norm;
    }
    return out;
}

// ------------------------------------------------- KMeans + ARI (ops/iris)

// sklearn.cluster.KMeans equivalent: k-means++ init, Lloyd iterations,
// n_init=10 restarts keeping the lowest inertia. Seeded (upstream leaves
// KMeans unseeded; only the clustering quality matters for the assertions).
inline std::vector<int> kmeans_predict(const float* X, int64_t n, int64_t dim,
                                       int k, uint32_t seed = 0) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::vector<double> best_centers;
    double best_inertia = std::numeric_limits<double>::infinity();
    std::vector<int> best_labels(n, 0), labels(n, 0);

    const auto sqdist = [&](const double* c, int64_t i) {
        double s = 0.0;
        for (int64_t d = 0; d < dim; ++d) {
            const double diff = c[d] - static_cast<double>(X[i * dim + d]);
            s += diff * diff;
        }
        return s;
    };

    for (int init = 0; init < 10; ++init) {
        // k-means++ seeding.
        std::vector<double> centers(static_cast<std::size_t>(k) * dim);
        std::vector<double> mindist(n, std::numeric_limits<double>::infinity());
        int64_t first = static_cast<int64_t>(unif(rng) * n) % n;
        for (int64_t d = 0; d < dim; ++d)
            centers[d] = static_cast<double>(X[first * dim + d]);
        for (int c = 1; c < k; ++c) {
            double total = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                mindist[i] = std::min(mindist[i], sqdist(&centers[(c - 1) * dim], i));
                total += mindist[i];
            }
            double target = unif(rng) * total;
            int64_t chosen = n - 1;
            double acc = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                acc += mindist[i];
                if (acc >= target) {
                    chosen = i;
                    break;
                }
            }
            for (int64_t d = 0; d < dim; ++d)
                centers[c * dim + d] = static_cast<double>(X[chosen * dim + d]);
        }
        // Lloyd.
        double inertia = 0.0;
        for (int iter = 0; iter < 300; ++iter) {
            inertia = 0.0;
            bool changed = false;
            for (int64_t i = 0; i < n; ++i) {
                double best = std::numeric_limits<double>::infinity();
                int arg = 0;
                for (int c = 0; c < k; ++c) {
                    const double s = sqdist(&centers[c * dim], i);
                    if (s < best) {
                        best = s;
                        arg = c;
                    }
                }
                if (labels[i] != arg) {
                    labels[i] = arg;
                    changed = true;
                }
                inertia += best;
            }
            std::vector<double> sums(static_cast<std::size_t>(k) * dim, 0.0);
            std::vector<int64_t> counts(k, 0);
            for (int64_t i = 0; i < n; ++i) {
                counts[labels[i]]++;
                for (int64_t d = 0; d < dim; ++d)
                    sums[labels[i] * dim + d] += static_cast<double>(X[i * dim + d]);
            }
            for (int c = 0; c < k; ++c)
                if (counts[c] > 0)
                    for (int64_t d = 0; d < dim; ++d)
                        centers[c * dim + d] = sums[c * dim + d] / counts[c];
            if (!changed) break;
        }
        if (inertia < best_inertia) {
            best_inertia = inertia;
            best_labels = labels;
            best_centers = centers;
        }
    }
    (void)best_centers;
    return best_labels;
}

// sklearn.metrics.adjusted_rand_score.
inline double adjusted_rand_score(const std::vector<int>& a,
                                  const std::vector<int>& b) {
    const int64_t n = static_cast<int64_t>(a.size());
    const int ka = *std::max_element(a.begin(), a.end()) + 1;
    const int kb = *std::max_element(b.begin(), b.end()) + 1;
    std::vector<int64_t> table(static_cast<std::size_t>(ka) * kb, 0);
    std::vector<int64_t> ra(ka, 0), rb(kb, 0);
    for (int64_t i = 0; i < n; ++i) {
        table[a[i] * kb + b[i]]++;
        ra[a[i]]++;
        rb[b[i]]++;
    }
    const auto comb2 = [](int64_t m) { return m * (m - 1) / 2.0; };
    double sum_comb = 0.0, sum_a = 0.0, sum_b = 0.0;
    for (int64_t v : table) sum_comb += comb2(v);
    for (int64_t v : ra) sum_a += comb2(v);
    for (int64_t v : rb) sum_b += comb2(v);
    const double total = comb2(n);
    const double expected = sum_a * sum_b / total;
    const double max_index = 0.5 * (sum_a + sum_b);
    if (max_index == expected) return 1.0;
    return (sum_comb - expected) / (max_index - expected);
}

// ------------------------------------------------ graph utilities (ops)

// umap/utils.py::disconnected_vertices for a fitted model's graph_:
// vertices whose graph row sums to zero.
inline std::vector<bool> disconnected_vertices(const umap::SparseMatrix& g) {
    std::vector<bool> out(g.n_rows, false);
    for (int64_t i = 0; i < g.n_rows; ++i) {
        double s = 0.0;
        for (int32_t k = g.indptr[i]; k < g.indptr[i + 1]; ++k)
            s += g.data[k];
        out[i] = (s == 0.0);
    }
    return out;
}

}  // namespace testutil
