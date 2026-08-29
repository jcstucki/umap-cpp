// Port of sklearn.manifold.trustworthiness, double precision, factored out of
// tests/test_umap.cpp for the Phase 2 port of the upstream test suite. Ranks
// are computed exactly like sklearn: for each point, the ranks of its k
// embedded nearest neighbours within the ordering of the original-space
// distances (self excluded), penalising rank > k. The input-space metric is
// "euclidean" (default) or "cosine" (sklearn's `metric` kwarg, used by
// test_umap_trustworthiness_on_sphere_iris); the embedding side is always
// euclidean like sklearn's.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace testutil {

inline double trustworthiness(const float* X, int64_t n, int64_t dim,
                              const float* emb, int64_t edim, int k,
                              const char* metric = "euclidean") {
    const bool cosine = std::strcmp(metric, "cosine") == 0;
    std::vector<double> drow(n);
    std::vector<int32_t> order(n);
    std::vector<int32_t> inverted(static_cast<std::size_t>(n) * n, 0);
    std::vector<double> norms(n, 0.0);
    if (cosine) {
        for (int64_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t d = 0; d < dim; ++d)
                s += static_cast<double>(X[i * dim + d]) *
                     static_cast<double>(X[i * dim + d]);
            norms[i] = std::sqrt(s);
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            if (cosine) {
                double dot = 0.0;
                for (int64_t d = 0; d < dim; ++d)
                    dot += static_cast<double>(X[i * dim + d]) *
                           static_cast<double>(X[j * dim + d]);
                const double denom = norms[i] * norms[j];
                s = denom > 0.0 ? 1.0 - dot / denom : 1.0;
            } else {
                for (int64_t d = 0; d < dim; ++d) {
                    const double diff = static_cast<double>(X[i * dim + d]) -
                                        static_cast<double>(X[j * dim + d]);
                    s += diff * diff;
                }
            }
            drow[j] = (i == j) ? std::numeric_limits<double>::infinity() : s;
        }
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return drow[a] < drow[b];
        });
        for (int64_t r = 0; r < n; ++r)
            inverted[i * n + order[r]] = static_cast<int32_t>(r + 1);
    }
    double t = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t d = 0; d < edim; ++d) {
                const double diff = static_cast<double>(emb[i * edim + d]) -
                                    static_cast<double>(emb[j * edim + d]);
                s += diff * diff;
            }
            drow[j] = (i == j) ? std::numeric_limits<double>::infinity() : s;
        }
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return drow[a] < drow[b];
        });
        for (int r = 0; r < k; ++r) {
            const int rank = inverted[i * n + order[r]] - k;
            if (rank > 0) t += rank;
        }
    }
    return 1.0 - t * (2.0 / (static_cast<double>(n) * k *
                             (2.0 * n - 3.0 * k - 1.0)));
}

}  // namespace testutil
