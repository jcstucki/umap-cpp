// Faithful C++ port of the supervised simplicial-set intersection machinery
// from reference/umap/umap/umap_.py + sparse.py (lmcinnes/umap, BSD 3
// clause). See include/umap/supervised.hpp for scope notes.
#include "umap/supervised.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace umap {

void fast_intersection(SparseMatrix& graph, const double* target,
                       double unknown_dist, double far_dist) {
    // for nz in range(rows.shape[0]): values[nz] *= np.exp(-dist) -- the
    // float32 value is promoted to float64 for the product and rounded back
    // on store. Iterating the canonical CSR row-major reproduces tocoo()
    // order (values are mutated independently, so order only matters for
    // documentation).
    const double unknown_factor = std::exp(-unknown_dist);
    const double far_factor = std::exp(-far_dist);
    for (int32_t i = 0; i < graph.n_rows; ++i) {
        for (int32_t idx = graph.indptr[i]; idx < graph.indptr[i + 1]; ++idx) {
            const int32_t j = graph.indices[idx];
            const double ti = target[i];
            const double tj = target[j];
            if (ti == -1.0 || tj == -1.0) {
                graph.data[idx] = static_cast<float>(
                    static_cast<double>(graph.data[idx]) * unknown_factor);
            } else if (ti != tj) {
                graph.data[idx] = static_cast<float>(
                    static_cast<double>(graph.data[idx]) * far_factor);
            }
        }
    }
}

void fast_metric_intersection(
    SparseMatrix& graph, const double* target,
    const std::function<double(double, double)>& metric, double scale) {
    for (int32_t i = 0; i < graph.n_rows; ++i) {
        for (int32_t idx = graph.indptr[i]; idx < graph.indptr[i + 1]; ++idx) {
            const int32_t j = graph.indices[idx];
            const double dist = metric(target[i], target[j]);
            graph.data[idx] = static_cast<float>(
                static_cast<double>(graph.data[idx]) *
                std::exp(-(scale * dist)));
        }
    }
}

std::vector<float> reprocess_row(const float* probabilities, int64_t n,
                                 double k, int n_iters) {
    // umap_.py::reprocess_row: bisection on the exponent so that
    // sum(p ** mid) == log2(k) (SMOOTH_K_TOLERANCE = 1e-5).
    const double target = std::log2(k);
    double lo = 0.0;
    double hi = std::numeric_limits<double>::infinity();
    double mid = 1.0;

    for (int it = 0; it < n_iters; ++it) {
        double psum = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            // pow(float32, float64) promotes to float64.
            psum += std::pow(static_cast<double>(probabilities[j]), mid);
        }
        if (std::fabs(psum - target) < 1e-5) break;
        if (psum < target) {
            hi = mid;
            mid = (lo + hi) / 2.0;
        } else {
            lo = mid;
            if (hi == std::numeric_limits<double>::infinity()) {
                mid *= 2.0;
            } else {
                mid = (lo + hi) / 2.0;
            }
        }
    }

    // return np.power(probabilities, mid): float64 result rounded to float32
    // on assignment into the float32 data slice.
    std::vector<float> out(static_cast<std::size_t>(n));
    for (int64_t j = 0; j < n; ++j) {
        out[static_cast<std::size_t>(j)] = static_cast<float>(
            std::pow(static_cast<double>(probabilities[j]), mid));
    }
    return out;
}

void reset_local_metrics(SparseMatrix& graph) {
    for (int32_t i = 0; i < graph.n_rows; ++i) {
        const int32_t start = graph.indptr[i];
        const int32_t end = graph.indptr[i + 1];
        if (end <= start) continue;
        const std::vector<float> row =
            reprocess_row(graph.data.data() + start, end - start);
        std::copy(row.begin(), row.end(), graph.data.begin() + start);
    }
}

namespace {

// sklearn normalize(X, norm="max") sparse branch: per-row norms =
// np.maximum(abs(min), max) where min/max include implicit zeros
// (min_max_axis semantics), then float32 in-place division of the stored
// values; rows whose norm is 0 are left untouched.
SparseMatrix normalize_max(const SparseMatrix& X) {
    SparseMatrix out = X;
    for (int32_t i = 0; i < out.n_rows; ++i) {
        const int32_t start = out.indptr[i];
        const int32_t end = out.indptr[i + 1];
        if (end <= start) continue;
        const bool has_implicit_zero = (end - start) < out.n_cols;
        float mn = has_implicit_zero ? 0.0f
                                     : std::numeric_limits<float>::infinity();
        float mx = has_implicit_zero ? 0.0f
                                     : -std::numeric_limits<float>::infinity();
        for (int32_t k = start; k < end; ++k) {
            mn = std::min(mn, out.data[k]);
            mx = std::max(mx, out.data[k]);
        }
        const float norm = std::max(std::fabs(mn), mx);
        if (norm == 0.0f) continue;
        for (int32_t k = start; k < end; ++k) out.data[k] /= norm;
    }
    return out;
}

}  // namespace

SparseMatrix reset_local_connectivity(const SparseMatrix& simplicial_set,
                                      bool reset_local_metric) {
    SparseMatrix ss = normalize_max(simplicial_set);
    if (reset_local_metric) reset_local_metrics(ss);
    const SparseMatrix transpose = ss.transpose();
    const SparseMatrix prod_matrix = ss.multiply(transpose);
    SparseMatrix result = ss.add(transpose).subtract(prod_matrix);
    result.eliminate_zeros();
    return result;
}

SparseMatrix discrete_metric_simplicial_set_intersection(
    const SparseMatrix& simplicial_set, const double* discrete_space,
    double unknown_dist, double far_dist,
    const std::function<double(double, double)>& metric, double metric_scale) {
    SparseMatrix graph = simplicial_set;
    if (metric) {
        fast_metric_intersection(graph, discrete_space, metric, metric_scale);
    } else {
        fast_intersection(graph, discrete_space, unknown_dist, far_dist);
    }
    graph.eliminate_zeros();
    return reset_local_connectivity(graph);
}

SparseMatrix general_simplicial_set_intersection(const SparseMatrix& ss1,
                                                 const SparseMatrix& ss2,
                                                 double weight,
                                                 bool right_complement) {
    // result = ss1.tocoo() (right_complement) or (ss1 + ss2).tocoo();
    // left/right stay the CSR operands.
    SparseMatrix result = right_complement ? ss1 : ss1.add(ss2);

    // sparse.py::general_sset_intersection. left_min / right_min come from
    // float32 .min() reductions, promoted to float64 by the /2.0 division.
    float d1_min = std::numeric_limits<float>::infinity();
    for (const float v : ss1.data) d1_min = std::min(d1_min, v);
    const double left_min =
        std::max(static_cast<double>(d1_min) / 2.0, 1.0e-8);

    float d2_min = std::numeric_limits<float>::infinity();
    if (right_complement) {
        for (const float v : ss2.data)
            d2_min = std::min(d2_min, static_cast<float>(1.0f - v));
    } else {
        for (const float v : ss2.data) d2_min = std::min(d2_min, v);
    }
    const double right_min =
        std::min(std::max(static_cast<double>(d2_min) / 2.0, 1.0e-8), 1e-4);

    for (int32_t i = 0; i < result.n_rows; ++i) {
        for (int32_t idx = result.indptr[i]; idx < result.indptr[i + 1];
             ++idx) {
            const int32_t j = result.indices[idx];

            double left_val = left_min;
            for (int32_t k = ss1.indptr[i]; k < ss1.indptr[i + 1]; ++k) {
                if (ss1.indices[k] == j)
                    left_val = static_cast<double>(ss1.data[k]);
            }
            double right_val = right_min;
            for (int32_t k = ss2.indptr[i]; k < ss2.indptr[i + 1]; ++k) {
                if (ss2.indices[k] == j) {
                    right_val = right_complement
                                    ? 1.0 - static_cast<double>(ss2.data[k])
                                    : static_cast<double>(ss2.data[k]);
                }
            }

            // NOTE: when neither side exceeds its minimum the (ss1+ss2) sum
            // value is left in place, exactly like upstream.
            if (left_val > left_min || right_val > right_min) {
                if (weight < 0.5) {
                    result.data[idx] = static_cast<float>(
                        left_val * std::pow(right_val,
                                            weight / (1.0 - weight)));
                } else {
                    result.data[idx] = static_cast<float>(
                        std::pow(left_val, (1.0 - weight) / weight) *
                        right_val);
                }
            }
        }
    }
    return result;
}

// --- composite additions ------------------------------------------------------
SparseMatrix general_simplicial_set_union(const SparseMatrix& ss1,
                                          const SparseMatrix& ss2) {
    // result = (ss1 + ss2).tocoo(); left/right stay the CSR operands.
    SparseMatrix result = ss1.add(ss2);

    // sparse.py::general_sset_union. left_min / right_min come from float32
    // .min() reductions, promoted to float64 by the /2.0 division; the union
    // has no 1e-4 clamp on right_min (unlike the intersection).
    float d1_min = std::numeric_limits<float>::infinity();
    for (const float v : ss1.data) d1_min = std::min(d1_min, v);
    const double left_min =
        std::max(static_cast<double>(d1_min) / 2.0, 1.0e-8);

    float d2_min = std::numeric_limits<float>::infinity();
    for (const float v : ss2.data) d2_min = std::min(d2_min, v);
    const double right_min =
        std::max(static_cast<double>(d2_min) / 2.0, 1.0e-8);

    for (int32_t i = 0; i < result.n_rows; ++i) {
        for (int32_t idx = result.indptr[i]; idx < result.indptr[i + 1];
             ++idx) {
            const int32_t j = result.indices[idx];

            double left_val = left_min;
            for (int32_t k = ss1.indptr[i]; k < ss1.indptr[i + 1]; ++k) {
                if (ss1.indices[k] == j)
                    left_val = static_cast<double>(ss1.data[k]);
            }
            double right_val = right_min;
            for (int32_t k = ss2.indptr[i]; k < ss2.indptr[i + 1]; ++k) {
                if (ss2.indices[k] == j)
                    right_val = static_cast<double>(ss2.data[k]);
            }

            result.data[idx] = static_cast<float>(
                left_val + right_val - left_val * right_val);
        }
    }
    return result;
}
// ------------------------------------------------------------------------------

}  // namespace umap
