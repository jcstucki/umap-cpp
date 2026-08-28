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
#include <string>

#include <Eigen/Dense>

#include "umap/distances.hpp"
#include "umap/find_ab_params.hpp"
#include "umap/fuzzy_simplicial_set.hpp"
#include "umap/layouts.hpp"
#include "umap/spectral.hpp"
// --- supervised-sparse additions --------------------------------------------
#include "umap/sparse_distances.hpp"
#include "umap/sparse_nn_descent.hpp"
#include "umap/supervised.hpp"
// ----------------------------------------------------------------------------

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
                                              bool x_is_y,
                                              const MetricKwds& kwds) {
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

    // Other named metrics: numba-f4-faithful kernels where available
    // (manhattan / minkowski / seuclidean), otherwise umap's distance kernels
    // via the double-precision registry in distances.hpp (documented: matches
    // the numba float32 kernels only to float32 rounding of the double
    // result). metric_kwds are bound positionally like upstream.
    DistanceF32Fn fn;
    try {
        fn = make_metric_f32(metric, kwds);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(
            "pairwise_distances_sklearn: unsupported metric '" + metric + "'");
    }
    for (int64_t i = 0; i < nx; ++i) {
        const float* xi = X + i * dim;
        const int64_t j0 = x_is_y ? i : 0;
        for (int64_t j = j0; j < ny; ++j) {
            const float d =
                (x_is_y && i == j)
                    ? fn(xi, xi, static_cast<std::size_t>(dim))
                    : fn(xi, Y + j * dim, static_cast<std::size_t>(dim));
            out[static_cast<std::size_t>(i) * ny + j] = d;
            if (x_is_y && i != j) out[static_cast<std::size_t>(j) * ny + i] = d;
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

// --- densmap additions ------------------------------------------------------
// np.mean / np.std on float32 arrays (numpy _methods._mean/_std: plain
// pairwise summation over the whole array -- unlike np.add.reduceat above --
// then all-float32 division/sqrt; verified against the reference numpy).
float numpy_mean_f32(const float* a, int64_t n) {
    return numpy_pairwise_sum_f32(a, n) / static_cast<float>(n);
}

float numpy_std_f32(const float* a, int64_t n) {
    const float mean = numpy_mean_f32(a, n);
    std::vector<float> x(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const float d = a[i] - mean;
        x[static_cast<std::size_t>(i)] = d * d;
    }
    return std::sqrt(numpy_pairwise_sum_f32(x.data(), n) /
                     static_cast<float>(n));
}

// umap_.py::_densmap_original_densities (square=true: D = d_val * d_val) and
// _densmap_embedding_densities (square=false: weighted = mu * d_val), fused:
// both walk the graph edges in COO order (row-major over our canonical CSR),
// look the edge distance up in the dists CSR row of the head vertex (d_val
// stays 0.0 when absent) and accumulate float32 r / mu_sum. numba typing:
// d_val is float64 (initialised 0.0), mu float32; `r[j] += mu * D` promotes
// to float64 and rounds back to float32 on store; mu_sum stays float32.
// The caller applies the Python-level `np.log(epsilon + (r / mu_sum))`
// (float32 under NEP 50 -- unlike the float64 log inside the numba epoch
// init) via densmap_log_radii below.
void densmap_densities(const SparseMatrix& graph, const SparseMatrix& dists,
                       bool square, std::vector<float>& r,
                       std::vector<float>& mu_sum) {
    const std::size_t nv = static_cast<std::size_t>(graph.n_cols);
    r.assign(nv, 0.0f);
    mu_sum.assign(nv, 0.0f);
    for (int32_t row = 0; row < graph.n_rows; ++row) {
        for (int32_t idx = graph.indptr[row]; idx < graph.indptr[row + 1];
             ++idx) {
            const int32_t j = row;
            const int32_t k = graph.indices[idx];
            double d_val = 0.0;
            for (int32_t p = dists.indptr[j]; p < dists.indptr[j + 1]; ++p) {
                if (dists.indices[p] == k) {
                    d_val = static_cast<double>(dists.data[p]);
                    break;
                }
            }
            const double contrib =
                square ? d_val * d_val : d_val;  // D vs plain distance
            const float mu = graph.data[idx];
            r[static_cast<std::size_t>(j)] = static_cast<float>(
                r[static_cast<std::size_t>(j)] +
                static_cast<double>(mu) * contrib);
            r[static_cast<std::size_t>(k)] = static_cast<float>(
                r[static_cast<std::size_t>(k)] +
                static_cast<double>(mu) * contrib);
            mu_sum[static_cast<std::size_t>(j)] += mu;
            mu_sum[static_cast<std::size_t>(k)] += mu;
        }
    }
}

// ro = np.log(epsilon + (ro / mu_sum)) at the Python level: NEP 50 keeps the
// whole expression float32 (the weak Python-float epsilon casts to float32).
// np.log on float32 arrays matches the float32 rounding of the float64 log
// (verified empirically over 2e5 samples in the reference environment).
void densmap_log_radii(std::vector<float>& r,
                       const std::vector<float>& mu_sum) {
    for (std::size_t i = 0; i < r.size(); ++i) {
        const float arg = 1e-8f + (r[i] / mu_sum[i]);
        r[i] = static_cast<float>(std::log(static_cast<double>(arg)));
    }
}
// ----------------------------------------------------------------------------

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

// --- feature-bundle additions -----------------------------------------------

double default_disconnection_distance(const std::string& metric) {
    // umap_.py::DISCONNECTION_DISTANCES; .get(metric, np.inf).
    if (metric == "correlation" || metric == "cosine") return 2.0;
    if (metric == "hellinger" || metric == "jaccard" ||
        metric == "bit_jaccard" || metric == "dice")
        return 1.0;
    return std::numeric_limits<double>::infinity();
}

void unique_rows(const float* X, int64_t n, int64_t dim,
                 std::vector<int64_t>& index, std::vector<int64_t>& inverse) {
    // np.unique(X, axis=0, ...): rows viewed as a structured dtype are
    // compared field by field (i.e. lexicographic on the float values, with
    // -0.0 == 0.0), argsorted with mergesort (stable, because return_index is
    // requested), then mask/cumsum produce index and inverse.
    std::vector<int64_t> perm(static_cast<std::size_t>(n));
    std::iota(perm.begin(), perm.end(), 0);
    const auto row_less = [X, dim](int64_t a, int64_t b) {
        const float* ra = X + a * dim;
        const float* rb = X + b * dim;
        for (int64_t c = 0; c < dim; ++c) {
            if (ra[c] < rb[c]) return true;
            if (rb[c] < ra[c]) return false;
        }
        return false;
    };
    std::stable_sort(perm.begin(), perm.end(), row_less);
    const auto row_eq = [X, dim](int64_t a, int64_t b) {
        const float* ra = X + a * dim;
        const float* rb = X + b * dim;
        for (int64_t c = 0; c < dim; ++c) {
            if (ra[c] != rb[c]) return false;
        }
        return true;
    };
    index.clear();
    inverse.assign(static_cast<std::size_t>(n), 0);
    int64_t group = -1;
    for (int64_t j = 0; j < n; ++j) {
        if (j == 0 || !row_eq(perm[static_cast<std::size_t>(j) - 1],
                              perm[static_cast<std::size_t>(j)])) {
            index.push_back(perm[static_cast<std::size_t>(j)]);
            ++group;
        }
        inverse[static_cast<std::size_t>(perm[static_cast<std::size_t>(j)])] =
            group;
    }
}

// --- supervised-sparse additions --------------------------------------------

void csr_unique_rows(const int32_t* indptr, const int32_t* indices,
                     const float* data, int64_t n,
                     std::vector<int64_t>& index,
                     std::vector<int64_t>& inverse) {
    // utils.py::csr_unique: each lil row becomes the tuple
    // (column indices..., values...); np.unique on the object array sorts the
    // tuples (Python comparison: elementwise numeric, then by length) with a
    // stable sort because return_index is requested.
    std::vector<std::vector<double>> keys(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        std::vector<double>& key = keys[static_cast<std::size_t>(i)];
        const int32_t start = indptr[i];
        const int32_t end = indptr[i + 1];
        key.reserve(static_cast<std::size_t>(end - start) * 2);
        for (int32_t k = start; k < end; ++k)
            key.push_back(static_cast<double>(indices[k]));
        for (int32_t k = start; k < end; ++k)
            key.push_back(static_cast<double>(data[k]));
    }
    std::vector<int64_t> perm(static_cast<std::size_t>(n));
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&keys](int64_t a, int64_t b) {
        return keys[static_cast<std::size_t>(a)] <
               keys[static_cast<std::size_t>(b)];
    });
    index.clear();
    inverse.assign(static_cast<std::size_t>(n), 0);
    int64_t group = -1;
    for (int64_t j = 0; j < n; ++j) {
        if (j == 0 || keys[static_cast<std::size_t>(
                          perm[static_cast<std::size_t>(j) - 1])] !=
                          keys[static_cast<std::size_t>(
                              perm[static_cast<std::size_t>(j)])]) {
            index.push_back(perm[static_cast<std::size_t>(j)]);
            ++group;
        }
        inverse[static_cast<std::size_t>(perm[static_cast<std::size_t>(j)])] =
            group;
    }
}

std::vector<float> sparse_pairwise_distances_sklearn(
    const int32_t* x_indptr, const int32_t* x_indices, const float* x_data,
    int64_t nx, const int32_t* y_indptr, const int32_t* y_indices,
    const float* y_data, int64_t ny, const std::string& metric, bool x_is_y) {
    std::vector<float> out(static_cast<std::size_t>(nx) * ny);

    if (metric == "euclidean" || metric == "l2") {
        // sklearn _euclidean_distances_upcast on float32 CSR data: chunks are
        // upcast to float64 (exact) -- row norms accumulate sequentially over
        // the stored order, dot products merge the two rows by ascending
        // column (scipy SMMP accumulation order), and each element is rounded
        // to float32 as ((-2*dot) + XX_i) + YY_j.
        const auto row_norms = [](const int32_t* indptr, const float* data,
                                  int64_t n) {
            std::vector<double> norms(static_cast<std::size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (int32_t k = indptr[i]; k < indptr[i + 1]; ++k) {
                    const double v = static_cast<double>(data[k]);
                    s += v * v;
                }
                norms[static_cast<std::size_t>(i)] = s;
            }
            return norms;
        };
        const std::vector<double> XX = row_norms(x_indptr, x_data, nx);
        const std::vector<double> YY =
            x_is_y ? XX : row_norms(y_indptr, y_data, ny);
        for (int64_t i = 0; i < nx; ++i) {
            for (int64_t j = 0; j < ny; ++j) {
                double s = 0.0;
                int32_t ai = x_indptr[i];
                int32_t aj = y_indptr[j];
                while (ai < x_indptr[i + 1] && aj < y_indptr[j + 1]) {
                    const int32_t ci = x_indices[ai];
                    const int32_t cj = y_indices[aj];
                    if (ci == cj) {
                        s += static_cast<double>(x_data[ai]) *
                             static_cast<double>(y_data[aj]);
                        ++ai;
                        ++aj;
                    } else if (ci < cj) {
                        ++ai;
                    } else {
                        ++aj;
                    }
                }
                float d = static_cast<float>(
                    ((-2.0 * s) + XX[static_cast<std::size_t>(i)]) +
                    YY[static_cast<std::size_t>(j)]);
                d = std::max(d, 0.0f);  // np.maximum(distances, 0)
                if (x_is_y && i == j) d = 0.0f;  // fill_diagonal
                out[static_cast<std::size_t>(i) * ny + j] = std::sqrt(d);
            }
        }
        return out;
    }

    if (metric == "cosine") {
        // cosine_distances = 1 - cosine_similarity. normalize(X) replicates
        // sklearn's inplace_csr_row_normalize_l2 (float32 squares accumulated
        // into a float64 sum, float64 division rounded to float32); the
        // normalized sparse matmul replicates scipy's float32 kernel, whose
        // multiply-accumulate is contracted to fmaf in the shipped wheels.
        const auto normalize_l2 = [](const int32_t* indptr, const float* data,
                                     int64_t n, int64_t nnz) {
            std::vector<float> nd(data, data + nnz);
            for (int64_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (int32_t k = indptr[i]; k < indptr[i + 1]; ++k) {
                    const float sq = data[k] * data[k];
                    s += static_cast<double>(sq);
                }
                if (s == 0.0) continue;
                const double root = std::sqrt(s);
                for (int32_t k = indptr[i]; k < indptr[i + 1]; ++k)
                    nd[static_cast<std::size_t>(k)] = static_cast<float>(
                        static_cast<double>(data[k]) / root);
            }
            return nd;
        };
        const std::vector<float> xn =
            normalize_l2(x_indptr, x_data, nx, x_indptr[nx]);
        const std::vector<float> yn =
            x_is_y ? xn : normalize_l2(y_indptr, y_data, ny, y_indptr[ny]);
        for (int64_t i = 0; i < nx; ++i) {
            for (int64_t j = 0; j < ny; ++j) {
                float s = 0.0f;
                int32_t ai = x_indptr[i];
                int32_t aj = y_indptr[j];
                while (ai < x_indptr[i + 1] && aj < y_indptr[j + 1]) {
                    const int32_t ci = x_indices[ai];
                    const int32_t cj = y_indices[aj];
                    if (ci == cj) {
                        s = std::fmaf(xn[static_cast<std::size_t>(ai)],
                                      yn[static_cast<std::size_t>(aj)], s);
                        ++ai;
                        ++aj;
                    } else if (ci < cj) {
                        ++ai;
                    } else {
                        ++aj;
                    }
                }
                float d = (-s) + 1.0f;               // S *= -1; S += 1
                d = std::min(std::max(d, 0.0f), 2.0f);  // np.clip(S, 0, 2)
                if (x_is_y && i == j) d = 0.0f;
                out[static_cast<std::size_t>(i) * ny + j] = d;
            }
        }
        return out;
    }

    throw std::invalid_argument(
        "sparse_pairwise_distances_sklearn: metric '" + metric +
        "' is not supported for sparse data in the C++ port (upstream "
        "densifies via pairwise_special_metric)");
}

// ----------------------------------------------------------------------------

namespace {

// numpy's DOUBLE_pairwise_sum / np.add.reduce seeding, mirroring the float32
// helpers above (np.mean(float64_array) = add.reduce(a) / n with a[0] seeding
// the accumulator and the remainder pairwise-summed).
double numpy_pairwise_sum_f64(const double* a, int64_t n) {
    if (n < 8) {
        double res = -0.0;
        for (int64_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        int64_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (int k = 0; k < 8; ++k) r[k] += a[i + k];
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) +
                     ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return numpy_pairwise_sum_f64(a, n2) + numpy_pairwise_sum_f64(a + n2, n - n2);
}

double numpy_mean_f64(const double* a, int64_t n) {
    // np.mean = plain pairwise sum over the whole array / n (the seed-first
    // form only applies to np.add.reduceat; verified vs the reference numpy).
    if (n <= 0) return 0.0;
    return numpy_pairwise_sum_f64(a, n) / static_cast<double>(n);
}

// warn() / verbose prints go to stderr (no warnings module in C++).
void warn_msg(const std::string& msg) {
    std::fprintf(stderr, "%s\n", msg.c_str());
}

// umap_.py::raise_disconnected_warning (threshold = 0.1).
void raise_disconnected_warning(int64_t edges_removed,
                                int64_t vertices_disconnected,
                                double disconnection_distance,
                                int64_t total_rows, bool verbose) {
    const double threshold = 0.1;
    char dd_buf[64];
    std::snprintf(dd_buf, sizeof(dd_buf), "%g", disconnection_distance);
    if (verbose && vertices_disconnected == 0 && edges_removed > 0) {
        std::fprintf(stderr,
                     "Disconnection_distance = %s has removed %lld edges.  "
                     "This is not a problem as no vertices were "
                     "disconnected.\n",
                     dd_buf, static_cast<long long>(edges_removed));
    } else if (vertices_disconnected > 0 &&
               static_cast<double>(vertices_disconnected) <=
                   threshold * static_cast<double>(total_rows)) {
        std::fprintf(
            stderr,
            "A few of your vertices were disconnected from the manifold.  "
            "This shouldn't cause problems.\n"
            "Disconnection_distance = %s has removed %lld edges.\n"
            "It has only fully disconnected %lld vertices.\n"
            "Use umap.utils.disconnected_vertices() to identify them.\n",
            dd_buf, static_cast<long long>(edges_removed),
            static_cast<long long>(vertices_disconnected));
    } else if (static_cast<double>(vertices_disconnected) >
               threshold * static_cast<double>(total_rows)) {
        std::fprintf(
            stderr,
            "A large number of your vertices were disconnected from the "
            "manifold.\n"
            "Disconnection_distance = %s has removed %lld edges.\n"
            "It has fully disconnected %lld vertices.\n"
            "You might consider using find_disconnected_points() to find and "
            "remove these points from your data.\n"
            "Use umap.utils.disconnected_vertices() to identify them.\n",
            dd_buf, static_cast<long long>(edges_removed),
            static_cast<long long>(vertices_disconnected));
    }
}

}  // namespace

std::vector<float> pca_fit_transform(const float* X, int64_t n, int64_t dim,
                                     int n_components) {
    if (n_components > std::min<int64_t>(n, dim)) {
        throw std::invalid_argument(
            "n_components must be between 0 and min(n_samples, n_features) "
            "for PCA initialisation");
    }
    Eigen::MatrixXd M(n, dim);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < dim; ++j)
            M(i, j) = static_cast<double>(X[i * dim + j]);
    // self.mean_ = np.mean(X, axis=0); X -= mean.
    const Eigen::RowVectorXd mean = M.colwise().mean();
    M.rowwise() -= mean;
    // U, S, Vt = linalg.svd(X, full_matrices=False)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::MatrixXd& U = svd.matrixU();
    const Eigen::MatrixXd& V = svd.matrixV();
    const Eigen::VectorXd& S = svd.singularValues();
    std::vector<float> out(static_cast<std::size_t>(n) * n_components);
    for (int j = 0; j < n_components; ++j) {
        // svd_flip(U, Vt, u_based_decision=False): sign of the largest-|.|
        // entry of row j of Vt (np.argmax takes the first maximum on ties).
        int64_t imax = 0;
        double amax = -1.0;
        for (int64_t c = 0; c < dim; ++c) {
            const double av = std::abs(V(c, j));
            if (av > amax) {
                amax = av;
                imax = c;
            }
        }
        const double sign = V(imax, j) < 0.0 ? -1.0 : 1.0;
        // fit_transform returns U[:, :k] * S[:k].
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<std::size_t>(i) * n_components + j] =
                static_cast<float>(U(i, j) * S(j) * sign);
        }
    }
    return out;
}

// ----------------------------------------------------------------------------

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
    if (!p.init_array.has_value() && p.init != "pca" && p.init != "spectral" &&
        p.init != "random" && p.init != "tswspectral")
        throw std::invalid_argument(
            "string init values must be one of: \"pca\", \"tswspectral\","
            " \"spectral\" or \"random\"");
    // if isinstance(self.init, np.ndarray) and init.shape[1] != n_components
    if (p.init_array.has_value() && p.init_array_cols >= 0 &&
        p.init_array_cols != p.n_components)
        throw std::invalid_argument(
            "init ndarray must match n_components value");
    if (p.negative_sample_rate < 0)
        throw std::invalid_argument("negative sample rate must be positive");
    if (p.learning_rate < 0.0)
        throw std::invalid_argument("learning_rate must be positive");
    if (p.n_neighbors < 2)
        throw std::invalid_argument("n_neighbors must be greater than 1");
    // --- supervised-sparse additions ----------------------------------------
    if (p.target_n_neighbors < 2 && p.target_n_neighbors != -1)
        throw std::invalid_argument(
            "target_n_neighbors must be greater than 1");
    // ------------------------------------------------------------------------
    if (p.n_components < 1)
        throw std::invalid_argument("n_components must be greater than 0");
    if (p.n_epochs.has_value() && *p.n_epochs < 0)
        throw std::invalid_argument(
            "n_epochs must be a nonnegative integer "
            "or a list of nonnegative integers");
    // --- feature-bundle additions -------------------------------------------
    if (p.n_epochs_list.has_value()) {
        if (p.n_epochs.has_value())
            throw std::invalid_argument(
                "set only one of n_epochs and n_epochs_list");
        // Upstream: an empty list has a float dtype and negative entries fail
        // the >= 0 check; both hit the same ValueError.
        if (p.n_epochs_list->empty() ||
            std::any_of(p.n_epochs_list->begin(), p.n_epochs_list->end(),
                        [](int e) { return e < 0; }))
            throw std::invalid_argument(
                "n_epochs must be a nonnegative integer "
                "or a list of nonnegative integers");
    }
    if (p.metric == "precomputed") {
        if (p.unique)
            throw std::invalid_argument(
                "unique is poorly defined on a precomputed metric");
        warn_msg(
            "using precomputed metric; inverse_transform will be unavailable");
    }
    // --- densmap additions --------------------------------------------------
    // set output distance metric (upstream's elif chain; callable output
    // metrics are out of C++ scope). Note the reference's "cannnot" typo.
    if (p.output_metric == "precomputed") {
        throw std::invalid_argument("output_metric cannnot be 'precomputed'");
    } else if (named_distances_with_gradients().count(p.output_metric) > 0) {
        // ok -- resolved at embedding time via make_output_metric
    } else if (named_distances().count(p.output_metric) > 0) {
        throw std::invalid_argument(
            "gradient function is not yet implemented for " + p.output_metric +
            ".");
    } else {
        throw std::invalid_argument(
            "output_metric is neither callable nor a recognised string");
    }
    if (p.dens_lambda < 0.0)
        throw std::invalid_argument("dens_lambda cannot be negative");
    if (p.dens_frac < 0.0 || p.dens_frac > 1.0)
        throw std::invalid_argument("dens_frac must be between 0.0 and 1.0");
    if (p.dens_var_shift < 0.0)
        throw std::invalid_argument("dens_var_shift cannot be negative");
    if (p.densmap && p.output_metric != "euclidean" && p.output_metric != "l2")
        throw std::invalid_argument(
            "Non-Euclidean output metric not supported for densMAP.");
    // ------------------------------------------------------------------------
    if (p.precomputed_knn.has_value()) {
        const PrecomputedKnn& k = *p.precomputed_knn;
        if (p.unique)
            throw std::invalid_argument(
                "unique is not currently available for precomputed_knn.");
        const std::size_t expected = static_cast<std::size_t>(k.n_samples) *
                                     static_cast<std::size_t>(k.n_neighbors);
        if (k.indices.size() != expected || k.dists.size() != expected)
            throw std::invalid_argument(
                "precomputed_knn[0] and precomputed_knn[1]"
                " must be numpy arrays of the same size.");
        // #848: warn but proceed if no search index is present.
        if (!k.search_index)
            warn_msg(
                "precomputed_knn[2] (knn_search_index) "
                "is not an NNDescent object: transforming new data with "
                "transform will be unavailable.");
    }
    // ------------------------------------------------------------------------
}

std::vector<float> UMAP::make_fit_init(const float* data, int64_t n_index,
                                       const SparseMatrix& filtered_graph,
                                       nnd::NumpyRandomState& rs) const {
    const int dim = params_.n_components;
    const int64_t n = filtered_graph.n_rows;
    const std::size_t total = static_cast<std::size_t>(n) * dim;

    // init as a user ndarray (the final `else` of simplicial_set_embedding's
    // init dispatch): no noisy_scale_coords, but duplicate rows trigger the
    // KDTree jitter branch.
    if (params_.init_array.has_value()) {
        const std::vector<float>& arr = *params_.init_array;
        if (arr.size() != total) {
            throw std::invalid_argument(
                "UMAP: init_array must have shape "
                "(n_samples[, unique], n_components)");
        }
        // if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
        std::vector<int64_t> uidx, uinv;
        unique_rows(arr.data(), n, dim, uidx, uinv);
        if (static_cast<int64_t>(uidx.size()) == n) return arr;
        // tree = KDTree(init_data); dist, ind = tree.query(init_data, k=2)
        // nndist = np.mean(dist[:, 1])
        // The second column of an exact self-query is the distance to the
        // nearest *other* point (the self-match occupies column 0; exact
        // duplicates make both 0). sklearn's KDTree computes in float64.
        std::vector<double> d2(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            double best = std::numeric_limits<double>::infinity();
            const float* ri = arr.data() + static_cast<std::size_t>(i) * dim;
            for (int64_t j = 0; j < n; ++j) {
                if (j == i) continue;
                const float* rj =
                    arr.data() + static_cast<std::size_t>(j) * dim;
                double acc = 0.0;
                for (int c = 0; c < dim; ++c) {
                    const double diff = static_cast<double>(ri[c]) -
                                        static_cast<double>(rj[c]);
                    acc += diff * diff;
                }
                best = std::min(best, acc);
            }
            d2[static_cast<std::size_t>(i)] = std::sqrt(best);
        }
        const double nndist = numpy_mean_f64(d2.data(), n);
        // embedding = init_data + random_state.normal(
        //     scale=0.001 * nndist, size=init_data.shape).astype(np.float32)
        std::vector<float> out(total);
        for (std::size_t i = 0; i < total; ++i) {
            out[i] = arr[i] +
                     static_cast<float>(rs.normal(0.0, 0.001 * nndist));
        }
        return out;
    }

    if (params_.init == "random") {
        // random_state.uniform(low=-10, high=10, size=(n, dim)).astype(f32)
        std::vector<float> out(total);
        for (auto& v : out)
            v = static_cast<float>(rs.uniform(-10.0, 10.0));
        return out;
    }
    if (params_.init == "pca") {
        // pca = PCA(n_components=n_components, random_state=random_state)
        // embedding = pca.fit_transform(data).astype(np.float32)
        // (the full LAPACK solver never touches random_state)
        // embedding = noisy_scale_coords(embedding, random_state,
        //                                max_coord=10, noise=0.0001)
        std::vector<float> coords = pca_fit_transform(data, n_index, dim_, dim);
        float max_abs = 0.0f;
        for (float v : coords) max_abs = std::max(max_abs, std::abs(v));
        const double expansion = 10.0 / static_cast<double>(max_abs);
        std::vector<float> out(total);
        for (std::size_t i = 0; i < total; ++i) {
            out[i] = static_cast<float>(static_cast<double>(coords[i]) *
                                        expansion) +
                     static_cast<float>(rs.normal(0.0, 0.0001));
        }
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
            n_index > 0 ? data : nullptr, dim_, filtered_graph, n_comp, labels,
            dim, tau_state);
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

UMAP& UMAP::fit(const float* X, int64_t n_samples, int64_t n_features,
                const float* y) {
    sparse_data_ = false;
    raw_indptr_.clear();
    raw_indices_.clear();
    raw_values_.clear();
    sparse_knn_index_.reset();
    return fit_impl(X, n_samples, n_features, y);
}

// --- supervised-sparse additions --------------------------------------------
UMAP& UMAP::fit(const int32_t* indptr, const int32_t* indices,
                const float* data, int64_t n_samples, int64_t n_features,
                const float* y) {
    // Branches upstream supports for sparse input but this port does not:
    if (params_.metric == "precomputed")
        throw std::invalid_argument(
            "UMAP: metric='precomputed' with sparse input (the upstream "
            "sparse precomputed branch) is not ported");
    if (params_.precomputed_knn.has_value())
        throw std::invalid_argument(
            "UMAP: precomputed_knn with sparse input is not ported");
    if (!params_.init_array.has_value() && params_.init == "pca")
        throw std::invalid_argument(
            "UMAP: init='pca' with sparse input (upstream TruncatedSVD) is "
            "not ported");
    sparse_data_ = true;
    knn_index_.reset();
    raw_indptr_.assign(indptr, indptr + n_samples + 1);
    raw_indices_.assign(indices, indices + indptr[n_samples]);
    raw_values_.assign(data, data + indptr[n_samples]);
    // if self._sparse_data and not X.has_sorted_indices: X.sort_indices()
    for (int64_t i = 0; i < n_samples; ++i) {
        const int32_t start = raw_indptr_[static_cast<std::size_t>(i)];
        const int32_t end = raw_indptr_[static_cast<std::size_t>(i) + 1];
        bool sorted = true;
        for (int32_t k = start + 1; k < end; ++k) {
            if (raw_indices_[static_cast<std::size_t>(k) - 1] >
                raw_indices_[static_cast<std::size_t>(k)]) {
                sorted = false;
                break;
            }
        }
        if (sorted) continue;
        std::vector<std::pair<int32_t, float>> row;
        row.reserve(static_cast<std::size_t>(end - start));
        for (int32_t k = start; k < end; ++k)
            row.emplace_back(raw_indices_[static_cast<std::size_t>(k)],
                             raw_values_[static_cast<std::size_t>(k)]);
        std::sort(row.begin(), row.end(),
                  [](const std::pair<int32_t, float>& a,
                     const std::pair<int32_t, float>& b) {
                      return a.first < b.first;
                  });
        for (int32_t k = start; k < end; ++k) {
            raw_indices_[static_cast<std::size_t>(k)] =
                row[static_cast<std::size_t>(k - start)].first;
            raw_values_[static_cast<std::size_t>(k)] =
                row[static_cast<std::size_t>(k - start)].second;
        }
    }
    return fit_impl(nullptr, n_samples, n_features, y);
}
// ----------------------------------------------------------------------------

UMAP& UMAP::fit_impl(const float* X, int64_t n_samples, int64_t n_features,
                     const float* y) {
    validate_parameters();
    n_ = n_samples;
    dim_ = n_features;
    if (sparse_data_) {
        raw_data_.clear();
    } else {
        raw_data_.assign(X,
                         X + static_cast<std::size_t>(n_samples) * n_features);
    }
    embedding_list_.clear();
    unique_inverse_.clear();
    edges_removed_ = 0;
    vertices_disconnected_ = 0;
    // --- densmap additions --------------------------------------------------
    graph_dists_ = SparseMatrix();
    rad_orig_.clear();
    rad_emb_.clear();
    // return_dists for fuzzy_simplicial_set (upstream passes
    // self.densmap or self.output_dens in all fit branches).
    const bool want_graph_dists = params_.densmap || params_.output_dens;
    // _densmap_kwds["n_neighbors"] = self.n_neighbors (overridden to
    // _n_neighbors below only when truncation happens AND densmap is on).
    int dens_n_neighbors = params_.n_neighbors;
    // ------------------------------------------------------------------------

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

    // self._disconnection_distance (resolved in _validate_parameters).
    disconnection_distance_ =
        params_.disconnection_distance.has_value()
            ? *params_.disconnection_distance
            : default_disconnection_distance(params_.metric);

    // metric="precomputed": X must be a square distance matrix (sklearn's
    // check_pairwise_arrays raises for non-square input on the small-data
    // branch; we enforce it for all branches).
    if (params_.metric == "precomputed" && n_features != n_samples) {
        throw std::invalid_argument(
            "Precomputed metric requires shape (n_queries, n_indexed). Got (" +
            std::to_string(n_samples) + ", " + std::to_string(n_features) +
            ") for " + std::to_string(n_samples) + " indexed.");
    }

    // precomputed_knn validation chain (the raw-data-dependent part of
    // _validate_parameters; the shape/type checks live in
    // validate_parameters()). Mirrors the upstream elif chain exactly,
    // including the quirk that n < 4096 only sets
    // force_approximation_algorithm and *skips* the column pruning.
    force_approximation_algorithm_ = params_.force_approximation_algorithm;
    const PrecomputedKnn* pknn = params_.precomputed_knn.has_value()
                                     ? &*params_.precomputed_knn
                                     : nullptr;
    bool prune_pknn = false;
    if (pknn != nullptr) {
        if (pknn->n_neighbors < params_.n_neighbors) {
            warn_msg(
                "precomputed_knn has a lower number of neighbors than "
                "n_neighbors parameter. precomputed_knn will be ignored"
                " and the k-nn will be computed normally.");
            pknn = nullptr;
        } else if (pknn->n_samples != n_samples) {
            warn_msg(
                "precomputed_knn has a different number of samples than the"
                " data you are fitting. precomputed_knn will be ignored and"
                "the k-nn will be computed normally.");
            pknn = nullptr;
        } else if (pknn->n_samples < 4096 &&
                   !force_approximation_algorithm_) {
            // force_approximation_algorithm is irrelevant for pre-computed
            // knn; always set it to True to keep downstream code paths
            // working.
            force_approximation_algorithm_ = true;
        } else if (pknn->n_neighbors > params_.n_neighbors) {
            // if k for precomputed_knn larger than n_neighbors simply prune.
            prune_pknn = true;
        }
    }

    // unique=True: np.unique(X, axis=0) row dedup (csr_unique for sparse
    // input); otherwise the identity.
    std::vector<int64_t> index;
    std::vector<int64_t> inverse;
    if (params_.unique) {
        if (sparse_data_) {
            csr_unique_rows(raw_indptr_.data(), raw_indices_.data(),
                            raw_values_.data(), n_samples, index, inverse);
        } else {
            unique_rows(raw_data_.data(), n_samples, n_features, index,
                        inverse);
        }
        unique_inverse_ = inverse;
    } else {
        index.resize(static_cast<std::size_t>(n_samples));
        std::iota(index.begin(), index.end(), 0);
        inverse = index;
    }
    const int64_t n_index = static_cast<int64_t>(index.size());
    // X_indexed = X[index] (a fresh copy upstream too -- fancy indexing).
    std::vector<float> X_indexed_store;
    const float* X_indexed = raw_data_.data();
    if (params_.unique && !sparse_data_) {
        X_indexed_store.resize(static_cast<std::size_t>(n_index) * n_features);
        for (int64_t i = 0; i < n_index; ++i) {
            std::memcpy(X_indexed_store.data() + i * n_features,
                        raw_data_.data() + index[static_cast<std::size_t>(i)] *
                                               n_features,
                        static_cast<std::size_t>(n_features) * sizeof(float));
        }
        X_indexed = X_indexed_store.data();
    }
    // --- supervised-sparse additions ----------------------------------------
    // Sparse X[index]: row-gathered CSR (identity reuses the raw arrays).
    std::vector<int32_t> sp_indptr_store, sp_indices_store;
    std::vector<float> sp_values_store;
    const int32_t* sp_indptr = raw_indptr_.data();
    const int32_t* sp_indices = raw_indices_.data();
    const float* sp_values = raw_values_.data();
    if (params_.unique && sparse_data_) {
        sp_indptr_store.assign(1, 0);
        for (int64_t i = 0; i < n_index; ++i) {
            const int64_t src = index[static_cast<std::size_t>(i)];
            for (int32_t k = raw_indptr_[static_cast<std::size_t>(src)];
                 k < raw_indptr_[static_cast<std::size_t>(src) + 1]; ++k) {
                sp_indices_store.push_back(
                    raw_indices_[static_cast<std::size_t>(k)]);
                sp_values_store.push_back(
                    raw_values_[static_cast<std::size_t>(k)]);
            }
            sp_indptr_store.push_back(
                static_cast<int32_t>(sp_indices_store.size()));
        }
        sp_indptr = sp_indptr_store.data();
        sp_indices = sp_indices_store.data();
        sp_values = sp_values_store.data();
    }
    // ------------------------------------------------------------------------

    if (n_index <= params_.n_neighbors) {
        if (n_index == 1) {
            // np.zeros((1, n_components)) -- note: *not* inverse-expanded,
            // exactly like the reference's early return.
            embedding_.assign(params_.n_components, 0.0f);
            graph_ = SparseMatrix();
            knn_index_.reset();
            knn_indices_.clear();
            knn_dists_.clear();
            fitted_ = true;
            return *this;
        }
        std::fprintf(stderr,
                     "n_neighbors is larger than the dataset size; truncating "
                     "to X.shape[0] - 1\n");
        n_neighbors_ = static_cast<int>(n_index - 1);
        // if self.densmap: self._densmap_kwds["n_neighbors"] = _n_neighbors
        if (params_.densmap) dens_n_neighbors = n_neighbors_;
    } else {
        n_neighbors_ = params_.n_neighbors;
    }

    // random_state = check_random_state(self.random_state)
    nnd::NumpyRandomState rs(params_.random_state.has_value()
                                 ? *params_.random_state
                                 : std::random_device{}());

    FuzzySimplicialSetResult fuzzy;
    edges_removed_ = 0;
    if (n_index < 4096 && !force_approximation_algorithm_) {
        // Handle small cases efficiently by computing all distances.
        small_data_ = true;
        knn_index_.reset();
        sparse_knn_index_.reset();
        std::vector<float> dmat;
        if (params_.metric == "precomputed") {
            // pairwise_distances(X, metric="precomputed") is a validated
            // passthrough of the input distance matrix.
            dmat.assign(X_indexed,
                        X_indexed + static_cast<std::size_t>(n_index) * n_index);
        } else if (sparse_data_) {
            // --- supervised-sparse additions --------------------------------
            // pairwise_distances(X_csr, metric=self.metric): sklearn's native
            // sparse float32 kernels (euclidean/l2, cosine).
            dmat = sparse_pairwise_distances_sklearn(
                sp_indptr, sp_indices, sp_values, n_index, sp_indptr,
                sp_indices, sp_values, n_index, params_.metric,
                /*x_is_y=*/true);
            // ----------------------------------------------------------------
        } else {
            dmat = pairwise_distances_sklearn(X_indexed, n_index, X_indexed,
                                              n_index, n_features,
                                              params_.metric, /*x_is_y=*/true,
                                              params_.metric_kwds);
        }
        // dmat[dmat >= self._disconnection_distance] = np.inf (skipped for
        // the default +inf threshold, where the mask is all-False).
        if (std::isfinite(disconnection_distance_)) {
            for (float& v : dmat) {
                if (static_cast<double>(v) >= disconnection_distance_) {
                    ++edges_removed_;
                    v = std::numeric_limits<float>::infinity();
                }
            }
        }
        // fuzzy_simplicial_set(dmat, "precomputed") -> nearest_neighbors
        // precomputed branch: fast_knn_indices + row gather, inf -> -1.
        knn_cols_ = n_neighbors_;
        knn_indices_.assign(static_cast<std::size_t>(n_index) * n_neighbors_,
                            -1);
        knn_dists_.assign(static_cast<std::size_t>(n_index) * n_neighbors_,
                          0.0f);
        fast_knn_from_dmat(dmat, n_index, n_index, n_neighbors_,
                           knn_indices_.data(), knn_dists_.data());
        for (std::size_t i = 0; i < knn_dists_.size(); ++i) {
            if (std::isinf(knn_dists_[i])) knn_indices_[i] = -1;
        }
        fuzzy = fuzzy_simplicial_set(
            static_cast<int>(n_index), knn_indices_.data(), knn_dists_.data(),
            n_neighbors_, params_.set_op_mix_ratio, params_.local_connectivity,
            /*apply_set_operations=*/true,
            /*return_dists=*/want_graph_dists);
    } else {
        // Standard case.
        small_data_ = false;
        if (params_.metric == "precomputed") {
            // nearest_neighbors' precomputed branch: fast_knn_indices over the
            // distance-matrix rows, then prune infinite-distance neighbours
            // (knn_indices[knn_dists == inf] = -1); no search index.
            knn_index_.reset();
            knn_cols_ = n_neighbors_;
            knn_indices_.assign(
                static_cast<std::size_t>(n_index) * n_neighbors_, -1);
            knn_dists_.assign(static_cast<std::size_t>(n_index) * n_neighbors_,
                              0.0f);
            const std::vector<float> dmat(
                X_indexed,
                X_indexed + static_cast<std::size_t>(n_index) * n_index);
            fast_knn_from_dmat(dmat, n_index, n_index, n_neighbors_,
                               knn_indices_.data(), knn_dists_.data());
            for (std::size_t i = 0; i < knn_dists_.size(); ++i) {
                if (std::isinf(knn_dists_[i])) knn_indices_[i] = -1;
            }
        } else if (pknn != nullptr) {
            // Bypass the kNN computation with the user-provided graph.
            knn_index_ = pknn->search_index;
            if (prune_pknn) {
                knn_cols_ = params_.n_neighbors;
                knn_indices_.resize(static_cast<std::size_t>(n_index) *
                                    knn_cols_);
                knn_dists_.resize(static_cast<std::size_t>(n_index) *
                                  knn_cols_);
                for (int64_t i = 0; i < n_index; ++i) {
                    for (int64_t j = 0; j < knn_cols_; ++j) {
                        knn_indices_[i * knn_cols_ + j] =
                            pknn->indices[i * pknn->n_neighbors + j];
                        knn_dists_[i * knn_cols_ + j] =
                            pknn->dists[i * pknn->n_neighbors + j];
                    }
                }
            } else {
                knn_cols_ = pknn->n_neighbors;
                knn_indices_ = pknn->indices;
                knn_dists_ = pknn->dists;
            }
        } else if (sparse_data_) {
            // --- supervised-sparse additions --------------------------------
            // Sparse NN-descent (continuing the numpy RandomState stream)
            // with the prepared sparse search index kept for transform().
            sparse_knn_index_ = std::make_shared<spnnd::SparsePreparedIndex>(
                spnnd::SparsePreparedIndex::build(
                    sp_indptr, sp_indices, sp_values, n_index, n_features,
                    n_neighbors_, params_.metric, rs, params_.verbose));
            knn_indices_ = sparse_knn_index_->knn_indices();
            knn_dists_ = sparse_knn_index_->knn_dists();
            knn_cols_ = n_neighbors_;
            // ----------------------------------------------------------------
        } else {
            // NN-descent (continuing the numpy RandomState stream) with the
            // prepared search index kept for transform().
            knn_index_ = std::make_shared<nnd::PreparedIndex>(
                nnd::PreparedIndex::build(X_indexed, n_index, n_features,
                                          n_neighbors_, params_.metric,
                                          params_.metric_kwds, rs,
                                          params_.verbose));
            knn_indices_ = knn_index_->knn_indices();
            knn_dists_ = knn_index_->knn_dists();
            knn_cols_ = n_neighbors_;
        }
        // Disconnect any vertices farther apart than _disconnection_distance
        // (default +inf: only unfilled heap slots at +inf are masked; note
        // inf >= inf is true, and upstream counts those as removed edges).
        for (std::size_t i = 0; i < knn_dists_.size(); ++i) {
            if (static_cast<double>(knn_dists_[i]) >= disconnection_distance_) {
                knn_indices_[i] = -1;
                knn_dists_[i] = std::numeric_limits<float>::infinity();
                ++edges_removed_;
            }
        }
        // Upstream passes self.n_neighbors (not _n_neighbors) here; the knn
        // array width may differ from it (wider precomputed_knn arrays).
        fuzzy = fuzzy_simplicial_set(
            static_cast<int>(n_index), knn_indices_.data(), knn_dists_.data(),
            params_.n_neighbors, params_.set_op_mix_ratio,
            params_.local_connectivity, /*apply_set_operations=*/true,
            /*return_dists=*/want_graph_dists, static_cast<int>(knn_cols_));
    }
    graph_ = fuzzy.graph;
    sigmas_ = std::move(fuzzy.sigmas);
    rhos_ = std::move(fuzzy.rhos);
    // self.graph_dists_ (and self._densmap_kwds["graph_dists"]).
    if (fuzzy.has_dists) graph_dists_ = std::move(fuzzy.dists);

    // Report the number of vertices with degree 0 in our graph_ (they were
    // properly disconnected) and mirror raise_disconnected_warning.
    vertices_disconnected_ = 0;
    for (int32_t i = 0; i < graph_.n_rows; ++i) {
        float row_sum = 0.0f;
        for (int32_t k = graph_.indptr[i]; k < graph_.indptr[i + 1]; ++k)
            row_sum += graph_.data[k];
        if (row_sum == 0.0f) ++vertices_disconnected_;
    }
    raise_disconnected_warning(edges_removed_, vertices_disconnected_,
                               disconnection_distance_, n_,
                               params_.verbose);

    // --- supervised-sparse additions ----------------------------------------
    // Supervised / semi-supervised fit: intersect graph_ with the target
    // simplicial set built from y (self._supervised path of umap_.py::fit).
    if (y != nullptr) supervise_graph(y, index, n_samples, rs);
    // ------------------------------------------------------------------------

    if (params_.transform_mode == "embedding") {
        // _fit_embed_data -> simplicial_set_embedding. The weight filtering
        // runs *before* initialisation upstream (spectral init sees the
        // filtered graph), so resolve/filter first, then init, then hand the
        // prefiltered graph to the optimisation glue.
        SparseMatrix g = graph_;
        const std::vector<int>* epochs_list =
            params_.n_epochs_list.has_value() ? &*params_.n_epochs_list
                                              : nullptr;
        int requested_epochs = params_.n_epochs.value_or(-1);
        if (epochs_list != nullptr) {
            // n_epochs_max = max(n_epochs) for filtering/epochs-per-sample.
            requested_epochs = *std::max_element(epochs_list->begin(),
                                                 epochs_list->end());
        }
        // Use more epochs for densMAP (upstream: default_epochs += 200).
        const int n_epochs = resolve_epochs_and_filter_graph(
            g, requested_epochs, params_.densmap ? 200 : 0,
            /*write_through=*/&graph_);
        // disconnected_vertices for the NaN mask: computed upstream *after*
        // _fit_embed_data from self.graph_ -- which the epoch filter just
        // mutated through the shared scipy data array (write_through above).
        // A target_weight near 1.0 supervised fit can fully disconnect
        // vertices whose neighbours all carry other labels.
        std::vector<bool> disconnected(static_cast<std::size_t>(graph_.n_rows),
                                       false);
        for (int32_t i = 0; i < graph_.n_rows; ++i) {
            float row_sum = 0.0f;
            for (int32_t k = graph_.indptr[i]; k < graph_.indptr[i + 1]; ++k)
                row_sum += graph_.data[k];
            if (row_sum == 0.0f)
                disconnected[static_cast<std::size_t>(i)] = true;
        }
        fit_init_ = make_fit_init(sparse_data_ ? nullptr : X_indexed, n_index,
                                  g, rs);
        const std::vector<float>& init_emb = fit_init_;
        const auto r3 = rs.rng_state3();
        const int64_t rng_state[3] = {r3[0], r3[1], r3[2]};

        // --- densmap additions ----------------------------------------------
        // simplicial_set_embedding's `if densmap or output_dens:` block:
        // original densities from the *filtered* graph + graph_dists_, then
        // ro = log(eps + ro/mu_sum) (float32); R standardises ro for densmap.
        DensmapParams dens_kwds;
        std::vector<float> ro;
        if (params_.densmap || params_.output_dens) {
            std::vector<float> mu_sum;
            densmap_densities(g, graph_dists_, /*square=*/true, ro, mu_sum);
            densmap_log_radii(ro, mu_sum);
            if (params_.densmap) {
                // R = (ro - np.mean(ro)) / np.std(ro), all float32.
                const int64_t nv = static_cast<int64_t>(ro.size());
                const float mean = numpy_mean_f32(ro.data(), nv);
                const float sd = numpy_std_f32(ro.data(), nv);
                dens_kwds.R.resize(ro.size());
                for (std::size_t i = 0; i < ro.size(); ++i)
                    dens_kwds.R[i] = (ro[i] - mean) / sd;
                dens_kwds.mu = g.data;  // filtered graph.data
                dens_kwds.mu_sum = std::move(mu_sum);
                // _densmap_kwds: lambda/frac only non-zero when densmap.
                dens_kwds.lambda = params_.dens_lambda;
                dens_kwds.frac = params_.dens_frac;
                dens_kwds.var_shift = params_.dens_var_shift;
            }
        }
        // euclidean_output = self.output_metric in ("euclidean", "l2").
        const bool euclidean_output = (params_.output_metric == "euclidean" ||
                                       params_.output_metric == "l2");
        OutputMetricFn out_metric;
        if (!euclidean_output)
            out_metric = make_output_metric(params_.output_metric,
                                            params_.output_metric_kwds);
        // --------------------------------------------------------------------

        std::vector<std::vector<float>> emb_list;
        embedding_ = simplicial_set_embedding_from_init(
            g, init_emb.data(), params_.n_components, a_, b_, rng_state,
            n_epochs, params_.repulsion_strength, initial_alpha_,
            static_cast<double>(params_.negative_sample_rate),
            /*parallel=*/!params_.random_state.has_value(),
            /*graph_prefiltered=*/true, epochs_list,
            epochs_list != nullptr ? &emb_list : nullptr,
            params_.densmap ? &dens_kwds : nullptr,
            euclidean_output ? nullptr : &out_metric);

        // --- densmap additions ----------------------------------------------
        // simplicial_set_embedding's `if output_dens:` epilogue (runs on the
        // optimised embedding *before* the disconnected-NaN masking, and
        // continues the same numpy RandomState through nearest_neighbors).
        if (params_.output_dens) {
            rad_orig_ = ro;
            const int64_t k = dens_n_neighbors;
            const int64_t n_emb = g.n_rows;
            std::vector<int32_t> emb_knn_i(static_cast<std::size_t>(n_emb) *
                                           k);
            std::vector<float> emb_knn_d(static_cast<std::size_t>(n_emb) * k);
            nnd::nearest_neighbors(embedding_.data(), n_emb,
                                   params_.n_components, k, "euclidean",
                                   MetricKwds{}, /*angular=*/false, rs,
                                   emb_knn_i.data(), emb_knn_d.data(),
                                   params_.verbose);
            // fuzzy_simplicial_set(embedding, k, rs, "euclidean", {}, knn...,
            // return_dists=True) -- upstream's *default* set_op_mix_ratio=1.0
            // and local_connectivity=1.0, not self's parameters.
            FuzzySimplicialSetResult emb_fuzzy = fuzzy_simplicial_set(
                static_cast<int>(n_emb), emb_knn_i.data(), emb_knn_d.data(),
                static_cast<int>(k), /*set_op_mix_ratio=*/1.0,
                /*local_connectivity=*/1.0, /*apply_set_operations=*/true,
                /*return_dists=*/true);
            SparseMatrix emb_graph = emb_fuzzy.graph;
            emb_graph.eliminate_zeros();
            std::vector<float> emb_mu_sum;
            densmap_densities(emb_graph, emb_fuzzy.dists, /*square=*/false,
                              rad_emb_, emb_mu_sum);
            densmap_log_radii(rad_emb_, emb_mu_sum);
        }
        // --------------------------------------------------------------------

        // self.embedding_list_ = [e[inverse] for e in aux["embedding_list"]]
        // (snapshots are inverse-expanded but *not* NaN-masked upstream).
        if (epochs_list != nullptr) {
            embedding_list_.reserve(emb_list.size());
            const int dim_out = params_.n_components;
            for (const std::vector<float>& e : emb_list) {
                std::vector<float> expanded(
                    static_cast<std::size_t>(n_samples) * dim_out);
                for (int64_t i = 0; i < n_samples; ++i) {
                    std::memcpy(
                        expanded.data() + i * dim_out,
                        e.data() + inverse[static_cast<std::size_t>(i)] *
                                       dim_out,
                        static_cast<std::size_t>(dim_out) * sizeof(float));
                }
                embedding_list_.push_back(std::move(expanded));
            }
        }

        // Fully-disconnected vertices embed as NaN (before the inverse
        // expansion, matching fit()).
        for (int32_t i = 0; i < graph_.n_rows; ++i) {
            if (!disconnected[static_cast<std::size_t>(i)]) continue;
            for (int d = 0; d < params_.n_components; ++d)
                embedding_[static_cast<std::size_t>(i) * params_.n_components +
                           d] = std::numeric_limits<float>::quiet_NaN();
        }

        // self.embedding_ = self.embedding_[inverse]
        if (params_.unique) {
            const int dim_out = params_.n_components;
            std::vector<float> expanded(
                static_cast<std::size_t>(n_samples) * dim_out);
            for (int64_t i = 0; i < n_samples; ++i) {
                std::memcpy(expanded.data() + i * dim_out,
                            embedding_.data() +
                                inverse[static_cast<std::size_t>(i)] * dim_out,
                            static_cast<std::size_t>(dim_out) * sizeof(float));
            }
            embedding_ = std::move(expanded);
            // self.rad_orig_ = aux_data["rad_orig"][inverse] (and rad_emb_;
            // the radii are inverse-expanded but never NaN-masked upstream).
            if (params_.output_dens) {
                std::vector<float> rad_o(static_cast<std::size_t>(n_samples));
                std::vector<float> rad_e(static_cast<std::size_t>(n_samples));
                for (int64_t i = 0; i < n_samples; ++i) {
                    rad_o[static_cast<std::size_t>(i)] =
                        rad_orig_[static_cast<std::size_t>(
                            inverse[static_cast<std::size_t>(i)])];
                    rad_e[static_cast<std::size_t>(i)] =
                        rad_emb_[static_cast<std::size_t>(
                            inverse[static_cast<std::size_t>(i)])];
                }
                rad_orig_ = std::move(rad_o);
                rad_emb_ = std::move(rad_e);
            }
        }
    } else {
        // transform_mode != "embedding": the reference skips the embedding
        // entirely (fit_transform then returns self.graph_).
        embedding_.clear();
        fit_init_.clear();
    }

    fitted_ = true;
    return *this;
}

// --- supervised-sparse additions --------------------------------------------

namespace {

// utils.py::fast_knn_indices semantics on a float64 distance matrix (the
// supervised small-data target path: sklearn pairwise on float64 y), with the
// row gather still in float64; fuzzy_simplicial_set then casts to float32
// (knn_dists.astype(np.float32)).
void fast_knn_from_dmat_f64(const std::vector<double>& dmat, int64_t n,
                            int64_t m, int64_t n_neighbors,
                            int32_t* knn_indices, float* knn_dists) {
    std::vector<int32_t> order(static_cast<std::size_t>(m));
    for (int64_t row = 0; row < n; ++row) {
        const double* d = dmat.data() + row * m;
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [d](int32_t a, int32_t b) { return d[a] < d[b]; });
        for (int64_t j = 0; j < n_neighbors; ++j) {
            knn_indices[row * n_neighbors + j] = order[j];
            knn_dists[row * n_neighbors + j] =
                static_cast<float>(d[order[j]]);
        }
    }
}

// sklearn pairwise_distances on the (n, 1) float64 target column.
// "euclidean"/"l2" replicate euclidean_distances' float64 dot-trick
// (distances = (-2 * X @ X.T + XX) + YY, maximum(0), zero diagonal, sqrt) --
// exact for 1-d input. Other named metrics evaluate the double-precision
// registry directly (upstream routes through sklearn / pairwise_special_
// metric; documented deviation, exact for the usual 1-d integer-ish labels).
std::vector<double> target_pairwise_f64(const std::vector<double>& y,
                                        const std::string& metric,
                                        const MetricKwds& kwds) {
    const int64_t n = static_cast<int64_t>(y.size());
    std::vector<double> out(static_cast<std::size_t>(n) * n);
    if (metric == "euclidean" || metric == "l2") {
        for (int64_t i = 0; i < n; ++i) {
            const double xx = y[static_cast<std::size_t>(i)] *
                              y[static_cast<std::size_t>(i)];
            for (int64_t j = 0; j < n; ++j) {
                const double yy = y[static_cast<std::size_t>(j)] *
                                  y[static_cast<std::size_t>(j)];
                double v = ((-2.0 * (y[static_cast<std::size_t>(i)] *
                                     y[static_cast<std::size_t>(j)])) +
                            xx) +
                           yy;
                v = std::max(v, 0.0);
                if (i == j) v = 0.0;
                out[static_cast<std::size_t>(i) * n + j] = std::sqrt(v);
            }
        }
        return out;
    }
    const DistanceFn fn = make_metric(metric, kwds);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            out[static_cast<std::size_t>(i) * n + j] =
                fn(&y[static_cast<std::size_t>(i)],
                   &y[static_cast<std::size_t>(j)], 1);
        }
    }
    return out;
}

}  // namespace

void UMAP::supervise_graph(const float* y, const std::vector<int64_t>& index,
                           int64_t n_samples, nnd::NumpyRandomState& rs) {
    const int64_t n_index = static_cast<int64_t>(index.size());
    // y_ = check_array(y, ensure_2d=False)[index] (float64).
    std::vector<double> y_(static_cast<std::size_t>(n_index));
    for (int64_t i = 0; i < n_index; ++i)
        y_[static_cast<std::size_t>(i)] = static_cast<double>(
            y[index[static_cast<std::size_t>(i)]]);

    const std::string& tm = params_.target_metric;
    if (tm == "categorical") {
        // far_dist from target_weight; unknown labels (-1) use the default
        // unknown_dist = 1.0.
        const double far_dist =
            params_.target_weight < 1.0
                ? 2.5 * (1.0 / (1.0 - params_.target_weight))
                : 1.0e12;
        graph_ = discrete_metric_simplicial_set_intersection(
            graph_, y_.data(), /*unknown_dist=*/1.0, far_dist);
        return;
    }
    if (tm == "hierarchical_categorical" || tm == "ordinal" || tm == "count" ||
        tm == "string" || tm == "myers") {
        // dist.DISCRETE_METRICS branch: metric_kws come from
        // get_discrete_params(y_, target_metric), NOT target_metric_kwds.
        const double scale = params_.target_weight < 1.0
                                 ? 2.5 * (1.0 / (1.0 - params_.target_weight))
                                 : 1.0e12;
        std::function<double(double, double)> metric_fn;
        if (tm == "ordinal") {
            // {"support_size": float(data.max() - data.min()) / 2.0}
            double mn = y_[0], mx = y_[0];
            for (const double v : y_) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            const double support_size = (mx - mn) / 2.0;
            metric_fn = [support_size](double a, double b) {
                return ordinal_distance(a, b, support_size);
            };
        } else if (tm == "count") {
            // scipy.stats.tmin/tmax/tmean with no limits are plain
            // min/max/np.mean; normalisation = count_distance(min, max,
            // poisson_lambda=lambda) / 2.
            double mn = y_[0], mx = y_[0];
            for (const double v : y_) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            const double lambda_ =
                numpy_mean_f64(y_.data(), static_cast<int64_t>(y_.size()));
            const double normalisation =
                count_distance(mn, mx, lambda_, 1.0) / 2.0;
            metric_fn = [lambda_, normalisation](double a, double b) {
                return count_distance(a, b, lambda_, normalisation);
            };
        } else {
            // "string"/"myers" need object-dtype y; "hierarchical_
            // categorical" gets {} from get_discrete_params and KeyErrors
            // upstream (cat_hierarchy=[{}] has no entries).
            throw std::invalid_argument(
                "UMAP: target_metric '" + tm +
                "' is not supported in the C++ port");
        }
        graph_ = discrete_metric_simplicial_set_intersection(
            graph_, y_.data(), /*unknown_dist=*/1.0, /*far_dist=*/5.0,
            metric_fn, scale);
        return;
    }

    // General (non-discrete) target metric: build the target fuzzy
    // simplicial set (apply_set_operations=False!) and intersect.
    const int target_n_neighbors = params_.target_n_neighbors == -1
                                       ? n_neighbors_
                                       : params_.target_n_neighbors;
    FuzzySimplicialSetResult tfuzzy;
    std::vector<int32_t> tknn_i(static_cast<std::size_t>(n_index) *
                                target_n_neighbors);
    std::vector<float> tknn_d(static_cast<std::size_t>(n_index) *
                              target_n_neighbors);
    if (n_samples < 4096) {
        // "Handle the small case as precomputed as before" -- upstream tests
        // *original* y length (y.shape[0]), not the deduplicated length.
        const std::vector<double> ydmat =
            target_pairwise_f64(y_, tm, params_.target_metric_kwds);
        fast_knn_from_dmat_f64(ydmat, n_index, n_index, target_n_neighbors,
                               tknn_i.data(), tknn_d.data());
    } else {
        // Standard case: fuzzy_simplicial_set(y_, ...) -> NN-descent on the
        // float32-cast target column, continuing the RandomState stream.
        std::vector<float> y32(static_cast<std::size_t>(n_index));
        for (int64_t i = 0; i < n_index; ++i)
            y32[static_cast<std::size_t>(i)] =
                static_cast<float>(y_[static_cast<std::size_t>(i)]);
        nnd::nearest_neighbors(y32.data(), n_index, 1, target_n_neighbors, tm,
                               params_.target_metric_kwds, /*angular=*/false,
                               rs, tknn_i.data(), tknn_d.data(),
                               params_.verbose);
    }
    tfuzzy = fuzzy_simplicial_set(
        static_cast<int>(n_index), tknn_i.data(), tknn_d.data(),
        target_n_neighbors, /*set_op_mix_ratio=*/1.0,
        /*local_connectivity=*/1.0, /*apply_set_operations=*/false,
        /*return_dists=*/false);
    graph_ = general_simplicial_set_intersection(graph_, tfuzzy.graph,
                                                 params_.target_weight);
    graph_ = reset_local_connectivity(graph_);
}

// ----------------------------------------------------------------------------

std::vector<float> UMAP::fit_transform(const float* X, int64_t n_samples,
                                       int64_t n_features, const float* y) {
    if (params_.transform_mode == "graph") {
        throw std::invalid_argument(
            "fit_transform returns the graph when transform_mode='graph'; "
            "use fit_transform_graph()");
    }
    if (params_.transform_mode != "embedding") {
        throw std::invalid_argument(
            "Unrecognized transform mode " + params_.transform_mode +
            "; should be one of 'embedding' or 'graph'");
    }
    fit(X, n_samples, n_features, y);
    return embedding_;
}

// --- supervised-sparse additions --------------------------------------------
std::vector<float> UMAP::fit_transform(const int32_t* indptr,
                                       const int32_t* indices,
                                       const float* data, int64_t n_samples,
                                       int64_t n_features, const float* y) {
    if (params_.transform_mode == "graph") {
        throw std::invalid_argument(
            "fit_transform returns the graph when transform_mode='graph'; "
            "use fit_transform_graph()");
    }
    if (params_.transform_mode != "embedding") {
        throw std::invalid_argument(
            "Unrecognized transform mode " + params_.transform_mode +
            "; should be one of 'embedding' or 'graph'");
    }
    fit(indptr, indices, data, n_samples, n_features, y);
    return embedding_;
}
// ----------------------------------------------------------------------------

const SparseMatrix& UMAP::fit_transform_graph(const float* X,
                                              int64_t n_samples,
                                              int64_t n_features) {
    if (params_.transform_mode != "graph") {
        throw std::invalid_argument(
            "fit_transform_graph requires transform_mode='graph'");
    }
    fit(X, n_samples, n_features);
    return graph_;
}

MembershipStrengths UMAP::transform_membership(const float* X_new,
                                               int64_t n_queries) const {
    // #848: raise when a search index would be needed but none exists
    // (metric="precomputed" standard-branch fits, precomputed_knn without a
    // search index).
    if (!small_data_ && !knn_index_) {
        throw std::runtime_error(
            "No search index available: transforming data"
            " into an existing embedding is not supported");
    }

    // kNN of the new points against the training data.
    int64_t cols;
    std::vector<int32_t> indices;
    std::vector<float> dists;
    if (params_.metric == "precomputed") {
        warn_msg(
            "Transforming new data with precomputed metric. "
            "We are assuming the input data is a matrix of distances from the "
            "new points to the points in the training set. If the input "
            "matrix is sparse, it should contain distances from the new "
            "points to their nearest neighbours or approximate nearest "
            "neighbours in the training set.");
        // assert X.shape[1] == self._raw_data.shape[0] holds by construction
        // (this transform takes rows of width n_features == n_train).
        cols = n_neighbors_;
        indices.resize(static_cast<std::size_t>(n_queries) * cols);
        dists.resize(static_cast<std::size_t>(n_queries) * cols);
        // argpartition + take_along_axis + argsort == k smallest per row
        // sorted by distance.
        const std::vector<float> dmat(
            X_new, X_new + static_cast<std::size_t>(n_queries) * n_);
        fast_knn_from_dmat(dmat, n_queries, n_, cols, indices.data(),
                           dists.data());
        // assert np.min(indices) >= 0 and np.min(dists) >= 0.0
        for (float d : dists) {
            if (!(d >= 0.0f)) {
                throw std::invalid_argument(
                    "transform with metric='precomputed': distances must be "
                    "non-negative");
            }
        }
    } else if (small_data_) {
        cols = n_neighbors_;
        const std::vector<float> dmat = pairwise_distances_sklearn(
            X_new, n_queries, raw_data_.data(), n_, dim_, params_.metric,
            /*x_is_y=*/false, params_.metric_kwds);
        indices.resize(static_cast<std::size_t>(n_queries) * cols);
        dists.resize(static_cast<std::size_t>(n_queries) * cols);
        // argpartition + argsort: k smallest per row, sorted by distance
        // (ties by ascending index).
        fast_knn_from_dmat(dmat, n_queries, n_, cols, indices.data(),
                           dists.data());
    } else {
        cols = params_.n_neighbors;
        // epsilon = 0.24 if self._knn_search_index._angular_trees else 0.12
        // (the kwds-aware resolve_metric mirrors the flag the index was
        // built with, including the generic-metric hook's angular list).
        const double epsilon =
            nnd::resolve_metric(params_.metric, params_.metric_kwds)
                    .angular_trees
                ? 0.24
                : 0.12;
        indices.resize(static_cast<std::size_t>(n_queries) * cols);
        dists.resize(static_cast<std::size_t>(n_queries) * cols);
        knn_index_->query(X_new, n_queries, cols, epsilon, indices.data(),
                          dists.data());
    }

    // Remove any nearest neighbours whose distances are greater than our
    // disconnection_distance: indices[dists >= dd] = -1.
    for (std::size_t i = 0; i < dists.size(); ++i) {
        if (static_cast<double>(dists[i]) >= disconnection_distance_)
            indices[i] = -1;
    }

    // smooth_knn_dist with local_connectivity reduced by 1 for transform.
    const double adjusted_local_connectivity =
        std::max(0.0, params_.local_connectivity - 1.0);
    const SmoothKnnResult sk = smooth_knn_dist(
        dists.data(), static_cast<int>(n_queries), static_cast<int>(cols),
        static_cast<double>(n_neighbors_), 64, adjusted_local_connectivity);
    return compute_membership_strengths(
        indices.data(), dists.data(), sk.sigmas.data(), sk.rhos.data(),
        static_cast<int>(n_queries), static_cast<int>(cols),
        /*return_dists=*/false, /*bipartite=*/true);
}

SparseMatrix UMAP::transform_graph(const float* X_new,
                                   int64_t n_queries) const {
    if (!fitted_)
        throw std::runtime_error("UMAP: transform_graph() called before fit()");
    if (params_.transform_mode != "graph") {
        throw std::invalid_argument(
            "transform_graph requires transform_mode='graph'");
    }
    if (n_ == 1)
        throw std::invalid_argument(
            "Transform unavailable when model was fit with only a single "
            "data sample.");
    // --- supervised-sparse additions ----------------------------------------
    if (sparse_data_)
        throw std::invalid_argument(
            "UMAP: model was fit with sparse data; use the CSR "
            "transform_graph overload");
    // ------------------------------------------------------------------------
    // Input-hash shortcut: upstream returns self.graph_.
    if (n_queries == n_ &&
        std::memcmp(X_new, raw_data_.data(),
                    raw_data_.size() * sizeof(float)) == 0) {
        return graph_;
    }
    // --- densmap additions --------------------------------------------------
    // Raised after the hash shortcut, exactly like upstream's transform().
    if (params_.densmap)
        throw std::runtime_error(
            "Transforming data into an existing embedding not supported "
            "for densMAP.");
    // ------------------------------------------------------------------------
    const MembershipStrengths ms = transform_membership(X_new, n_queries);
    // The reference returns the raw COO matrix (explicit zeros retained; our
    // CSR keeps them too since eliminate_zeros() is not called).
    return SparseMatrix::from_coo(static_cast<int32_t>(n_queries),
                                  static_cast<int32_t>(n_), ms.rows, ms.cols,
                                  ms.vals);
}

std::vector<float> UMAP::transform(const float* X_new,
                                   int64_t n_queries) const {
    if (!fitted_)
        throw std::runtime_error("UMAP: transform() called before fit()");
    if (params_.transform_mode == "graph") {
        throw std::invalid_argument(
            "transform returns the graph when transform_mode='graph'; use "
            "transform_graph()");
    }
    if (n_ == 1)
        throw std::invalid_argument(
            "Transform unavailable when model was fit with only a single "
            "data sample.");
    // --- supervised-sparse additions ----------------------------------------
    if (sparse_data_)
        throw std::invalid_argument(
            "UMAP: model was fit with sparse data; use the CSR transform "
            "overload");
    // ------------------------------------------------------------------------

    // Input-hash shortcut: transforming the training data returns embedding_.
    if (n_queries == n_ &&
        std::memcmp(X_new, raw_data_.data(),
                    raw_data_.size() * sizeof(float)) == 0) {
        return embedding_;
    }
    // --- densmap additions --------------------------------------------------
    // NotImplementedError raised *after* the hash shortcut (transforming the
    // training data itself still returns embedding_ for densMAP models).
    if (params_.densmap)
        throw std::runtime_error(
            "Transforming data into an existing embedding not supported "
            "for densMAP.");
    // ------------------------------------------------------------------------
    if (params_.n_epochs_list.has_value()) {
        // Upstream: self.n_epochs is the list and `self.n_epochs // 3.0`
        // raises a TypeError inside transform.
        throw std::invalid_argument(
            "transform: n_epochs as a list is not supported (the reference "
            "raises a TypeError)");
    }
    const MembershipStrengths ms = transform_membership(X_new, n_queries);
    return transform_from_membership(ms, n_queries);
}

std::vector<float> UMAP::transform_from_membership(
    const MembershipStrengths& ms, int64_t n_queries) const {
    const int dim_out = params_.n_components;
    nnd::NumpyRandomState rs(params_.transform_seed);
    const auto r3 = rs.rng_state3();
    const int64_t rng_state[3] = {r3[0], r3[1], r3[2]};

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
    // (float32 copy, fixes #179 & #217), move_other=False, alpha/4); the
    // non-euclidean output metrics take optimize_layout_generic (upstream's
    // `if self.output_metric == "euclidean"` -- note: "l2" takes the generic
    // branch here, unlike fit's euclidean_output test).
    std::vector<float> tail_embedding = embedding_;
    if (params_.output_metric == "euclidean") {
        optimize_layout_euclidean(
            embedding.data(), static_cast<int>(n_queries),
            tail_embedding.data(), dim_out, head, tail, n_epochs,
            static_cast<int>(n_), epochs_per_sample, a_, b_, rng_state,
            params_.repulsion_strength, initial_alpha_ / 4.0,
            static_cast<double>(params_.negative_sample_rate),
            /*parallel=*/!params_.random_state.has_value(), /*densmap=*/false,
            /*move_other=*/false);
    } else {
        // --- densmap additions ----------------------------------------------
        const OutputMetricFn out_metric = make_output_metric(
            params_.output_metric, params_.output_metric_kwds);
        optimize_layout_generic(
            embedding.data(), static_cast<int>(n_queries),
            tail_embedding.data(), dim_out, head, tail, n_epochs,
            static_cast<int>(n_), epochs_per_sample, a_, b_, rng_state,
            out_metric, params_.repulsion_strength, initial_alpha_ / 4.0,
            static_cast<double>(params_.negative_sample_rate),
            /*move_other=*/false);
        // --------------------------------------------------------------------
    }
    return embedding;
}

// --- supervised-sparse additions --------------------------------------------

MembershipStrengths UMAP::transform_membership_sparse(
    const int32_t* q_indptr, const int32_t* q_indices, const float* q_data,
    int64_t n_queries) const {
    int64_t cols;
    std::vector<int32_t> indices;
    std::vector<float> dists;
    if (small_data_) {
        // pairwise_distances(X_new, self._raw_data, metric=self.metric) +
        // argpartition/argsort (== stable k-smallest per row).
        cols = n_neighbors_;
        const std::vector<float> dmat = sparse_pairwise_distances_sklearn(
            q_indptr, q_indices, q_data, n_queries, raw_indptr_.data(),
            raw_indices_.data(), raw_values_.data(), n_, params_.metric,
            /*x_is_y=*/false);
        indices.resize(static_cast<std::size_t>(n_queries) * cols);
        dists.resize(static_cast<std::size_t>(n_queries) * cols);
        fast_knn_from_dmat(dmat, n_queries, n_, cols, indices.data(),
                           dists.data());
    } else {
        if (!sparse_knn_index_) {
            throw std::runtime_error(
                "No search index available: transforming data"
                " into an existing embedding is not supported");
        }
        cols = params_.n_neighbors;
        // epsilon = 0.24 if self._knn_search_index._angular_trees else 0.12.
        const double epsilon =
            spdist::resolve_sparse_metric(params_.metric).angular_trees ? 0.24
                                                                        : 0.12;
        indices.resize(static_cast<std::size_t>(n_queries) * cols);
        dists.resize(static_cast<std::size_t>(n_queries) * cols);
        sparse_knn_index_->query(q_indptr, q_indices, q_data, n_queries, cols,
                                 epsilon, indices.data(), dists.data());
    }

    // indices[dists >= self._disconnection_distance] = -1.
    for (std::size_t i = 0; i < dists.size(); ++i) {
        if (static_cast<double>(dists[i]) >= disconnection_distance_)
            indices[i] = -1;
    }

    const double adjusted_local_connectivity =
        std::max(0.0, params_.local_connectivity - 1.0);
    const SmoothKnnResult sk = smooth_knn_dist(
        dists.data(), static_cast<int>(n_queries), static_cast<int>(cols),
        static_cast<double>(n_neighbors_), 64, adjusted_local_connectivity);
    return compute_membership_strengths(
        indices.data(), dists.data(), sk.sigmas.data(), sk.rhos.data(),
        static_cast<int>(n_queries), static_cast<int>(cols),
        /*return_dists=*/false, /*bipartite=*/true);
}

namespace {

// The transform input-hash shortcut for sparse queries: same CSR content as
// the training data.
bool same_csr(const int32_t* q_indptr, const int32_t* q_indices,
              const float* q_data, int64_t n_queries,
              const std::vector<int32_t>& indptr,
              const std::vector<int32_t>& indices,
              const std::vector<float>& data) {
    if (n_queries + 1 != static_cast<int64_t>(indptr.size())) return false;
    for (int64_t i = 0; i <= n_queries; ++i) {
        if (q_indptr[i] != indptr[static_cast<std::size_t>(i)]) return false;
    }
    const std::size_t nnz = indices.size();
    return std::memcmp(q_indices, indices.data(), nnz * sizeof(int32_t)) ==
               0 &&
           std::memcmp(q_data, data.data(), nnz * sizeof(float)) == 0;
}

}  // namespace

std::vector<float> UMAP::transform(const int32_t* q_indptr,
                                   const int32_t* q_indices,
                                   const float* q_data,
                                   int64_t n_queries) const {
    if (!fitted_)
        throw std::runtime_error("UMAP: transform() called before fit()");
    if (params_.transform_mode == "graph") {
        throw std::invalid_argument(
            "transform returns the graph when transform_mode='graph'; use "
            "transform_graph()");
    }
    if (n_ == 1)
        throw std::invalid_argument(
            "Transform unavailable when model was fit with only a single "
            "data sample.");
    if (!sparse_data_)
        throw std::invalid_argument(
            "UMAP: model was fit with dense data; use the dense transform "
            "overload");

    // Input-hash shortcut: transforming the training data returns embedding_.
    if (same_csr(q_indptr, q_indices, q_data, n_queries, raw_indptr_,
                 raw_indices_, raw_values_)) {
        return embedding_;
    }
    if (params_.densmap)
        throw std::runtime_error(
            "Transforming data into an existing embedding not supported "
            "for densMAP.");
    if (params_.n_epochs_list.has_value()) {
        throw std::invalid_argument(
            "transform: n_epochs as a list is not supported (the reference "
            "raises a TypeError)");
    }

    const MembershipStrengths ms =
        transform_membership_sparse(q_indptr, q_indices, q_data, n_queries);
    return transform_from_membership(ms, n_queries);
}

SparseMatrix UMAP::transform_graph(const int32_t* q_indptr,
                                   const int32_t* q_indices,
                                   const float* q_data,
                                   int64_t n_queries) const {
    if (!fitted_)
        throw std::runtime_error("UMAP: transform_graph() called before fit()");
    if (params_.transform_mode != "graph") {
        throw std::invalid_argument(
            "transform_graph requires transform_mode='graph'");
    }
    if (n_ == 1)
        throw std::invalid_argument(
            "Transform unavailable when model was fit with only a single "
            "data sample.");
    if (!sparse_data_)
        throw std::invalid_argument(
            "UMAP: model was fit with dense data; use the dense "
            "transform_graph overload");
    if (same_csr(q_indptr, q_indices, q_data, n_queries, raw_indptr_,
                 raw_indices_, raw_values_)) {
        return graph_;
    }
    if (params_.densmap)
        throw std::runtime_error(
            "Transforming data into an existing embedding not supported "
            "for densMAP.");
    const MembershipStrengths ms =
        transform_membership_sparse(q_indptr, q_indices, q_data, n_queries);
    return SparseMatrix::from_coo(static_cast<int32_t>(n_queries),
                                  static_cast<int32_t>(n_), ms.rows, ms.cols,
                                  ms.vals);
}

// ----------------------------------------------------------------------------

}  // namespace umap
