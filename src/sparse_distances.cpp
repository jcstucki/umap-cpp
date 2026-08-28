// Implementation of include/umap/sparse_distances.hpp. See that header for
// scope and faithfulness notes. Line references are to
// reference/pynndescent/pynndescent/sparse.py (identical implementations for
// these metrics live in reference/umap/umap/sparse.py).

#include "umap/sparse_distances.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace umap {
namespace spdist {

namespace {
constexpr float kFloat32Max = std::numeric_limits<float>::max();
}  // namespace

// ------------------------------------------------------------------- aux ops

float norm(const float* data, int64_t nnz) {
    // utils.py::norm: result starts as float64; each term vec[i]*vec[i] is a
    // float32 product promoted to float64 for the accumulation; np.sqrt in
    // float64, cast to float32 by the f4 return signature.
    double result = 0.0;
    for (int64_t i = 0; i < nnz; ++i) {
        float prod = data[i] * data[i];
        result += static_cast<double>(prod);
    }
    return static_cast<float>(std::sqrt(result));
}

std::vector<int32_t> arr_union(const int32_t* ar1, int64_t n1,
                               const int32_t* ar2, int64_t n2) {
    // arr_union == arr_unique(concatenate) for sorted duplicate-free inputs:
    // an ordinary sorted-set union.
    std::vector<int32_t> out;
    out.reserve(static_cast<size_t>(n1 + n2));
    int64_t i1 = 0, i2 = 0;
    while (i1 < n1 && i2 < n2) {
        if (ar1[i1] == ar2[i2]) {
            out.push_back(ar1[i1]);
            ++i1;
            ++i2;
        } else if (ar1[i1] < ar2[i2]) {
            out.push_back(ar1[i1]);
            ++i1;
        } else {
            out.push_back(ar2[i2]);
            ++i2;
        }
    }
    while (i1 < n1) out.push_back(ar1[i1++]);
    while (i2 < n2) out.push_back(ar2[i2++]);
    return out;
}

std::vector<int32_t> arr_intersect(const int32_t* ar1, int64_t n1,
                                   const int32_t* ar2, int64_t n2) {
    // arr_intersect: sort(concatenate), keep values equal to their successor.
    std::vector<int32_t> out;
    int64_t i1 = 0, i2 = 0;
    while (i1 < n1 && i2 < n2) {
        if (ar1[i1] == ar2[i2]) {
            out.push_back(ar1[i1]);
            ++i1;
            ++i2;
        } else if (ar1[i1] < ar2[i2]) {
            ++i1;
        } else {
            ++i2;
        }
    }
    return out;
}

void sparse_sum(const int32_t* ind1, const float* data1, int64_t n1,
                const int32_t* ind2, const float* data2, int64_t n2,
                std::vector<int32_t>& result_ind,
                std::vector<float>& result_data) {
    result_ind.clear();
    result_data.clear();
    result_ind.reserve(static_cast<size_t>(n1 + n2));
    result_data.reserve(static_cast<size_t>(n1 + n2));

    int64_t i1 = 0, i2 = 0;

    // pass through both index lists
    while (i1 < n1 && i2 < n2) {
        int32_t j1 = ind1[i1];
        int32_t j2 = ind2[i2];
        if (j1 == j2) {
            float val = data1[i1] + data2[i2];
            if (val != 0.0f) {
                result_ind.push_back(j1);
                result_data.push_back(val);
            }
            ++i1;
            ++i2;
        } else if (j1 < j2) {
            float val = data1[i1];
            if (val != 0.0f) {
                result_ind.push_back(j1);
                result_data.push_back(val);
            }
            ++i1;
        } else {
            float val = data2[i2];
            if (val != 0.0f) {
                result_ind.push_back(j2);
                result_data.push_back(val);
            }
            ++i2;
        }
    }

    // pass over the tails
    while (i1 < n1) {
        float val = data1[i1];
        if (val != 0.0f) {
            result_ind.push_back(ind1[i1]);
            result_data.push_back(val);
        }
        ++i1;
    }
    while (i2 < n2) {
        float val = data2[i2];
        if (val != 0.0f) {
            result_ind.push_back(ind2[i2]);
            result_data.push_back(val);
        }
        ++i2;
    }
}

void sparse_diff(const int32_t* ind1, const float* data1, int64_t n1,
                 const int32_t* ind2, const float* data2, int64_t n2,
                 std::vector<int32_t>& result_ind,
                 std::vector<float>& result_data) {
    // sparse_sum(ind1, data1, ind2, -data2); the negation is exact in f32.
    std::vector<float> neg(data2, data2 + n2);
    for (float& v : neg) v = -v;
    sparse_sum(ind1, data1, n1, ind2, neg.data(), n2, result_ind, result_data);
}

void sparse_mul(const int32_t* ind1, const float* data1, int64_t n1,
                const int32_t* ind2, const float* data2, int64_t n2,
                std::vector<int32_t>& result_ind,
                std::vector<float>& result_data) {
    result_ind.clear();
    result_data.clear();

    int64_t i1 = 0, i2 = 0;
    while (i1 < n1 && i2 < n2) {
        int32_t j1 = ind1[i1];
        int32_t j2 = ind2[i2];
        if (j1 == j2) {
            float val = data1[i1] * data2[i2];
            if (val != 0.0f) {
                result_ind.push_back(j1);
                result_data.push_back(val);
            }
            ++i1;
            ++i2;
        } else if (j1 < j2) {
            ++i1;
        } else {
            ++i2;
        }
    }
}

// ------------------------------------------------------------------- metrics
// The kernels below inline the sorted merges instead of materialising the
// sparse_diff / sparse_mul temporaries: the accumulated terms are the same
// float32 values in the same (index-sorted) order, so results are
// bit-identical while avoiding per-call allocation. Terms that the Python
// aux ops drop for being exactly zero contribute nothing to any of these
// accumulations.

namespace {

// Merge |x - y| difference values in float32, invoking `accum` on each.
template <typename F>
inline void for_each_diff(const int32_t* ind1, const float* data1, int64_t n1,
                          const int32_t* ind2, const float* data2, int64_t n2,
                          F&& accum) {
    int64_t i1 = 0, i2 = 0;
    while (i1 < n1 && i2 < n2) {
        int32_t j1 = ind1[i1];
        int32_t j2 = ind2[i2];
        if (j1 == j2) {
            accum(data1[i1] + (-data2[i2]));
            ++i1;
            ++i2;
        } else if (j1 < j2) {
            accum(data1[i1]);
            ++i1;
        } else {
            accum(-data2[i2]);
            ++i2;
        }
    }
    while (i1 < n1) accum(data1[i1++]);
    while (i2 < n2) accum(-data2[i2++]);
}

// Merge x * y product values (index intersection) in float32.
template <typename F>
inline void for_each_mul(const int32_t* ind1, const float* data1, int64_t n1,
                         const int32_t* ind2, const float* data2, int64_t n2,
                         F&& accum) {
    int64_t i1 = 0, i2 = 0;
    while (i1 < n1 && i2 < n2) {
        int32_t j1 = ind1[i1];
        int32_t j2 = ind2[i2];
        if (j1 == j2) {
            accum(data1[i1] * data2[i2]);
            ++i1;
            ++i2;
        } else if (j1 < j2) {
            ++i1;
        } else {
            ++i2;
        }
    }
}

}  // namespace

float sparse_euclidean(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2) {
    // result = 0.0 (f8); result += aux_data[i] ** 2 (f4 square promoted to
    // f8); np.sqrt in f8, cast on return.
    double result = 0.0;
    for_each_diff(ind1, data1, n1, ind2, data2, n2, [&](float v) {
        float sq = v * v;
        result += static_cast<double>(sq);
    });
    return static_cast<float>(std::sqrt(result));
}

float sparse_dot_product(const int32_t* ind1, const float* data1, int64_t n1,
                         const int32_t* ind2, const float* data2, int64_t n2) {
    // sparse.py::sparse_dot_product: f4 result/val locals, merge over the
    // index intersection with early returns on exhaustion. The numba version
    // reads ind1[0]/ind2[0] unconditionally (UB for empty inputs); we guard.
    if (n1 == 0 || n2 == 0) return 0.0f;
    float result = 0.0f;
    int64_t i1 = 0, i2 = 0;
    int32_t j1 = ind1[i1];
    int32_t j2 = ind2[i2];
    while (true) {
        if (j1 == j2) {
            float val = data1[i1] * data2[i2];
            result += val;
            ++i1;
            if (i1 >= n1) return result;
            j1 = ind1[i1];
            ++i2;
            if (i2 >= n2) return result;
            j2 = ind2[i2];
        } else if (j1 < j2) {
            ++i1;
            if (i1 >= n1) return result;
            j1 = ind1[i1];
        } else {
            ++i2;
            if (i2 >= n2) return result;
            j2 = ind2[i2];
        }
    }
}

float sparse_squared_euclidean(const int32_t* ind1, const float* data1,
                               int64_t n1, const int32_t* ind2,
                               const float* data2, int64_t n2) {
    // Explicit f4 signature with result: float32 -- pure float32 accumulation.
    float result = 0.0f;
    for_each_diff(ind1, data1, n1, ind2, data2, n2,
                  [&](float v) { result += v * v; });
    return result;
}

float sparse_manhattan(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2) {
    // result = 0.0 (f8); += np.abs(f4 value) promoted to f8.
    double result = 0.0;
    for_each_diff(ind1, data1, n1, ind2, data2, n2,
                  [&](float v) { result += static_cast<double>(std::fabs(v)); });
    return static_cast<float>(result);
}

float sparse_cosine(const int32_t* ind1, const float* data1, int64_t n1,
                    const int32_t* ind2, const float* data2, int64_t n2) {
    // result = 0.0 (f8) accumulating f4 products; norms are f4 (utils.norm);
    // the final expression evaluates norm1*norm2 in f4, the division and
    // subtraction in f8.
    double result = 0.0;
    float norm1 = norm(data1, n1);
    float norm2 = norm(data2, n2);
    for_each_mul(ind1, data1, n1, ind2, data2, n2,
                 [&](float v) { result += static_cast<double>(v); });

    if (norm1 == 0.0f && norm2 == 0.0f) {
        return 0.0f;
    } else if (norm1 == 0.0f || norm2 == 0.0f) {
        return 1.0f;
    } else {
        float norm_prod = norm1 * norm2;
        return static_cast<float>(1.0 - (result / static_cast<double>(norm_prod)));
    }
}

float sparse_alternative_cosine(const int32_t* ind1, const float* data1,
                                int64_t n1, const int32_t* ind2,
                                const float* data2, int64_t n2) {
    // locals: result/norm_x/norm_y are float32 -- all-f4 arithmetic.
    float result = 0.0f;
    float norm_x = norm(data1, n1);
    float norm_y = norm(data2, n2);
    for_each_mul(ind1, data1, n1, ind2, data2, n2,
                 [&](float v) { result += v; });

    if (norm_x == 0.0f && norm_y == 0.0f) {
        return 0.0f;
    } else if (norm_x == 0.0f || norm_y == 0.0f) {
        return kFloat32Max;
    } else if (result <= 0.0f) {
        return kFloat32Max;
    } else {
        result = (norm_x * norm_y) / result;
        return std::log2(result);
    }
}

float sparse_sqrt_correction(float d) {
    // np.sqrt on a float32 array: single-precision sqrt.
    return std::sqrt(d);
}

float sparse_correct_alternative_cosine(float d) {
    // @numba.vectorize: isclose(0.0, abs(d), atol=1e-7) uses the default
    // rtol=1e-5: |0 - |d|| <= 1e-7 + 1e-5 * |d| (evaluated in f8).
    double ad = std::fabs(static_cast<double>(d));
    if (ad <= 1e-7 + 1.0e-5 * ad || d < 0.0f) {
        return 0.0f;
    }
    return static_cast<float>(1.0 - std::pow(2.0, -static_cast<double>(d)));
}

// ------------------------------------------------------------------ registry

SparseMetricInfo resolve_sparse_metric(const std::string& name) {
    SparseMetricInfo m;
    if (name == "euclidean" || name == "l2") {
        m.dist = sparse_squared_euclidean;  // sparse_fast_distance_alternatives
        m.correction = sparse_sqrt_correction;  // np.sqrt
        m.true_dist = sparse_euclidean;
        m.angular_trees = false;
    } else if (name == "sqeuclidean") {
        m.dist = sparse_squared_euclidean;
        m.correction = nullptr;
        m.true_dist = sparse_squared_euclidean;
        m.angular_trees = false;
    } else if (name == "manhattan" || name == "taxicab" || name == "l1") {
        m.dist = sparse_manhattan;
        m.correction = nullptr;
        m.true_dist = sparse_manhattan;
        m.angular_trees = false;
    } else if (name == "cosine") {
        m.dist = sparse_alternative_cosine;  // sparse_fast_distance_alternatives
        m.correction = sparse_correct_alternative_cosine;
        m.true_dist = sparse_cosine;
        m.angular_trees = true;  // pynndescent angular metric list
    } else {
        throw std::invalid_argument(
            "umap::spdist::resolve_sparse_metric: metric '" + name +
            "' not implemented for sparse input (supported: euclidean/l2, "
            "sqeuclidean, manhattan/l1/taxicab, cosine)");
    }
    return m;
}

}  // namespace spdist
}  // namespace umap
