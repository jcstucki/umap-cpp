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

// Sorted-merge intersection size (pynndescent's fast_intersection_size /
// umap's arr_intersect(...).shape[0] -- identical counts for CSR rows).
inline int64_t intersection_size(const int32_t* ind1, int64_t n1,
                                 const int32_t* ind2, int64_t n2) {
    int64_t i1 = 0, i2 = 0, count = 0;
    while (i1 < n1 && i2 < n2) {
        if (ind1[i1] == ind2[i2]) {
            ++count;
            ++i1;
            ++i2;
        } else if (ind1[i1] < ind2[i2]) {
            ++i1;
        } else {
            ++i2;
        }
    }
    return count;
}

// arr_union(ind1, ind2).shape[0] without materialising the union.
inline int64_t union_size(const int32_t* ind1, int64_t n1, const int32_t* ind2,
                          int64_t n2) {
    return n1 + n2 - intersection_size(ind1, n1, ind2, n2);
}

// numba np.sum over a float32 *array* inlined into a fastmath=True kernel:
// LLVM vectorizes the reduction (VF=4, IC=1 on this target) -- a 4-lane
// accumulator, (l0 + l1) + (l2 + l3) shuffle reduction, scalar tail, and a
// purely scalar loop when n < 4. Verified bit-exact against the reference
// numba (arm64 macOS) for n in [0, 40] x 30 random trials; codegen-dependent
// upstream, so exactness is empirical (documented deviation risk). Loops over
// numba *typed lists* (sparse_mul results) are NOT vectorized and stay
// sequential -- do not use this for those.
inline float fastmath_f4_sum(const float* a, int64_t n) {
    if (n < 4) {
        float acc = 0.0f;
        for (int64_t i = 0; i < n; ++i) acc += a[i];
        return acc;
    }
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
    const int64_t nv = n - (n % 4);
    for (int64_t i = 0; i < nv; i += 4) {
        l0 += a[i];
        l1 += a[i + 1];
        l2 += a[i + 2];
        l3 += a[i + 3];
    }
    float red = (l0 + l1) + (l2 + l3);
    for (int64_t i = nv; i < n; ++i) red += a[i];
    return red;
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

float sparse_chebyshev(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2) {
    // result = 0.0 (f8); result = max(result, np.abs(f4 value)) -- the f4
    // absolute value is exact, the max compares in f8.
    double result = 0.0;
    for_each_diff(ind1, data1, n1, ind2, data2, n2, [&](float v) {
        result = std::max(result, static_cast<double>(std::fabs(v)));
    });
    return static_cast<float>(result);
}

float sparse_minkowski(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2,
                       double p) {
    // result = 0.0 (f8); += np.abs(f4)**p -- f4 ** f8 promotes to an f8 pow;
    // final result ** (1.0 / p) in f8.
    double result = 0.0;
    for_each_diff(ind1, data1, n1, ind2, data2, n2, [&](float v) {
        result += std::pow(static_cast<double>(std::fabs(v)), p);
    });
    return static_cast<float>(std::pow(result, 1.0 / p));
}

float sparse_hamming(const int32_t* ind1, const float* data1, int64_t n1,
                     const int32_t* ind2, const float* data2, int64_t n2,
                     double n_features) {
    // num_not_equal = sparse_diff(...)[0].shape[0]: exact-zero differences
    // (equal stored values) are DROPPED by sparse_sum, so they don't count.
    int64_t num_not_equal = 0;
    for_each_diff(ind1, data1, n1, ind2, data2, n2,
                  [&](float v) { num_not_equal += (v != 0.0f); });
    return static_cast<float>(static_cast<double>(num_not_equal) / n_features);
}

float sparse_canberra(const int32_t* ind1, const float* data1, int64_t n1,
                      const int32_t* ind2, const float* data2, int64_t n2) {
    // denom = sparse_sum(|x|, |y|) (f4, exact zeros dropped);
    // denom = (1.0 / denom).astype(np.float32)  [f8 reciprocal, f4 store];
    // numer = |sparse_diff(x, y)| (f4);
    // vals = sparse_mul(numer, denom) (f4 products, exact zeros dropped);
    // result = 0.0 (f8) += val.
    std::vector<int32_t> denom_ind, numer_ind, val_ind;
    std::vector<float> denom_data, numer_data, val_data;
    std::vector<float> abs1(data1, data1 + n1), abs2(data2, data2 + n2);
    for (float& v : abs1) v = std::fabs(v);
    for (float& v : abs2) v = std::fabs(v);
    sparse_sum(ind1, abs1.data(), n1, ind2, abs2.data(), n2, denom_ind,
               denom_data);
    for (float& v : denom_data)
        v = static_cast<float>(1.0 / static_cast<double>(v));
    sparse_diff(ind1, data1, n1, ind2, data2, n2, numer_ind, numer_data);
    for (float& v : numer_data) v = std::fabs(v);
    sparse_mul(numer_ind.data(), numer_data.data(),
               static_cast<int64_t>(numer_ind.size()), denom_ind.data(),
               denom_data.data(), static_cast<int64_t>(denom_ind.size()),
               val_ind, val_data);
    double result = 0.0;
    for (float v : val_data) result += static_cast<double>(v);
    return static_cast<float>(result);
}

float sparse_bray_curtis(const int32_t* ind1, const float* data1, int64_t n1,
                         const int32_t* ind2, const float* data2, int64_t n2) {
    // Explicit f4 signature + fastmath upstream: np.sum of the f4 |sum| /
    // |diff| arrays takes numba's *vectorized* fastmath reduction
    // (fastmath_f4_sum); float(numerator) / denominator is an f8 division
    // cast back to f4 on return. Empty-union and zero denominators -> 0.0.
    std::vector<int32_t> tmp_ind;
    std::vector<float> denom_data;
    sparse_sum(ind1, data1, n1, ind2, data2, n2, tmp_ind, denom_data);
    if (denom_data.empty()) return 0.0f;
    for (float& v : denom_data) v = std::fabs(v);
    float denominator = fastmath_f4_sum(denom_data.data(),
                                        static_cast<int64_t>(denom_data.size()));
    if (denominator == 0.0f) return 0.0f;
    std::vector<float> numer_data;
    sparse_diff(ind1, data1, n1, ind2, data2, n2, tmp_ind, numer_data);
    for (float& v : numer_data) v = std::fabs(v);
    float numerator = fastmath_f4_sum(numer_data.data(),
                                      static_cast<int64_t>(numer_data.size()));
    return static_cast<float>(static_cast<double>(numerator) /
                              static_cast<double>(denominator));
}

float sparse_jaccard(const int32_t* ind1, const float* data1, int64_t n1,
                     const int32_t* ind2, const float* data2, int64_t n2) {
    (void)data1;
    (void)data2;
    int64_t num_equal = intersection_size(ind1, n1, ind2, n2);
    int64_t num_non_zero = n1 + n2 - num_equal;
    if (num_non_zero == 0) return 0.0f;
    return static_cast<float>(static_cast<double>(num_non_zero - num_equal) /
                              static_cast<double>(num_non_zero));
}

float sparse_matching(const int32_t* ind1, const float* data1, int64_t n1,
                      const int32_t* ind2, const float* data2, int64_t n2,
                      double n_features) {
    (void)data1;
    (void)data2;
    int64_t num_true_true = intersection_size(ind1, n1, ind2, n2);
    int64_t num_not_equal = n1 + n2 - 2 * num_true_true;
    return static_cast<float>(static_cast<double>(num_not_equal) / n_features);
}

float sparse_dice(const int32_t* ind1, const float* data1, int64_t n1,
                  const int32_t* ind2, const float* data2, int64_t n2) {
    (void)data1;
    (void)data2;
    int64_t num_true_true = intersection_size(ind1, n1, ind2, n2);
    int64_t num_not_equal = n1 + n2 - 2 * num_true_true;
    if (num_not_equal == 0) return 0.0f;
    return static_cast<float>(
        static_cast<double>(num_not_equal) /
        (2.0 * static_cast<double>(num_true_true) + num_not_equal));
}

float sparse_kulsinski(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2,
                       double n_features) {
    (void)data1;
    (void)data2;
    int64_t num_true_true = intersection_size(ind1, n1, ind2, n2);
    int64_t num_not_equal = n1 + n2 - 2 * num_true_true;
    if (num_not_equal == 0) return 0.0f;
    return static_cast<float>(
        (static_cast<double>(num_not_equal - num_true_true) + n_features) /
        (static_cast<double>(num_not_equal) + n_features));
}

float sparse_rogers_tanimoto(const int32_t* ind1, const float* data1,
                             int64_t n1, const int32_t* ind2,
                             const float* data2, int64_t n2,
                             double n_features) {
    (void)data1;
    (void)data2;
    int64_t num_not_equal =
        n1 + n2 - 2 * intersection_size(ind1, n1, ind2, n2);
    return static_cast<float>((2.0 * static_cast<double>(num_not_equal)) /
                              (n_features + num_not_equal));
}

float sparse_russellrao(const int32_t* ind1, const float* data1, int64_t n1,
                        const int32_t* ind2, const float* data2, int64_t n2,
                        double n_features) {
    // if ind1.shape[0] == ind2.shape[0] and np.all(ind1 == ind2): 0.0
    if (n1 == n2) {
        bool all_eq = true;
        for (int64_t i = 0; i < n1; ++i)
            if (ind1[i] != ind2[i]) {
                all_eq = false;
                break;
            }
        if (all_eq) return 0.0f;
    }
    int64_t num_true_true = intersection_size(ind1, n1, ind2, n2);
    // np.sum(data != 0): stored values, so explicit zeros do not count.
    int64_t nz1 = 0, nz2 = 0;
    for (int64_t i = 0; i < n1; ++i) nz1 += (data1[i] != 0.0f);
    for (int64_t i = 0; i < n2; ++i) nz2 += (data2[i] != 0.0f);
    if (num_true_true == nz1 && num_true_true == nz2) return 0.0f;
    return static_cast<float>(
        (n_features - static_cast<double>(num_true_true)) / n_features);
}

float sparse_sokal_michener(const int32_t* ind1, const float* data1,
                            int64_t n1, const int32_t* ind2,
                            const float* data2, int64_t n2,
                            double n_features) {
    return sparse_rogers_tanimoto(ind1, data1, n1, ind2, data2, n2,
                                  n_features);  // identical formula upstream
}

float sparse_sokal_sneath(const int32_t* ind1, const float* data1, int64_t n1,
                          const int32_t* ind2, const float* data2,
                          int64_t n2) {
    (void)data1;
    (void)data2;
    int64_t num_true_true = intersection_size(ind1, n1, ind2, n2);
    int64_t num_not_equal = n1 + n2 - 2 * num_true_true;
    if (num_not_equal == 0) return 0.0f;
    return static_cast<float>(
        static_cast<double>(num_not_equal) /
        (0.5 * static_cast<double>(num_true_true) + num_not_equal));
}

float sparse_correlation(const int32_t* ind1, const float* data1, int64_t n1,
                         const int32_t* ind2, const float* data2, int64_t n2,
                         double n_features) {
    // Faithful to the numba typing: mu_x/mu_y/dot_product/norm1/norm2 in f8,
    // shifted_data arrays stored f4, norm(shifted)**2 squared in *f4*
    // (float32 ** 2 stays float32 in numba), sparse_mul products f4 with
    // exact zeros dropped from common_indices.
    if (n1 == 0 && n2 == 0) return 0.0f;
    if (n1 == 0 || n2 == 0) return 1.0f;

    double mu_x = 0.0, mu_y = 0.0;
    for (int64_t i = 0; i < n1; ++i) mu_x += static_cast<double>(data1[i]);
    for (int64_t i = 0; i < n2; ++i) mu_y += static_cast<double>(data2[i]);
    mu_x /= n_features;
    mu_y /= n_features;

    std::vector<float> shifted1(static_cast<size_t>(n1));
    std::vector<float> shifted2(static_cast<size_t>(n2));
    for (int64_t i = 0; i < n1; ++i)
        shifted1[i] = static_cast<float>(static_cast<double>(data1[i]) - mu_x);
    for (int64_t i = 0; i < n2; ++i)
        shifted2[i] = static_cast<float>(static_cast<double>(data2[i]) - mu_y);

    // (numba emits separate mul/add -- keep statements split so clang's
    // default fp-contract cannot fuse them into fmas)
    float norm1_f4 = norm(shifted1.data(), n1);
    float norm2_f4 = norm(shifted2.data(), n2);
    float nsq1 = norm1_f4 * norm1_f4;  // f4 ** 2 stays f4 in numba
    float nsq2 = norm2_f4 * norm2_f4;
    double tail1 = (n_features - static_cast<double>(n1)) * (mu_x * mu_x);
    double tail2 = (n_features - static_cast<double>(n2)) * (mu_y * mu_y);
    double norm1 = std::sqrt(static_cast<double>(nsq1) + tail1);
    double norm2 = std::sqrt(static_cast<double>(nsq2) + tail2);

    std::vector<int32_t> prod_ind;
    std::vector<float> prod_data;
    sparse_mul(ind1, shifted1.data(), n1, ind2, shifted2.data(), n2, prod_ind,
               prod_data);

    double dot_product = 0.0;
    for (float v : prod_data) dot_product += static_cast<double>(v);

    // common_indices = set(dot_prod_inds); both ind arrays and prod_ind are
    // sorted, so membership is a pointer walk.
    for (int64_t i = 0, k = 0; i < n1; ++i) {
        while (k < static_cast<int64_t>(prod_ind.size()) &&
               prod_ind[k] < ind1[i])
            ++k;
        bool common =
            k < static_cast<int64_t>(prod_ind.size()) && prod_ind[k] == ind1[i];
        if (!common) {
            double prod = static_cast<double>(shifted1[i]) * mu_y;
            dot_product -= prod;  // no fma (numba emits mul then sub)
        }
    }
    for (int64_t i = 0, k = 0; i < n2; ++i) {
        while (k < static_cast<int64_t>(prod_ind.size()) &&
               prod_ind[k] < ind2[i])
            ++k;
        bool common =
            k < static_cast<int64_t>(prod_ind.size()) && prod_ind[k] == ind2[i];
        if (!common) {
            double prod = static_cast<double>(shifted2[i]) * mu_x;
            dot_product -= prod;
        }
    }
    int64_t n_union = union_size(ind1, n1, ind2, n2);
    {
        double prod =
            mu_x * mu_y * (n_features - static_cast<double>(n_union));
        dot_product += prod;
    }

    if (norm1 == 0.0 && norm2 == 0.0) return 0.0f;
    if (dot_product == 0.0) return 1.0f;
    return static_cast<float>(1.0 - (dot_product / (norm1 * norm2)));
}

float sparse_hellinger(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2) {
    // result = 0.0 (f8) accumulating np.sqrt(f4 product) values (f4 sqrt);
    // norms are numba np.sum(f4 array) -- sequential float32;
    // sqrt_norm_prod = np.sqrt(f4 * f4) stays f4.
    double result = 0.0;
    float norm1 = 0.0f, norm2 = 0.0f;
    for (int64_t i = 0; i < n1; ++i) norm1 += data1[i];
    for (int64_t i = 0; i < n2; ++i) norm2 += data2[i];
    float sqrt_norm_prod = std::sqrt(norm1 * norm2);
    for_each_mul(ind1, data1, n1, ind2, data2, n2, [&](float v) {
        result += static_cast<double>(std::sqrt(v));
    });
    if (norm1 == 0.0f && norm2 == 0.0f) return 0.0f;
    if (norm1 == 0.0f || norm2 == 0.0f) return 1.0f;
    if (result > static_cast<double>(sqrt_norm_prod)) return 0.0f;
    return static_cast<float>(
        std::sqrt(1.0 - (result / static_cast<double>(sqrt_norm_prod))));
}

// ---------------------------------------------------------------- dirichlet
// float32-input replicas of umap/sparse.py's approx_log_Gamma / log_beta /
// log_single_beta as numba types them for f4 data (the dense port's versions
// take f8 inputs, which rounds differently in the f4 subexpressions).
namespace {

inline double sp_approx_log_Gamma(float x) {
    if (x == 1.0f) return 0.0;
    // x*np.log(x) - x evaluates in f4 (left-assoc), the remaining terms in
    // f8. Statements kept split so clang cannot contract mul+add into fmas
    // (numba emits separate ops).
    float xl = x * std::log(x);
    float t = xl - x;
    double xd = static_cast<double>(x);
    double r = static_cast<double>(t);
    double mid = 0.5 * std::log(2.0 * M_PI / xd);
    r += mid;
    r += 1.0 / (xd * 12.0);
    return r;
}

inline double sp_log_beta(float x, float y) {
    float a = std::min(x, y);
    float b = std::max(x, y);
    if (b < 5.0f) {
        // value unifies to f8 (loop adds f8 terms); the initial -np.log(b)
        // is still computed in f4 then widened.
        double value = static_cast<double>(-std::log(b));
        for (int64_t i = 1; i < static_cast<int64_t>(a); ++i)
            value += std::log(static_cast<double>(i)) -
                     std::log(static_cast<double>(b) + static_cast<double>(i));
        return value;
    }
    return sp_approx_log_Gamma(x) + sp_approx_log_Gamma(y) -
           sp_approx_log_Gamma(x + y);  // x + y in f4
}

inline double sp_log_single_beta(float x) {
    double xd = static_cast<double>(x);
    double lin = -2.0 * xd;
    lin += 0.5;
    double t1 = std::log(2.0) * lin;
    double t2 = 0.5 * std::log(2.0 * M_PI / xd);
    double r = t1 + t2;
    r += 0.125 / xd;
    return r;
}

}  // namespace

float sparse_ll_dirichlet(const int32_t* ind1, const float* data1, int64_t n1,
                          const int32_t* ind2, const float* data2,
                          int64_t n2) {
    // n1/n2 are numba np.sum(f4 array): sequential float32.
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int64_t i = 0; i < n1; ++i) sum1 += data1[i];
    for (int64_t i = 0; i < n2; ++i) sum2 += data2[i];
    if (sum1 == 0.0f && sum2 == 0.0f) return 0.0f;
    if (sum1 == 0.0f || sum2 == 0.0f) return 1e8f;

    double log_b = 0.0;
    int64_t i1 = 0, i2 = 0;
    while (i1 < n1 && i2 < n2) {
        if (ind1[i1] == ind2[i2]) {
            if (data1[i1] * data2[i2] != 0.0f)
                log_b += sp_log_beta(data1[i1], data2[i2]);
            ++i1;
            ++i2;
        } else if (ind1[i1] < ind2[i2]) {
            ++i1;
        } else {
            ++i2;
        }
    }
    double self_denom1 = 0.0, self_denom2 = 0.0;
    for (int64_t i = 0; i < n1; ++i) self_denom1 += sp_log_single_beta(data1[i]);
    for (int64_t i = 0; i < n2; ++i) self_denom2 += sp_log_single_beta(data2[i]);

    // (statements split to keep clang from fusing the 1/n * (...) products
    // with the outer addition)
    double term2 = log_b - sp_log_beta(sum1, sum2) -
                   (self_denom2 - sp_log_single_beta(sum2));
    double term1 = log_b - sp_log_beta(sum2, sum1) -
                   (self_denom1 - sp_log_single_beta(sum1));
    double s2 = 1.0 / static_cast<double>(sum2) * term2;
    double s1 = 1.0 / static_cast<double>(sum1) * term1;
    return static_cast<float>(std::sqrt(s2 + s1));
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

float sparse_alternative_jaccard(const int32_t* ind1, const float* data1,
                                 int64_t n1, const int32_t* ind2,
                                 const float* data2, int64_t n2) {
    (void)data1;
    (void)data2;
    // locals num_equal/num_non_zero are intp; num_equal / num_non_zero is a
    // true (f8) division; -np.log2 in f8, cast to f4 by the signature.
    int64_t num_equal = intersection_size(ind1, n1, ind2, n2);
    int64_t num_non_zero = n1 + n2 - num_equal;
    if (num_non_zero == 0) return 0.0f;
    if (num_equal == 0) return kFloat32Max;
    return static_cast<float>(-std::log2(static_cast<double>(num_equal) /
                                         static_cast<double>(num_non_zero)));
}

float sparse_alternative_hellinger(const int32_t* ind1, const float* data1,
                                   int64_t n1, const int32_t* ind2,
                                   const float* data2, int64_t n2) {
    // locals: result / l1_norm_x / l1_norm_y are float32; np.sqrt and
    // np.log2 of f4 stay f4 -- an all-f4 kernel like alternative_cosine.
    // np.sum(data) over the row arrays inlines with fastmath -> vectorized
    // reduction; the result loop iterates a typed List -> sequential.
    float result = 0.0f;
    float l1_norm_x = fastmath_f4_sum(data1, n1);
    float l1_norm_y = fastmath_f4_sum(data2, n2);
    for_each_mul(ind1, data1, n1, ind2, data2, n2,
                 [&](float v) { result += std::sqrt(v); });
    if (l1_norm_x == 0.0f && l1_norm_y == 0.0f) return 0.0f;
    if (l1_norm_x == 0.0f || l1_norm_y == 0.0f) return kFloat32Max;
    if (result <= 0.0f) return kFloat32Max;
    result = std::sqrt(l1_norm_x * l1_norm_y) / result;
    return std::log2(result);
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

float sparse_correct_alternative_jaccard(float d) {
    // @numba.vectorize correct_alternative_jaccard: 1.0 - pow(2.0, -v) in
    // f8 (no isclose guard, no negative guard -- unlike cosine/hellinger).
    return static_cast<float>(1.0 - std::pow(2.0, -static_cast<double>(d)));
}

float sparse_correct_alternative_hellinger(float d) {
    // Same isclose(0, |d|, atol=1e-7) + negative guard as the cosine
    // correction, with np.sqrt around the corrected similarity.
    double ad = std::fabs(static_cast<double>(d));
    if (ad <= 1e-7 + 1.0e-5 * ad || d < 0.0f) {
        return 0.0f;
    }
    return static_cast<float>(
        std::sqrt(1.0 - std::pow(2.0, -static_cast<double>(d))));
}

// ------------------------------------------------------------------ registry

SparseMetricInfo resolve_sparse_metric(const std::string& name,
                                       int64_t n_features,
                                       const SparseMetricKwds& kwds) {
    // Mirrors NNDescent.__init__'s sparse branch: sparse_named_distances
    // lookup, fast alternative + correction when one exists, n_features
    // bound for sparse_need_n_features metrics, and the name-based angular
    // metric list ("cosine", "dot", "correlation", "dice", "jaccard",
    // "hellinger", "hamming", ...). ll_dirichlet is umap-only: NN-descent
    // upstream receives it as a callable (euclidean trees unless UMAP's
    // angular_rp_forest is set -- angular_from_param).
    auto need_n_features = [&](const char* metric) -> double {
        if (n_features <= 0)
            throw std::invalid_argument(
                std::string("umap::spdist::resolve_sparse_metric: metric '") +
                metric +
                "' requires n_features (the number of data columns) > 0");
        return static_cast<double>(n_features);
    };

    SparseMetricInfo m;
    if (name == "euclidean" || name == "l2") {
        m.dist = sparse_squared_euclidean;  // sparse_fast_distance_alternatives
        m.correction = sparse_sqrt_correction;  // np.sqrt
        m.true_dist = sparse_euclidean;
    } else if (name == "sqeuclidean") {
        m.dist = sparse_squared_euclidean;
        m.true_dist = sparse_squared_euclidean;
    } else if (name == "manhattan" || name == "taxicab" || name == "l1") {
        m.dist = sparse_manhattan;
        m.true_dist = sparse_manhattan;
    } else if (name == "chebyshev" || name == "linf" || name == "linfty" ||
               name == "linfinity") {
        m.dist = sparse_chebyshev;
        m.true_dist = sparse_chebyshev;
    } else if (name == "minkowski") {
        m.dist = SparseDist(sparse_minkowski, kwds.p);
        m.true_dist = m.dist;
    } else if (name == "canberra") {
        m.dist = sparse_canberra;
        m.true_dist = sparse_canberra;
    } else if (name == "braycurtis") {
        m.dist = sparse_bray_curtis;
        m.true_dist = sparse_bray_curtis;
    } else if (name == "hamming") {
        m.dist = SparseDist(sparse_hamming, need_n_features("hamming"));
        m.true_dist = m.dist;
        m.angular_trees = true;
    } else if (name == "jaccard") {
        m.dist = sparse_alternative_jaccard;  // fast alternative
        m.correction = sparse_correct_alternative_jaccard;
        m.true_dist = sparse_jaccard;
        m.angular_trees = true;
    } else if (name == "dice") {
        m.dist = sparse_dice;
        m.true_dist = sparse_dice;
        m.angular_trees = true;
    } else if (name == "matching") {
        m.dist = SparseDist(sparse_matching, need_n_features("matching"));
        m.true_dist = m.dist;
    } else if (name == "kulsinski") {
        m.dist = SparseDist(sparse_kulsinski, need_n_features("kulsinski"));
        m.true_dist = m.dist;
    } else if (name == "rogerstanimoto") {
        m.dist = SparseDist(sparse_rogers_tanimoto,
                            need_n_features("rogerstanimoto"));
        m.true_dist = m.dist;
    } else if (name == "russellrao") {
        m.dist = SparseDist(sparse_russellrao, need_n_features("russellrao"));
        m.true_dist = m.dist;
    } else if (name == "sokalmichener") {
        m.dist = SparseDist(sparse_sokal_michener,
                            need_n_features("sokalmichener"));
        m.true_dist = m.dist;
    } else if (name == "sokalsneath") {
        m.dist = sparse_sokal_sneath;
        m.true_dist = sparse_sokal_sneath;
    } else if (name == "cosine") {
        m.dist = sparse_alternative_cosine;  // sparse_fast_distance_alternatives
        m.correction = sparse_correct_alternative_cosine;
        m.true_dist = sparse_cosine;
        m.angular_trees = true;
    } else if (name == "correlation") {
        m.dist = SparseDist(sparse_correlation, need_n_features("correlation"));
        m.true_dist = m.dist;
        m.angular_trees = true;
    } else if (name == "hellinger") {
        m.dist = sparse_alternative_hellinger;  // fast alternative
        m.correction = sparse_correct_alternative_hellinger;
        m.true_dist = sparse_hellinger;
        m.angular_trees = true;
    } else if (name == "ll_dirichlet") {
        // umap/sparse.py only; a *callable* to NN-descent upstream.
        m.dist = sparse_ll_dirichlet;
        m.true_dist = sparse_ll_dirichlet;
        m.angular_from_param = true;
    } else {
        throw std::invalid_argument(
            "umap::spdist::resolve_sparse_metric: metric '" + name +
            "' not implemented for sparse input (absent upstream or not "
            "ported -- see sparse_distances.hpp for the list)");
    }
    return m;
}

}  // namespace spdist
}  // namespace umap
