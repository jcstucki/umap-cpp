// Implementation of include/umap/nn_descent.hpp. See that header for scope
// and faithfulness notes. Line references in comments are to
// reference/pynndescent/pynndescent/{utils,rp_trees,pynndescent_}.py.

#include "umap/nn_descent.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "umap/rng.hpp"

namespace umap {
namespace nnd {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kFloat32Max = std::numeric_limits<float>::max();
constexpr double kEps = 1e-8;  // rp_trees.py EPS

// numba np.abs on int32 (INT32_MIN stays INT32_MIN, like C).
inline int32_t iabs32(int32_t v) {
    return v < 0 ? static_cast<int32_t>(0u - static_cast<uint32_t>(v)) : v;
}

// utils.py::norm -- float64 accumulation, float32 return.
inline float l2_norm(const float* vec, int64_t dim) {
    double result = 0.0;
    for (int64_t i = 0; i < dim; ++i)
        result += static_cast<double>(vec[i]) * static_cast<double>(vec[i]);
    return static_cast<float>(std::sqrt(result));
}

inline float l2_norm_strided(const float* row, int64_t dim) { return l2_norm(row, dim); }

}  // namespace

// ---------------------------------------------------------------- heap utils

NNHeap NNHeap::make(int64_t n_points, int64_t size) {
    NNHeap h;
    h.n_points = n_points;
    h.size = size;
    h.ind.assign(static_cast<size_t>(n_points * size), -1);
    h.dist.assign(static_cast<size_t>(n_points * size), kInf);
    h.flag.assign(static_cast<size_t>(n_points * size), 0);
    return h;
}

int simple_heap_push(float* priorities, int32_t* indices, int64_t size,
                     float p, int32_t n) {
    if (p >= priorities[0]) return 0;

    priorities[0] = p;
    indices[0] = n;

    int64_t i = 0;
    while (true) {
        int64_t ic1 = 2 * i + 1;
        int64_t ic2 = ic1 + 1;
        int64_t i_swap;

        if (ic1 >= size) {
            break;
        } else if (ic2 >= size) {
            if (priorities[ic1] > p) {
                i_swap = ic1;
            } else {
                break;
            }
        } else if (priorities[ic1] >= priorities[ic2]) {
            if (p < priorities[ic1]) {
                i_swap = ic1;
            } else {
                break;
            }
        } else {
            if (p < priorities[ic2]) {
                i_swap = ic2;
            } else {
                break;
            }
        }

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];
        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;
    return 1;
}

int checked_heap_push(float* priorities, int32_t* indices, int64_t size,
                      float p, int32_t n) {
    if (p >= priorities[0]) return 0;
    for (int64_t i = 0; i < size; ++i)
        if (n == indices[i]) return 0;
    return simple_heap_push(priorities, indices, size, p, n);
}

int checked_flagged_heap_push(float* priorities, int32_t* indices,
                              uint8_t* flags, int64_t size, float p, int32_t n,
                              uint8_t f) {
    if (p >= priorities[0]) return 0;
    for (int64_t i = 0; i < size; ++i)
        if (n == indices[i]) return 0;

    priorities[0] = p;
    indices[0] = n;
    flags[0] = f;

    int64_t i = 0;
    while (true) {
        int64_t ic1 = 2 * i + 1;
        int64_t ic2 = ic1 + 1;
        int64_t i_swap;

        if (ic1 >= size) {
            break;
        } else if (ic2 >= size) {
            if (priorities[ic1] > p) {
                i_swap = ic1;
            } else {
                break;
            }
        } else if (priorities[ic1] >= priorities[ic2]) {
            if (p < priorities[ic1]) {
                i_swap = ic1;
            } else {
                break;
            }
        } else {
            if (p < priorities[ic2]) {
                i_swap = ic2;
            } else {
                break;
            }
        }

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];
        flags[i] = flags[i_swap];
        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;
    flags[i] = f;
    return 1;
}

void siftdown(float* heap1, int32_t* heap2, int64_t size, int64_t elt) {
    while (elt * 2 + 1 < size) {
        int64_t left_child = elt * 2 + 1;
        int64_t right_child = left_child + 1;
        int64_t swap = elt;

        if (heap1[swap] < heap1[left_child]) swap = left_child;
        if (right_child < size && heap1[swap] < heap1[right_child])
            swap = right_child;

        if (swap == elt) {
            break;
        } else {
            std::swap(heap1[elt], heap1[swap]);
            std::swap(heap2[elt], heap2[swap]);
            elt = swap;
        }
    }
}

void deheap_sort(int32_t* indices, float* distances, int64_t n_points,
                 int64_t size) {
    for (int64_t i = 0; i < n_points; ++i) {
        int32_t* ind = indices + i * size;
        float* dist = distances + i * size;
        for (int64_t j = size - 1; j > 0; --j) {
            std::swap(ind[0], ind[j]);
            std::swap(dist[0], dist[j]);
            siftdown(dist, ind, j, 0);
        }
    }
}

// ------------------------------------------------------------------- metrics

float squared_euclidean(const float* x, const float* y, int64_t dim) {
    float result = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }
    return result;
}

float manhattan(const float* x, const float* y, int64_t dim) {
    float result = 0.0f;
    for (int64_t i = 0; i < dim; ++i) result += std::fabs(x[i] - y[i]);
    return result;
}

float alternative_cosine(const float* x, const float* y, int64_t dim) {
    float result = 0.0f;
    float norm_x = 0.0f;
    float norm_y = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        result += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }
    if (norm_x == 0.0f && norm_y == 0.0f) {
        return 0.0f;
    } else if (norm_x == 0.0f || norm_y == 0.0f) {
        return kFloat32Max;
    } else if (result <= 0.0f) {
        return kFloat32Max;
    } else {
        result = std::sqrt(norm_x * norm_y) / result;
        return std::log2(result);
    }
}

float true_euclidean(const float* x, const float* y, int64_t dim) {
    // distances.py::euclidean -- float64 accumulation under numba's default
    // lazy typing (result = 0.0), sqrt at the end.
    double result = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        double diff = static_cast<double>(x[i]) - static_cast<double>(y[i]);
        result += diff * diff;
    }
    return static_cast<float>(std::sqrt(result));
}

float true_cosine(const float* x, const float* y, int64_t dim) {
    double result = 0.0, norm_x = 0.0, norm_y = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        result += static_cast<double>(x[i]) * static_cast<double>(y[i]);
        norm_x += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        norm_y += static_cast<double>(y[i]) * static_cast<double>(y[i]);
    }
    if (norm_x == 0.0 && norm_y == 0.0) return 0.0f;
    if (norm_x == 0.0 || norm_y == 0.0) return 1.0f;
    return static_cast<float>(1.0 - (result / std::sqrt(norm_x * norm_y)));
}

namespace {

float sqrt_correction(float d) { return std::sqrt(d); }
// distances.py::correct_alternative_cosine: 1.0 - pow(2.0, -d) (computed in
// float64 by numba's lazy vectorize; we cast the result back to float32).
float alt_cosine_correction(float d) {
    return static_cast<float>(1.0 - std::pow(2.0, -static_cast<double>(d)));
}
float identity_sq(const float* x, const float* y, int64_t dim) {
    return squared_euclidean(x, y, dim);
}

// --- tests-port additions (branch tests-port) -------------------------------
// jaccard + hellinger support for the dense NN-descent path, required by the
// upstream test-suite port (test_umap_ops.py::test_disconnected_data runs
// UMAP(metric="jaccard"/"hellinger", force_approximation_algorithm=True)).
// Faithful ports of pynndescent/distances.py fast_distance_alternatives:
// alternative_jaccard / correct_alternative_jaccard and alternative_hellinger
// / correct_alternative_hellinger (numba float32 locals), plus the plain
// float32 kernels as true_dist. Both metrics are in pynndescent's
// angular-trees list (pynndescent_.py).
float alternative_jaccard(const float* x, const float* y, int64_t dim) {
    float num_non_zero = 0.0f;
    float num_equal = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        const bool x_true = x[i] != 0.0f;
        const bool y_true = y[i] != 0.0f;
        num_non_zero += (x_true || y_true) ? 1.0f : 0.0f;
        num_equal += (x_true && y_true) ? 1.0f : 0.0f;
    }
    if (num_non_zero == 0.0f) return 0.0f;
    return -std::log2(num_equal / num_non_zero);  // -log2(0) == +inf
}

float alt_jaccard_correction(float v) {
    return static_cast<float>(1.0 - std::pow(2.0, -static_cast<double>(v)));
}

float true_jaccard(const float* x, const float* y, int64_t dim) {
    float num_non_zero = 0.0f;
    float num_equal = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        const bool x_true = x[i] != 0.0f;
        const bool y_true = y[i] != 0.0f;
        num_non_zero += (x_true || y_true) ? 1.0f : 0.0f;
        num_equal += (x_true && y_true) ? 1.0f : 0.0f;
    }
    if (num_non_zero == 0.0f) return 0.0f;
    return (num_non_zero - num_equal) / num_non_zero;
}

float alternative_hellinger(const float* x, const float* y, int64_t dim) {
    float result = 0.0f;
    float l1_norm_x = 0.0f;
    float l1_norm_y = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        result += std::sqrt(x[i] * y[i]);
        l1_norm_x += x[i];
        l1_norm_y += y[i];
    }
    if (l1_norm_x == 0.0f && l1_norm_y == 0.0f) {
        return 0.0f;
    } else if (l1_norm_x == 0.0f || l1_norm_y == 0.0f) {
        return kFloat32Max;
    } else if (result <= 0.0f) {
        return kFloat32Max;
    } else {
        result = std::sqrt(l1_norm_x * l1_norm_y) / result;
        return std::log2(result);
    }
}

float alt_hellinger_correction(float d) {
    return static_cast<float>(
        std::sqrt(1.0 - std::pow(2.0, -static_cast<double>(d))));
}

float true_hellinger(const float* x, const float* y, int64_t dim) {
    float result = 0.0f;
    float l1_norm_x = 0.0f;
    float l1_norm_y = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        result += std::sqrt(x[i] * y[i]);
        l1_norm_x += x[i];
        l1_norm_y += y[i];
    }
    if (l1_norm_x == 0.0f && l1_norm_y == 0.0f) {
        return 0.0f;
    } else if (l1_norm_x == 0.0f || l1_norm_y == 0.0f) {
        return 1.0f;
    } else {
        return std::sqrt(1.0f - result / std::sqrt(l1_norm_x * l1_norm_y));
    }
}
// ----------------------------------------------------------------------------

}  // namespace

MetricInfo resolve_metric(const std::string& name) {
    MetricInfo m;
    if (name == "euclidean" || name == "l2") {
        m.dist = squared_euclidean;       // fast_distance_alternatives
        m.correction = sqrt_correction;   // np.sqrt
        m.true_dist = true_euclidean;
        m.angular_trees = false;
    } else if (name == "sqeuclidean") {
        m.dist = identity_sq;
        m.correction = nullptr;
        m.true_dist = identity_sq;
        m.angular_trees = false;
    } else if (name == "manhattan" || name == "taxicab" || name == "l1") {
        m.dist = manhattan;
        m.correction = nullptr;
        m.true_dist = manhattan;
        m.angular_trees = false;
    } else if (name == "cosine") {
        m.dist = alternative_cosine;      // fast_distance_alternatives
        m.correction = alt_cosine_correction;
        m.true_dist = true_cosine;
        m.angular_trees = true;           // pynndescent angular metric list
        // --- tests-port additions (branch tests-port): jaccard + hellinger --
    } else if (name == "jaccard") {
        m.dist = alternative_jaccard;     // fast_distance_alternatives
        m.correction = alt_jaccard_correction;
        m.true_dist = true_jaccard;
        m.angular_trees = true;           // pynndescent angular metric list
    } else if (name == "hellinger") {
        m.dist = alternative_hellinger;   // fast_distance_alternatives
        m.correction = alt_hellinger_correction;
        m.true_dist = true_hellinger;
        m.angular_trees = true;           // pynndescent angular metric list
        // --------------------------------------------------------------------
    } else {
        throw std::invalid_argument("umap::nnd::resolve_metric: unsupported metric '" +
                                    name + "'");
    }
    return m;
}

// --- densmap additions ------------------------------------------------------
MetricInfo resolve_metric(const std::string& name, const MetricKwds& kwds) {
    // Fast built-in metrics (none of them takes kwds upstream).
    if (name == "euclidean" || name == "l2" || name == "sqeuclidean" ||
        name == "manhattan" || name == "taxicab" || name == "l1" ||
        name == "cosine") {
        return resolve_metric(name);
    }
    // Generic-metric hook: pynndescent's no-fast-alternative branch. The
    // metric runs directly inside NN-descent (correction = identity) with the
    // kwds bound positionally, like `_partial_dist_func`.
    MetricInfo m;
    DistanceF32Fn f = make_metric_f32(name, kwds);
    DistFn wrapped = [f](const float* x, const float* y, int64_t dim) {
        return f(x, y, static_cast<std::size_t>(dim));
    };
    m.dist = wrapped;
    m.true_dist = wrapped;
    m.correction = nullptr;
    // pynndescent NNDescent.__init__ angular-trees metric list.
    m.angular_trees =
        (name == "cosine" || name == "dot" || name == "correlation" ||
         name == "dice" || name == "jaccard" || name == "hellinger" ||
         name == "hamming" || name == "bit_hamming" || name == "bit_jaccard");
    return m;
}
// ----------------------------------------------------------------------------

// --- hardening additions -----------------------------------------------------
// Callable-metric flavour (metric == "custom"): pynndescent treats a callable
// exactly like a no-fast-alternative metric -- the function runs raw inside
// NN-descent with no correction, and `metric in (...)` is False for the
// angular list, so euclidean rp-trees are used.
MetricInfo resolve_metric(const std::string& name, const MetricKwds& kwds,
                          const DistanceF32Fn* custom) {
    if (name != "custom" || custom == nullptr || !*custom)
        return resolve_metric(name, kwds);
    MetricInfo m;
    DistanceF32Fn f = *custom;
    DistFn wrapped = [f](const float* x, const float* y, int64_t dim) {
        return f(x, y, static_cast<std::size_t>(dim));
    };
    m.dist = wrapped;
    m.true_dist = wrapped;
    m.correction = nullptr;
    m.angular_trees = false;  // callables are never in the angular list
    return m;
}
// -----------------------------------------------------------------------------

// ------------------------------------------------------------------ RP trees

void euclidean_random_projection_split(
    const float* data, int64_t dim, const std::vector<int32_t>& indices,
    int64_t rng_state[3], std::vector<int32_t>& left,
    std::vector<int32_t>& right, std::vector<float>& hyperplane,
    float& offset) {
    const int64_t m = static_cast<int64_t>(indices.size());

    // Select two random points, set the hyperplane between them.
    int64_t left_index = pymod(tau_rand_int(rng_state), m);
    int64_t right_index = pymod(tau_rand_int(rng_state), m);
    right_index += (left_index == right_index) ? 1 : 0;
    right_index = right_index % m;
    const float* pl = data + static_cast<int64_t>(indices[left_index]) * dim;
    const float* pr = data + static_cast<int64_t>(indices[right_index]) * dim;

    // hyperplane_offset is a float32 local in numba, but the "/ 2.0" promotes
    // each term to float64 before the subtraction; mirror that exactly.
    float hyperplane_offset = 0.0f;
    hyperplane.assign(static_cast<size_t>(dim), 0.0f);
    for (int64_t d = 0; d < dim; ++d) {
        hyperplane[d] = pl[d] - pr[d];
        hyperplane_offset = static_cast<float>(
            static_cast<double>(hyperplane_offset) -
            static_cast<double>(hyperplane[d] * (pl[d] + pr[d])) / 2.0);
    }

    // For each point compute the margin (project into normal vector, add
    // offset); boundary hits are decided by a coin flip.
    int64_t n_left = 0, n_right = 0;
    std::vector<int8_t> side(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        float margin = hyperplane_offset;
        const float* pt = data + static_cast<int64_t>(indices[i]) * dim;
        for (int64_t d = 0; d < dim; ++d) margin += hyperplane[d] * pt[d];

        if (std::fabs(static_cast<double>(margin)) < kEps) {
            side[i] = static_cast<int8_t>(pymod(iabs32(tau_rand_int(rng_state)), 2));
            if (side[i] == 0)
                ++n_left;
            else
                ++n_right;
        } else if (margin > 0) {
            side[i] = 0;
            ++n_left;
        } else {
            side[i] = 1;
            ++n_right;
        }
    }

    // If all points fell on one side, assign randomly.
    if (n_left == 0 || n_right == 0) {
        n_left = 0;
        n_right = 0;
        for (int64_t i = 0; i < m; ++i) {
            side[i] = static_cast<int8_t>(pymod(tau_rand_int(rng_state), 2));
            if (side[i] == 0)
                ++n_left;
            else
                ++n_right;
        }
    }

    left.clear();
    right.clear();
    left.reserve(static_cast<size_t>(n_left));
    right.reserve(static_cast<size_t>(n_right));
    for (int64_t i = 0; i < m; ++i) {
        if (side[i] == 0)
            left.push_back(indices[i]);
        else
            right.push_back(indices[i]);
    }
    offset = hyperplane_offset;
}

void angular_random_projection_split(
    const float* data, int64_t dim, const std::vector<int32_t>& indices,
    int64_t rng_state[3], std::vector<int32_t>& left,
    std::vector<int32_t>& right, std::vector<float>& hyperplane,
    float& offset) {
    const int64_t m = static_cast<int64_t>(indices.size());

    int64_t left_index = pymod(tau_rand_int(rng_state), m);
    int64_t right_index = pymod(tau_rand_int(rng_state), m);
    right_index += (left_index == right_index) ? 1 : 0;
    right_index = right_index % m;
    const float* pl = data + static_cast<int64_t>(indices[left_index]) * dim;
    const float* pr = data + static_cast<int64_t>(indices[right_index]) * dim;

    float left_norm = l2_norm_strided(pl, dim);
    float right_norm = l2_norm_strided(pr, dim);
    if (std::fabs(static_cast<double>(left_norm)) < kEps) left_norm = 1.0f;
    if (std::fabs(static_cast<double>(right_norm)) < kEps) right_norm = 1.0f;

    hyperplane.assign(static_cast<size_t>(dim), 0.0f);
    for (int64_t d = 0; d < dim; ++d)
        hyperplane[d] = (pl[d] / left_norm) - (pr[d] / right_norm);

    float hyperplane_norm = l2_norm(hyperplane.data(), dim);
    if (std::fabs(static_cast<double>(hyperplane_norm)) < kEps)
        hyperplane_norm = 1.0f;
    for (int64_t d = 0; d < dim; ++d) hyperplane[d] = hyperplane[d] / hyperplane_norm;

    int64_t n_left = 0, n_right = 0;
    std::vector<int8_t> side(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        float margin = 0.0f;
        const float* pt = data + static_cast<int64_t>(indices[i]) * dim;
        for (int64_t d = 0; d < dim; ++d) margin += hyperplane[d] * pt[d];

        if (std::fabs(static_cast<double>(margin)) < kEps) {
            // NB: angular split uses tau_rand_int % 2 (no abs); Python modulo
            // keeps the result in {0, 1}.
            side[i] = static_cast<int8_t>(pymod(tau_rand_int(rng_state), 2));
            if (side[i] == 0)
                ++n_left;
            else
                ++n_right;
        } else if (margin > 0) {
            side[i] = 0;
            ++n_left;
        } else {
            side[i] = 1;
            ++n_right;
        }
    }

    if (n_left == 0 || n_right == 0) {
        n_left = 0;
        n_right = 0;
        for (int64_t i = 0; i < m; ++i) {
            side[i] = static_cast<int8_t>(pymod(tau_rand_int(rng_state), 2));
            if (side[i] == 0)
                ++n_left;
            else
                ++n_right;
        }
    }

    left.clear();
    right.clear();
    left.reserve(static_cast<size_t>(n_left));
    right.reserve(static_cast<size_t>(n_right));
    for (int64_t i = 0; i < m; ++i) {
        if (side[i] == 0)
            left.push_back(indices[i]);
        else
            right.push_back(indices[i]);
    }
    offset = 0.0f;
}

namespace {

void make_tree_recursive(const float* data, int64_t dim,
                         const std::vector<int32_t>& indices,
                         int64_t rng_state[3], int64_t leaf_size, bool angular,
                         int64_t max_depth,
                         std::vector<std::vector<int32_t>>& leaves_out) {
    if (static_cast<int64_t>(indices.size()) > leaf_size && max_depth > 0) {
        std::vector<int32_t> left_indices, right_indices;
        std::vector<float> hyperplane;
        float off;
        if (angular) {
            angular_random_projection_split(data, dim, indices, rng_state,
                                            left_indices, right_indices,
                                            hyperplane, off);
        } else {
            euclidean_random_projection_split(data, dim, indices, rng_state,
                                              left_indices, right_indices,
                                              hyperplane, off);
        }
        make_tree_recursive(data, dim, left_indices, rng_state, leaf_size,
                            angular, max_depth - 1, leaves_out);
        make_tree_recursive(data, dim, right_indices, rng_state, leaf_size,
                            angular, max_depth - 1, leaves_out);
    } else {
        leaves_out.push_back(indices);
    }
}

}  // namespace

int64_t make_dense_tree_leaves(const float* data, int64_t n, int64_t dim,
                               int64_t rng_state[3], int64_t leaf_size,
                               bool angular, int64_t max_depth,
                               std::vector<std::vector<int32_t>>& leaves_out) {
    std::vector<int32_t> indices(static_cast<size_t>(n));
    std::iota(indices.begin(), indices.end(), 0);
    leaves_out.clear();
    make_tree_recursive(data, dim, indices, rng_state, leaf_size, angular,
                        max_depth, leaves_out);
    int64_t max_leaf_size = leaf_size;
    for (const auto& leaf : leaves_out)
        max_leaf_size =
            std::max(max_leaf_size, static_cast<int64_t>(leaf.size()));
    return max_leaf_size;
}

namespace {

LeafArray make_forest_leaf_array_from_states(
    const float* data, int64_t n, int64_t dim, int64_t n_neighbors,
    int64_t leaf_size, bool angular, int64_t max_depth,
    const std::vector<std::array<int64_t, 3>>& tree_rng_states) {
    if (leaf_size < 0)
        leaf_size = std::max<int64_t>(
            60, std::min<int64_t>(256, 5 * n_neighbors));  // make_forest default

    std::vector<std::vector<std::vector<int32_t>>> forest_leaves;
    int64_t global_max_leaf = 0;
    for (const auto& st : tree_rng_states) {
        int64_t rng_state[3] = {st[0], st[1], st[2]};
        std::vector<std::vector<int32_t>> leaves;
        int64_t tree_leaf_size = make_dense_tree_leaves(
            data, n, dim, rng_state, leaf_size, angular, max_depth, leaves);
        global_max_leaf = std::max(global_max_leaf, tree_leaf_size);
        forest_leaves.push_back(std::move(leaves));
    }

    LeafArray la;
    la.row_len = global_max_leaf;
    for (const auto& tl : forest_leaves)
        la.n_rows += static_cast<int64_t>(tl.size());
    la.idx.assign(static_cast<size_t>(la.n_rows * la.row_len), -1);
    int64_t r = 0;
    for (const auto& tl : forest_leaves) {
        for (const auto& leaf : tl) {
            std::copy(leaf.begin(), leaf.end(),
                      la.idx.begin() + r * la.row_len);
            ++r;
        }
    }
    return la;
}

}  // namespace

LeafArray make_forest_leaf_array(const float* data, int64_t n, int64_t dim,
                                 int64_t n_neighbors, int64_t n_trees,
                                 int64_t leaf_size, bool angular,
                                 int64_t max_depth,
                                 NumpyRandomState& numpy_rs) {
    // make_forest draws rng_states with a single randint(size=(n_trees, 3))
    // call: row-major, 3 consecutive draws per tree.
    std::vector<std::array<int64_t, 3>> states;
    states.reserve(static_cast<size_t>(n_trees));
    for (int64_t t = 0; t < n_trees; ++t) states.push_back(numpy_rs.rng_state3());
    return make_forest_leaf_array_from_states(data, n, dim, n_neighbors,
                                              leaf_size, angular, max_depth,
                                              states);
}

// ---------------------------------------------------------------- NN-descent

namespace {

struct Update {
    int32_t p;
    int32_t q;
    float d;
};

}  // namespace

void init_rp_tree(const float* data, int64_t dim, DistFn dist, NNHeap& graph,
                  const LeafArray& leaf_array) {
    // pynndescent_.py::init_rp_tree with its default n_threads=8 "virtual
    // thread" partitioning (deterministic regardless of real parallelism).
    const int64_t n_threads = 8;
    const int64_t n_leaves = leaf_array.n_rows;
    const int64_t block_size = n_threads * 64;
    const int64_t n_blocks = n_leaves / block_size;
    const int64_t mls = leaf_array.row_len;
    const int64_t updates_per_thread =
        static_cast<int64_t>(static_cast<double>(block_size) * mls * (mls - 1) /
                             (2.0 * n_threads)) +
        1;

    std::vector<std::vector<Update>> updates(static_cast<size_t>(n_threads));

    for (int64_t bi = 0; bi < n_blocks + 1; ++bi) {
        const int64_t block_start = bi * block_size;
        const int64_t block_end = std::min(n_leaves, (bi + 1) * block_size);
        const int64_t block_len = block_end - block_start;

        // generate_leaf_updates: thresholds are read live from the heap roots
        // (no pushes happen until the apply phase below).
        const int64_t leaves_per_thread = block_len / n_threads + 1;
        for (int64_t t = 0; t < n_threads; ++t) {
            updates[t].clear();
            const int64_t start_leaf = t * leaves_per_thread;
            const int64_t end_leaf =
                std::min(start_leaf + leaves_per_thread, block_len);
            int64_t count = 0;
            for (int64_t leaf = start_leaf; leaf < end_leaf; ++leaf) {
                const int32_t* row = leaf_array.row(block_start + leaf);
                for (int64_t i = 0; i < mls; ++i) {
                    int32_t p = row[i];
                    if (p < 0) break;
                    for (int64_t j = i + 1; j < mls; ++j) {
                        int32_t q = row[j];
                        if (q < 0) break;
                        float d = dist(data + static_cast<int64_t>(p) * dim,
                                       data + static_cast<int64_t>(q) * dim, dim);
                        float max_threshold =
                            std::max(graph.dist_row(p)[0], graph.dist_row(q)[0]);
                        if (d < max_threshold) {
                            if (count < updates_per_thread) {
                                updates[t].push_back({p, q, d});
                                ++count;
                            }
                        }
                    }
                }
            }
        }

        // Apply phase: vertex-block partitioning is a no-op serially (each
        // update touches exactly rows p and q, in generation-thread order).
        for (int64_t t = 0; t < n_threads; ++t) {
            for (const Update& u : updates[t]) {
                checked_flagged_heap_push(graph.dist_row(u.p), graph.ind_row(u.p),
                                          graph.flag_row(u.p), graph.size, u.d,
                                          u.q, 1);
                checked_flagged_heap_push(graph.dist_row(u.q), graph.ind_row(u.q),
                                          graph.flag_row(u.q), graph.size, u.d,
                                          u.p, 1);
            }
        }
    }
}

void init_random(int64_t n_neighbors, const float* data, int64_t n,
                 int64_t dim, NNHeap& heap, DistFn dist, int64_t rng_state[3]) {
    for (int64_t i = 0; i < n; ++i) {
        if (heap.ind_row(i)[0] < 0) {
            int64_t filled = 0;
            for (int64_t k = 0; k < heap.size; ++k)
                if (heap.ind_row(i)[k] >= 0) ++filled;
            const int64_t reps = n_neighbors - filled;
            for (int64_t j = 0; j < reps; ++j) {
                int32_t idx = static_cast<int32_t>(
                    pymod(iabs32(tau_rand_int(rng_state)), n));
                float d = dist(data + static_cast<int64_t>(idx) * dim,
                               data + i * dim, dim);
                checked_flagged_heap_push(heap.dist_row(i), heap.ind_row(i),
                                          heap.flag_row(i), heap.size, d, idx,
                                          1);
            }
        }
    }
}

void new_build_candidates(NNHeap& current_graph, int64_t max_candidates,
                          const int64_t rng_state[3], int64_t n_threads,
                          std::vector<int32_t>& new_candidates,
                          std::vector<int32_t>& old_candidates) {
    const int64_t n_vertices = current_graph.n_points;
    const int64_t n_neighbors = current_graph.size;
    const int64_t mc = max_candidates;

    new_candidates.assign(static_cast<size_t>(n_vertices * mc), -1);
    old_candidates.assign(static_cast<size_t>(n_vertices * mc), -1);
    std::vector<float> new_priority(static_cast<size_t>(n_vertices * mc), kInf);
    std::vector<float> old_priority(static_cast<size_t>(n_vertices * mc), kInf);

    const int64_t block_size = n_vertices / n_threads + 1;

    for (int64_t t = 0; t < n_threads; ++t) {
        // local_rng_state = rng_state + n : a *copy*; the caller's state is
        // deliberately never advanced (matches utils.new_build_candidates).
        int64_t local_rng_state[3] = {rng_state[0] + t, rng_state[1] + t,
                                      rng_state[2] + t};
        const int64_t block_start = t * block_size;
        const int64_t block_end = std::min(block_start + block_size, n_vertices);

        for (int64_t i = 0; i < n_vertices; ++i) {
            for (int64_t j = 0; j < n_neighbors; ++j) {
                int32_t idx = current_graph.ind_row(i)[j];
                bool i_in = (i >= block_start && i < block_end);
                bool idx_in = (idx >= block_start && idx < block_end);
                if (idx >= 0 && (i_in || idx_in)) {
                    uint8_t isn = current_graph.flag_row(i)[j];
                    float d = tau_rand(local_rng_state);
                    if (isn) {
                        if (i_in)
                            checked_heap_push(new_priority.data() + i * mc,
                                              new_candidates.data() + i * mc,
                                              mc, d, idx);
                        if (idx_in)
                            checked_heap_push(new_priority.data() + idx * mc,
                                              new_candidates.data() + idx * mc,
                                              mc, d, static_cast<int32_t>(i));
                    } else {
                        if (i_in)
                            checked_heap_push(old_priority.data() + i * mc,
                                              old_candidates.data() + i * mc,
                                              mc, d, idx);
                        if (idx_in)
                            checked_heap_push(old_priority.data() + idx * mc,
                                              old_candidates.data() + idx * mc,
                                              mc, d, static_cast<int32_t>(i));
                    }
                }
            }
        }
    }

    // Clear the "new" flag for anything that was selected as a new candidate.
    for (int64_t i = 0; i < n_vertices; ++i) {
        for (int64_t j = 0; j < n_neighbors; ++j) {
            int32_t idx = current_graph.ind_row(i)[j];
            for (int64_t k = 0; k < mc; ++k) {
                if (new_candidates[i * mc + k] == idx) {
                    current_graph.flag_row(i)[j] = 0;
                    break;
                }
            }
        }
    }
}

namespace {

// pynndescent_.py::nn_descent_internal (array-update path), n_threads=1.
void nn_descent_internal(const float* data, int64_t n, int64_t dim,
                         NNHeap& current_graph, int64_t n_neighbors,
                         int64_t rng_state[3], int64_t max_candidates,
                         DistFn dist, int64_t n_iters, double delta,
                         bool verbose) {
    const int64_t block_size = 16384;
    const int64_t n_blocks = n / block_size;
    const int64_t n_threads = 1;
    const int64_t mc = max_candidates;
    const int64_t max_updates_per_thread =
        static_cast<int64_t>(
            (static_cast<double>(mc) * mc + mc * (mc - 1) / 2.0) * block_size /
            static_cast<double>(n_threads)) +
        1024;

    std::vector<int32_t> new_cand, old_cand;
    std::vector<Update> updates;

    for (int64_t iter = 0; iter < n_iters; ++iter) {
        if (verbose)
            std::fprintf(stderr, "\t%lld / %lld\n",
                         static_cast<long long>(iter + 1),
                         static_cast<long long>(n_iters));

        new_build_candidates(current_graph, mc, rng_state, n_threads, new_cand,
                             old_cand);

        int64_t c = 0;
        for (int64_t bi = 0; bi < n_blocks + 1; ++bi) {
            const int64_t block_start = bi * block_size;
            const int64_t block_end = std::min(n, (bi + 1) * block_size);

            // generate_graph_update_array (single thread covers all rows).
            updates.clear();
            int64_t idx_count = 0;
            for (int64_t i = block_start;
                 i < block_end && idx_count < max_updates_per_thread; ++i) {
                for (int64_t j = 0; j < mc; ++j) {
                    if (idx_count >= max_updates_per_thread) break;
                    int32_t p = new_cand[i * mc + j];
                    if (p < 0) continue;
                    const float* data_p = data + static_cast<int64_t>(p) * dim;
                    const float dist_thresh_p = current_graph.dist_row(p)[0];

                    // vs other new candidates (k starts at j: includes q == p,
                    // which is how each point acquires itself as a neighbor)
                    for (int64_t k = j; k < mc; ++k) {
                        if (idx_count >= max_updates_per_thread) break;
                        int32_t q = new_cand[i * mc + k];
                        if (q < 0) continue;
                        float d = dist(data_p,
                                       data + static_cast<int64_t>(q) * dim, dim);
                        float max_threshold =
                            std::max(dist_thresh_p, current_graph.dist_row(q)[0]);
                        if (d <= max_threshold) {
                            updates.push_back({p, q, d});
                            ++idx_count;
                        }
                    }
                    // vs old candidates
                    for (int64_t k = 0; k < mc; ++k) {
                        if (idx_count >= max_updates_per_thread) break;
                        int32_t q = old_cand[i * mc + k];
                        if (q < 0) continue;
                        float d = dist(data_p,
                                       data + static_cast<int64_t>(q) * dim, dim);
                        float max_threshold =
                            std::max(dist_thresh_p, current_graph.dist_row(q)[0]);
                        if (d <= max_threshold) {
                            updates.push_back({p, q, d});
                            ++idx_count;
                        }
                    }
                }
            }

            // apply_graph_update_array
            for (const Update& u : updates) {
                c += checked_flagged_heap_push(
                    current_graph.dist_row(u.p), current_graph.ind_row(u.p),
                    current_graph.flag_row(u.p), current_graph.size, u.d, u.q, 1);
                c += checked_flagged_heap_push(
                    current_graph.dist_row(u.q), current_graph.ind_row(u.q),
                    current_graph.flag_row(u.q), current_graph.size, u.d, u.p, 1);
            }
        }

        if (c <= delta * static_cast<double>(n_neighbors) * static_cast<double>(n)) {
            if (verbose)
                std::fprintf(stderr,
                             "\tStopping threshold met -- exiting after %lld "
                             "iterations\n",
                             static_cast<long long>(iter + 1));
            return;
        }
    }
}

}  // namespace

NNHeap nn_descent(const float* data, int64_t n, int64_t dim,
                  int64_t n_neighbors, int64_t rng_state[3],
                  int64_t max_candidates, DistFn dist, int64_t n_iters,
                  double delta, const LeafArray& leaf_array, bool verbose) {
    NNHeap current_graph = NNHeap::make(n, n_neighbors);

    if (leaf_array.n_rows > 0)
        init_rp_tree(data, dim, dist, current_graph, leaf_array);
    init_random(n_neighbors, data, n, dim, current_graph, dist, rng_state);

    nn_descent_internal(data, n, dim, current_graph, n_neighbors, rng_state,
                        max_candidates, dist, n_iters, delta, verbose);

    deheap_sort(current_graph.ind.data(), current_graph.dist.data(), n,
                n_neighbors);
    return current_graph;
}

// --- inverse-update additions ------------------------------------------------

void init_from_neighbor_graph(NNHeap& heap, const int32_t* indices,
                              const float* distances, int64_t n_rows,
                              int64_t n_cols) {
    // pynndescent_.py::init_from_neighbor_graph: flag-0 pushes in (p, k)
    // order.
    for (int64_t p = 0; p < n_rows; ++p) {
        for (int64_t k = 0; k < n_cols; ++k) {
            const int32_t q = indices[p * n_cols + k];
            const float d = distances[p * n_cols + k];
            checked_flagged_heap_push(heap.dist_row(p), heap.ind_row(p),
                                      heap.flag_row(p), heap.size, d, q, 0);
        }
    }
}

void nn_descent_with_init_graph(NNHeap& current_graph, const float* data,
                                int64_t n, int64_t dim, int64_t n_neighbors,
                                int64_t rng_state[3], int64_t max_candidates,
                                DistFn dist, int64_t n_iters, double delta,
                                bool verbose) {
    // nn_descent with init_graph=current_graph (rp_tree_init handled by the
    // caller): straight to the candidate loop, then deheap_sort.
    nn_descent_internal(data, n, dim, current_graph, n_neighbors, rng_state,
                        max_candidates, dist, n_iters, delta, verbose);
    deheap_sort(current_graph.ind.data(), current_graph.dist.data(), n,
                n_neighbors);
}
// -----------------------------------------------------------------------------

// ------------------------------------------------------------------ toplevel

void brute_force_knn(const float* data, int64_t n, int64_t dim,
                     int64_t n_neighbors, const std::string& metric,
                     int32_t* knn_indices, float* knn_dists) {
    MetricInfo m = resolve_metric(metric);
    std::vector<float> row(static_cast<size_t>(n));
    std::vector<int32_t> order(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j)
            row[j] = m.true_dist(data + i * dim, data + j * dim, dim);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int32_t a, int32_t b) { return row[a] < row[b]; });
        for (int64_t k = 0; k < n_neighbors; ++k) {
            knn_indices[i * n_neighbors + k] = order[k];
            knn_dists[i * n_neighbors + k] = row[order[k]];
        }
    }
}

namespace {

void run_nearest_neighbors(const float* data, int64_t n, int64_t dim,
                           int64_t n_neighbors, const MetricInfo& m,
                           int64_t rng_state[3], const LeafArray& leaf_array,
                           int64_t n_iters, int32_t* knn_indices,
                           float* knn_dists, bool verbose) {
    const int64_t max_candidates = 60;  // umap_.py::nearest_neighbors
    const double delta = 0.001;         // NNDescent default

    NNHeap graph = nn_descent(data, n, dim, n_neighbors, rng_state,
                              max_candidates, m.dist, n_iters, delta,
                              leaf_array, verbose);

    for (int64_t i = 0; i < n * n_neighbors; ++i) {
        knn_indices[i] = graph.ind[i];
        knn_dists[i] =
            m.correction ? m.correction(graph.dist[i]) : graph.dist[i];
    }
}

}  // namespace

void nearest_neighbors(const float* data, int64_t n, int64_t dim,
                       int64_t n_neighbors, const std::string& metric,
                       bool angular, uint32_t random_seed,
                       int32_t* knn_indices, float* knn_dists, bool verbose) {
    NumpyRandomState numpy_rs(random_seed);
    nearest_neighbors(data, n, dim, n_neighbors, metric, angular, numpy_rs,
                      knn_indices, knn_dists, verbose);
}

void nearest_neighbors(const float* data, int64_t n, int64_t dim,
                       int64_t n_neighbors, const std::string& metric,
                       bool angular, NumpyRandomState& numpy_rs,
                       int32_t* knn_indices, float* knn_dists, bool verbose) {
    nearest_neighbors(data, n, dim, n_neighbors, metric, MetricKwds{}, angular,
                      numpy_rs, knn_indices, knn_dists, verbose);
}

// --- densmap additions ------------------------------------------------------
void nearest_neighbors(const float* data, int64_t n, int64_t dim,
                       int64_t n_neighbors, const std::string& metric,
                       const MetricKwds& metric_kwds, bool angular,
                       NumpyRandomState& numpy_rs, int32_t* knn_indices,
                       float* knn_dists, bool verbose) {
    (void)angular;  // only affects callable metrics in pynndescent
    MetricInfo m = resolve_metric(metric, metric_kwds);

    // umap_.py::nearest_neighbors "hacked values"
    const int64_t n_trees = std::min<int64_t>(
        64, 5 + static_cast<int64_t>(
                    std::nearbyint(std::sqrt(static_cast<double>(n)) / 20.0)));
    const int64_t n_iters = std::max<int64_t>(
        5, static_cast<int64_t>(std::nearbyint(std::log2(static_cast<double>(n)))));

    // NNDescent.__init__ seeding order: rng_state, search_rng_state, then the
    // per-tree states inside make_forest.
    std::array<int64_t, 3> rng_state_arr = numpy_rs.rng_state3();
    std::array<int64_t, 3> search_rng_state = numpy_rs.rng_state3();
    (void)search_rng_state;  // consumed for draw-order parity; search not ported

    LeafArray leaf_array = make_forest_leaf_array(
        data, n, dim, n_neighbors, n_trees, /*leaf_size=*/-1, m.angular_trees,
        /*max_depth=*/200, numpy_rs);

    int64_t rng_state[3] = {rng_state_arr[0], rng_state_arr[1], rng_state_arr[2]};
    run_nearest_neighbors(data, n, dim, n_neighbors, m, rng_state,
                          leaf_array, n_iters, knn_indices, knn_dists, verbose);
}
// ----------------------------------------------------------------------------

void nearest_neighbors_with_state(
    const float* data, int64_t n, int64_t dim, int64_t n_neighbors,
    const std::string& metric, bool angular, const int64_t rng_state_in[3],
    const std::vector<std::array<int64_t, 3>>& tree_rng_states,
    int32_t* knn_indices, float* knn_dists, bool verbose) {
    (void)angular;
    MetricInfo m = resolve_metric(metric);

    const int64_t n_iters = std::max<int64_t>(
        5, static_cast<int64_t>(std::nearbyint(std::log2(static_cast<double>(n)))));

    LeafArray leaf_array = make_forest_leaf_array_from_states(
        data, n, dim, n_neighbors, /*leaf_size=*/-1, m.angular_trees,
        /*max_depth=*/200, tree_rng_states);

    int64_t rng_state[3] = {rng_state_in[0], rng_state_in[1], rng_state_in[2]};
    run_nearest_neighbors(data, n, dim, n_neighbors, m, rng_state,
                          leaf_array, n_iters, knn_indices, knn_dists, verbose);
}

// --- hardening additions -----------------------------------------------------
// Callable-metric overload: identical routing to the kwds-aware overload with
// the MetricInfo built from the user function (euclidean trees, no fast
// alternative -- pynndescent's callable branch).
void nearest_neighbors(const float* data, int64_t n, int64_t dim,
                       int64_t n_neighbors, const DistanceF32Fn& custom,
                       NumpyRandomState& numpy_rs, int32_t* knn_indices,
                       float* knn_dists, bool verbose) {
    MetricInfo m = resolve_metric("custom", MetricKwds{}, &custom);

    // umap_.py::nearest_neighbors "hacked values"
    const int64_t n_trees = std::min<int64_t>(
        64, 5 + static_cast<int64_t>(
                    std::nearbyint(std::sqrt(static_cast<double>(n)) / 20.0)));
    const int64_t n_iters = std::max<int64_t>(
        5,
        static_cast<int64_t>(std::nearbyint(std::log2(static_cast<double>(n)))));

    std::array<int64_t, 3> rng_state_arr = numpy_rs.rng_state3();
    std::array<int64_t, 3> search_rng_state = numpy_rs.rng_state3();
    (void)search_rng_state;  // draw-order parity; search index not built here

    LeafArray leaf_array = make_forest_leaf_array(
        data, n, dim, n_neighbors, n_trees, /*leaf_size=*/-1, m.angular_trees,
        /*max_depth=*/200, numpy_rs);

    int64_t rng_state[3] = {rng_state_arr[0], rng_state_arr[1],
                            rng_state_arr[2]};
    run_nearest_neighbors(data, n, dim, n_neighbors, m, rng_state, leaf_array,
                          n_iters, knn_indices, knn_dists, verbose);
}
// -----------------------------------------------------------------------------

}  // namespace nnd
}  // namespace umap
