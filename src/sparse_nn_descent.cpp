// Implementation of include/umap/sparse_nn_descent.hpp. See that header for
// scope, faithfulness and integration notes. Line references in comments are
// to reference/pynndescent/pynndescent/{sparse_nndescent,rp_trees,sparse}.py.

#include "umap/sparse_nn_descent.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "umap/rng.hpp"

namespace umap {
namespace spnnd {

namespace {

using nnd::pymod;

constexpr double kEps = 1e-8;  // rp_trees.py EPS

// numba np.abs on int32 (INT32_MIN stays INT32_MIN, like C).
inline int32_t iabs32(int32_t v) {
    return v < 0 ? static_cast<int32_t>(0u - static_cast<uint32_t>(v)) : v;
}

struct Update {
    int32_t p;
    int32_t q;
    float d;
};

// Row slice convenience: dist(X[i], X[j]) through a SparseDistFn.
inline float row_dist(SparseDistFn dist, const CsrView& X, int64_t i,
                      int64_t j) {
    return dist(X.row_ind(i), X.row_data(i), X.row_nnz(i), X.row_ind(j),
                X.row_data(j), X.row_nnz(j));
}

}  // namespace

// ------------------------------------------------------------------ RP trees

void sparse_euclidean_random_projection_split(
    const CsrView& X, const std::vector<int32_t>& indices,
    int64_t rng_state[3], std::vector<int32_t>& left,
    std::vector<int32_t>& right) {
    const int64_t m = static_cast<int64_t>(indices.size());

    // Select two random points, set the hyperplane between them. NB: unlike
    // the dense euclidean split, the sparse one takes np.abs of the raw draw
    // before the (Python-semantics) modulo.
    int64_t left_index = pymod(iabs32(tau_rand_int(rng_state)), m);
    int64_t right_index = pymod(iabs32(tau_rand_int(rng_state)), m);
    right_index += (left_index == right_index) ? 1 : 0;
    right_index = right_index % m;
    const int64_t li = indices[left_index];
    const int64_t ri = indices[right_index];

    // Compute the normal vector to the hyperplane (the vector between the two
    // points) and the offset from the origin:
    //   hyperplane = sparse_diff(left, right)                      (float32)
    //   offset_data = sparse_sum(left, right) / 2.0                (f4 -> f8
    //     division by the float64 literal, cast back via .astype(np.float32))
    //   hyperplane_offset -= sum(sparse_mul(hyperplane, offset))   (float64)
    std::vector<int32_t> hyperplane_inds;
    std::vector<float> hyperplane_data;
    spdist::sparse_diff(X.row_ind(li), X.row_data(li), X.row_nnz(li),
                        X.row_ind(ri), X.row_data(ri), X.row_nnz(ri),
                        hyperplane_inds, hyperplane_data);

    std::vector<int32_t> offset_inds;
    std::vector<float> offset_data;
    spdist::sparse_sum(X.row_ind(li), X.row_data(li), X.row_nnz(li),
                       X.row_ind(ri), X.row_data(ri), X.row_nnz(ri),
                       offset_inds, offset_data);
    for (float& v : offset_data)
        v = static_cast<float>(static_cast<double>(v) / 2.0);

    std::vector<int32_t> mul_inds;
    std::vector<float> mul_data;
    spdist::sparse_mul(hyperplane_inds.data(), hyperplane_data.data(),
                       static_cast<int64_t>(hyperplane_inds.size()),
                       offset_inds.data(), offset_data.data(),
                       static_cast<int64_t>(offset_inds.size()), mul_inds,
                       mul_data);
    double hyperplane_offset = 0.0;
    for (float v : mul_data) hyperplane_offset -= static_cast<double>(v);

    // For each point compute the margin (project into normal vector, add
    // offset); boundary hits are decided by a coin flip.
    int64_t n_left = 0, n_right = 0;
    std::vector<int8_t> side(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        double margin = hyperplane_offset;
        const int64_t pi = indices[i];
        spdist::sparse_mul(hyperplane_inds.data(), hyperplane_data.data(),
                           static_cast<int64_t>(hyperplane_inds.size()),
                           X.row_ind(pi), X.row_data(pi), X.row_nnz(pi),
                           mul_inds, mul_data);
        for (float v : mul_data) margin += static_cast<double>(v);

        if (std::fabs(margin) < kEps) {
            side[i] =
                static_cast<int8_t>(pymod(iabs32(tau_rand_int(rng_state)), 2));
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

    // If all points fell on one side, assign randomly (abs draw again).
    if (n_left == 0 || n_right == 0) {
        n_left = 0;
        n_right = 0;
        for (int64_t i = 0; i < m; ++i) {
            side[i] =
                static_cast<int8_t>(pymod(iabs32(tau_rand_int(rng_state)), 2));
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
}

void sparse_angular_random_projection_split(
    const CsrView& X, const std::vector<int32_t>& indices,
    int64_t rng_state[3], std::vector<int32_t>& left,
    std::vector<int32_t>& right) {
    const int64_t m = static_cast<int64_t>(indices.size());

    // No np.abs on the anchor draws here (Python modulo keeps them in range).
    int64_t left_index = pymod(tau_rand_int(rng_state), m);
    int64_t right_index = pymod(tau_rand_int(rng_state), m);
    right_index += (left_index == right_index) ? 1 : 0;
    right_index = right_index % m;
    const int64_t li = indices[left_index];
    const int64_t ri = indices[right_index];

    // utils.norm returns float32; the `norm = 1.0` fallback unifies the numba
    // local to float64, so the normalising division happens in float64 with
    // the f4-rounded norm, cast back to float32 per element.
    double left_norm =
        static_cast<double>(spdist::norm(X.row_data(li), X.row_nnz(li)));
    double right_norm =
        static_cast<double>(spdist::norm(X.row_data(ri), X.row_nnz(ri)));
    if (std::fabs(left_norm) < kEps) left_norm = 1.0;
    if (std::fabs(right_norm) < kEps) right_norm = 1.0;

    std::vector<float> normalized_left(X.row_data(li),
                                       X.row_data(li) + X.row_nnz(li));
    for (float& v : normalized_left)
        v = static_cast<float>(static_cast<double>(v) / left_norm);
    std::vector<float> normalized_right(X.row_data(ri),
                                        X.row_data(ri) + X.row_nnz(ri));
    for (float& v : normalized_right)
        v = static_cast<float>(static_cast<double>(v) / right_norm);

    std::vector<int32_t> hyperplane_inds;
    std::vector<float> hyperplane_data;
    spdist::sparse_diff(X.row_ind(li), normalized_left.data(), X.row_nnz(li),
                        X.row_ind(ri), normalized_right.data(), X.row_nnz(ri),
                        hyperplane_inds, hyperplane_data);

    // hyperplane_norm is a declared float32 numba local: all-f4 normalise.
    float hyperplane_norm = spdist::norm(
        hyperplane_data.data(), static_cast<int64_t>(hyperplane_data.size()));
    if (std::fabs(static_cast<double>(hyperplane_norm)) < kEps)
        hyperplane_norm = 1.0f;
    for (float& v : hyperplane_data) v = v / hyperplane_norm;

    // For each point compute the margin (project into normal vector).
    int64_t n_left = 0, n_right = 0;
    std::vector<int8_t> side(static_cast<size_t>(m));
    std::vector<int32_t> mul_inds;
    std::vector<float> mul_data;
    for (int64_t i = 0; i < m; ++i) {
        double margin = 0.0;
        const int64_t pi = indices[i];
        spdist::sparse_mul(hyperplane_inds.data(), hyperplane_data.data(),
                           static_cast<int64_t>(hyperplane_inds.size()),
                           X.row_ind(pi), X.row_data(pi), X.row_nnz(pi),
                           mul_inds, mul_data);
        for (float v : mul_data) margin += static_cast<double>(v);

        if (std::fabs(margin) < kEps) {
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
}

namespace {

void make_sparse_tree_recursive(
    const CsrView& X, const std::vector<int32_t>& indices,
    int64_t rng_state[3], int64_t leaf_size, bool angular, int64_t max_depth,
    std::vector<std::vector<int32_t>>& leaves_out) {
    if (static_cast<int64_t>(indices.size()) > leaf_size && max_depth > 0) {
        std::vector<int32_t> left_indices, right_indices;
        if (angular) {
            sparse_angular_random_projection_split(X, indices, rng_state,
                                                   left_indices, right_indices);
        } else {
            sparse_euclidean_random_projection_split(
                X, indices, rng_state, left_indices, right_indices);
        }
        make_sparse_tree_recursive(X, left_indices, rng_state, leaf_size,
                                   angular, max_depth - 1, leaves_out);
        make_sparse_tree_recursive(X, right_indices, rng_state, leaf_size,
                                   angular, max_depth - 1, leaves_out);
    } else {
        leaves_out.push_back(indices);
    }
}

}  // namespace

int64_t make_sparse_tree_leaves(const CsrView& X, int64_t rng_state[3],
                                int64_t leaf_size, bool angular,
                                int64_t max_depth,
                                std::vector<std::vector<int32_t>>& leaves_out) {
    std::vector<int32_t> indices(static_cast<size_t>(X.n_rows));
    std::iota(indices.begin(), indices.end(), 0);
    leaves_out.clear();
    make_sparse_tree_recursive(X, indices, rng_state, leaf_size, angular,
                               max_depth, leaves_out);
    int64_t max_leaf_size = leaf_size;
    for (const auto& leaf : leaves_out)
        max_leaf_size =
            std::max(max_leaf_size, static_cast<int64_t>(leaf.size()));
    return max_leaf_size;
}

namespace {

LeafArray make_sparse_forest_leaf_array_from_states(
    const CsrView& X, int64_t n_neighbors, int64_t leaf_size, bool angular,
    int64_t max_depth,
    const std::vector<std::array<int64_t, 3>>& tree_rng_states) {
    if (leaf_size < 0)
        leaf_size = std::max<int64_t>(
            60, std::min<int64_t>(256, 5 * n_neighbors));  // make_forest default

    std::vector<std::vector<std::vector<int32_t>>> forest_leaves;
    int64_t global_max_leaf = 0;
    for (const auto& st : tree_rng_states) {
        int64_t rng_state[3] = {st[0], st[1], st[2]};
        std::vector<std::vector<int32_t>> leaves;
        int64_t tree_leaf_size = make_sparse_tree_leaves(
            X, rng_state, leaf_size, angular, max_depth, leaves);
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
            std::copy(leaf.begin(), leaf.end(), la.idx.begin() + r * la.row_len);
            ++r;
        }
    }
    return la;
}

}  // namespace

LeafArray make_sparse_forest_leaf_array(const CsrView& X, int64_t n_neighbors,
                                        int64_t n_trees, int64_t leaf_size,
                                        bool angular, int64_t max_depth,
                                        NumpyRandomState& numpy_rs) {
    // make_forest draws rng_states with a single randint(size=(n_trees, 3))
    // call: row-major, 3 consecutive draws per tree.
    std::vector<std::array<int64_t, 3>> states;
    states.reserve(static_cast<size_t>(n_trees));
    for (int64_t t = 0; t < n_trees; ++t) states.push_back(numpy_rs.rng_state3());
    return make_sparse_forest_leaf_array_from_states(X, n_neighbors, leaf_size,
                                                     angular, max_depth, states);
}

// ---------------------------------------------------------------- NN-descent

void sparse_init_rp_tree(const CsrView& X, SparseDistFn dist, NNHeap& graph,
                         const LeafArray& leaf_array) {
    // sparse_nndescent.py::init_rp_tree with its default n_threads=8 "virtual
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
        // (no pushes happen until the apply phase below). NB: the sparse
        // version splits leaves per thread with *ceil* division, unlike the
        // dense (block_len // t) + 1.
        const int64_t leaves_per_thread =
            (block_len + n_threads - 1) / n_threads;
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
                        float d = row_dist(dist, X, p, q);
                        // `d < thresholds[p] or d < thresholds[q]`
                        if (d < graph.dist_row(p)[0] ||
                            d < graph.dist_row(q)[0]) {
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
                nnd::checked_flagged_heap_push(
                    graph.dist_row(u.p), graph.ind_row(u.p), graph.flag_row(u.p),
                    graph.size, u.d, u.q, 1);
                nnd::checked_flagged_heap_push(
                    graph.dist_row(u.q), graph.ind_row(u.q), graph.flag_row(u.q),
                    graph.size, u.d, u.p, 1);
            }
        }
    }
}

void sparse_init_random(int64_t n_neighbors, const CsrView& X, NNHeap& heap,
                        SparseDistFn dist, int64_t rng_state[3]) {
    const int64_t n_samples = X.n_rows;
    for (int64_t i = 0; i < n_samples; ++i) {
        if (heap.ind_row(i)[0] < 0) {
            int64_t filled = 0;
            for (int64_t k = 0; k < heap.size; ++k)
                if (heap.ind_row(i)[k] >= 0) ++filled;
            const int64_t reps = n_neighbors - filled;
            for (int64_t j = 0; j < reps; ++j) {
                int32_t idx = static_cast<int32_t>(
                    pymod(iabs32(tau_rand_int(rng_state)), n_samples));
                // d = dist(X[idx], X[i]) -- from/to order as in Python.
                float d = row_dist(dist, X, idx, i);
                nnd::checked_flagged_heap_push(heap.dist_row(i),
                                               heap.ind_row(i),
                                               heap.flag_row(i), heap.size, d,
                                               idx, 1);
            }
        }
    }
}

namespace {

// sparse_nndescent.py::nn_descent_internal (array-update path), n_threads=1.
void sparse_nn_descent_internal(const CsrView& X, NNHeap& current_graph,
                                int64_t n_neighbors, int64_t rng_state[3],
                                int64_t max_candidates, SparseDistFn dist,
                                int64_t n_iters, double delta, bool verbose) {
    const int64_t n_vertices = X.n_rows;
    const int64_t block_size = 16384;
    const int64_t n_blocks = n_vertices / block_size;
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

        nnd::new_build_candidates(current_graph, mc, rng_state, n_threads,
                                  new_cand, old_cand);

        int64_t c = 0;
        for (int64_t bi = 0; bi < n_blocks + 1; ++bi) {
            const int64_t block_start = bi * block_size;
            const int64_t block_end =
                std::min(n_vertices, (bi + 1) * block_size);

            // sparse_generate_graph_update_array (single thread, all rows).
            updates.clear();
            int64_t idx_count = 0;
            for (int64_t i = block_start;
                 i < block_end && idx_count < max_updates_per_thread; ++i) {
                for (int64_t j = 0; j < mc; ++j) {
                    if (idx_count >= max_updates_per_thread) break;
                    int32_t p = new_cand[i * mc + j];
                    if (p < 0) continue;
                    const float dist_thresh_p = current_graph.dist_row(p)[0];

                    // vs other new candidates (k starts at j: includes q == p,
                    // which is how each point acquires itself as a neighbor)
                    for (int64_t k = j; k < mc; ++k) {
                        if (idx_count >= max_updates_per_thread) break;
                        int32_t q = new_cand[i * mc + k];
                        if (q < 0) continue;
                        float d = row_dist(dist, X, p, q);
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
                        float d = row_dist(dist, X, p, q);
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
                c += nnd::checked_flagged_heap_push(
                    current_graph.dist_row(u.p), current_graph.ind_row(u.p),
                    current_graph.flag_row(u.p), current_graph.size, u.d, u.q,
                    1);
                c += nnd::checked_flagged_heap_push(
                    current_graph.dist_row(u.q), current_graph.ind_row(u.q),
                    current_graph.flag_row(u.q), current_graph.size, u.d, u.p,
                    1);
            }
        }

        if (c <= delta * static_cast<double>(n_neighbors) *
                     static_cast<double>(n_vertices)) {
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

NNHeap sparse_nn_descent(const CsrView& X, int64_t n_neighbors,
                         int64_t rng_state[3], int64_t max_candidates,
                         SparseDistFn dist, int64_t n_iters, double delta,
                         const LeafArray& leaf_array, bool verbose) {
    NNHeap current_graph = NNHeap::make(X.n_rows, n_neighbors);

    if (leaf_array.n_rows > 0)
        sparse_init_rp_tree(X, dist, current_graph, leaf_array);
    sparse_init_random(n_neighbors, X, current_graph, dist, rng_state);

    sparse_nn_descent_internal(X, current_graph, n_neighbors, rng_state,
                               max_candidates, dist, n_iters, delta, verbose);

    nnd::deheap_sort(current_graph.ind.data(), current_graph.dist.data(),
                     X.n_rows, n_neighbors);
    return current_graph;
}

// ------------------------------------------------------------------ toplevel

void sparse_brute_force_knn(const CsrView& X, int64_t n_neighbors,
                            const std::string& metric, int32_t* knn_indices,
                            float* knn_dists) {
    spdist::SparseMetricInfo m = spdist::resolve_sparse_metric(metric);
    const int64_t n = X.n_rows;
    std::vector<float> row(static_cast<size_t>(n));
    std::vector<int32_t> order(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j)
            row[j] = m.true_dist(X.row_ind(i), X.row_data(i), X.row_nnz(i),
                                 X.row_ind(j), X.row_data(j), X.row_nnz(j));
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

void run_sparse_nearest_neighbors(const CsrView& X, int64_t n_neighbors,
                                  const std::string& metric,
                                  int64_t rng_state[3],
                                  const LeafArray& leaf_array, int64_t n_iters,
                                  int32_t* knn_indices, float* knn_dists,
                                  bool verbose) {
    spdist::SparseMetricInfo m = spdist::resolve_sparse_metric(metric);
    const int64_t max_candidates = 60;  // umap_.py::nearest_neighbors
    const double delta = 0.001;         // NNDescent default

    NNHeap graph = sparse_nn_descent(X, n_neighbors, rng_state, max_candidates,
                                     m.dist, n_iters, delta, leaf_array,
                                     verbose);

    for (int64_t i = 0; i < X.n_rows * n_neighbors; ++i) {
        knn_indices[i] = graph.ind[i];
        knn_dists[i] =
            m.correction ? m.correction(graph.dist[i]) : graph.dist[i];
    }
}

}  // namespace

void nearest_neighbors_sparse(const int32_t* indptr, const int32_t* indices,
                              const float* data, int64_t n, int64_t dim,
                              int64_t n_neighbors, const std::string& metric,
                              bool angular, uint32_t random_seed,
                              int32_t* knn_indices, float* knn_dists,
                              bool force_approximation, bool verbose) {
    (void)angular;  // only affects callable metrics in pynndescent
    CsrView X{indptr, indices, data, n, dim};
    spdist::SparseMetricInfo m = spdist::resolve_sparse_metric(metric);

    // UMAP.fit small-data branch (n < 4096, force_approximation_algorithm
    // False): exact pairwise distances + stable-argsort kNN.
    if (n < 4096 && !force_approximation) {
        sparse_brute_force_knn(X, n_neighbors, metric, knn_indices, knn_dists);
        return;
    }

    // umap_.py::nearest_neighbors "hacked values"
    const int64_t n_trees = std::min<int64_t>(
        64, 5 + static_cast<int64_t>(
                    std::nearbyint(std::sqrt(static_cast<double>(n)) / 20.0)));
    const int64_t n_iters = std::max<int64_t>(
        5, static_cast<int64_t>(
               std::nearbyint(std::log2(static_cast<double>(n)))));

    // NNDescent.__init__ seeding order: rng_state, search_rng_state, then the
    // per-tree states inside make_forest.
    NumpyRandomState numpy_rs(random_seed);
    std::array<int64_t, 3> rng_state_arr = numpy_rs.rng_state3();
    std::array<int64_t, 3> search_rng_state = numpy_rs.rng_state3();
    (void)search_rng_state;  // consumed for draw-order parity; search not ported

    LeafArray leaf_array = make_sparse_forest_leaf_array(
        X, n_neighbors, n_trees, /*leaf_size=*/-1, m.angular_trees,
        /*max_depth=*/200, numpy_rs);

    int64_t rng_state[3] = {rng_state_arr[0], rng_state_arr[1],
                            rng_state_arr[2]};
    run_sparse_nearest_neighbors(X, n_neighbors, metric, rng_state, leaf_array,
                                 n_iters, knn_indices, knn_dists, verbose);
}

void nearest_neighbors_sparse_with_state(
    const int32_t* indptr, const int32_t* indices, const float* data,
    int64_t n, int64_t dim, int64_t n_neighbors, const std::string& metric,
    bool angular, const int64_t rng_state_in[3],
    const std::vector<std::array<int64_t, 3>>& tree_rng_states,
    int32_t* knn_indices, float* knn_dists, bool verbose) {
    (void)angular;
    CsrView X{indptr, indices, data, n, dim};
    spdist::SparseMetricInfo m = spdist::resolve_sparse_metric(metric);

    const int64_t n_iters = std::max<int64_t>(
        5, static_cast<int64_t>(
               std::nearbyint(std::log2(static_cast<double>(n)))));

    LeafArray leaf_array = make_sparse_forest_leaf_array_from_states(
        X, n_neighbors, /*leaf_size=*/-1, m.angular_trees, /*max_depth=*/200,
        tree_rng_states);

    int64_t rng_state[3] = {rng_state_in[0], rng_state_in[1], rng_state_in[2]};
    run_sparse_nearest_neighbors(X, n_neighbors, metric, rng_state, leaf_array,
                                 n_iters, knn_indices, knn_dists, verbose);
}

}  // namespace spnnd
}  // namespace umap
