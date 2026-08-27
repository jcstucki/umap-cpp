// Faithful C++ port of the pynndescent *sparse* query path used by UMAP's
// transform() when the training data is sparse: NNDescent.prepare()
// (_init_search_graph's sparse branch: make_sparse_hub_tree +
// convert_tree_format, sparse.diversify / sparse.diversify_csr, shared
// degree_prune / symmetric-max / vertex re-order) and NNDescent.query()
// (_init_sparse_search_function's search_closure: sparse_select_side tree
// descent + epsilon-bounded graph search), from
// reference/pynndescent/pynndescent/{pynndescent_.py, rp_trees.py, sparse.py,
// utils.py} (BSD 2 clause). See include/umap/sparse_nn_descent.hpp for the
// API-level faithfulness notes.
//
// Faithfulness notes (numba typing):
// - sparse_euclidean_hub_split: hyperplane/offset vectors are float32
//   (sparse_diff/sum/mul values); the offset halving runs in float64 and is
//   cast back per element (exact: /2 is a power of two); hyperplane_offset
//   and margins accumulate float32 products into float64.
// - sparse_angular_hub_split: utils.norm returns float32 (declared f4
//   signature); the `norm = 1.0` fallback unifies the norm locals to
//   float64, so all normalising divisions run in float64 on the f4-rounded
//   norm and are cast back to float32 per element; margins accumulate f4
//   products into float64 (no offset).
// - Tie margins (|margin| <= EPS boundary handling matches the Python
//   `> EPS / < -EPS / else` ordering) assign side = i % 2 (no rng); only the
//   degenerate everything-on-one-side fallback consumes rng
//   (np.abs(tau_rand_int) % 2 in both split flavours).
// - Hub pair selection minimises edge cuts (uint32, strict <), not balance;
//   the recursion has NO MIN_SPLIT_BALANCE leaf fallback (unlike the dense
//   hub tree).
// - sparse_select_side: margin is float32 (offset f4 + sparse_dot_product's
//   f4 accumulation), |margin| < EPS coin flips use tau_rand_int % 2 with
//   Python modulo (no np.abs, unlike the dense select_side).
// - scipy's zero-copy-transpose quirk is replicated exactly as in the dense
//   port: the "reverse" diversify_csr pass runs over the *forward* CSR rows;
//   net effect M = max(F2, F2^T). sparse.diversify_csr indexes occluders
//   correctly via order[k] (no dense-style raw-position quirk).
// - The sparse search closure: distance bound is
//   float32(1 + epsilon) * heap_priorities[0] (no min_distance term), the
//   distance argument order is dist(candidate_row, query_row), and queries
//   are never normalised (the reference compares the sparse _distance_func
//   against the dense alternative_cosine/alternative_dot kernels, which
//   never match).

#include "umap/sparse_nn_descent.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>

#include "nn_query_detail.hpp"
#include "umap/rng.hpp"

namespace umap {
namespace spnnd {

namespace {

namespace qd = nnd::detail;

using nnd::pymod;
using qd::EPS;
using qd::FLOAT32_EPS;
using qd::iabs32;

const float INF = std::numeric_limits<float>::infinity();

// Row slice convenience: dist(X[i], X[j]) through a SparseDistFn.
inline float row_dist(SparseDistFn dist, const CsrView& X, int64_t i,
                      int64_t j) {
    return dist(X.row_ind(i), X.row_data(i), X.row_nnz(i), X.row_ind(j),
                X.row_data(j), X.row_nnz(j));
}

// ------------------------------------------------------------ hub tree build

struct SparseHubSplitResult {
    std::vector<int32_t> left;
    std::vector<int32_t> right;
    std::vector<int32_t> hyperplane_inds;  // {-1} for degenerate fallback
    std::vector<float> hyperplane_data;    // {-1.0f} for degenerate fallback
    double offset = 0.0;
};

// rp_trees.py::sparse_euclidean_hub_split
SparseHubSplitResult sparse_euclidean_hub_split(
    const CsrView& X, const std::vector<int32_t>& indices,
    const int32_t* knn_indices, int64_t n_neighbors,
    const std::vector<int32_t>& degrees, std::vector<int32_t>& idx_to_pos,
    int64_t rng_state[3]) {
    const int64_t n_points = static_cast<int64_t>(indices.size());
    std::vector<int32_t> top_hubs =
        qd::get_top_k_hub_indices(indices, degrees, 3);
    const int64_t n_hubs = static_cast<int64_t>(top_hubs.size());

    // Build lookup (idx_to_pos is a reused n-sized scratch buffer; upstream
    // reallocates np.full(n, -1) per split).
    std::fill(idx_to_pos.begin(), idx_to_pos.end(), -1);
    for (int64_t i = 0; i < n_points; ++i)
        idx_to_pos[indices[i]] = static_cast<int32_t>(i);

    SparseHubSplitResult res;
    res.hyperplane_inds.assign(1, -1);
    res.hyperplane_data.assign(1, -1.0f);
    double best_offset = 0.0;
    uint32_t best_edge_cuts = 0xFFFFFFFFu;
    uint32_t best_n_left = 0, best_n_right = 0;
    std::vector<int8_t> best_side(static_cast<size_t>(n_points), 0);
    std::vector<int8_t> side(static_cast<size_t>(n_points));

    std::vector<int32_t> hyperplane_inds, offset_inds, mul_inds;
    std::vector<float> hyperplane_data, offset_data, mul_data;

    for (int64_t hi = 0; hi < n_hubs; ++hi) {
        for (int64_t hj = hi + 1; hj < n_hubs; ++hj) {
            const int64_t left = top_hubs[hi];
            const int64_t right = top_hubs[hj];

            // hyperplane = sparse_diff(left, right)               (float32)
            // offset_data = sparse_sum(left, right) / 2.0         (f8, cast
            //   back to f4 via .astype(np.float32) at the sparse_mul call)
            // hyperplane_offset -= sum(sparse_mul(hp, offset))    (float64)
            double hyperplane_offset = 0.0;
            spdist::sparse_diff(X.row_ind(left), X.row_data(left),
                                X.row_nnz(left), X.row_ind(right),
                                X.row_data(right), X.row_nnz(right),
                                hyperplane_inds, hyperplane_data);
            spdist::sparse_sum(X.row_ind(left), X.row_data(left),
                               X.row_nnz(left), X.row_ind(right),
                               X.row_data(right), X.row_nnz(right),
                               offset_inds, offset_data);
            for (float& v : offset_data)
                v = static_cast<float>(static_cast<double>(v) / 2.0);
            spdist::sparse_mul(hyperplane_inds.data(), hyperplane_data.data(),
                               static_cast<int64_t>(hyperplane_inds.size()),
                               offset_inds.data(), offset_data.data(),
                               static_cast<int64_t>(offset_inds.size()),
                               mul_inds, mul_data);
            for (float v : mul_data)
                hyperplane_offset -= static_cast<double>(v);

            // Project all points.
            uint32_t n_left = 0, n_right = 0;
            for (int64_t i = 0; i < n_points; ++i) {
                double margin = hyperplane_offset;
                const int64_t pi = indices[i];
                spdist::sparse_mul(
                    hyperplane_inds.data(), hyperplane_data.data(),
                    static_cast<int64_t>(hyperplane_inds.size()),
                    X.row_ind(pi), X.row_data(pi), X.row_nnz(pi), mul_inds,
                    mul_data);
                for (float v : mul_data) margin += static_cast<double>(v);

                if (margin > EPS) {
                    side[i] = 0;
                    ++n_left;
                } else if (margin < -EPS) {
                    side[i] = 1;
                    ++n_right;
                } else {
                    side[i] = static_cast<int8_t>(i % 2);
                    if (side[i] == 0)
                        ++n_left;
                    else
                        ++n_right;
                }
            }

            if (n_left == 0 || n_right == 0) continue;

            // Count edge cuts.
            uint32_t edge_cuts = 0;
            for (int64_t i = 0; i < n_points; ++i) {
                const int32_t point_idx = indices[i];
                const int8_t point_side = side[i];
                for (int64_t j_nb = 0; j_nb < n_neighbors; ++j_nb) {
                    int32_t neighbor =
                        knn_indices[static_cast<int64_t>(point_idx) *
                                        n_neighbors +
                                    j_nb];
                    if (neighbor < 0) break;
                    int32_t neighbor_pos = idx_to_pos[neighbor];
                    if (neighbor_pos >= 0) {
                        if (side[neighbor_pos] != point_side) ++edge_cuts;
                    }
                }
            }
            edge_cuts = edge_cuts / 2;

            if (edge_cuts < best_edge_cuts) {
                best_edge_cuts = edge_cuts;
                best_n_left = n_left;
                best_n_right = n_right;
                res.hyperplane_inds = hyperplane_inds;
                res.hyperplane_data = hyperplane_data;
                best_offset = hyperplane_offset;
                best_side = side;
            }
        }
    }

    // Fallback
    if (best_n_left == 0 || best_n_right == 0) {
        best_n_left = 0;
        best_n_right = 0;
        for (int64_t i = 0; i < n_points; ++i) {
            best_side[i] =
                static_cast<int8_t>(pymod(iabs32(tau_rand_int(rng_state)), 2));
            if (best_side[i] == 0)
                ++best_n_left;
            else
                ++best_n_right;
        }
    }

    res.left.reserve(best_n_left);
    res.right.reserve(best_n_right);
    for (int64_t i = 0; i < n_points; ++i) {
        if (best_side[i] == 0)
            res.left.push_back(indices[i]);
        else
            res.right.push_back(indices[i]);
    }
    res.offset = best_offset;
    return res;
}

// rp_trees.py::sparse_angular_hub_split
SparseHubSplitResult sparse_angular_hub_split(
    const CsrView& X, const std::vector<int32_t>& indices,
    const int32_t* knn_indices, int64_t n_neighbors,
    const std::vector<int32_t>& degrees, std::vector<int32_t>& idx_to_pos,
    int64_t rng_state[3]) {
    const int64_t n_points = static_cast<int64_t>(indices.size());
    std::vector<int32_t> top_hubs =
        qd::get_top_k_hub_indices(indices, degrees, 3);
    const int64_t n_hubs = static_cast<int64_t>(top_hubs.size());

    std::fill(idx_to_pos.begin(), idx_to_pos.end(), -1);
    for (int64_t i = 0; i < n_points; ++i)
        idx_to_pos[indices[i]] = static_cast<int32_t>(i);

    SparseHubSplitResult res;
    res.hyperplane_inds.assign(1, -1);
    res.hyperplane_data.assign(1, -1.0f);
    uint32_t best_edge_cuts = 0xFFFFFFFFu;
    uint32_t best_n_left = 0, best_n_right = 0;
    std::vector<int8_t> best_side(static_cast<size_t>(n_points), 0);
    std::vector<int8_t> side(static_cast<size_t>(n_points));

    std::vector<int32_t> hyperplane_inds, mul_inds;
    std::vector<float> hyperplane_data, mul_data;
    std::vector<float> normalized_left, normalized_right;

    for (int64_t hi = 0; hi < n_hubs; ++hi) {
        for (int64_t hj = hi + 1; hj < n_hubs; ++hj) {
            const int64_t left = top_hubs[hi];
            const int64_t right = top_hubs[hj];

            // utils.norm returns float32; the `= 1.0` fallback unifies the
            // local to float64, so divisions run in f8 and cast back to f4.
            double left_norm = static_cast<double>(
                spdist::norm(X.row_data(left), X.row_nnz(left)));
            double right_norm = static_cast<double>(
                spdist::norm(X.row_data(right), X.row_nnz(right)));
            if (std::fabs(left_norm) < EPS) left_norm = 1.0;
            if (std::fabs(right_norm) < EPS) right_norm = 1.0;

            normalized_left.assign(X.row_data(left),
                                   X.row_data(left) + X.row_nnz(left));
            for (float& v : normalized_left)
                v = static_cast<float>(static_cast<double>(v) / left_norm);
            normalized_right.assign(X.row_data(right),
                                    X.row_data(right) + X.row_nnz(right));
            for (float& v : normalized_right)
                v = static_cast<float>(static_cast<double>(v) / right_norm);

            spdist::sparse_diff(X.row_ind(left), normalized_left.data(),
                                X.row_nnz(left), X.row_ind(right),
                                normalized_right.data(), X.row_nnz(right),
                                hyperplane_inds, hyperplane_data);

            // hyperplane_norm is likewise f4 from utils.norm, unified to f8
            // by the fallback assignment.
            double hyperplane_norm = static_cast<double>(spdist::norm(
                hyperplane_data.data(),
                static_cast<int64_t>(hyperplane_data.size())));
            if (std::fabs(hyperplane_norm) < EPS) hyperplane_norm = 1.0;
            for (float& v : hyperplane_data)
                v = static_cast<float>(static_cast<double>(v) /
                                       hyperplane_norm);

            // Project all points.
            uint32_t n_left = 0, n_right = 0;
            for (int64_t i = 0; i < n_points; ++i) {
                double margin = 0.0;
                const int64_t pi = indices[i];
                spdist::sparse_mul(
                    hyperplane_inds.data(), hyperplane_data.data(),
                    static_cast<int64_t>(hyperplane_inds.size()),
                    X.row_ind(pi), X.row_data(pi), X.row_nnz(pi), mul_inds,
                    mul_data);
                for (float v : mul_data) margin += static_cast<double>(v);

                if (margin > EPS) {
                    side[i] = 0;
                    ++n_left;
                } else if (margin < -EPS) {
                    side[i] = 1;
                    ++n_right;
                } else {
                    side[i] = static_cast<int8_t>(i % 2);
                    if (side[i] == 0)
                        ++n_left;
                    else
                        ++n_right;
                }
            }

            if (n_left == 0 || n_right == 0) continue;

            // Count edge cuts.
            uint32_t edge_cuts = 0;
            for (int64_t i = 0; i < n_points; ++i) {
                const int32_t point_idx = indices[i];
                const int8_t point_side = side[i];
                for (int64_t j_nb = 0; j_nb < n_neighbors; ++j_nb) {
                    int32_t neighbor =
                        knn_indices[static_cast<int64_t>(point_idx) *
                                        n_neighbors +
                                    j_nb];
                    if (neighbor < 0) break;
                    int32_t neighbor_pos = idx_to_pos[neighbor];
                    if (neighbor_pos >= 0) {
                        if (side[neighbor_pos] != point_side) ++edge_cuts;
                    }
                }
            }
            edge_cuts = edge_cuts / 2;

            if (edge_cuts < best_edge_cuts) {
                best_edge_cuts = edge_cuts;
                best_n_left = n_left;
                best_n_right = n_right;
                res.hyperplane_inds = hyperplane_inds;
                res.hyperplane_data = hyperplane_data;
                best_side = side;
            }
        }
    }

    // Fallback
    if (best_n_left == 0 || best_n_right == 0) {
        best_n_left = 0;
        best_n_right = 0;
        for (int64_t i = 0; i < n_points; ++i) {
            best_side[i] =
                static_cast<int8_t>(pymod(iabs32(tau_rand_int(rng_state)), 2));
            if (best_side[i] == 0)
                ++best_n_left;
            else
                ++best_n_right;
        }
    }

    res.left.reserve(best_n_left);
    res.right.reserve(best_n_right);
    for (int64_t i = 0; i < n_points; ++i) {
        if (best_side[i] == 0)
            res.left.push_back(indices[i]);
        else
            res.right.push_back(indices[i]);
    }
    res.offset = 0.0;  // np.float64(0.0)
    return res;
}

// Node storage mirroring the typed-list appends of
// make_sparse_hub_{euclidean,angular}_tree (post-order; root is last).
struct SparseHubTreeNodes {
    std::vector<std::vector<int32_t>> hyperplane_inds;
    std::vector<std::vector<float>> hyperplane_data;
    std::vector<double> offsets;                        // -inf for leaves
    std::vector<std::pair<int32_t, int32_t>> children;  // (-1,-1) for leaves
    std::vector<std::vector<int32_t>> point_indices;    // {-1} for internal
};

void append_sparse_leaf(SparseHubTreeNodes& t, std::vector<int32_t> indices) {
    // hyperplanes.append([[-1.0], [-1.0]]) -- sentinel, trims to empty.
    t.hyperplane_inds.push_back({-1});
    t.hyperplane_data.push_back({-1.0f});
    t.offsets.push_back(-std::numeric_limits<double>::infinity());
    t.children.emplace_back(-1, -1);
    t.point_indices.push_back(std::move(indices));
}

// rp_trees.py::make_sparse_hub_euclidean_tree / make_sparse_hub_angular_tree
void make_sparse_hub_tree_recursive(
    const CsrView& X, std::vector<int32_t> indices,
    const int32_t* knn_indices, int64_t n_neighbors,
    const std::vector<int32_t>& degrees, std::vector<int32_t>& idx_to_pos,
    SparseHubTreeNodes& t, int64_t rng_state[3], int64_t leaf_size,
    bool angular, int64_t max_depth) {
    if (static_cast<int64_t>(indices.size()) > leaf_size && max_depth > 0) {
        SparseHubSplitResult s =
            angular ? sparse_angular_hub_split(X, indices, knn_indices,
                                               n_neighbors, degrees,
                                               idx_to_pos, rng_state)
                    : sparse_euclidean_hub_split(X, indices, knn_indices,
                                                 n_neighbors, degrees,
                                                 idx_to_pos, rng_state);
        make_sparse_hub_tree_recursive(X, std::move(s.left), knn_indices,
                                       n_neighbors, degrees, idx_to_pos, t,
                                       rng_state, leaf_size, angular,
                                       max_depth - 1);
        int32_t left_node_num =
            static_cast<int32_t>(t.point_indices.size()) - 1;
        make_sparse_hub_tree_recursive(X, std::move(s.right), knn_indices,
                                       n_neighbors, degrees, idx_to_pos, t,
                                       rng_state, leaf_size, angular,
                                       max_depth - 1);
        int32_t right_node_num =
            static_cast<int32_t>(t.point_indices.size()) - 1;
        t.hyperplane_inds.push_back(std::move(s.hyperplane_inds));
        t.hyperplane_data.push_back(std::move(s.hyperplane_data));
        t.offsets.push_back(s.offset);
        t.children.emplace_back(left_node_num, right_node_num);
        t.point_indices.push_back({-1});
    } else {
        append_sparse_leaf(t, std::move(indices));
    }
}

// rp_trees.py::recursive_convert_sparse. Hyperplanes are stored trimmed:
// the padded flat row 0 is -1 beyond the true length, and
// sparse_select_side's trailing-negative scan recovers exactly the stored
// (inds, data) prefix; the sentinel [-1] hyperplane trims to empty.
std::pair<int64_t, int64_t> recursive_convert_sparse(
    const SparseHubTreeNodes& t, SparseFlatTree& out, int64_t node_num,
    int64_t leaf_start, int64_t tree_node) {
    if (t.children[tree_node].first < 0) {
        const std::vector<int32_t>& leaf = t.point_indices[tree_node];
        int64_t leaf_end = leaf_start + static_cast<int64_t>(leaf.size());
        out.children[node_num * 2] = static_cast<int32_t>(-leaf_start);
        out.children[node_num * 2 + 1] = static_cast<int32_t>(-leaf_end);
        std::copy(leaf.begin(), leaf.end(), out.indices.begin() + leaf_start);
        return {node_num, leaf_end};
    }
    // Trim trailing negative "indices" (the -1 padding / sentinel), exactly
    // what sparse_select_side's scan does on the padded float32 row.
    const std::vector<int32_t>& hin = t.hyperplane_inds[tree_node];
    const std::vector<float>& hda = t.hyperplane_data[tree_node];
    size_t hp_size = hin.size();
    while (hp_size > 0 && static_cast<float>(hin[hp_size - 1]) < 0.0f)
        --hp_size;
    out.hyperplane_inds[node_num].assign(hin.begin(), hin.begin() + hp_size);
    out.hyperplane_data[node_num].assign(hda.begin(), hda.begin() + hp_size);
    out.offsets[node_num] = static_cast<float>(t.offsets[tree_node]);
    out.children[node_num * 2] = static_cast<int32_t>(node_num + 1);
    int64_t old_node_num = node_num;
    auto r = recursive_convert_sparse(t, out, node_num + 1, leaf_start,
                                      t.children[tree_node].first);
    node_num = r.first;
    leaf_start = r.second;
    out.children[old_node_num * 2 + 1] = static_cast<int32_t>(node_num + 1);
    return recursive_convert_sparse(t, out, node_num + 1, leaf_start,
                                    t.children[tree_node].second);
}

// ------------------------------------------------------------ search graph

// sparse.py::diversify (operates in place on copies of the kNN graph arrays).
void sparse_diversify(int32_t* indices, float* distances, int64_t n,
                      int64_t k, const CsrView& X, SparseDistFn dist,
                      int64_t rng_state[3], double prune_probability) {
    std::vector<int32_t> new_indices;
    std::vector<float> new_distances;
    for (int64_t i = 0; i < n; ++i) {
        new_indices.assign(1, indices[i * k]);
        new_distances.assign(1, distances[i * k]);
        for (int64_t j = 1; j < k; ++j) {
            if (indices[i * k + j] < 0) break;
            bool flag = true;
            for (size_t kk = 0; kk < new_indices.size(); ++kk) {
                int32_t c = new_indices[kk];
                float d = row_dist(dist, X, indices[i * k + j], c);
                if (new_distances[kk] > FLOAT32_EPS &&
                    d < distances[i * k + j]) {
                    if (static_cast<double>(tau_rand(rng_state)) <
                        prune_probability) {
                        flag = false;
                        break;
                    }
                }
            }
            if (flag) {
                new_indices.push_back(indices[i * k + j]);
                new_distances.push_back(distances[i * k + j]);
            }
        }
        for (int64_t j = 0; j < k; ++j) {
            if (j < static_cast<int64_t>(new_indices.size())) {
                indices[i * k + j] = new_indices[j];
                distances[i * k + j] = new_distances[j];
            } else {
                indices[i * k + j] = -1;
                distances[i * k + j] = INF;
            }
        }
    }
}

// sparse.py::diversify_csr applied to the forward CSR rows (see the
// transpose-sharing note in nn_query.cpp / the header). Prunes entries in
// place. NB: unlike the dense diversify_csr, the occluding *data point* is
// correctly indexed via order[k] here (upstream difference, kept).
void sparse_diversify_csr_rows(qd::GraphRows& g, const CsrView& X,
                               SparseDistFn dist, int64_t rng_state[3],
                               double prune_probability) {
    const int64_t n = static_cast<int64_t>(g.col.size());
    std::vector<int64_t> order;
    std::vector<int8_t> retained;
    for (int64_t i = 0; i < n; ++i) {
        std::vector<int32_t>& cols = g.col[i];
        std::vector<float>& vals = g.val[i];
        const int64_t m = static_cast<int64_t>(cols.size());
        order.resize(m);
        std::iota(order.begin(), order.end(), 0);
        // np.argsort(current_data); rows are ascending so this is the
        // identity except for exact ties (numpy's introsort falls back to a
        // stable insertion sort below 16 elements, and rows here have at
        // most n_neighbors entries).
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return vals[a] < vals[b];
        });
        retained.assign(m, 1);
        for (int64_t idx = 1; idx < m; ++idx) {
            int64_t j = order[idx];
            for (int64_t kk = 0; kk < idx; ++kk) {
                int64_t l = order[kk];
                if (retained[l] == 1) {
                    int32_t p = cols[j];
                    int32_t q = cols[l];
                    float d = row_dist(dist, X, p, q);
                    if (vals[l] > FLOAT32_EPS && d < vals[j]) {
                        if (static_cast<double>(tau_rand(rng_state)) <
                            prune_probability) {
                            retained[j] = 0;
                            break;
                        }
                    }
                }
            }
        }
        // zero + eliminate (order-preserving compaction)
        int64_t w = 0;
        for (int64_t p = 0; p < m; ++p) {
            if (retained[p]) {
                cols[w] = cols[p];
                vals[w] = vals[p];
                ++w;
            }
        }
        cols.resize(w);
        vals.resize(w);
    }
}

}  // namespace

// ---------------------------------------------------------------- toplevels

SparseFlatTree make_sparse_hub_tree(const CsrView& X,
                                    const int32_t* knn_indices,
                                    int64_t n_neighbors, int64_t rng_state[3],
                                    int64_t leaf_size, bool angular,
                                    int64_t max_depth) {
    const int64_t n = X.n_rows;
    std::vector<int32_t> degrees =
        qd::compute_global_degrees(knn_indices, n, n_neighbors);
    std::vector<int32_t> all_indices(static_cast<size_t>(n));
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::vector<int32_t> idx_to_pos(static_cast<size_t>(n), -1);

    SparseHubTreeNodes t;
    make_sparse_hub_tree_recursive(X, std::move(all_indices), knn_indices,
                                   n_neighbors, degrees, idx_to_pos, t,
                                   rng_state, leaf_size, angular, max_depth);

    SparseFlatTree out;
    out.n_nodes = static_cast<int64_t>(t.children.size());
    out.hyperplane_inds.assign(static_cast<size_t>(out.n_nodes), {});
    out.hyperplane_data.assign(static_cast<size_t>(out.n_nodes), {});
    out.offsets.assign(static_cast<size_t>(out.n_nodes), 0.0f);
    out.children.assign(static_cast<size_t>(out.n_nodes * 2), -1);
    out.indices.assign(static_cast<size_t>(n), -1);
    recursive_convert_sparse(t, out, 0, 0, out.n_nodes - 1);
    return out;
}

nnd::SearchGraph build_sparse_search_graph(
    const CsrView& X, const int32_t* knn_indices, const float* knn_dists,
    int64_t n_neighbors, SparseDistFn dist, int64_t rng_state[3],
    double diversify_prob, double pruning_degree_multiplier) {
    const int64_t n = X.n_rows;
    const int64_t k = n_neighbors;

    // Forward diversify on copies (compressed=False path).
    std::vector<int32_t> div_ind(knn_indices, knn_indices + n * k);
    std::vector<float> div_dist(knn_dists, knn_dists + n * k);
    sparse_diversify(div_ind.data(), div_dist.data(), n, k, X, dist,
                     rng_state, diversify_prob);

    // COO -> CSR: rows keep insertion (distance-ascending) order; zero
    // distances become FLOAT32_EPS; -1 entries are dropped.
    qd::GraphRows fwd;
    float min_distance =
        qd::assemble_forward_rows(div_ind.data(), div_dist.data(), n, k, fwd);

    // "Reverse" diversification (operates on the forward rows).
    sparse_diversify_csr_rows(fwd, X, dist, rng_state, diversify_prob);

    // M = elementwise max(F2, F2^T); drop the diagonal; degree_prune.
    const int64_t max_degree = static_cast<int64_t>(
        std::nearbyint(pruning_degree_multiplier * static_cast<double>(k)));
    nnd::SearchGraph out = qd::finalize_search_graph(fwd, n, max_degree);
    out.min_distance = min_distance;
    return out;
}

// ------------------------------------------------------ SparsePreparedIndex

SparsePreparedIndex SparsePreparedIndex::build(
    const int32_t* indptr, const int32_t* indices, const float* data,
    int64_t n, int64_t dim, int64_t n_neighbors, const std::string& metric,
    uint32_t random_seed, bool verbose) {
    NumpyRandomState numpy_rs(random_seed);
    return build(indptr, indices, data, n, dim, n_neighbors, metric, numpy_rs,
                 verbose);
}

SparsePreparedIndex SparsePreparedIndex::build(
    const int32_t* indptr, const int32_t* indices, const float* data,
    int64_t n, int64_t dim, int64_t n_neighbors, const std::string& metric,
    NumpyRandomState& numpy_rs, bool verbose) {
    SparsePreparedIndex idx;
    idx.n_ = n;
    idx.dim_ = dim;
    idx.n_neighbors_ = n_neighbors;
    idx.metric_ = spdist::resolve_sparse_metric(metric);
    CsrView X{indptr, indices, data, n, dim};

    // umap_.py::nearest_neighbors "hacked values"
    const int64_t n_trees = std::min<int64_t>(
        64, 5 + static_cast<int64_t>(
                    std::nearbyint(std::sqrt(static_cast<double>(n)) / 20.0)));
    const int64_t n_iters = std::max<int64_t>(
        5,
        static_cast<int64_t>(std::nearbyint(std::log2(static_cast<double>(n)))));

    // NNDescent.__init__ seeding order: rng_state, search_rng_state (then a
    // 10x tau_rand_int warm-up), then the per-tree states inside make_forest.
    std::array<int64_t, 3> rng_state_arr = numpy_rs.rng_state3();
    idx.search_rng_state_ = numpy_rs.rng_state3();
    {
        int64_t warm[3] = {idx.search_rng_state_[0], idx.search_rng_state_[1],
                           idx.search_rng_state_[2]};
        for (int i = 0; i < 10; ++i) (void)tau_rand_int(warm);
        idx.search_rng_state_ = {warm[0], warm[1], warm[2]};
    }

    LeafArray leaf_array = make_sparse_forest_leaf_array(
        X, n_neighbors, n_trees, /*leaf_size=*/-1, idx.metric_.angular_trees,
        /*max_depth=*/200, numpy_rs);

    int64_t rng_state[3] = {rng_state_arr[0], rng_state_arr[1],
                            rng_state_arr[2]};
    NNHeap graph = sparse_nn_descent(X, n_neighbors, rng_state,
                                     /*max_candidates=*/60, idx.metric_.dist,
                                     n_iters, /*delta=*/0.001, leaf_array,
                                     verbose);

    // NNDescent.neighbor_graph (corrected distances, original data order).
    idx.knn_indices_.assign(graph.ind.begin(), graph.ind.end());
    idx.knn_dists_.resize(graph.dist.size());
    for (size_t i = 0; i < graph.dist.size(); ++i)
        idx.knn_dists_[i] = idx.metric_.correction
                                ? idx.metric_.correction(graph.dist[i])
                                : graph.dist[i];

    // ---- prepare() / _init_search_graph (sparse branch) --------------------
    // Sparse hub search tree first (search_leaf_size=30 when leaf_size is
    // None, max_search_tree_depth defaults to max_rptree_depth=200);
    // rng_state continues from where nn_descent left it.
    idx.tree_ = spnnd::make_sparse_hub_tree(X, graph.ind.data(), n_neighbors,
                                            rng_state, /*leaf_size=*/30,
                                            idx.metric_.angular_trees,
                                            /*max_depth=*/200);

    // Search graph (UMAP defaults: diversify_prob=1.0,
    // pruning_degree_multiplier=1.5), still in original data order.
    nnd::SearchGraph sg = build_sparse_search_graph(
        X, graph.ind.data(), graph.dist.data(), n_neighbors, idx.metric_.dist,
        rng_state, /*diversify_prob=*/1.0, /*pruning_degree_multiplier=*/1.5);

    // Re-order data + graph by the search tree leaf order (_vertex_order).
    idx.vertex_order_ = idx.tree_.indices;
    std::vector<int32_t> inv(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        inv[idx.vertex_order_[i]] = static_cast<int32_t>(i);

    // raw_data[vertex_order, :] -- CSR row selection keeps each row's
    // (sorted) internal order.
    idx.data_indptr_.assign(static_cast<size_t>(n + 1), 0);
    idx.data_indices_.clear();
    idx.data_data_.clear();
    idx.data_indices_.reserve(static_cast<size_t>(indptr[n]));
    idx.data_data_.reserve(static_cast<size_t>(indptr[n]));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t old_row = idx.vertex_order_[i];
        for (int32_t p = indptr[old_row]; p < indptr[old_row + 1]; ++p) {
            idx.data_indices_.push_back(indices[p]);
            idx.data_data_.push_back(data[p]);
        }
        idx.data_indptr_[i + 1] =
            static_cast<int32_t>(idx.data_indices_.size());
    }

    idx.search_graph_ =
        qd::reorder_search_graph(sg, n, idx.vertex_order_, inv);

    // resort_tree_indices: tree.indices[argsort(vertex_order)] == arange(n).
    for (int64_t i = 0; i < n; ++i)
        idx.tree_.indices[i] = static_cast<int32_t>(i);

    return idx;
}

void SparsePreparedIndex::query(const int32_t* q_indptr,
                                const int32_t* q_indices, const float* q_data,
                                int64_t n_queries, int64_t k, double epsilon,
                                int32_t* out_indices,
                                float* out_dists) const {
    NNHeap result = NNHeap::make(n_queries, k);
    int64_t internal_rng_state[3] = {search_rng_state_[0], search_rng_state_[1],
                                     search_rng_state_[2]};
    std::vector<uint8_t> visited(static_cast<size_t>((n_ / 8) + 1), 0);
    std::vector<qd::SeedItem> seed_set;
    const SparseDistFn dist = metric_.dist;
    const CsrView D{data_indptr_.data(), data_indices_.data(),
                    data_data_.data(), n_, dim_};

    // distance_scale is a declared float32 numba local: f4(1 + epsilon), and
    // the bound stays float32 throughout (no min_distance term, unlike the
    // dense closure).
    const float distance_scale = static_cast<float>(1.0 + epsilon);

    for (int64_t i = 0; i < n_queries; ++i) {
        std::fill(visited.begin(), visited.end(), 0);

        const int32_t* cq_inds = q_indices + q_indptr[i];
        const float* cq_data = q_data + q_indptr[i];
        const int64_t cq_nnz =
            static_cast<int64_t>(q_indptr[i + 1]) - q_indptr[i];
        // NB: no query normalisation -- the reference's alternative_dot /
        // alternative_cosine identity check never matches a sparse
        // _distance_func (see header notes).

        float* heap_priorities = result.dist_row(i);
        int32_t* heap_indices = result.ind_row(i);
        seed_set.clear();

        // ---- init from tree ------------------------------------------------
        uint32_t node = 0;
        while (tree_.children[node * 2] > 0) {
            // sparse_select_side: float32 margin (f4 offset + f4
            // sparse_dot_product), EPS coin flip via tau_rand_int % 2
            // (Python modulo, no abs).
            float margin = tree_.offsets[node];
            const std::vector<int32_t>& hin = tree_.hyperplane_inds[node];
            const std::vector<float>& hda = tree_.hyperplane_data[node];
            margin += spdist::sparse_dot_product(
                hin.data(), hda.data(), static_cast<int64_t>(hin.size()),
                cq_inds, cq_data, cq_nnz);
            int side;
            if (std::fabs(static_cast<double>(margin)) < EPS)
                side = static_cast<int>(
                    pymod(tau_rand_int(internal_rng_state), 2));
            else
                side = margin > 0.0f ? 0 : 1;
            node = static_cast<uint32_t>(tree_.children[node * 2 + side]);
        }
        const int64_t leaf_start = -tree_.children[node * 2];
        const int64_t leaf_end = -tree_.children[node * 2 + 1];
        const int64_t n_initial_points = leaf_end - leaf_start;
        const int64_t n_random_samples =
            std::min(k, n_neighbors_) - n_initial_points;

        for (int64_t j = leaf_start; j < leaf_end; ++j) {
            int32_t candidate = tree_.indices[j];
            // d = dist(data[candidate], current_query)
            float d = dist(D.row_ind(candidate), D.row_data(candidate),
                           D.row_nnz(candidate), cq_inds, cq_data, cq_nnz);
            nnd::simple_heap_push(heap_priorities, heap_indices, k, d,
                                  candidate);
            qd::heapq_push(seed_set, {d, candidate});
            qd::mark_visited(visited, candidate);
        }

        // ---- random samples if needed -------------------------------------
        for (int64_t j = 0; j < n_random_samples; ++j) {
            int32_t candidate = static_cast<int32_t>(
                pymod(iabs32(tau_rand_int(internal_rng_state)), n_));
            if (qd::check_and_mark_visited(visited, candidate) == 0) {
                float d = dist(D.row_ind(candidate), D.row_data(candidate),
                               D.row_nnz(candidate), cq_inds, cq_data,
                               cq_nnz);
                nnd::simple_heap_push(heap_priorities, heap_indices, k, d,
                                      candidate);
                qd::heapq_push(seed_set, {d, candidate});
            }
        }

        // ---- search --------------------------------------------------------
        float distance_bound = distance_scale * heap_priorities[0];

        if (seed_set.empty()) continue;
        qd::SeedItem top = qd::heapq_pop(seed_set);
        float d_vertex = top.d;
        int32_t vertex = top.idx;

        while (d_vertex < distance_bound) {
            for (int32_t j = search_graph_.indptr[vertex];
                 j < search_graph_.indptr[vertex + 1]; ++j) {
                int32_t candidate = search_graph_.indices[j];
                if (qd::check_and_mark_visited(visited, candidate) == 0) {
                    float d = dist(D.row_ind(candidate),
                                   D.row_data(candidate),
                                   D.row_nnz(candidate), cq_inds, cq_data,
                                   cq_nnz);
                    if (d < distance_bound) {
                        nnd::simple_heap_push(heap_priorities, heap_indices,
                                              k, d, candidate);
                        qd::heapq_push(seed_set, {d, candidate});
                        distance_bound = distance_scale * heap_priorities[0];
                    }
                }
            }
            if (seed_set.empty()) break;
            top = qd::heapq_pop(seed_set);
            d_vertex = top.d;
            vertex = top.idx;
        }
    }

    nnd::deheap_sort(result.ind.data(), result.dist.data(), n_queries, k);

    for (int64_t i = 0; i < n_queries * k; ++i) {
        int32_t ind = result.ind[i];
        // numpy fancy indexing: vertex_order[-1] wraps to the last vertex.
        out_indices[i] = vertex_order_[ind >= 0 ? ind : n_ + ind];
        float d = result.dist[i];
        out_dists[i] = metric_.correction ? metric_.correction(d) : d;
    }
}

}  // namespace spnnd
}  // namespace umap
