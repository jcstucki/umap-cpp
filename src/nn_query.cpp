// Faithful C++ port of the pynndescent query path used by UMAP's transform():
// NNDescent.prepare() (_init_search_graph: hub search tree + diversified /
// degree-pruned CSR search graph) and NNDescent.query() (tree-descent search
// init + epsilon-bounded graph search), from
// reference/pynndescent/pynndescent/{pynndescent_.py, rp_trees.py, utils.py}
// (BSD 2 clause).
//
// Faithfulness notes:
// - All float32/float64 promotions mirror the numba typing of the reference:
//   euclidean hub-split margins/offsets accumulate in float64 (offset_type is
//   numba.float64 and the /2.0 promotes), angular hub-split margins and
//   select_side margins accumulate in float32 (locals={"margin": float32}).
// - scipy quirk replicated: `reverse_graph = search_graph.transpose()` shares
//   the CSR arrays, so diversify_csr (the "reverse diversification") actually
//   runs over the *forward* CSR rows, and its pruning is reflected in both the
//   forward graph and its transpose.  The net effect implemented here:
//     F2 = diversify_csr(forward rows);  M = elementwise max(F2, F2^T).
// - diversify_csr quirk replicated: the occluding point's distance is looked
//   up via `order[k]` but the occluding *data point* via raw position `k`
//   (upstream indexing bug, kept for parity).
// - np.argsort in diversify_csr is applied to rows that are already ascending
//   (kNN rows are deheap-sorted and forward diversify preserves order), so a
//   stable argsort is exact except for exactly-tied distances.
// - The Python query maps heap slots that stayed empty (index -1) through
//   numpy fancy indexing, yielding vertex_order[-1] (the *last* vertex); this
//   wrap is replicated.
#include "umap/nn_descent.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

#include "nn_query_detail.hpp"
#include "umap/rng.hpp"

namespace umap {
namespace nnd {

namespace {

using detail::EPS;
using detail::FLOAT32_EPS;
using detail::GraphRows;
using detail::compute_global_degrees;
using detail::get_top_k_hub_indices;
using detail::iabs32;

constexpr float MIN_SPLIT_BALANCE = 0.1f;       // rp_trees.py
const float INF = std::numeric_limits<float>::infinity();

// utils.py::norm -- float64 accumulation, float64 return.
inline double l2_norm(const float* vec, int64_t dim) {
    double result = 0.0;
    for (int64_t i = 0; i < dim; ++i)
        result += static_cast<double>(vec[i]) * static_cast<double>(vec[i]);
    return std::sqrt(result);
}

// ------------------------------------------------------------ hub tree build

struct HubSplitResult {
    std::vector<int32_t> left;
    std::vector<int32_t> right;
    std::vector<float> hyperplane;
    double offset = 0.0;
    float balance = 0.0f;
};

// rp_trees.py::euclidean_hub_split. Margin/offset math promotes to float64
// exactly as numba types it (offset accumulates f8 via the /2.0; margin adds
// f4 products into the f8 offset).
HubSplitResult euclidean_hub_split(const float* data, int64_t dim,
                                   const std::vector<int32_t>& indices,
                                   const std::vector<int32_t>& degrees,
                                   int64_t rng_state[3]) {
    const int64_t n_points = static_cast<int64_t>(indices.size());
    std::vector<int32_t> top_hubs = get_top_k_hub_indices(indices, degrees, 3);
    const int64_t n_hubs = static_cast<int64_t>(top_hubs.size());

    HubSplitResult res;
    float best_balance = 0.0f;
    uint32_t best_n_left = 0, best_n_right = 0;
    std::vector<float> best_hyperplane(static_cast<size_t>(dim), 0.0f);
    double best_offset = 0.0;
    std::vector<int8_t> best_side(static_cast<size_t>(n_points), 0);
    std::vector<int8_t> side(static_cast<size_t>(n_points));

    for (int64_t hi = 0; hi < n_hubs; ++hi) {
        for (int64_t hj = hi + 1; hj < n_hubs; ++hj) {
            const float* left = data + static_cast<int64_t>(top_hubs[hi]) * dim;
            const float* right = data + static_cast<int64_t>(top_hubs[hj]) * dim;

            double hyperplane_offset = 0.0;
            std::vector<float> hyperplane_vector(static_cast<size_t>(dim));
            for (int64_t d = 0; d < dim; ++d) {
                hyperplane_vector[d] = left[d] - right[d];
                // f4 * (f4 + f4) product, / 2.0 promotes to f8
                float prod = hyperplane_vector[d] * (left[d] + right[d]);
                hyperplane_offset -= static_cast<double>(prod) / 2.0;
            }

            uint32_t n_left = 0, n_right = 0;
            for (int64_t i = 0; i < n_points; ++i) {
                double margin = hyperplane_offset;
                const float* p = data + static_cast<int64_t>(indices[i]) * dim;
                for (int64_t d = 0; d < dim; ++d)
                    margin += static_cast<double>(hyperplane_vector[d] * p[d]);

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

            float balance = static_cast<float>(std::min(n_left, n_right)) /
                            static_cast<float>(n_points);
            if (balance > best_balance) {
                best_balance = balance;
                best_n_left = n_left;
                best_n_right = n_right;
                best_offset = hyperplane_offset;
                best_hyperplane = hyperplane_vector;
                best_side = side;
            }
        }
    }

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
    res.hyperplane = std::move(best_hyperplane);
    res.offset = best_offset;
    res.balance = best_balance;
    return res;
}

// rp_trees.py::angular_hub_split. Margin accumulates in float32 (numba
// unifies it as f4 there, unlike the euclidean variant).
HubSplitResult angular_hub_split(const float* data, int64_t dim,
                                 const std::vector<int32_t>& indices,
                                 const std::vector<int32_t>& degrees,
                                 int64_t rng_state[3]) {
    const int64_t n_points = static_cast<int64_t>(indices.size());
    std::vector<int32_t> top_hubs = get_top_k_hub_indices(indices, degrees, 3);
    const int64_t n_hubs = static_cast<int64_t>(top_hubs.size());

    HubSplitResult res;
    float best_balance = 0.0f;
    uint32_t best_n_left = 0, best_n_right = 0;
    std::vector<float> best_hyperplane(static_cast<size_t>(dim), 0.0f);
    std::vector<int8_t> best_side(static_cast<size_t>(n_points), 0);
    std::vector<int8_t> side(static_cast<size_t>(n_points));

    for (int64_t hi = 0; hi < n_hubs; ++hi) {
        for (int64_t hj = hi + 1; hj < n_hubs; ++hj) {
            const float* left = data + static_cast<int64_t>(top_hubs[hi]) * dim;
            const float* right = data + static_cast<int64_t>(top_hubs[hj]) * dim;

            double left_norm = l2_norm(left, dim);
            double right_norm = l2_norm(right, dim);
            if (std::fabs(left_norm) < EPS) left_norm = 1.0;
            if (std::fabs(right_norm) < EPS) right_norm = 1.0;

            std::vector<float> hyperplane_vector(static_cast<size_t>(dim));
            for (int64_t d = 0; d < dim; ++d)
                hyperplane_vector[d] = static_cast<float>(
                    static_cast<double>(left[d]) / left_norm -
                    static_cast<double>(right[d]) / right_norm);

            double hyperplane_norm = l2_norm(hyperplane_vector.data(), dim);
            if (std::fabs(hyperplane_norm) < EPS) hyperplane_norm = 1.0;
            for (int64_t d = 0; d < dim; ++d)
                hyperplane_vector[d] = static_cast<float>(
                    static_cast<double>(hyperplane_vector[d]) / hyperplane_norm);

            uint32_t n_left = 0, n_right = 0;
            for (int64_t i = 0; i < n_points; ++i) {
                float margin = 0.0f;
                const float* p = data + static_cast<int64_t>(indices[i]) * dim;
                for (int64_t d = 0; d < dim; ++d)
                    margin += hyperplane_vector[d] * p[d];

                if (static_cast<double>(margin) > EPS) {
                    side[i] = 0;
                    ++n_left;
                } else if (static_cast<double>(margin) < -EPS) {
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

            float balance = static_cast<float>(std::min(n_left, n_right)) /
                            static_cast<float>(n_points);
            if (balance > best_balance) {
                best_balance = balance;
                best_n_left = n_left;
                best_n_right = n_right;
                best_hyperplane = hyperplane_vector;
                best_side = side;
            }
        }
    }

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
    res.hyperplane = std::move(best_hyperplane);
    res.offset = 0.0;  // np.float32(0.0)
    res.balance = best_balance;
    return res;
}

// Node storage mirroring the typed-list appends of make_hub_{euclidean,
// angular}_tree (children appended post-order; root is the last node).
struct HubTreeNodes {
    std::vector<std::vector<float>> hyperplanes;  // empty vec for leaves
    std::vector<double> offsets;                  // -inf for leaves
    std::vector<std::pair<int32_t, int32_t>> children;  // (-1,-1) for leaves
    std::vector<std::vector<int32_t>> point_indices;    // {-1} for internal
};

void append_leaf(HubTreeNodes& t, std::vector<int32_t> indices) {
    t.hyperplanes.emplace_back();
    t.offsets.push_back(-std::numeric_limits<double>::infinity());
    t.children.emplace_back(-1, -1);
    t.point_indices.push_back(std::move(indices));
}

// rp_trees.py::make_hub_euclidean_tree / make_hub_angular_tree
void make_hub_tree_recursive(const float* data, int64_t dim,
                             std::vector<int32_t> indices,
                             const std::vector<int32_t>& degrees,
                             HubTreeNodes& t, int64_t rng_state[3],
                             int64_t leaf_size, bool angular,
                             int64_t max_depth) {
    if (static_cast<int64_t>(indices.size()) > leaf_size && max_depth > 0) {
        HubSplitResult s =
            angular ? angular_hub_split(data, dim, indices, degrees, rng_state)
                    : euclidean_hub_split(data, dim, indices, degrees,
                                          rng_state);
        if (s.balance < MIN_SPLIT_BALANCE) {
            append_leaf(t, std::move(indices));
            return;
        }
        make_hub_tree_recursive(data, dim, std::move(s.left), degrees, t,
                                rng_state, leaf_size, angular, max_depth - 1);
        int32_t left_node_num =
            static_cast<int32_t>(t.point_indices.size()) - 1;
        make_hub_tree_recursive(data, dim, std::move(s.right), degrees, t,
                                rng_state, leaf_size, angular, max_depth - 1);
        int32_t right_node_num =
            static_cast<int32_t>(t.point_indices.size()) - 1;
        t.hyperplanes.push_back(std::move(s.hyperplane));
        t.offsets.push_back(s.offset);
        t.children.emplace_back(left_node_num, right_node_num);
        t.point_indices.push_back({-1});
    } else {
        append_leaf(t, std::move(indices));
    }
}

// rp_trees.py::recursive_convert
std::pair<int64_t, int64_t> recursive_convert(const HubTreeNodes& t,
                                              FlatTree& out, int64_t node_num,
                                              int64_t leaf_start,
                                              int64_t tree_node) {
    if (t.children[tree_node].first < 0) {
        const std::vector<int32_t>& leaf = t.point_indices[tree_node];
        int64_t leaf_end = leaf_start + static_cast<int64_t>(leaf.size());
        out.children[node_num * 2] = static_cast<int32_t>(-leaf_start);
        out.children[node_num * 2 + 1] = static_cast<int32_t>(-leaf_end);
        std::copy(leaf.begin(), leaf.end(), out.indices.begin() + leaf_start);
        return {node_num, leaf_end};
    }
    const std::vector<float>& hp = t.hyperplanes[tree_node];
    std::copy(hp.begin(), hp.end(), out.hyperplanes.begin() + node_num * out.dim);
    out.offsets[node_num] = static_cast<float>(t.offsets[tree_node]);
    out.children[node_num * 2] = static_cast<int32_t>(node_num + 1);
    int64_t old_node_num = node_num;
    auto r = recursive_convert(t, out, node_num + 1, leaf_start,
                               t.children[tree_node].first);
    node_num = r.first;
    leaf_start = r.second;
    out.children[old_node_num * 2 + 1] = static_cast<int32_t>(node_num + 1);
    return recursive_convert(t, out, node_num + 1, leaf_start,
                             t.children[tree_node].second);
}

// ------------------------------------------------------------ search graph

// pynndescent_.py::diversify (dense; operates in place on copies of the kNN
// graph arrays).
void diversify_dense(int32_t* indices, float* distances, int64_t n, int64_t k,
                     const float* data, int64_t dim, DistFn dist,
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
                float d = dist(
                    data + static_cast<int64_t>(indices[i * k + j]) * dim,
                    data + static_cast<int64_t>(c) * dim, dim);
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

// pynndescent_.py::diversify_csr applied to the forward CSR rows (see the
// transpose-sharing note at the top of this file). Prunes entries in place.
void diversify_csr_rows(GraphRows& g, const float* data, int64_t dim,
                        DistFn dist, int64_t rng_state[3],
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
        // identity except for exact ties (stable here, unspecified upstream).
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return vals[a] < vals[b];
        });
        retained.assign(m, 1);
        for (int64_t idx = 1; idx < m; ++idx) {
            int64_t j = order[idx];
            for (int64_t kk = 0; kk < idx; ++kk) {
                int64_t l = order[kk];
                if (retained[l] == 1) {
                    // NB: upstream compares against current_indices[k] (raw
                    // position), not current_indices[order[k]].
                    float d =
                        dist(data + static_cast<int64_t>(cols[j]) * dim,
                             data + static_cast<int64_t>(cols[kk]) * dim, dim);
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

FlatTree make_hub_tree(const float* data, int64_t n, int64_t dim,
                       const int32_t* knn_indices, int64_t n_neighbors,
                       int64_t rng_state[3], int64_t leaf_size, bool angular,
                       int64_t max_depth) {
    std::vector<int32_t> degrees =
        compute_global_degrees(knn_indices, n, n_neighbors);
    std::vector<int32_t> all_indices(static_cast<size_t>(n));
    std::iota(all_indices.begin(), all_indices.end(), 0);

    HubTreeNodes t;
    make_hub_tree_recursive(data, dim, std::move(all_indices), degrees, t,
                            rng_state, leaf_size, angular, max_depth);

    FlatTree out;
    out.n_nodes = static_cast<int64_t>(t.children.size());
    out.dim = dim;
    out.hyperplanes.assign(static_cast<size_t>(out.n_nodes * dim), 0.0f);
    out.offsets.assign(static_cast<size_t>(out.n_nodes), 0.0f);
    out.children.assign(static_cast<size_t>(out.n_nodes * 2), -1);
    out.indices.assign(static_cast<size_t>(n), -1);
    recursive_convert(t, out, 0, 0, out.n_nodes - 1);
    return out;
}

SearchGraph build_search_graph(const float* data, int64_t n, int64_t dim,
                               const int32_t* knn_indices,
                               const float* knn_dists, int64_t n_neighbors,
                               DistFn dist, int64_t rng_state[3],
                               double diversify_prob,
                               double pruning_degree_multiplier) {
    const int64_t k = n_neighbors;

    // Forward diversify on copies (compressed=False path).
    std::vector<int32_t> div_ind(knn_indices, knn_indices + n * k);
    std::vector<float> div_dist(knn_dists, knn_dists + n * k);
    diversify_dense(div_ind.data(), div_dist.data(), n, k, data, dim, dist,
                    rng_state, diversify_prob);

    // COO -> CSR: rows keep insertion (distance-ascending) order; zero
    // distances become FLOAT32_EPS; -1 entries are dropped.
    GraphRows fwd;
    float min_distance = detail::assemble_forward_rows(
        div_ind.data(), div_dist.data(), n, k, fwd);

    // "Reverse" diversification (operates on the forward rows; see header).
    diversify_csr_rows(fwd, data, dim, dist, rng_state, diversify_prob);

    // M = elementwise max(F2, F2^T); drop the diagonal; degree_prune.
    const int64_t max_degree = static_cast<int64_t>(
        std::nearbyint(pruning_degree_multiplier * static_cast<double>(k)));
    SearchGraph out = detail::finalize_search_graph(fwd, n, max_degree);
    out.min_distance = min_distance;
    return out;
}

// ------------------------------------------------------------ PreparedIndex

namespace {

using detail::SeedItem;
using detail::heapq_pop;
using detail::heapq_push;
using detail::mark_visited;
using detail::check_and_mark_visited;

}  // namespace

PreparedIndex PreparedIndex::build(const float* data, int64_t n, int64_t dim,
                                   int64_t n_neighbors,
                                   const std::string& metric,
                                   uint32_t random_seed, bool verbose) {
    NumpyRandomState numpy_rs(random_seed);
    return build(data, n, dim, n_neighbors, metric, numpy_rs, verbose);
}

PreparedIndex PreparedIndex::build(const float* data, int64_t n, int64_t dim,
                                   int64_t n_neighbors,
                                   const std::string& metric,
                                   NumpyRandomState& numpy_rs, bool verbose) {
    PreparedIndex idx;
    idx.n_ = n;
    idx.dim_ = dim;
    idx.n_neighbors_ = n_neighbors;
    idx.metric_ = resolve_metric(metric);
    idx.normalize_query_ = (metric == "cosine" || metric == "dot");

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

    LeafArray leaf_array = make_forest_leaf_array(
        data, n, dim, n_neighbors, n_trees, /*leaf_size=*/-1,
        idx.metric_.angular_trees, /*max_depth=*/200, numpy_rs);

    int64_t rng_state[3] = {rng_state_arr[0], rng_state_arr[1],
                            rng_state_arr[2]};
    NNHeap graph = nn_descent(data, n, dim, n_neighbors, rng_state,
                              /*max_candidates=*/60, idx.metric_.dist, n_iters,
                              /*delta=*/0.001, leaf_array, verbose);

    // NNDescent.neighbor_graph (corrected distances, original data order).
    idx.knn_indices_.assign(graph.ind.begin(), graph.ind.end());
    idx.knn_dists_.resize(graph.dist.size());
    for (size_t i = 0; i < graph.dist.size(); ++i)
        idx.knn_dists_[i] = idx.metric_.correction
                                ? idx.metric_.correction(graph.dist[i])
                                : graph.dist[i];

    // ---- prepare() / _init_search_graph -----------------------------------
    // Hub search tree first (search_leaf_size=30 when leaf_size is None,
    // max_search_tree_depth defaults to max_rptree_depth=200); rng_state
    // continues from where nn_descent left it.
    idx.tree_ = nnd::make_hub_tree(data, n, dim, graph.ind.data(), n_neighbors,
                                   rng_state, /*leaf_size=*/30,
                                   idx.metric_.angular_trees,
                                   /*max_depth=*/200);

    // Search graph (UMAP defaults: diversify_prob=1.0,
    // pruning_degree_multiplier=1.5), still in original data order.
    SearchGraph sg = build_search_graph(
        data, n, dim, graph.ind.data(), graph.dist.data(), n_neighbors,
        idx.metric_.dist, rng_state, /*diversify_prob=*/1.0,
        /*pruning_degree_multiplier=*/1.5);

    // Re-order data + graph by the search tree leaf order (_vertex_order).
    idx.vertex_order_ = idx.tree_.indices;
    std::vector<int32_t> inv(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) inv[idx.vertex_order_[i]] = static_cast<int32_t>(i);

    idx.data_.resize(static_cast<size_t>(n * dim));
    for (int64_t i = 0; i < n; ++i)
        std::memcpy(idx.data_.data() + i * dim,
                    data + static_cast<int64_t>(idx.vertex_order_[i]) * dim,
                    sizeof(float) * static_cast<size_t>(dim));

    idx.search_graph_ =
        detail::reorder_search_graph(sg, n, idx.vertex_order_, inv);

    // resort_tree_indices: tree.indices[argsort(vertex_order)] == arange(n).
    for (int64_t i = 0; i < n; ++i) idx.tree_.indices[i] = static_cast<int32_t>(i);

    return idx;
}

void PreparedIndex::query(const float* query_data, int64_t n_queries,
                          int64_t k, double epsilon, int32_t* out_indices,
                          float* out_dists) const {
    NNHeap result = NNHeap::make(n_queries, k);
    int64_t internal_rng_state[3] = {search_rng_state_[0], search_rng_state_[1],
                                     search_rng_state_[2]};
    std::vector<uint8_t> visited(static_cast<size_t>((n_ / 8) + 1), 0);
    std::vector<SeedItem> seed_set;
    std::vector<float> normalized(static_cast<size_t>(dim_));
    const DistFn dist = metric_.dist;

    for (int64_t i = 0; i < n_queries; ++i) {
        std::fill(visited.begin(), visited.end(), 0);

        const float* current_query = query_data + i * dim_;
        if (normalize_query_) {
            float sq = 0.0f;  // np.float32 (q**2).sum()
            for (int64_t d = 0; d < dim_; ++d)
                sq += current_query[d] * current_query[d];
            float norm = std::sqrt(sq);
            if (!(norm > 0.0f)) continue;
            for (int64_t d = 0; d < dim_; ++d)
                normalized[d] = current_query[d] / norm;
            current_query = normalized.data();
        }

        float* heap_priorities = result.dist_row(i);
        int32_t* heap_indices = result.ind_row(i);
        seed_set.clear();

        // ---- init from tree ------------------------------------------------
        uint32_t node = 0;
        while (tree_.children[node * 2] > 0) {
            // select_side: float32 margin accumulation, EPS coin flip.
            float margin = tree_.offsets[node];
            const float* h = tree_.hyperplanes.data() + node * dim_;
            for (int64_t d = 0; d < dim_; ++d)
                margin += h[d] * current_query[d];
            int side;
            if (std::fabs(static_cast<double>(margin)) < EPS)
                side = static_cast<int>(
                    pymod(iabs32(tau_rand_int(internal_rng_state)), 2));
            else
                side = margin > 0.0f ? 0 : 1;
            node = static_cast<uint32_t>(tree_.children[node * 2 + side]);
        }
        const int64_t leaf_start = -tree_.children[node * 2];
        const int64_t leaf_end = -tree_.children[node * 2 + 1];
        const int64_t n_initial_points = leaf_end - leaf_start;

        for (int64_t j = leaf_start; j < leaf_end; ++j) {
            int32_t candidate = tree_.indices[j];
            float d = dist(current_query,
                           data_.data() + static_cast<int64_t>(candidate) * dim_,
                           dim_);
            simple_heap_push(heap_priorities, heap_indices, k, d, candidate);
            heapq_push(seed_set, {d, candidate});
            mark_visited(visited, candidate);
        }

        // ---- random samples if needed -------------------------------------
        const int64_t n_random_samples =
            std::min(k, n_neighbors_) - n_initial_points;
        for (int64_t j = 0; j < n_random_samples; ++j) {
            int32_t candidate = static_cast<int32_t>(
                pymod(iabs32(tau_rand_int(internal_rng_state)), n_));
            if (check_and_mark_visited(visited, candidate) == 0) {
                float d = dist(
                    current_query,
                    data_.data() + static_cast<int64_t>(candidate) * dim_,
                    dim_);
                simple_heap_push(heap_priorities, heap_indices, k, d,
                                 candidate);
                heapq_push(seed_set, {d, candidate});
            }
        }

        // ---- search --------------------------------------------------------
        // distance_bound is float32; epsilon math promotes through float64.
        float distance_bound = static_cast<float>(
            static_cast<double>(heap_priorities[0]) +
            epsilon * static_cast<double>(heap_priorities[0] -
                                          search_graph_.min_distance));

        if (seed_set.empty()) continue;
        SeedItem top = heapq_pop(seed_set);
        float d_vertex = top.d;
        int32_t vertex = top.idx;

        while (d_vertex < distance_bound) {
            for (int32_t j = search_graph_.indptr[vertex];
                 j < search_graph_.indptr[vertex + 1]; ++j) {
                int32_t candidate = search_graph_.indices[j];
                if (check_and_mark_visited(visited, candidate) == 0) {
                    float d = dist(
                        current_query,
                        data_.data() + static_cast<int64_t>(candidate) * dim_,
                        dim_);
                    if (d < distance_bound) {
                        simple_heap_push(heap_priorities, heap_indices, k, d,
                                         candidate);
                        heapq_push(seed_set, {d, candidate});
                        distance_bound = static_cast<float>(
                            static_cast<double>(heap_priorities[0]) +
                            epsilon *
                                static_cast<double>(
                                    heap_priorities[0] -
                                    search_graph_.min_distance));
                    }
                }
            }
            if (seed_set.empty()) break;
            top = heapq_pop(seed_set);
            d_vertex = top.d;
            vertex = top.idx;
        }
    }

    deheap_sort(result.ind.data(), result.dist.data(), n_queries, k);

    for (int64_t i = 0; i < n_queries * k; ++i) {
        int32_t ind = result.ind[i];
        // numpy fancy indexing: vertex_order[-1] wraps to the last vertex.
        out_indices[i] = vertex_order_[ind >= 0 ? ind : n_ + ind];
        float d = result.dist[i];
        out_dists[i] = metric_.correction ? metric_.correction(d) : d;
    }
}

}  // namespace nnd
}  // namespace umap
