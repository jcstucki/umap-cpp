// Internal building blocks shared by the dense (src/nn_query.cpp) and sparse
// (src/sparse_nn_query.cpp) ports of the pynndescent query path. These are
// pieces the Python reference genuinely shares between its dense and sparse
// code paths:
//   * rp_trees.py::compute_global_degrees / get_top_k_hub_indices (hub trees)
//   * the CSR-assembly / symmetric-max / degree_prune half of
//     NNDescent._init_search_graph (identical for dense and sparse)
//   * the vertex re-order of the search graph by search-tree leaf order
//   * CPython-heapq seed-set semantics and the visited bitset used by both
//     search closures (utils.py::mark_visited / check_and_mark_visited)
// Moved verbatim out of src/nn_query.cpp's anonymous namespace -- behaviour
// is identical to the pre-refactor dense port (all dense goldens unchanged).
//
// This header is internal to the library (lives in src/, not include/).
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

#include "umap/nn_descent.hpp"

namespace umap {
namespace nnd {
namespace detail {

constexpr double EPS = 1e-8;                    // rp_trees.py
constexpr float FLOAT32_EPS = 1.1920929e-07f;   // np.finfo(np.float32).eps
constexpr float INF_F = std::numeric_limits<float>::infinity();

// numba np.abs on int32 (INT32_MIN stays INT32_MIN, like C).
inline int32_t iabs32(int32_t v) {
    return v < 0 ? static_cast<int32_t>(0u - static_cast<uint32_t>(v)) : v;
}

// ------------------------------------------------------------ hub tree utils

// rp_trees.py::compute_global_degrees
inline std::vector<int32_t> compute_global_degrees(const int32_t* knn_indices,
                                                   int64_t n, int64_t k) {
    std::vector<int32_t> degrees(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < k; ++j) {
            int32_t nb = knn_indices[i * k + j];
            if (nb >= 0 && nb < n) ++degrees[nb];
        }
    return degrees;
}

// rp_trees.py::get_top_k_hub_indices (insertion-based top-k, ties keep the
// earlier-inserted entry).
inline std::vector<int32_t> get_top_k_hub_indices(
    const std::vector<int32_t>& indices, const std::vector<int32_t>& degrees,
    int64_t k) {
    const int64_t n_points = static_cast<int64_t>(indices.size());
    const int64_t actual_k = std::min<int64_t>(k, n_points);
    std::vector<int32_t> top_degrees(static_cast<size_t>(actual_k), -1);
    std::vector<int32_t> top_indices(static_cast<size_t>(actual_k), 0);
    for (int64_t i = 0; i < n_points; ++i) {
        int32_t deg = degrees[indices[i]];
        if (deg > top_degrees[actual_k - 1]) {
            int64_t insert_pos = actual_k - 1;
            while (insert_pos > 0 && deg > top_degrees[insert_pos - 1])
                --insert_pos;
            for (int64_t j = actual_k - 1; j > insert_pos; --j) {
                top_degrees[j] = top_degrees[j - 1];
                top_indices[j] = top_indices[j - 1];
            }
            top_degrees[insert_pos] = deg;
            top_indices[insert_pos] = indices[i];
        }
    }
    return top_indices;
}

// ------------------------------------------------------------ search graph

// One CSR row held as parallel (col, val) arrays in insertion order.
struct GraphRows {
    std::vector<std::vector<int32_t>> col;
    std::vector<std::vector<float>> val;
};

// The COO -> CSR assembly of the forward-diversified kNN graph
// (_init_search_graph): rows keep insertion (distance-ascending) order; zero
// distances become FLOAT32_EPS; -1 entries are dropped. Returns
// NNDescent._min_distance (min over the assembled forward data).
inline float assemble_forward_rows(const int32_t* div_ind,
                                   const float* div_dist, int64_t n, int64_t k,
                                   GraphRows& fwd) {
    fwd.col.assign(static_cast<size_t>(n), {});
    fwd.val.assign(static_cast<size_t>(n), {});
    float min_distance = INF_F;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            int32_t c = div_ind[i * k + j];
            if (c == -1) continue;
            float v = div_dist[i * k + j];
            if (v == 0.0f) v = FLOAT32_EPS;
            fwd.col[i].push_back(c);
            fwd.val[i].push_back(v);
            if (v < min_distance) min_distance = v;
        }
    }
    return min_distance;
}

// The tail of _init_search_graph after the reverse diversification:
// M = elementwise max(F2, F2^T), diagonal removal (setdiag(0) +
// eliminate_zeros), degree_prune to `max_degree`, binarisation. Rows of the
// result are column-sorted (scipy sort_indices). min_distance is left at its
// default; callers set it from assemble_forward_rows.
inline SearchGraph finalize_search_graph(const GraphRows& fwd, int64_t n,
                                         int64_t max_degree) {
    std::vector<std::map<int32_t, float>> sym(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        for (size_t p = 0; p < fwd.col[i].size(); ++p) {
            int32_t j = fwd.col[i][p];
            float v = fwd.val[i][p];
            if (j == i) continue;  // setdiag(0) + eliminate_zeros
            auto upd = [v](std::map<int32_t, float>& row, int32_t c) {
                auto it = row.find(c);
                if (it == row.end())
                    row.emplace(c, v);
                else if (v > it->second)
                    it->second = v;
            };
            upd(sym[i], j);
            upd(sym[j], static_cast<int32_t>(i));
        }
    }

    // degree_prune: cut at the (max_degree+1)-th smallest value per row, keep
    // entries <= cut (ties at the cut survive, as upstream).
    std::vector<float> row_vals;
    SearchGraph out;
    out.indptr.assign(static_cast<size_t>(n + 1), 0);
    for (int64_t i = 0; i < n; ++i) {
        std::map<int32_t, float>& row = sym[i];
        if (static_cast<int64_t>(row.size()) > max_degree) {
            row_vals.clear();
            for (const auto& kv : row) row_vals.push_back(kv.second);
            std::sort(row_vals.begin(), row_vals.end());
            float cut_value = row_vals[max_degree];
            for (auto it = row.begin(); it != row.end();) {
                if (it->second > cut_value)
                    it = row.erase(it);
                else
                    ++it;
            }
        }
        for (const auto& kv : row) out.indices.push_back(kv.first);
        out.indptr[i + 1] = static_cast<int32_t>(out.indices.size());
    }
    return out;
}

// Re-order the (binarised) search graph rows and columns by the search-tree
// leaf order: search_graph[vertex_order, :][:, vertex_order], rows
// column-sorted (csr sort_indices). `inv` is the inverse permutation of
// `vertex_order` (inv[vertex_order[i]] == i).
inline SearchGraph reorder_search_graph(const SearchGraph& sg, int64_t n,
                                        const std::vector<int32_t>& vertex_order,
                                        const std::vector<int32_t>& inv) {
    SearchGraph out;
    out.min_distance = sg.min_distance;
    out.indptr.assign(static_cast<size_t>(n + 1), 0);
    out.indices.reserve(sg.indices.size());
    std::vector<int32_t> row_buf;
    for (int64_t i = 0; i < n; ++i) {
        int32_t old_row = vertex_order[i];
        row_buf.clear();
        for (int32_t p = sg.indptr[old_row]; p < sg.indptr[old_row + 1]; ++p)
            row_buf.push_back(inv[sg.indices[p]]);
        std::sort(row_buf.begin(), row_buf.end());  // csr sort_indices
        out.indices.insert(out.indices.end(), row_buf.begin(), row_buf.end());
        out.indptr[i + 1] = static_cast<int32_t>(out.indices.size());
    }
    return out;
}

// ------------------------------------------------------------ query search

// CPython heapq on (float32 distance, int32 index) tuples, replicated
// operation-for-operation so tie handling and heap layout match numba's
// heapq implementation.
struct SeedItem {
    float d;
    int32_t idx;
};

inline bool seed_less(const SeedItem& a, const SeedItem& b) {
    return a.d < b.d || (a.d == b.d && a.idx < b.idx);
}

inline void heapq_siftdown(std::vector<SeedItem>& heap, int64_t startpos,
                           int64_t pos) {
    SeedItem newitem = heap[pos];
    while (pos > startpos) {
        int64_t parentpos = (pos - 1) >> 1;
        SeedItem parent = heap[parentpos];
        if (seed_less(newitem, parent)) {
            heap[pos] = parent;
            pos = parentpos;
            continue;
        }
        break;
    }
    heap[pos] = newitem;
}

inline void heapq_siftup(std::vector<SeedItem>& heap, int64_t pos) {
    const int64_t endpos = static_cast<int64_t>(heap.size());
    const int64_t startpos = pos;
    SeedItem newitem = heap[pos];
    int64_t childpos = 2 * pos + 1;
    while (childpos < endpos) {
        int64_t rightpos = childpos + 1;
        if (rightpos < endpos &&
            !seed_less(heap[childpos], heap[rightpos]))
            childpos = rightpos;
        heap[pos] = heap[childpos];
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    heap[pos] = newitem;
    heapq_siftdown(heap, startpos, pos);
}

inline void heapq_push(std::vector<SeedItem>& heap, SeedItem item) {
    heap.push_back(item);
    heapq_siftdown(heap, 0, static_cast<int64_t>(heap.size()) - 1);
}

inline SeedItem heapq_pop(std::vector<SeedItem>& heap) {
    SeedItem lastelt = heap.back();
    heap.pop_back();
    if (!heap.empty()) {
        SeedItem returnitem = heap[0];
        heap[0] = lastelt;
        heapq_siftup(heap, 0);
        return returnitem;
    }
    return lastelt;
}

// utils.py::mark_visited / check_and_mark_visited (bitset over uint8 table).
inline void mark_visited(std::vector<uint8_t>& table, int32_t candidate) {
    table[candidate >> 3] |=
        static_cast<uint8_t>(1u << (candidate & 7));
}

inline uint8_t check_and_mark_visited(std::vector<uint8_t>& table,
                                      int32_t candidate) {
    int32_t loc = candidate >> 3;
    uint8_t mask = static_cast<uint8_t>(1u << (candidate & 7));
    uint8_t was_visited = table[loc] & mask;
    table[loc] |= mask;
    return was_visited;
}

}  // namespace detail
}  // namespace nnd
}  // namespace umap
