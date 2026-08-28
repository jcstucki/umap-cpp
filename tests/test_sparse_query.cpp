// Tests for the sparse pynndescent query path port (SparsePreparedIndex:
// prepare + search), compared stage-by-stage and end-to-end against goldens
// generated from the reference pynndescent (tests/golden_sparse_query.hpp,
// NUMBA_NUM_THREADS=1), plus recall against exact brute-force kNN of the
// query points under the true sparse metric.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_sparse_query.hpp"
#include "umap/sparse_distances.hpp"
#include "umap/sparse_nn_descent.hpp"

using namespace umap::spnnd;

namespace {

// Exact kNN of each sparse query row against the sparse training rows (true
// metric, stable tie-breaking by ascending index).
void sparse_brute_force_query(const CsrView& train, const int32_t* q_indptr,
                              const int32_t* q_indices, const float* q_data,
                              int64_t nq, int64_t k, const std::string& metric,
                              int32_t* out_ind) {
    umap::spdist::SparseMetricInfo m =
        umap::spdist::resolve_sparse_metric(metric);
    const int64_t n = train.n_rows;
    std::vector<float> row(static_cast<size_t>(n));
    std::vector<int32_t> order(static_cast<size_t>(n));
    for (int64_t i = 0; i < nq; ++i) {
        const int32_t* qi = q_indices + q_indptr[i];
        const float* qd = q_data + q_indptr[i];
        const int64_t qn = static_cast<int64_t>(q_indptr[i + 1]) - q_indptr[i];
        for (int64_t j = 0; j < n; ++j)
            row[j] = m.true_dist(qi, qd, qn, train.row_ind(j),
                                 train.row_data(j), train.row_nnz(j));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int32_t a, int32_t b) { return row[a] < row[b]; });
        for (int64_t j = 0; j < k; ++j) out_ind[i * k + j] = order[j];
    }
}

double query_recall(const int32_t* approx, const int32_t* exact, int64_t nq,
                    int64_t k) {
    int64_t hits = 0;
    for (int64_t i = 0; i < nq; ++i) {
        std::set<int32_t> truth(exact + i * k, exact + (i + 1) * k);
        for (int64_t j = 0; j < k; ++j)
            if (truth.count(approx[i * k + j])) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(nq * k);
}

struct Case {
    const char* name;
    const char* metric;
    int64_t n, dim, nq;
    const int32_t* indptr;
    const int32_t* indices;
    const float* data;
    const int32_t* q_indptr;
    const int32_t* q_indices;
    const float* q_data;
    const int64_t* search_rng_state;
    const int32_t* knn_ind;
    const float* knn_dist;
    const int32_t* vertex_order;
    const int32_t* graph_indptr;
    const int32_t* graph_indices;
    int64_t graph_nnz;
    float min_distance;
    int64_t tree_n_nodes;
    const int32_t* tree_children;
    const float* tree_offsets;
    const int32_t* tree_hp_indptr;
    const int32_t* tree_hp_inds;
    const float* tree_hp_data;
    const int32_t* query_ind;        // epsilon = 0.1
    const float* query_dist;
    float umap_epsilon;              // what UMAP's transform() passes
    const int32_t* query_ind_umap;   // epsilon = umap_epsilon
    const float* query_dist_umap;
    // Recall of the *Python reference* vs exact brute force on this case
    // (the C++ port is index-exact vs the reference, so its recall must
    // match these to the digit). The 0.95 aspiration does not hold for the
    // reference itself at these epsilons on sparse data -- e.g. cosine at
    // epsilon 0.1 is 0.895 in Python -- so the assertion pins the measured
    // reference recall instead (verified via sklearn pairwise_distances).
    double recall_eps01;
    double recall_umap;
};

Case euclidean_case() {
    using namespace golden_sparse_query;
    return {"euclidean", "euclidean", spe_n, spe_dim, spe_nq, spe_indptr,
            spe_indices, spe_data, spe_q_indptr, spe_q_indices, spe_q_data,
            spe_search_rng_state, spe_knn_ind, spe_knn_dist, spe_vertex_order,
            spe_graph_indptr, spe_graph_indices, spe_graph_indptr[spe_n],
            spe_min_distance, spe_tree_n_nodes, spe_tree_children,
            spe_tree_offsets, spe_tree_hp_indptr, spe_tree_hp_inds,
            spe_tree_hp_data, spe_query_ind, spe_query_dist, spe_umap_epsilon,
            spe_query_ind_umap_eps, spe_query_dist_umap_eps, 0.94, 0.955};
}

Case cosine_case() {
    using namespace golden_sparse_query;
    return {"cosine", "cosine", spc_n, spc_dim, spc_nq, spc_indptr,
            spc_indices, spc_data, spc_q_indptr, spc_q_indices, spc_q_data,
            spc_search_rng_state, spc_knn_ind, spc_knn_dist, spc_vertex_order,
            spc_graph_indptr, spc_graph_indices, spc_graph_indptr[spc_n],
            spc_min_distance, spc_tree_n_nodes, spc_tree_children,
            spc_tree_offsets, spc_tree_hp_indptr, spc_tree_hp_inds,
            spc_tree_hp_data, spc_query_ind, spc_query_dist, spc_umap_epsilon,
            spc_query_ind_umap_eps, spc_query_dist_umap_eps, 0.895, 0.94};
}

SparsePreparedIndex build_index(const Case& c) {
    return SparsePreparedIndex::build(c.indptr, c.indices, c.data, c.n, c.dim,
                                      10, c.metric, 42);
}

void check_prepare_artifacts(const Case& c) {
    SparsePreparedIndex idx = build_index(c);

    // search_rng_state after the 10x tau_rand_int warm-up
    for (int i = 0; i < 3; ++i)
        CHECK(idx.search_rng_state()[i] == c.search_rng_state[i]);

    // fit-time kNN graph (corrected distances, original order)
    for (int64_t i = 0; i < c.n * 10; ++i) {
        CHECK(idx.knn_indices()[i] == c.knn_ind[i]);
        CHECK(idx.knn_dists()[i] ==
              doctest::Approx(c.knn_dist[i]).epsilon(1e-5));
    }

    // converted sparse hub search tree
    REQUIRE(idx.search_tree().n_nodes == c.tree_n_nodes);
    for (int64_t i = 0; i < c.tree_n_nodes * 2; ++i)
        CHECK(idx.search_tree().children[i] == c.tree_children[i]);
    for (int64_t i = 0; i < c.tree_n_nodes; ++i)
        CHECK(idx.search_tree().offsets[i] ==
              doctest::Approx(c.tree_offsets[i]).epsilon(1e-6));
    for (int64_t i = 0; i < c.tree_n_nodes; ++i) {
        const auto& hin = idx.search_tree().hyperplane_inds[i];
        const auto& hda = idx.search_tree().hyperplane_data[i];
        const int64_t lo = c.tree_hp_indptr[i], hi = c.tree_hp_indptr[i + 1];
        REQUIRE(static_cast<int64_t>(hin.size()) == hi - lo);
        for (int64_t p = lo; p < hi; ++p) {
            CHECK(hin[p - lo] == c.tree_hp_inds[p]);
            CHECK(hda[p - lo] ==
                  doctest::Approx(c.tree_hp_data[p]).epsilon(1e-6));
        }
    }
    // resorted tree indices are the identity permutation
    for (int64_t i = 0; i < c.n; ++i)
        CHECK(idx.search_tree().indices[i] == i);

    // vertex order (= pre-resort tree leaf order)
    for (int64_t i = 0; i < c.n; ++i)
        CHECK(idx.vertex_order()[i] == c.vertex_order[i]);

    // reordered, binarised CSR search graph
    REQUIRE(static_cast<int64_t>(idx.search_graph().indices.size()) ==
            c.graph_nnz);
    for (int64_t i = 0; i <= c.n; ++i)
        CHECK(idx.search_graph().indptr[i] == c.graph_indptr[i]);
    for (int64_t i = 0; i < c.graph_nnz; ++i)
        CHECK(idx.search_graph().indices[i] == c.graph_indices[i]);

    CHECK(idx.min_distance() == doctest::Approx(c.min_distance));
}

void check_query(const Case& c, double dist_eps) {
    SparsePreparedIndex idx = build_index(c);

    const int64_t k = 10;
    std::vector<int32_t> ind(c.nq * k);
    std::vector<float> dist(c.nq * k);
    std::vector<int32_t> exact(c.nq * k);
    CsrView train{c.indptr, c.indices, c.data, c.n, c.dim};
    sparse_brute_force_query(train, c.q_indptr, c.q_indices, c.q_data, c.nq,
                             k, c.metric, exact.data());

    // ---- epsilon = 0.1: index-exact parity with the Python reference ------
    idx.query(c.q_indptr, c.q_indices, c.q_data, c.nq, k, /*epsilon=*/0.1,
              ind.data(), dist.data());
    int64_t mismatches = 0;
    for (int64_t i = 0; i < c.nq * k; ++i)
        if (ind[i] != c.query_ind[i]) ++mismatches;
    CHECK(mismatches == 0);
    for (int64_t i = 0; i < c.nq * k; ++i)
        CHECK(dist[i] == doctest::Approx(c.query_dist[i]).epsilon(dist_eps));

    double recall01 = query_recall(ind.data(), exact.data(), c.nq, k);
    INFO("recall at epsilon 0.1 = " << recall01);
    CHECK(recall01 == doctest::Approx(c.recall_eps01));

    // ---- UMAP transform() epsilon (0.12 / 0.24): parity + recall ----------
    // (queries are independent: search_rng_state is copied per call, so the
    // second query sees the same state, exactly as in Python)
    idx.query(c.q_indptr, c.q_indices, c.q_data, c.nq, k, c.umap_epsilon,
              ind.data(), dist.data());
    mismatches = 0;
    for (int64_t i = 0; i < c.nq * k; ++i)
        if (ind[i] != c.query_ind_umap[i]) ++mismatches;
    CHECK(mismatches == 0);
    for (int64_t i = 0; i < c.nq * k; ++i)
        CHECK(dist[i] ==
              doctest::Approx(c.query_dist_umap[i]).epsilon(dist_eps));

    double recall = query_recall(ind.data(), exact.data(), c.nq, k);
    INFO("recall at umap epsilon = " << recall);
    CHECK(recall == doctest::Approx(c.recall_umap));
    CHECK(recall >= 0.94);  // documented reference recall floor (see Case)
}

}  // namespace

TEST_CASE(
    "SparsePreparedIndex prepare() artifacts match pynndescent (euclidean)") {
    check_prepare_artifacts(euclidean_case());
}

TEST_CASE(
    "SparsePreparedIndex prepare() artifacts match pynndescent (cosine)") {
    check_prepare_artifacts(cosine_case());
}

TEST_CASE("SparsePreparedIndex query matches pynndescent + recall (euclidean)") {
    check_query(euclidean_case(), 1e-4);
}

TEST_CASE("SparsePreparedIndex query matches pynndescent + recall (cosine)") {
    check_query(cosine_case(), 2e-3);
}

TEST_CASE("SparsePreparedIndex knn graph matches nearest_neighbors_sparse") {
    // The fit-time kNN graph kept by SparsePreparedIndex is what UMAP's fit
    // consumes; it must equal the standalone NN-descent path (force the
    // approximation branch: n = 300 < 4096 would brute-force otherwise).
    using namespace golden_sparse_query;
    const int64_t k = 10;
    SparsePreparedIndex idx = SparsePreparedIndex::build(
        spe_indptr, spe_indices, spe_data, spe_n, spe_dim, k, "euclidean", 42);
    std::vector<int32_t> ind(spe_n * k);
    std::vector<float> dist(spe_n * k);
    nearest_neighbors_sparse(spe_indptr, spe_indices, spe_data, spe_n, spe_dim,
                             k, "euclidean", false, 42, ind.data(),
                             dist.data(), /*force_approximation=*/true);
    for (int64_t i = 0; i < spe_n * k; ++i) {
        CHECK(idx.knn_indices()[i] == ind[i]);
        CHECK(idx.knn_dists()[i] == dist[i]);
    }
}
