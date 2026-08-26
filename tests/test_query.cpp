// Tests for the pynndescent query path port (PreparedIndex: prepare + search),
// compared stage-by-stage and end-to-end against goldens generated from the
// reference pynndescent (tests/golden_query.hpp, NUMBA_NUM_THREADS=1), plus
// recall against exact brute-force kNN of the query points.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <vector>

#include "doctest.h"
#include "golden_query.hpp"
#include "umap/nn_descent.hpp"

using namespace umap::nnd;

namespace {

// Exact kNN of each query point against the training data (true metric,
// stable tie-breaking by ascending index).
void brute_force_query(const float* train, int64_t n, int64_t dim,
                       const float* queries, int64_t nq, int64_t k,
                       const std::string& metric, int32_t* out_ind) {
    MetricInfo m = resolve_metric(metric);
    std::vector<float> row(static_cast<size_t>(n));
    std::vector<int32_t> order(static_cast<size_t>(n));
    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t j = 0; j < n; ++j)
            row[j] = m.true_dist(queries + i * dim, train + j * dim, dim);
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
    const float* train;
    const float* queries;
    const int64_t* search_rng_state;
    const int32_t* vertex_order;
    const int32_t* graph_indptr;
    const int32_t* graph_indices;
    int64_t graph_nnz;
    float min_distance;
    int64_t tree_n_nodes;
    const int32_t* tree_children;
    const float* tree_offsets;
    const float* tree_hyperplanes;
    const int32_t* query_ind;        // epsilon = 0.1
    const float* query_dist;
    float umap_epsilon;              // what UMAP's transform() passes
    const int32_t* query_ind_umap;   // epsilon = umap_epsilon
    const float* query_dist_umap;
};

Case euclidean_case() {
    using namespace golden_query;
    return {"euclidean", "euclidean", euc_n, euc_dim, euc_nq, euc_train,
            euc_queries, euc_search_rng_state, euc_vertex_order,
            euc_graph_indptr, euc_graph_indices, euc_graph_indptr[euc_n],
            euc_min_distance, euc_tree_n_nodes, euc_tree_children,
            euc_tree_offsets, euc_tree_hyperplanes, euc_query_ind,
            euc_query_dist, euc_umap_epsilon, euc_query_ind_umap_eps,
            euc_query_dist_umap_eps};
}

Case cosine_case() {
    using namespace golden_query;
    return {"cosine", "cosine", cos_n, cos_dim, cos_nq, cos_train,
            cos_queries, cos_search_rng_state, cos_vertex_order,
            cos_graph_indptr, cos_graph_indices, cos_graph_indptr[cos_n],
            cos_min_distance, cos_tree_n_nodes, cos_tree_children,
            cos_tree_offsets, cos_tree_hyperplanes, cos_query_ind,
            cos_query_dist, cos_umap_epsilon, cos_query_ind_umap_eps,
            cos_query_dist_umap_eps};
}

void check_prepare_artifacts(const Case& c) {
    PreparedIndex idx =
        PreparedIndex::build(c.train, c.n, c.dim, 10, c.metric, 42);

    // search_rng_state after the 10x tau_rand_int warm-up
    for (int i = 0; i < 3; ++i)
        CHECK(idx.search_rng_state()[i] == c.search_rng_state[i]);

    // converted hub search tree
    REQUIRE(idx.search_tree().n_nodes == c.tree_n_nodes);
    for (int64_t i = 0; i < c.tree_n_nodes * 2; ++i)
        CHECK(idx.search_tree().children[i] == c.tree_children[i]);
    for (int64_t i = 0; i < c.tree_n_nodes; ++i)
        CHECK(idx.search_tree().offsets[i] ==
              doctest::Approx(c.tree_offsets[i]).epsilon(1e-6));
    for (int64_t i = 0; i < c.tree_n_nodes * c.dim; ++i)
        CHECK(idx.search_tree().hyperplanes[i] ==
              doctest::Approx(c.tree_hyperplanes[i]).epsilon(1e-6));
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
    PreparedIndex idx =
        PreparedIndex::build(c.train, c.n, c.dim, 10, c.metric, 42);

    const int64_t k = 10;
    std::vector<int32_t> ind(c.nq * k);
    std::vector<float> dist(c.nq * k);
    std::vector<int32_t> exact(c.nq * k);
    brute_force_query(c.train, c.n, c.dim, c.queries, c.nq, k, c.metric,
                      exact.data());

    // ---- epsilon = 0.1: index-exact parity with the Python reference ------
    idx.query(c.queries, c.nq, k, /*epsilon=*/0.1, ind.data(), dist.data());
    int64_t mismatches = 0;
    for (int64_t i = 0; i < c.nq * k; ++i)
        if (ind[i] != c.query_ind[i]) ++mismatches;
    CHECK(mismatches == 0);
    for (int64_t i = 0; i < c.nq * k; ++i)
        CHECK(dist[i] == doctest::Approx(c.query_dist[i]).epsilon(dist_eps));

    // ---- UMAP transform() epsilon (0.12 / 0.24): parity + recall ----------
    // (queries are independent: search_rng_state is copied per call, so the
    // second query sees the same state, exactly as in Python)
    idx.query(c.queries, c.nq, k, c.umap_epsilon, ind.data(), dist.data());
    mismatches = 0;
    for (int64_t i = 0; i < c.nq * k; ++i)
        if (ind[i] != c.query_ind_umap[i]) ++mismatches;
    CHECK(mismatches == 0);
    for (int64_t i = 0; i < c.nq * k; ++i)
        CHECK(dist[i] ==
              doctest::Approx(c.query_dist_umap[i]).epsilon(dist_eps));

    double recall = query_recall(ind.data(), exact.data(), c.nq, k);
    INFO("recall at umap epsilon = " << recall);
    CHECK(recall >= 0.95);
}

}  // namespace

TEST_CASE("PreparedIndex prepare() artifacts match pynndescent (euclidean)") {
    check_prepare_artifacts(euclidean_case());
}

TEST_CASE("PreparedIndex prepare() artifacts match pynndescent (cosine)") {
    check_prepare_artifacts(cosine_case());
}

TEST_CASE("PreparedIndex query matches pynndescent + recall (euclidean)") {
    check_query(euclidean_case(), 1e-4);
}

TEST_CASE("PreparedIndex query matches pynndescent + recall (cosine)") {
    check_query(cosine_case(), 2e-3);
}

TEST_CASE("PreparedIndex knn graph matches nearest_neighbors output") {
    // The fit-time kNN graph kept by PreparedIndex is what UMAP's fit
    // consumes; it must equal the standalone nearest_neighbors path.
    using namespace golden_query;
    const int64_t k = 10;
    PreparedIndex idx =
        PreparedIndex::build(euc_train, euc_n, euc_dim, k, "euclidean", 42);
    std::vector<int32_t> ind(euc_n * k);
    std::vector<float> dist(euc_n * k);
    nearest_neighbors(euc_train, euc_n, euc_dim, k, "euclidean", false, 42,
                      ind.data(), dist.data());
    for (int64_t i = 0; i < euc_n * k; ++i) {
        CHECK(idx.knn_indices()[i] == ind[i]);
        CHECK(idx.knn_dists()[i] == dist[i]);
    }
}
