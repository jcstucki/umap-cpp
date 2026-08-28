// Tests for the sparse-input support: sparse distance kernels
// (umap/sparse_distances.hpp) and sparse RP-tree forest + NN-descent
// (umap/sparse_nn_descent.hpp). Deterministic pieces and end-to-end kNN
// graphs are compared against goldens generated from the reference
// pynndescent + sklearn (tests/golden_sparse.hpp, NUMBA_NUM_THREADS=1);
// the NN-descent result is additionally validated by recall against exact
// sparse brute force.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_nn.hpp"      // seed-42 rng/tree states (shared with dense)
#include "golden_sparse.hpp"
#include "umap/sparse_distances.hpp"
#include "umap/sparse_nn_descent.hpp"

using namespace umap;

namespace {

spnnd::CsrView make_view(const std::int32_t* indptr, const std::int32_t* ind,
                         const float* data, int64_t n, int64_t dim) {
    spnnd::CsrView v;
    v.indptr = indptr;
    v.indices = ind;
    v.data = data;
    v.n_rows = n;
    v.n_cols = dim;
    return v;
}

// Distance between the two vectors of golden pair `p` under `fn`.
float pair_dist(spdist::SparseDistFn fn, int64_t p) {
    const std::int32_t* off = golden_sparse::metric_vec_offsets;
    const std::int32_t* ind = golden_sparse::metric_vec_ind;
    const float* dat = golden_sparse::metric_vec_data;
    const int64_t a = 2 * p, b = 2 * p + 1;
    return fn(ind + off[a], dat + off[a], off[a + 1] - off[a], ind + off[b],
              dat + off[b], off[b + 1] - off[b]);
}

// Units-in-the-last-place distance between two floats (0 when bit-equal).
int64_t ulp_diff(float a, float b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b)) return INT64_MAX;
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = static_cast<int32_t>(0x80000000u) - ia;
    if (ib < 0) ib = static_cast<int32_t>(0x80000000u) - ib;
    return std::llabs(static_cast<int64_t>(ia) - static_cast<int64_t>(ib));
}

double neighbor_recall(const int32_t* approx, const int32_t* exact, int64_t n,
                       int64_t k) {
    int64_t hits = 0;
    for (int64_t i = 0; i < n; ++i) {
        std::set<int32_t> truth(exact + i * k, exact + (i + 1) * k);
        for (int64_t j = 0; j < k; ++j)
            if (truth.count(approx[i * k + j])) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(n * k);
}

}  // namespace

TEST_CASE("sparse aux ops match pynndescent (union/intersect/sum/diff/mul)") {
    using namespace golden_sparse;
    const int64_t n1 = sizeof(aux_v1_ind) / sizeof(aux_v1_ind[0]);
    const int64_t n2 = sizeof(aux_v2_ind) / sizeof(aux_v2_ind[0]);

    auto u = spdist::arr_union(aux_v1_ind, n1, aux_v2_ind, n2);
    REQUIRE(u.size() == sizeof(aux_union) / sizeof(aux_union[0]));
    for (size_t i = 0; i < u.size(); ++i) CHECK(u[i] == aux_union[i]);

    auto ix = spdist::arr_intersect(aux_v1_ind, n1, aux_v2_ind, n2);
    REQUIRE(ix.size() == sizeof(aux_intersect) / sizeof(aux_intersect[0]));
    for (size_t i = 0; i < ix.size(); ++i) CHECK(ix[i] == aux_intersect[i]);

    std::vector<int32_t> rind;
    std::vector<float> rdat;
    spdist::sparse_sum(aux_v1_ind, aux_v1_data, n1, aux_v2_ind, aux_v2_data,
                       n2, rind, rdat);
    REQUIRE(rind.size() == sizeof(aux_sum_ind) / sizeof(aux_sum_ind[0]));
    for (size_t i = 0; i < rind.size(); ++i) {
        CHECK(rind[i] == aux_sum_ind[i]);
        CHECK(rdat[i] == aux_sum_data[i]);
    }

    spdist::sparse_diff(aux_v1_ind, aux_v1_data, n1, aux_v2_ind, aux_v2_data,
                        n2, rind, rdat);
    REQUIRE(rind.size() == sizeof(aux_diff_ind) / sizeof(aux_diff_ind[0]));
    for (size_t i = 0; i < rind.size(); ++i) {
        CHECK(rind[i] == aux_diff_ind[i]);
        CHECK(rdat[i] == aux_diff_data[i]);
    }

    spdist::sparse_mul(aux_v1_ind, aux_v1_data, n1, aux_v2_ind, aux_v2_data,
                       n2, rind, rdat);
    REQUIRE(rind.size() == sizeof(aux_mul_ind) / sizeof(aux_mul_ind[0]));
    for (size_t i = 0; i < rind.size(); ++i) {
        CHECK(rind[i] == aux_mul_ind[i]);
        CHECK(rdat[i] == aux_mul_data[i]);
    }
}

TEST_CASE("sparse metrics match pynndescent goldens (incl. empty/disjoint/identical rows)") {
    using namespace golden_sparse;
    // Bit-exact except where numba fastmath reassociates the float32
    // accumulation of sparse_squared_euclidean on longer vectors: allow a
    // couple of ulps there (the f8-accumulating metrics are exact).
    for (int64_t p = 0; p < metric_n_pairs; ++p) {
        CAPTURE(p);
        CHECK(pair_dist(spdist::sparse_euclidean, p) == metric_euclidean[p]);
        CHECK(ulp_diff(pair_dist(spdist::sparse_squared_euclidean, p),
                       metric_sqeuclidean[p]) <= 2);
        CHECK(pair_dist(spdist::sparse_manhattan, p) == metric_manhattan[p]);
        CHECK(pair_dist(spdist::sparse_cosine, p) == metric_cosine[p]);
        CHECK(ulp_diff(pair_dist(spdist::sparse_alternative_cosine, p),
                       metric_alt_cosine[p]) <= 2);
    }
}

TEST_CASE("sparse metrics are symmetric and zero on identical rows") {
    using namespace golden_sparse;
    const std::int32_t* off = metric_vec_offsets;
    for (int64_t p = 0; p < metric_n_pairs; ++p) {
        const int64_t a = 2 * p, b = 2 * p + 1;
        float d_ab = spdist::sparse_euclidean(
            metric_vec_ind + off[a], metric_vec_data + off[a], off[a + 1] - off[a],
            metric_vec_ind + off[b], metric_vec_data + off[b], off[b + 1] - off[b]);
        float d_ba = spdist::sparse_euclidean(
            metric_vec_ind + off[b], metric_vec_data + off[b], off[b + 1] - off[b],
            metric_vec_ind + off[a], metric_vec_data + off[a], off[a + 1] - off[a]);
        CHECK(d_ab == d_ba);
        float d_aa = spdist::sparse_euclidean(
            metric_vec_ind + off[a], metric_vec_data + off[a], off[a + 1] - off[a],
            metric_vec_ind + off[a], metric_vec_data + off[a], off[a + 1] - off[a]);
        CHECK(d_aa == 0.0f);
    }
}

TEST_CASE("sparse distance corrections match numpy/pynndescent") {
    using namespace golden_sparse;
    const int64_t ns = sizeof(corr_sqrt_in) / sizeof(corr_sqrt_in[0]);
    for (int64_t i = 0; i < ns; ++i)
        CHECK(spdist::sparse_sqrt_correction(corr_sqrt_in[i]) == corr_sqrt[i]);
    const int64_t na = sizeof(corr_alt_in) / sizeof(corr_alt_in[0]);
    for (int64_t i = 0; i < na; ++i) {
        CAPTURE(i);
        CHECK(spdist::sparse_correct_alternative_cosine(corr_alt_in[i]) ==
              corr_alt_cosine[i]);
    }
}

TEST_CASE("resolve_sparse_metric registry: supported set + not-implemented throws") {
    auto e = spdist::resolve_sparse_metric("euclidean");
    CHECK(e.dist == &spdist::sparse_squared_euclidean);
    CHECK(e.correction == &spdist::sparse_sqrt_correction);
    CHECK(e.true_dist == &spdist::sparse_euclidean);
    CHECK_FALSE(e.angular_trees);
    CHECK(spdist::resolve_sparse_metric("l2").dist == e.dist);

    auto sq = spdist::resolve_sparse_metric("sqeuclidean");
    CHECK(sq.correction == nullptr);

    for (const char* name : {"manhattan", "l1", "taxicab"}) {
        auto m = spdist::resolve_sparse_metric(name);
        CHECK(m.dist == &spdist::sparse_manhattan);
        CHECK(m.correction == nullptr);
        CHECK_FALSE(m.angular_trees);
    }

    auto c = spdist::resolve_sparse_metric("cosine");
    CHECK(c.dist == &spdist::sparse_alternative_cosine);
    CHECK(c.correction == &spdist::sparse_correct_alternative_cosine);
    CHECK(c.angular_trees);

    // absent upstream or unported -> throws (documented behaviour; the full
    // registry is exercised in tests/test_sparse_metrics_full.cpp)
    for (const char* name :
         {"yule", "dot", "kantorovich", "wasserstein", "wasserstein_1d",
          "jensen-shannon", "symmetric_kl", "bit_jaccard", "not_a_metric"}) {
        CHECK_THROWS_AS(spdist::resolve_sparse_metric(name),
                        std::invalid_argument);
    }
}

TEST_CASE("sparse_euclidean_random_projection_split matches pynndescent") {
    using namespace golden_sparse;
    auto X = make_view(split_indptr, split_indices, split_data, split_n,
                       split_dim);
    std::vector<int32_t> indices(split_n);
    for (int i = 0; i < split_n; ++i) indices[i] = i;
    int64_t st[3] = {split_rng_state[0], split_rng_state[1], split_rng_state[2]};
    std::vector<int32_t> left, right;
    spnnd::sparse_euclidean_random_projection_split(X, indices, st, left, right);

    const size_t nl = sizeof(esplit_left) / sizeof(esplit_left[0]);
    const size_t nr = sizeof(esplit_right) / sizeof(esplit_right[0]);
    REQUIRE(left.size() == nl);
    REQUIRE(right.size() == nr);
    for (size_t i = 0; i < nl; ++i) CHECK(left[i] == esplit_left[i]);
    for (size_t i = 0; i < nr; ++i) CHECK(right[i] == esplit_right[i]);
    for (int i = 0; i < 3; ++i) CHECK(st[i] == esplit_rng_after[i]);
}

TEST_CASE("sparse_angular_random_projection_split matches pynndescent") {
    using namespace golden_sparse;
    auto X = make_view(split_indptr, split_indices, split_data, split_n,
                       split_dim);
    std::vector<int32_t> indices(split_n);
    for (int i = 0; i < split_n; ++i) indices[i] = i;
    int64_t st[3] = {split_rng_state[0], split_rng_state[1], split_rng_state[2]};
    std::vector<int32_t> left, right;
    spnnd::sparse_angular_random_projection_split(X, indices, st, left, right);

    const size_t nl = sizeof(asplit_left) / sizeof(asplit_left[0]);
    const size_t nr = sizeof(asplit_right) / sizeof(asplit_right[0]);
    REQUIRE(left.size() == nl);
    REQUIRE(right.size() == nr);
    for (size_t i = 0; i < nl; ++i) CHECK(left[i] == asplit_left[i]);
    for (size_t i = 0; i < nr; ++i) CHECK(right[i] == asplit_right[i]);
    for (int i = 0; i < 3; ++i) CHECK(st[i] == asplit_rng_after[i]);
}

TEST_CASE("make_sparse_tree leaves match pynndescent (euclidean and angular)") {
    using namespace golden_sparse;
    auto X = make_view(tree_indptr, tree_indices, tree_data, tree_n, tree_dim);
    for (int angular = 0; angular < 2; ++angular) {
        CAPTURE(angular);
        int64_t rng_state[3] = {tree_rng_state[0], tree_rng_state[1],
                                tree_rng_state[2]};
        std::vector<std::vector<int32_t>> leaves;
        int64_t width = spnnd::make_sparse_tree_leaves(X, rng_state, 15,
                                                       angular != 0, 200,
                                                       leaves);
        const int64_t g_n = angular ? atree_n_leaves : etree_n_leaves;
        const int64_t g_w = angular ? atree_leaf_width : etree_leaf_width;
        const std::int32_t* g_leaves = angular ? atree_leaves : etree_leaves;

        REQUIRE(static_cast<int64_t>(leaves.size()) == g_n);
        REQUIRE(width == g_w);
        for (int64_t r = 0; r < g_n; ++r) {
            for (int64_t c = 0; c < g_w; ++c) {
                int32_t expect = g_leaves[r * g_w + c];
                int32_t got = (c < static_cast<int64_t>(leaves[r].size()))
                                  ? leaves[r][c]
                                  : -1;
                CHECK(got == expect);
            }
        }
    }
}

TEST_CASE("sparse NN-descent parity: index-identical to pynndescent (euclidean, seed 42)") {
    using namespace golden_sparse;
    const int64_t n = nnd_e_n, k = nnd_e_k;
    std::vector<int32_t> ind(n * k);
    std::vector<float> dist(n * k);
    spnnd::nearest_neighbors_sparse(nnd_e_indptr, nnd_e_indices, nnd_e_data, n,
                                    nnd_e_dim, k, "euclidean", false, 42,
                                    ind.data(), dist.data(),
                                    /*force_approximation=*/true);
    int64_t index_mismatches = 0;
    for (int64_t i = 0; i < n * k; ++i) {
        if (ind[i] != nnd_e_knn_ind[i]) ++index_mismatches;
        CHECK(dist[i] == doctest::Approx(nnd_e_knn_dist[i]).epsilon(1e-5));
    }
    CHECK(index_mismatches == 0);
}

TEST_CASE("sparse NN-descent parity: index-identical to pynndescent (cosine, seed 42)") {
    using namespace golden_sparse;
    const int64_t n = nnd_c_n, k = nnd_c_k;
    std::vector<int32_t> ind(n * k);
    std::vector<float> dist(n * k);
    spnnd::nearest_neighbors_sparse(nnd_c_indptr, nnd_c_indices, nnd_c_data, n,
                                    nnd_c_dim, k, "cosine", true, 42,
                                    ind.data(), dist.data(),
                                    /*force_approximation=*/true);
    int64_t index_mismatches = 0;
    for (int64_t i = 0; i < n * k; ++i) {
        if (ind[i] != nnd_c_knn_ind[i]) ++index_mismatches;
        CHECK(dist[i] == doctest::Approx(nnd_c_knn_dist[i]).epsilon(1e-5));
    }
    CHECK(index_mismatches == 0);
}

TEST_CASE("sparse NN-descent recall >= 0.95 vs sparse brute force") {
    using namespace golden_sparse;
    const int64_t n = nnd_e_n, k = nnd_e_k;
    auto X = make_view(nnd_e_indptr, nnd_e_indices, nnd_e_data, n, nnd_e_dim);
    std::vector<int32_t> ind(n * k), exact_ind(n * k);
    std::vector<float> dist(n * k), exact_dist(n * k);
    spnnd::nearest_neighbors_sparse(nnd_e_indptr, nnd_e_indices, nnd_e_data, n,
                                    nnd_e_dim, k, "euclidean", false, 42,
                                    ind.data(), dist.data(),
                                    /*force_approximation=*/true);
    spnnd::sparse_brute_force_knn(X, k, "euclidean", exact_ind.data(),
                                  exact_dist.data());
    double recall = neighbor_recall(ind.data(), exact_ind.data(), n, k);
    INFO("sparse euclidean recall = " << recall);
    CHECK(recall >= 0.95);
    // First neighbor at distance 0, distances ascending. (The first *index*
    // is not always self: all-zero rows of the sparse matrix are exact
    // duplicates at distance 0 and NN-descent may order them arbitrarily --
    // the Python reference does the same, as the parity test shows.)
    for (int64_t i = 0; i < n; ++i) {
        CHECK(dist[i * k] == 0.0f);
        for (int64_t j = 1; j < k; ++j)
            CHECK(dist[i * k + j] >= dist[i * k + j - 1]);
    }
}

TEST_CASE("sparse brute-force kNN matches sklearn goldens (euclidean + cosine)") {
    using namespace golden_sparse;
    const int64_t n = bf_n, k = 10;
    auto X = make_view(bf_indptr, bf_indices, bf_data, n, bf_dim);
    std::vector<int32_t> ind(n * k);
    std::vector<float> dist(n * k);

    spnnd::sparse_brute_force_knn(X, k, "euclidean", ind.data(), dist.data());
    for (int64_t i = 0; i < n * k; ++i) {
        CHECK(ind[i] == bf_euclidean_knn_ind[i]);
        CHECK(dist[i] ==
              doctest::Approx(bf_euclidean_knn_dist[i]).epsilon(1e-5));
    }

    spnnd::sparse_brute_force_knn(X, k, "cosine", ind.data(), dist.data());
    for (int64_t i = 0; i < n * k; ++i) {
        CHECK(ind[i] == bf_cosine_knn_ind[i]);
        CHECK(dist[i] == doctest::Approx(bf_cosine_knn_dist[i]).epsilon(1e-5));
    }
}

TEST_CASE("nearest_neighbors_sparse routes small data (< 4096) to brute force") {
    using namespace golden_sparse;
    const int64_t n = bf_n, k = 10;
    auto X = make_view(bf_indptr, bf_indices, bf_data, n, bf_dim);
    std::vector<int32_t> ind_a(n * k), ind_b(n * k);
    std::vector<float> dist_a(n * k), dist_b(n * k);
    spnnd::nearest_neighbors_sparse(bf_indptr, bf_indices, bf_data, n, bf_dim,
                                    k, "euclidean", false, 42, ind_a.data(),
                                    dist_a.data());
    spnnd::sparse_brute_force_knn(X, k, "euclidean", ind_b.data(),
                                  dist_b.data());
    for (int64_t i = 0; i < n * k; ++i) {
        CHECK(ind_a[i] == ind_b[i]);
        CHECK(dist_a[i] == dist_b[i]);
    }
}

TEST_CASE("nearest_neighbors_sparse_with_state(seed-42 states) == seeded run") {
    using namespace golden_sparse;
    // nnd_e_n = 300 -> n_trees = 6, matching golden_nn's 6 seed-42 tree states
    const int64_t n = nnd_e_n, k = nnd_e_k;
    std::vector<int32_t> ind_a(n * k), ind_b(n * k);
    std::vector<float> dist_a(n * k), dist_b(n * k);

    spnnd::nearest_neighbors_sparse(nnd_e_indptr, nnd_e_indices, nnd_e_data, n,
                                    nnd_e_dim, k, "euclidean", false, 42,
                                    ind_a.data(), dist_a.data(),
                                    /*force_approximation=*/true);

    std::vector<std::array<int64_t, 3>> tree_states;
    for (int t = 0; t < 6; ++t)
        tree_states.push_back({golden_nn::tree_states_seed42_n6[t * 3],
                               golden_nn::tree_states_seed42_n6[t * 3 + 1],
                               golden_nn::tree_states_seed42_n6[t * 3 + 2]});
    spnnd::nearest_neighbors_sparse_with_state(
        nnd_e_indptr, nnd_e_indices, nnd_e_data, n, nnd_e_dim, k, "euclidean",
        false, golden_nn::rng_state_seed42, tree_states, ind_b.data(),
        dist_b.data());

    for (int64_t i = 0; i < n * k; ++i) {
        CHECK(ind_a[i] == ind_b[i]);
        CHECK(dist_a[i] == dist_b[i]);
    }
}

TEST_CASE("nearest_neighbors_sparse throws for unsupported metrics") {
    using namespace golden_sparse;
    std::vector<int32_t> ind(bf_n * 5);
    std::vector<float> dist(bf_n * 5);
    CHECK_THROWS_AS(
        spnnd::nearest_neighbors_sparse(bf_indptr, bf_indices, bf_data, bf_n,
                                        bf_dim, 5, "wasserstein_1d", false, 42,
                                        ind.data(), dist.data()),
        std::invalid_argument);
}
