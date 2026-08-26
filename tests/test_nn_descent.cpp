// Tests for the RP-tree forest + NN-descent kNN port (umap/nn_descent.hpp).
// Deterministic sub-components are compared against goldens generated from the
// reference pynndescent (tests/golden_nn.hpp); the end-to-end kNN graph is
// validated by neighbor recall against exact brute-force kNN, mirroring
// upstream test_umap_nn.py.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include "doctest.h"
#include "golden_nn.hpp"
#include "umap/nn_descent.hpp"

using namespace umap::nnd;

namespace {

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

TEST_CASE("NumpyRandomState replicates numpy RandomState.randint seeding") {
    NumpyRandomState rs(42);
    auto rng_state = rs.rng_state3();
    auto search_state = rs.rng_state3();
    for (int i = 0; i < 3; ++i) {
        CHECK(rng_state[i] == golden_nn::rng_state_seed42[i]);
        CHECK(search_state[i] == golden_nn::search_rng_state_seed42[i]);
    }
    // per-tree states drawn next, row-major
    for (int t = 0; t < 6; ++t) {
        auto st = rs.rng_state3();
        for (int i = 0; i < 3; ++i)
            CHECK(st[i] == golden_nn::tree_states_seed42_n6[t * 3 + i]);
    }

    NumpyRandomState rs2(12345);
    for (int i = 0; i < 12; ++i)
        CHECK(rs2.randint_pynn() == golden_nn::randint_draws_seed12345[i]);
}

TEST_CASE("n_trees / n_iters formulas match Python rounding") {
    const int64_t n_params =
        sizeof(golden_nn::param_n) / sizeof(golden_nn::param_n[0]);
    for (int64_t i = 0; i < n_params; ++i) {
        const double n = static_cast<double>(golden_nn::param_n[i]);
        int64_t n_trees = std::min<int64_t>(
            64, 5 + static_cast<int64_t>(std::nearbyint(std::sqrt(n) / 20.0)));
        int64_t n_iters = std::max<int64_t>(
            5, static_cast<int64_t>(std::nearbyint(std::log2(n))));
        CHECK(n_trees == golden_nn::param_n_trees[i]);
        CHECK(n_iters == golden_nn::param_n_iters[i]);
    }
}

TEST_CASE("checked_flagged_heap_push matches pynndescent semantics") {
    NNHeap h = NNHeap::make(1, 6);
    const int64_t n_ops =
        sizeof(golden_nn::push_ops_d) / sizeof(golden_nn::push_ops_d[0]);
    for (int64_t i = 0; i < n_ops; ++i) {
        int r = checked_flagged_heap_push(
            h.dist_row(0), h.ind_row(0), h.flag_row(0), 6,
            golden_nn::push_ops_d[i],
            static_cast<int32_t>(golden_nn::push_ops_idx[i]),
            static_cast<uint8_t>(golden_nn::push_ops_flag[i]));
        CHECK(r == golden_nn::push_rets[i]);
    }
    for (int64_t k = 0; k < 6; ++k) {
        CHECK(h.ind_row(0)[k] == golden_nn::push_final_ind[k]);
        CHECK(h.dist_row(0)[k] == golden_nn::push_final_dist[k]);
        CHECK(static_cast<int>(h.flag_row(0)[k]) == golden_nn::push_final_flag[k]);
    }
}

TEST_CASE("simple_heap_push matches pynndescent semantics") {
    std::vector<float> pri(6, std::numeric_limits<float>::infinity());
    std::vector<int32_t> ind(6, -1);
    const int64_t n_ops =
        sizeof(golden_nn::push_ops_d) / sizeof(golden_nn::push_ops_d[0]);
    for (int64_t i = 0; i < n_ops; ++i) {
        int r = simple_heap_push(pri.data(), ind.data(), 6,
                                 golden_nn::push_ops_d[i],
                                 static_cast<int32_t>(golden_nn::push_ops_idx[i]));
        CHECK(r == golden_nn::simple_push_rets[i]);
    }
    for (int64_t k = 0; k < 6; ++k) {
        CHECK(ind[k] == golden_nn::simple_push_final_ind[k]);
        CHECK(pri[k] == golden_nn::simple_push_final_dist[k]);
    }
}

TEST_CASE("deheap_sort matches pynndescent") {
    std::vector<int32_t> ind(golden_nn::deheap_ind_in,
                             golden_nn::deheap_ind_in + 28);
    std::vector<float> dist(golden_nn::deheap_dist_in,
                            golden_nn::deheap_dist_in + 28);
    deheap_sort(ind.data(), dist.data(), 4, 7);
    for (int64_t i = 0; i < 28; ++i) {
        CHECK(ind[i] == golden_nn::deheap_ind_out[i]);
        CHECK(dist[i] == golden_nn::deheap_dist_out[i]);
    }
}

TEST_CASE("new_build_candidates matches pynndescent (n_threads=1)") {
    NNHeap g = NNHeap::make(10, 4);
    for (int64_t i = 0; i < 40; ++i) {
        g.ind[i] = golden_nn::cand_graph_ind[i];
        g.flag[i] = static_cast<uint8_t>(golden_nn::cand_graph_flag_in[i]);
        g.dist[i] = 0.0f;  // unused by candidate building
    }
    std::vector<int32_t> new_c, old_c;
    new_build_candidates(g, 3, golden_nn::cand_rng_state, 1, new_c, old_c);
    for (int64_t i = 0; i < 30; ++i) {
        CHECK(new_c[i] == golden_nn::cand_new[i]);
        CHECK(old_c[i] == golden_nn::cand_old[i]);
    }
    for (int64_t i = 0; i < 40; ++i)
        CHECK(static_cast<int>(g.flag[i]) == golden_nn::cand_graph_flag_out[i]);
}

TEST_CASE("euclidean_random_projection_split matches pynndescent") {
    std::vector<int32_t> indices(12);
    for (int i = 0; i < 12; ++i) indices[i] = i;
    int64_t st[3] = {golden_nn::split_rng_state[0], golden_nn::split_rng_state[1],
                     golden_nn::split_rng_state[2]};
    std::vector<int32_t> left, right;
    std::vector<float> hp;
    float off;
    euclidean_random_projection_split(golden_nn::split_data, 4, indices, st,
                                      left, right, hp, off);

    const size_t nl = sizeof(golden_nn::esplit_left) / sizeof(int32_t);
    const size_t nr = sizeof(golden_nn::esplit_right) / sizeof(int32_t);
    REQUIRE(left.size() == nl);
    REQUIRE(right.size() == nr);
    for (size_t i = 0; i < nl; ++i) CHECK(left[i] == golden_nn::esplit_left[i]);
    for (size_t i = 0; i < nr; ++i) CHECK(right[i] == golden_nn::esplit_right[i]);
    for (int i = 0; i < 4; ++i)
        CHECK(hp[i] == doctest::Approx(golden_nn::esplit_hyperplane[i]).epsilon(1e-6));
    CHECK(off == doctest::Approx(golden_nn::esplit_offset).epsilon(1e-6));
    for (int i = 0; i < 3; ++i) CHECK(st[i] == golden_nn::esplit_rng_after[i]);
}

TEST_CASE("angular_random_projection_split matches pynndescent") {
    std::vector<int32_t> indices(12);
    for (int i = 0; i < 12; ++i) indices[i] = i;
    int64_t st[3] = {golden_nn::split_rng_state[0], golden_nn::split_rng_state[1],
                     golden_nn::split_rng_state[2]};
    std::vector<int32_t> left, right;
    std::vector<float> hp;
    float off;
    angular_random_projection_split(golden_nn::split_data, 4, indices, st, left,
                                    right, hp, off);

    const size_t nl = sizeof(golden_nn::asplit_left) / sizeof(int32_t);
    const size_t nr = sizeof(golden_nn::asplit_right) / sizeof(int32_t);
    REQUIRE(left.size() == nl);
    REQUIRE(right.size() == nr);
    for (size_t i = 0; i < nl; ++i) CHECK(left[i] == golden_nn::asplit_left[i]);
    for (size_t i = 0; i < nr; ++i) CHECK(right[i] == golden_nn::asplit_right[i]);
    for (int i = 0; i < 4; ++i)
        CHECK(hp[i] == doctest::Approx(golden_nn::asplit_hyperplane[i]).epsilon(1e-5));
    CHECK(off == doctest::Approx(golden_nn::asplit_offset));
    for (int i = 0; i < 3; ++i) CHECK(st[i] == golden_nn::asplit_rng_after[i]);
}

TEST_CASE("make_dense_tree leaves match pynndescent (euclidean and angular)") {
    for (int angular = 0; angular < 2; ++angular) {
        int64_t rng_state[3] = {golden_nn::tree_rng_state[0],
                                golden_nn::tree_rng_state[1],
                                golden_nn::tree_rng_state[2]};
        std::vector<std::vector<int32_t>> leaves;
        int64_t width = make_dense_tree_leaves(golden_nn::tree_data, 100, 4,
                                               rng_state, 15, angular != 0, 200,
                                               leaves);
        const int64_t g_n = angular ? golden_nn::atree_n_leaves
                                    : golden_nn::etree_n_leaves;
        const int64_t g_w = angular ? golden_nn::atree_leaf_width
                                    : golden_nn::etree_leaf_width;
        const std::int32_t* g_leaves =
            angular ? golden_nn::atree_leaves : golden_nn::etree_leaves;

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

TEST_CASE("brute_force_knn sanity: self first, ascending distances") {
    const int64_t n = golden_nn::small_n, dim = golden_nn::small_dim, k = 10;
    std::vector<int32_t> ind(n * k);
    std::vector<float> dist(n * k);
    brute_force_knn(golden_nn::small_data, n, dim, k, "euclidean", ind.data(),
                    dist.data());
    for (int64_t i = 0; i < n; ++i) {
        CHECK(ind[i * k] == i);
        CHECK(dist[i * k] == 0.0f);
        for (int64_t j = 1; j < k; ++j) CHECK(dist[i * k + j] >= dist[i * k + j - 1]);
    }
}

TEST_CASE("nearest_neighbors: full recall on small data (euclidean)") {
    const int64_t n = golden_nn::small_n, dim = golden_nn::small_dim, k = 10;
    std::vector<int32_t> ind(n * k), exact_ind(n * k);
    std::vector<float> dist(n * k), exact_dist(n * k);
    nearest_neighbors(golden_nn::small_data, n, dim, k, "euclidean", false, 42,
                      ind.data(), dist.data());
    brute_force_knn(golden_nn::small_data, n, dim, k, "euclidean",
                    exact_ind.data(), exact_dist.data());

    double recall = neighbor_recall(ind.data(), exact_ind.data(), n, k);
    CHECK(recall == doctest::Approx(1.0));

    // NN-descent should converge fully here: each point is its own nearest
    // neighbor at corrected distance 0, and distances are ascending.
    for (int64_t i = 0; i < n; ++i) {
        CHECK(ind[i * k] == i);
        CHECK(dist[i * k] == 0.0f);
        for (int64_t j = 1; j < k; ++j) CHECK(dist[i * k + j] >= dist[i * k + j - 1]);
        // corrected euclidean distances should match the exact metric closely
        for (int64_t j = 0; j < k; ++j)
            CHECK(dist[i * k + j] ==
                  doctest::Approx(exact_dist[i * k + j]).epsilon(1e-4));
    }
}

TEST_CASE("nearest_neighbors: recall >= 0.90 on 600-point data (euclidean)") {
    const int64_t n = golden_nn::med_n, dim = golden_nn::med_dim, k = 10;
    std::vector<int32_t> ind(n * k), exact_ind(n * k);
    std::vector<float> dist(n * k), exact_dist(n * k);
    nearest_neighbors(golden_nn::med_data, n, dim, k, "euclidean", false, 42,
                      ind.data(), dist.data());
    brute_force_knn(golden_nn::med_data, n, dim, k, "euclidean",
                    exact_ind.data(), exact_dist.data());
    double recall = neighbor_recall(ind.data(), exact_ind.data(), n, k);
    INFO("euclidean recall = " << recall);
    CHECK(recall >= 0.90);
}

TEST_CASE("nearest_neighbors: recall >= 0.85 on 600-point data (cosine)") {
    const int64_t n = golden_nn::med_n, dim = golden_nn::med_dim, k = 10;
    std::vector<int32_t> ind(n * k), exact_ind(n * k);
    std::vector<float> dist(n * k), exact_dist(n * k);
    nearest_neighbors(golden_nn::med_data, n, dim, k, "cosine", true, 42,
                      ind.data(), dist.data());
    brute_force_knn(golden_nn::med_data, n, dim, k, "cosine", exact_ind.data(),
                    exact_dist.data());
    double recall = neighbor_recall(ind.data(), exact_ind.data(), n, k);
    INFO("cosine recall = " << recall);
    CHECK(recall >= 0.85);
    // corrected cosine distances should approximate the true metric
    for (int64_t i = 0; i < 20; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            if (ind[i * k + j] < 0) continue;
            float truth = true_cosine(golden_nn::med_data + i * dim,
                                      golden_nn::med_data + ind[i * k + j] * dim,
                                      dim);
            CHECK(dist[i * k + j] == doctest::Approx(truth).epsilon(2e-3));
        }
    }
}

TEST_CASE("nearest_neighbors_with_state(seed-42 states) == nearest_neighbors(42)") {
    // med_n = 600 -> n_trees = 6, matching the goldens' 6 hardcoded tree states
    const int64_t n = golden_nn::med_n, dim = golden_nn::med_dim, k = 10;
    std::vector<int32_t> ind_a(n * k), ind_b(n * k);
    std::vector<float> dist_a(n * k), dist_b(n * k);

    nearest_neighbors(golden_nn::med_data, n, dim, k, "euclidean", false, 42,
                      ind_a.data(), dist_a.data());

    std::vector<std::array<int64_t, 3>> tree_states;
    for (int t = 0; t < 6; ++t)
        tree_states.push_back({golden_nn::tree_states_seed42_n6[t * 3],
                               golden_nn::tree_states_seed42_n6[t * 3 + 1],
                               golden_nn::tree_states_seed42_n6[t * 3 + 2]});
    nearest_neighbors_with_state(golden_nn::med_data, n, dim, k, "euclidean",
                                 false, golden_nn::rng_state_seed42, tree_states,
                                 ind_b.data(), dist_b.data());

    for (int64_t i = 0; i < n * k; ++i) {
        CHECK(ind_a[i] == ind_b[i]);
        CHECK(dist_a[i] == dist_b[i]);
    }
}
