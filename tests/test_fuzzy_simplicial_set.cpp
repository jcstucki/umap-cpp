// smooth_knn_dist / compute_membership_strengths / fuzzy_simplicial_set vs
// golden values from the reference Python UMAP (scripts/gen_golden.py).
#include <cstddef>
#include <vector>

#include "doctest.h"
#include "golden_data.hpp"
#include "umap/fuzzy_simplicial_set.hpp"

namespace {
constexpr int NS = golden::knn_n_samples;
constexpr int NN = golden::knn_n_neighbors;
// Goldens come from numba float32 arithmetic; ours is float32 too but exp/
// rounding may differ in the last ulp, so compare at float precision.
constexpr double FTOL = 1e-6;

void check_close(const std::vector<float>& actual, const float* expected,
                 std::size_t n, double tol = FTOL) {
    REQUIRE(actual.size() == n);
    for (std::size_t i = 0; i < n; ++i) {
        CAPTURE(i);
        CHECK(actual[i] ==
              doctest::Approx(expected[i]).epsilon(tol).scale(1.0));
    }
}
}  // namespace

TEST_CASE("smooth_knn_dist local_connectivity=1.0") {
    auto res = umap::smooth_knn_dist(golden::knn_dists, NS, NN,
                                     static_cast<double>(NN));
    check_close(res.sigmas, golden::smooth_sigmas_lc1, NS);
    check_close(res.rhos, golden::smooth_rhos_lc1, NS);
}

TEST_CASE("smooth_knn_dist local_connectivity=1.5 (interpolation)") {
    auto res = umap::smooth_knn_dist(golden::knn_dists, NS, NN,
                                     static_cast<double>(NN), 64, 1.5);
    check_close(res.sigmas, golden::smooth_sigmas_lc15, NS);
    check_close(res.rhos, golden::smooth_rhos_lc15, NS);
}

TEST_CASE("smooth_knn_dist bandwidth=0.5") {
    auto res = umap::smooth_knn_dist(golden::knn_dists, NS, NN,
                                     static_cast<double>(NN), 64, 1.0, 0.5);
    check_close(res.sigmas, golden::smooth_sigmas_bw05, NS);
    check_close(res.rhos, golden::smooth_rhos_bw05, NS);
}

TEST_CASE("compute_membership_strengths incl. return_dists") {
    auto smooth = umap::smooth_knn_dist(golden::knn_dists, NS, NN,
                                        static_cast<double>(NN));
    auto ms = umap::compute_membership_strengths(
        golden::knn_indices, golden::knn_dists, smooth.sigmas.data(),
        smooth.rhos.data(), NS, NN, /*return_dists=*/true);

    const std::size_t size = static_cast<std::size_t>(NS) * NN;
    REQUIRE(ms.rows.size() == size);
    REQUIRE(ms.cols.size() == size);
    for (std::size_t i = 0; i < size; ++i) {
        CAPTURE(i);
        CHECK(ms.rows[i] == golden::memb_rows[i]);
        CHECK(ms.cols[i] == golden::memb_cols[i]);
    }
    check_close(ms.vals, golden::memb_vals, size);
    check_close(ms.dists, golden::memb_dists, size);
}

TEST_CASE("compute_membership_strengths without return_dists") {
    auto smooth = umap::smooth_knn_dist(golden::knn_dists, NS, NN,
                                        static_cast<double>(NN));
    auto ms = umap::compute_membership_strengths(
        golden::knn_indices, golden::knn_dists, smooth.sigmas.data(),
        smooth.rhos.data(), NS, NN, /*return_dists=*/false);
    CHECK(ms.dists.empty());
    check_close(ms.vals, golden::memb_vals, static_cast<std::size_t>(NS) * NN);
}

TEST_CASE("fuzzy_simplicial_set set_op_mix_ratio=1.0 (fuzzy union)") {
    auto res = umap::fuzzy_simplicial_set(NS, golden::knn_indices,
                                          golden::knn_dists, NN, 1.0, 1.0);
    check_close(res.graph.to_dense(), golden::fuzzy_dense_mix10,
                static_cast<std::size_t>(NS) * NS);
    check_close(res.sigmas, golden::smooth_sigmas_lc1, NS);
    check_close(res.rhos, golden::smooth_rhos_lc1, NS);
    // symmetric
    auto dense = res.graph.to_dense();
    for (int i = 0; i < NS; ++i)
        for (int j = 0; j < NS; ++j)
            CHECK(dense[i * NS + j] == doctest::Approx(dense[j * NS + i])
                                           .epsilon(FTOL)
                                           .scale(1.0));
}

TEST_CASE("fuzzy_simplicial_set set_op_mix_ratio=0.5 (mixed)") {
    auto res = umap::fuzzy_simplicial_set(NS, golden::knn_indices,
                                          golden::knn_dists, NN, 0.5, 1.0);
    check_close(res.graph.to_dense(), golden::fuzzy_dense_mix05,
                static_cast<std::size_t>(NS) * NS);
}

TEST_CASE("fuzzy_simplicial_set set_op_mix_ratio=0.0 (fuzzy intersection)") {
    auto res = umap::fuzzy_simplicial_set(NS, golden::knn_indices,
                                          golden::knn_dists, NN, 0.0, 1.0);
    check_close(res.graph.to_dense(), golden::fuzzy_dense_mix00,
                static_cast<std::size_t>(NS) * NS);
}

TEST_CASE("fuzzy_simplicial_set with -1 (missing) knn entries") {
    std::vector<int32_t> knn_indices(
        golden::knn_indices,
        golden::knn_indices + static_cast<std::size_t>(NS) * NN);
    knn_indices[6 * NN + 3] = -1;  // same masking as gen_golden.py
    auto res = umap::fuzzy_simplicial_set(NS, knn_indices.data(),
                                          golden::knn_dists, NN, 1.0, 1.0);
    check_close(res.graph.to_dense(), golden::fuzzy_dense_masked,
                static_cast<std::size_t>(NS) * NS);
}

TEST_CASE("fuzzy_simplicial_set return_dists (max-symmetrised distances)") {
    auto res = umap::fuzzy_simplicial_set(NS, golden::knn_indices,
                                          golden::knn_dists, NN, 1.0, 1.0,
                                          true, /*return_dists=*/true);
    REQUIRE(res.has_dists);
    check_close(res.dists.to_dense(), golden::fuzzy_dists_dense,
                static_cast<std::size_t>(NS) * NS);
}

TEST_CASE("fuzzy_simplicial_set without set operations is the raw graph") {
    auto res = umap::fuzzy_simplicial_set(NS, golden::knn_indices,
                                          golden::knn_dists, NN, 1.0, 1.0,
                                          /*apply_set_operations=*/false);
    // Raw membership strengths (minus zeros): row-normalised local sets
    auto smooth = umap::smooth_knn_dist(golden::knn_dists, NS, NN,
                                        static_cast<double>(NN));
    auto ms = umap::compute_membership_strengths(
        golden::knn_indices, golden::knn_dists, smooth.sigmas.data(),
        smooth.rhos.data(), NS, NN);
    std::vector<float> expected(static_cast<std::size_t>(NS) * NS, 0.0f);
    for (std::size_t k = 0; k < ms.vals.size(); ++k) {
        expected[static_cast<std::size_t>(ms.rows[k]) * NS + ms.cols[k]] +=
            ms.vals[k];
    }
    CHECK(res.graph.to_dense() == expected);
}
