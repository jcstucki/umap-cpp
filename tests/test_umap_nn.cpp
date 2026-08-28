// Port of reference/umap/umap/tests/test_umap_nn.py to doctest (upstream test
// names preserved). Uses the conftest nn_data / sparse_nn_data fixtures baked
// in tests/upstream_data.hpp (np.random.seed(189212)).
//
// Adaptations (see tests/PORTING_NOTES.md):
//   * The nearest-neighbour accuracy tests are @pytest.mark.skip upstream;
//     they are run ACTIVELY here (the C++ NN-descent port is deterministic
//     under a seed) with upstream's thresholds. Upstream passes
//     np.random as random_state; the C++ port takes a seed (42).
//   * low_memory=True is the only NN-descent path ported (it is the one
//     UMAP.fit uses), so the low_memory variant runs the same code path.
//   * test_nn_descent_neighbor_accuracy_callable_metric: callable metrics are
//     not portable to C++ -- skipped.
//   * Dense nearest_neighbors() is the NNDescent branch of umap_.py's
//     nearest_neighbors (same as upstream's function); the sparse tests use
//     nearest_neighbors_sparse with force_approximation=true to match
//     upstream's unconditional NNDescent use.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "doctest.h"
#include "umap/fuzzy_simplicial_set.hpp"
#include "umap/nn_descent.hpp"
#include "umap/sparse_nn_descent.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"

namespace {

constexpr int64_t kNnN = 1002, kNnDim = 5;
constexpr int64_t kSparseN = 1000, kSparseDim = 50;

std::vector<float> nn_data_f32() {
    return testutil::to_f32(updata::nn_data,
                            static_cast<std::size_t>(kNnN) * kNnDim);
}

// smooth_knn() from test_umap_nn.py: row sums of the local fuzzy set
// memberships exp(-max(0, d - rho) / sigma).
std::vector<double> smooth_knn_norms(double local_connectivity) {
    const std::vector<float> data = nn_data_f32();
    std::vector<int32_t> knn_indices(kNnN * 10);
    std::vector<float> knn_dists(kNnN * 10);
    umap::nnd::nearest_neighbors(data.data(), kNnN, kNnDim, 10, "euclidean",
                                 false, 42u, knn_indices.data(),
                                 knn_dists.data());
    const umap::SmoothKnnResult sk = umap::smooth_knn_dist(
        knn_dists.data(), static_cast<int>(kNnN), 10, 10.0, 64,
        local_connectivity);
    std::vector<double> norms(kNnN, 0.0);
    for (int64_t i = 0; i < kNnN; ++i)
        for (int64_t j = 0; j < 10; ++j) {
            double shifted = static_cast<double>(knn_dists[i * 10 + j]) -
                             static_cast<double>(sk.rhos[i]);
            if (shifted < 0.0) shifted = 0.0;
            norms[i] += std::exp(-(shifted / static_cast<double>(sk.sigmas[i])));
        }
    return norms;
}

}  // namespace

TEST_SUITE("test_umap_nn.py") {

TEST_CASE("test_nn_bad_metric") {
    const std::vector<float> data = nn_data_f32();
    std::vector<int32_t> ind(kNnN * 10);
    std::vector<float> dist(kNnN * 10);
    CHECK_THROWS_AS(
        umap::nnd::nearest_neighbors(data.data(), kNnN, kNnDim, 10, "foobar",
                                     false, 42u, ind.data(), dist.data()),
        std::invalid_argument);
}

TEST_CASE("test_nn_bad_metric_sparse_data") {
    std::vector<int32_t> ind(kSparseN * 10);
    std::vector<float> dist(kSparseN * 10);
    CHECK_THROWS_AS(
        umap::spnnd::nearest_neighbors_sparse(
            updata::sparse_nn_indptr, updata::sparse_nn_indices,
            updata::sparse_nn_values, kSparseN, kSparseDim, 10, "seuclidean",
            false, 42u, ind.data(), dist.data()),
        std::invalid_argument);
}

TEST_CASE("test_nn_descent_neighbor_accuracy") {
    const std::vector<float> data = nn_data_f32();
    std::vector<int32_t> ind(kNnN * 10);
    std::vector<float> dist(kNnN * 10);
    umap::nnd::nearest_neighbors(data.data(), kNnN, kNnDim, 10, "euclidean",
                                 false, 42u, ind.data(), dist.data());
    const double pct = testutil::knn_accuracy(ind.data(), 10, updata::nn_data,
                                              kNnN, kNnDim);
    CAPTURE(pct);
    CHECK(pct >= 0.85);
}

TEST_CASE("test_nn_descent_neighbor_accuracy_low_memory") {
    // low_memory=True is the only ported path; same call, upstream's 0.89 bar.
    const std::vector<float> data = nn_data_f32();
    std::vector<int32_t> ind(kNnN * 10);
    std::vector<float> dist(kNnN * 10);
    umap::nnd::nearest_neighbors(data.data(), kNnN, kNnDim, 10, "euclidean",
                                 false, 42u, ind.data(), dist.data());
    const double pct = testutil::knn_accuracy(ind.data(), 10, updata::nn_data,
                                              kNnN, kNnDim);
    CAPTURE(pct);
    CHECK(pct >= 0.89);
}

TEST_CASE("test_angular_nn_descent_neighbor_accuracy") {
    const std::vector<float> data = nn_data_f32();
    std::vector<int32_t> ind(kNnN * 10);
    std::vector<float> dist(kNnN * 10);
    umap::nnd::nearest_neighbors(data.data(), kNnN, kNnDim, 10, "cosine", true,
                                 42u, ind.data(), dist.data());
    const std::vector<double> angular =
        testutil::l2_normalize_rows(updata::nn_data, kNnN, kNnDim);
    const double pct = testutil::knn_accuracy(ind.data(), 10, angular.data(),
                                              kNnN, kNnDim);
    CAPTURE(pct);
    CHECK(pct >= 0.85);
}

TEST_CASE("test_sparse_nn_descent_neighbor_accuracy") {
    std::vector<int32_t> ind(kSparseN * 20);
    std::vector<float> dist(kSparseN * 20);
    umap::spnnd::nearest_neighbors_sparse(
        updata::sparse_nn_indptr, updata::sparse_nn_indices,
        updata::sparse_nn_values, kSparseN, kSparseDim, 20, "euclidean", false,
        42u, ind.data(), dist.data(), /*force_approximation=*/true);
    // Dense float64 copy of the sparse data for the KDTree-equivalent truth.
    std::vector<double> dense(static_cast<std::size_t>(kSparseN) * kSparseDim,
                              0.0);
    for (int64_t i = 0; i < kSparseN; ++i)
        for (int32_t p = updata::sparse_nn_indptr[i];
             p < updata::sparse_nn_indptr[i + 1]; ++p)
            dense[i * kSparseDim + updata::sparse_nn_indices[p]] =
                static_cast<double>(updata::sparse_nn_values[p]);
    const double pct = testutil::knn_accuracy(ind.data(), 20, dense.data(),
                                              kSparseN, kSparseDim);
    CAPTURE(pct);
    CHECK(pct >= 0.75);
}

TEST_CASE("test_sparse_nn_descent_neighbor_accuracy_low_memory") {
    // Same single ported path (low_memory=True), upstream's 0.85 bar.
    std::vector<int32_t> ind(kSparseN * 20);
    std::vector<float> dist(kSparseN * 20);
    umap::spnnd::nearest_neighbors_sparse(
        updata::sparse_nn_indptr, updata::sparse_nn_indices,
        updata::sparse_nn_values, kSparseN, kSparseDim, 20, "euclidean", false,
        42u, ind.data(), dist.data(), /*force_approximation=*/true);
    std::vector<double> dense(static_cast<std::size_t>(kSparseN) * kSparseDim,
                              0.0);
    for (int64_t i = 0; i < kSparseN; ++i)
        for (int32_t p = updata::sparse_nn_indptr[i];
             p < updata::sparse_nn_indptr[i + 1]; ++p)
            dense[i * kSparseDim + updata::sparse_nn_indices[p]] =
                static_cast<double>(updata::sparse_nn_values[p]);
    const double pct = testutil::knn_accuracy(ind.data(), 20, dense.data(),
                                              kSparseN, kSparseDim);
    CAPTURE(pct);
    CHECK(pct >= 0.85);
}

// Callable metrics are not portable to the C++ surface.
TEST_CASE("test_nn_descent_neighbor_accuracy_callable_metric" *
          doctest::skip()) {}

TEST_CASE("test_sparse_angular_nn_descent_neighbor_accuracy") {
    std::vector<int32_t> ind(kSparseN * 20);
    std::vector<float> dist(kSparseN * 20);
    umap::spnnd::nearest_neighbors_sparse(
        updata::sparse_nn_indptr, updata::sparse_nn_indices,
        updata::sparse_nn_values, kSparseN, kSparseDim, 20, "cosine", true,
        42u, ind.data(), dist.data(), /*force_approximation=*/true);
    std::vector<double> dense(static_cast<std::size_t>(kSparseN) * kSparseDim,
                              0.0);
    for (int64_t i = 0; i < kSparseN; ++i)
        for (int32_t p = updata::sparse_nn_indptr[i];
             p < updata::sparse_nn_indptr[i + 1]; ++p)
            dense[i * kSparseDim + updata::sparse_nn_indices[p]] =
                static_cast<double>(updata::sparse_nn_values[p]);
    const std::vector<double> angular =
        testutil::l2_normalize_rows(dense.data(), kSparseN, kSparseDim);
    const double pct = testutil::knn_accuracy(ind.data(), 20, angular.data(),
                                              kSparseN, kSparseDim);
    CAPTURE(pct);
    CHECK(pct >= 0.90);
}

TEST_CASE("test_smooth_knn_dist_l1norms") {
    const std::vector<double> norms = smooth_knn_norms(1.0);
    const double expected = 1.0 + std::log2(10.0);
    for (std::size_t i = 0; i < norms.size(); ++i) {
        CAPTURE(i);
        CHECK(std::abs(norms[i] - expected) < 1.5e-3);  // decimal=3
    }
}

TEST_CASE("test_smooth_knn_dist_l1norms_w_connectivity") {
    const std::vector<double> norms = smooth_knn_norms(1.75);
    const double expected = 1.0 + std::log2(10.0);
    for (std::size_t i = 0; i < norms.size(); ++i) {
        CAPTURE(i);
        CHECK(std::abs(norms[i] - expected) < 1.5e-3);  // decimal=3
    }
}

}  // TEST_SUITE
