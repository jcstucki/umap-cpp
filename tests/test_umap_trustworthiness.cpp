// Port of reference/umap/umap/tests/test_umap_trustworthiness.py to doctest
// (upstream test names + thresholds preserved verbatim).
//
// Skips (features not yet ported; see tests/PORTING_NOTES.md):
//   * test_umap_sparse_trustworthiness -- sparse input is not wired into the
//     UMAP class yet (sparse NN + metrics are ported, the class glue is not).
//   * the six supervised/semi-supervised variants -- fit(y) is not ported.
// Adaptation:
//   * test_sparse_precomputed_metric_umap_trustworthiness: upstream wraps the
//     precomputed distance matrix in scipy CSR; the C++ precomputed path is
//     dense-only, so the same (fully dense) matrix is passed densely. The
//     make_blobs data is seeded upstream (random_state=42) and baked as
//     updata::blobs50.
#include <vector>

#include "doctest.h"
#include "trustworthiness.hpp"
#include "umap/umap.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"

namespace {

using umap::UMAP;
using umap::UMAPParams;

}  // namespace

TEST_SUITE("test_umap_trustworthiness.py") {

TEST_CASE("test_umap_sparse_trustworthiness" * doctest::skip()) {}

TEST_CASE("test_umap_trustworthiness_fast_approx") {
    const std::vector<float> data =
        testutil::to_f32(updata::nn_data, 50 * 5);  // nn_data[:50]
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.random_state = 42;
    p.n_epochs = 100;
    p.force_approximation_algorithm = true;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(data.data(), 50, 5);
    const double trust =
        testutil::trustworthiness(data.data(), 50, 5, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.8);
}

TEST_CASE("test_umap_trustworthiness_random_init") {
    const std::vector<float> data =
        testutil::to_f32(updata::nn_data, 50 * 5);  // nn_data[:50]
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.random_state = 42;
    p.n_epochs = 100;
    p.init = "random";
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(data.data(), 50, 5);
    const double trust =
        testutil::trustworthiness(data.data(), 50, 5, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.8);
}

// Supervised / semi-supervised fits (y, target_metric) are not ported yet.
TEST_CASE("test_supervised_umap_trustworthiness" * doctest::skip()) {}
TEST_CASE("test_semisupervised_umap_trustworthiness" * doctest::skip()) {}
TEST_CASE("test_metric_supervised_umap_trustworthiness" * doctest::skip()) {}
TEST_CASE("test_string_metric_supervised_umap_trustworthiness" *
          doctest::skip()) {}
TEST_CASE("test_discrete_metric_supervised_umap_trustworthiness" *
          doctest::skip()) {}
TEST_CASE("test_count_metric_supervised_umap_trustworthiness" *
          doctest::skip()) {}

TEST_CASE("test_sparse_precomputed_metric_umap_trustworthiness") {
    const std::vector<float> data = testutil::to_f32(updata::blobs50, 50 * 2);
    const std::vector<double> dmat64 = testutil::pairwise_euclidean_f64(
        updata::blobs50, 50, updata::blobs50, 50, 2);
    const std::vector<float> dmat = testutil::to_f32(dmat64.data(), 50 * 50);
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.random_state = 42;
    p.n_epochs = 100;
    p.metric = "precomputed";
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(dmat.data(), 50, 50);
    const double trust =
        testutil::trustworthiness(data.data(), 50, 2, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.75);
}

}  // TEST_SUITE
