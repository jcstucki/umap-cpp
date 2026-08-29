// Port of reference/umap/umap/tests/test_umap_trustworthiness.py to doctest
// (upstream test names + thresholds preserved verbatim).
//
// Skips (see tests/PORTING_NOTES.md):
//   * test_string_metric_supervised_umap_trustworthiness -- upstream's y is
//     an object array of strings; the C++ fit(y) surface is float-typed, and
//     target_metric="string" raises here like the other object-dtype targets.
// Adaptations:
//   * test_sparse_precomputed_metric_umap_trustworthiness: upstream wraps the
//     precomputed distance matrix in scipy CSR; the C++ precomputed path is
//     dense-only, so the same (fully dense) matrix is passed densely. The
//     make_blobs data is seeded upstream (random_state=42) and baked as
//     updata::blobs50 (+ blobs50_labels for the supervised cases).
//   * test_umap_sparse_trustworthiness: conftest's sparse_test_data =
//     csr(nn_data * binary_nn_data); the first 100 rows' product is built
//     densely here and converted to CSR (identical matrix). The fit is
//     unseeded like upstream.
#include <vector>

#include "doctest.h"
#include "trustworthiness.hpp"
#include "umap/umap.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"

namespace {

using umap::UMAP;
using umap::UMAPParams;

// UMAP(n_neighbors=10, min_dist=0.01, random_state=42, n_epochs=100) --
// the parameter set every supervised case uses.
UMAPParams blobs_params() {
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.random_state = 42;
    p.n_epochs = 100;
    return p;
}

// fit_transform(blobs50, labels) -> trustworthiness vs the blobs data.
double supervised_blobs_trust(const UMAPParams& p,
                              const std::vector<float>& labels) {
    const std::vector<float> data = testutil::to_f32(updata::blobs50, 50 * 2);
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(data.data(), 50, 2, labels.data());
    return testutil::trustworthiness(data.data(), 50, 2, emb.data(), 2, 10);
}

std::vector<float> blobs_labels_f32() {
    std::vector<float> y(50);
    for (int i = 0; i < 50; ++i)
        y[i] = static_cast<float>(updata::blobs50_labels[i]);
    return y;
}

}  // namespace

TEST_SUITE("test_umap_trustworthiness.py") {

TEST_CASE("test_umap_sparse_trustworthiness") {
    // sparse_test_data[:100] = csr((nn_data * binary_nn_data)[:100]).
    const int64_t n = 100, dim = 5;
    std::vector<float> dense(static_cast<std::size_t>(n) * dim);
    for (int64_t i = 0; i < n * dim; ++i)
        dense[i] = static_cast<float>(updata::nn_data[i] *
                                      updata::binary_nn_data[i]);
    const testutil::Csr csr = testutil::dense_to_csr(dense.data(), n, dim);
    UMAPParams p;
    p.n_neighbors = 10;
    p.n_epochs = 100;  // unseeded, like upstream
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        csr.indptr.data(), csr.indices.data(), csr.data.data(), n, dim);
    const double trust =
        testutil::trustworthiness(dense.data(), n, dim, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.88);
}

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

TEST_CASE("test_supervised_umap_trustworthiness") {
    const double trust = supervised_blobs_trust(blobs_params(),
                                                blobs_labels_f32());
    CAPTURE(trust);
    CHECK(trust >= 0.95);
}

TEST_CASE("test_semisupervised_umap_trustworthiness") {
    std::vector<float> labels = blobs_labels_f32();
    for (int i = 10; i < 30; ++i) labels[i] = -1.0f;  // labels[10:30] = -1
    const double trust = supervised_blobs_trust(blobs_params(), labels);
    CAPTURE(trust);
    CHECK(trust >= 0.95);
}

TEST_CASE("test_metric_supervised_umap_trustworthiness") {
    UMAPParams p = blobs_params();
    p.target_metric = "l1";
    p.target_weight = 0.8;
    const double trust = supervised_blobs_trust(p, blobs_labels_f32());
    CAPTURE(trust);
    CHECK(trust >= 0.95);
}

// Upstream's y is np.array(["this","that","other"])[labels] (object dtype);
// unrepresentable against the float-typed C++ y surface.
TEST_CASE("test_string_metric_supervised_umap_trustworthiness" *
          doctest::skip()) {}

TEST_CASE("test_discrete_metric_supervised_umap_trustworthiness") {
    UMAPParams p = blobs_params();
    p.target_metric = "ordinal";
    p.target_weight = 0.8;
    const double trust = supervised_blobs_trust(p, blobs_labels_f32());
    CAPTURE(trust);
    CHECK(trust >= 0.95);
}

TEST_CASE("test_count_metric_supervised_umap_trustworthiness") {
    // labels = labels**2 + 2*labels
    std::vector<float> labels = blobs_labels_f32();
    for (float& v : labels) v = v * v + 2.0f * v;
    UMAPParams p = blobs_params();
    p.target_metric = "count";
    p.target_weight = 0.8;
    const double trust = supervised_blobs_trust(p, labels);
    CAPTURE(trust);
    CHECK(trust >= 0.95);
}

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
