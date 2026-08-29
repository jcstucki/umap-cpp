// Port of reference/umap/umap/tests/test_umap_on_iris.py to doctest (upstream
// test names + thresholds preserved). iris is the real sklearn load_iris
// dataset and iris_selection the conftest np.random.choice mask, both baked in
// tests/upstream_data.hpp.
//
// Skips (see tests/PORTING_NOTES.md):
//   * test_umap_transform_on_iris_modified_dtype -- upstream re-casts
//     embedding_ to float64 and checks transform still works; the C++
//     embedding is always float32, there is no dtype to modify.
// Adaptations:
//   * test_umap_inverse_transform_on_iris: sklearn KDTree queries are ported
//     as exact brute-force kNN (the tree is exact too).
//   * test_precomputed_knn_on_iris: the pytest.warns bookkeeping is replaced
//     by behavioural checks (the C++ port emits the same warnings on stderr);
//     the NotImplementedError match "search index" maps to the C++
//     std::invalid_argument whose message contains "search index".
//   * test_umap_clusterability_on_supervised_iris: sklearn KMeans(3) +
//     adjusted_rand_score are replaced by the ports in upstream_helpers.hpp
//     (k-means++/Lloyd/n_init=10; upstream leaves KMeans unseeded).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "doctest.h"
#include "trustworthiness.hpp"
#include "umap/nn_descent.hpp"
#include "umap/umap.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"

namespace {

using umap::PrecomputedKnn;
using umap::UMAP;
using umap::UMAPParams;

constexpr int64_t kIrisN = 150, kIrisDim = 4;

const std::vector<float>& iris_f32() {
    static const std::vector<float> data =
        testutil::to_f32(updata::iris_data, kIrisN * kIrisDim);
    return data;
}

const std::vector<float>& iris_subset() {  // iris.data[iris_selection]
    static const std::vector<float> data = testutil::select_rows(
        iris_f32().data(), kIrisN, kIrisDim, updata::iris_selection, true);
    return data;
}

const std::vector<float>& iris_rest() {  // iris.data[~iris_selection]
    static const std::vector<float> data = testutil::select_rows(
        iris_f32().data(), kIrisN, kIrisDim, updata::iris_selection, false);
    return data;
}

int64_t subset_n() {
    return static_cast<int64_t>(iris_subset().size() / kIrisDim);
}
int64_t rest_n() { return static_cast<int64_t>(iris_rest().size() / kIrisDim); }

UMAPParams iris_params() {  // UMAP(n_neighbors=10, min_dist=0.01, rs=42)
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.random_state = 42;
    return p;
}

// conftest session fixtures, built once.
UMAP& iris_model() {
    static UMAP model = [] {
        UMAP m(iris_params());
        m.fit(iris_f32().data(), kIrisN, kIrisDim);
        return m;
    }();
    return model;
}

UMAP& iris_subset_model() {
    static UMAP model = [] {
        UMAP m(iris_params());
        m.fit(iris_subset().data(), subset_n(), kIrisDim);
        return m;
    }();
    return model;
}

bool throws_containing(const std::function<void()>& fn,
                       const std::string& needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

double graph_rms(const umap::SparseMatrix& a, const umap::SparseMatrix& b) {
    REQUIRE(a.n_rows == b.n_rows);
    REQUIRE(a.n_cols == b.n_cols);
    const std::size_t total =
        static_cast<std::size_t>(a.n_rows) * static_cast<std::size_t>(a.n_cols);
    std::vector<double> da(total, 0.0), db(total, 0.0);
    for (int64_t i = 0; i < a.n_rows; ++i)
        for (int32_t k = a.indptr[i]; k < a.indptr[i + 1]; ++k)
            da[i * a.n_cols + a.indices[k]] = a.data[k];
    for (int64_t i = 0; i < b.n_rows; ++i)
        for (int32_t k = b.indptr[i]; k < b.indptr[i + 1]; ++k)
            db[i * b.n_cols + b.indices[k]] = b.data[k];
    double s = 0.0;
    for (std::size_t i = 0; i < total; ++i) {
        const double diff = da[i] - db[i];
        s += diff * diff;
    }
    return std::sqrt(s / static_cast<double>(total));
}

}  // namespace

TEST_SUITE("test_umap_on_iris.py") {

TEST_CASE("test_umap_trustworthiness_on_iris") {
    const std::vector<float>& emb = iris_model().embedding();
    const double trust = testutil::trustworthiness(
        iris_f32().data(), kIrisN, kIrisDim, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.97);
}

TEST_CASE("test_initialized_umap_trustworthiness_on_iris") {
    // init = iris.data[:, 2:] (150 x 2 ndarray init).
    std::vector<float> init(kIrisN * 2);
    for (int64_t i = 0; i < kIrisN; ++i) {
        init[i * 2 + 0] = static_cast<float>(updata::iris_data[i * 4 + 2]);
        init[i * 2 + 1] = static_cast<float>(updata::iris_data[i * 4 + 3]);
    }
    UMAPParams p = iris_params();
    p.n_epochs = 200;
    p.init_array = init;
    p.init_array_cols = 2;
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(iris_f32().data(), kIrisN, kIrisDim);
    const double trust = testutil::trustworthiness(
        iris_f32().data(), kIrisN, kIrisDim, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.97);
}

TEST_CASE("test_umap_trustworthiness_on_sphere_iris") {
    UMAPParams p = iris_params();
    p.n_epochs = 200;
    p.output_metric = "haversine";
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(iris_f32().data(), kIrisN, kIrisDim);
    // Project onto a 3D embedding of the sphere (r = 3) and use cosine
    // distance for the input-space ranks, like upstream.
    const double r = 3.0;
    std::vector<float> projected(kIrisN * 3);
    for (int64_t i = 0; i < kIrisN; ++i) {
        const double theta = emb[i * 2 + 0], phi = emb[i * 2 + 1];
        projected[i * 3 + 0] =
            static_cast<float>(r * std::sin(theta) * std::cos(phi));
        projected[i * 3 + 1] =
            static_cast<float>(r * std::sin(theta) * std::sin(phi));
        projected[i * 3 + 2] = static_cast<float>(r * std::cos(theta));
    }
    const double trust =
        testutil::trustworthiness(iris_f32().data(), kIrisN, kIrisDim,
                                  projected.data(), 3, 10, "cosine");
    CAPTURE(trust);
    CHECK(trust >= 0.65);
}

TEST_CASE("test_umap_transform_on_iris") {
    const std::vector<float> emb =
        iris_subset_model().transform(iris_rest().data(), rest_n());
    const double trust = testutil::trustworthiness(
        iris_rest().data(), rest_n(), kIrisDim, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.80);
}

TEST_CASE("test_umap_transform_on_iris_w_pynndescent") {
    UMAPParams p = iris_params();
    p.n_epochs = 100;
    p.force_approximation_algorithm = true;
    UMAP model(p);
    model.fit(iris_subset().data(), subset_n(), kIrisDim);
    const std::vector<float> emb =
        model.transform(iris_rest().data(), rest_n());
    const double trust = testutil::trustworthiness(
        iris_rest().data(), rest_n(), kIrisDim, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.85);
}

TEST_CASE("test_umap_transform_on_iris_modified_dtype" * doctest::skip()) {}

TEST_CASE("test_umap_sparse_transform_on_iris") {
    const int64_t n = subset_n(), nq = rest_n();
    const testutil::Csr data =
        testutil::dense_to_csr(iris_subset().data(), n, kIrisDim);
    UMAPParams p = iris_params();
    p.n_epochs = 100;
    UMAP model(p);
    model.fit(data.indptr.data(), data.indices.data(), data.data.data(), n,
              kIrisDim);
    const testutil::Csr new_data =
        testutil::dense_to_csr(iris_rest().data(), nq, kIrisDim);
    const std::vector<float> emb = model.transform(
        new_data.indptr.data(), new_data.indices.data(), new_data.data.data(),
        nq);
    const double trust = testutil::trustworthiness(
        iris_rest().data(), nq, kIrisDim, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.80);
}

TEST_CASE("test_precomputed_transform_on_iris") {
    // distance_matrix = squareform(pdist(data)) in float64, cast like upstream.
    const std::vector<double> data64 = testutil::select_rows(
        updata::iris_data, kIrisN, kIrisDim, updata::iris_selection, true);
    const std::vector<double> rest64 = testutil::select_rows(
        updata::iris_data, kIrisN, kIrisDim, updata::iris_selection, false);
    const int64_t n = subset_n(), nq = rest_n();
    const std::vector<double> dmat64 = testutil::pairwise_euclidean_f64(
        data64.data(), n, data64.data(), n, kIrisDim);
    const std::vector<float> dmat =
        testutil::to_f32(dmat64.data(), dmat64.size());
    UMAPParams p = iris_params();
    p.n_epochs = 100;
    p.metric = "precomputed";
    UMAP model(p);
    model.fit(dmat.data(), n, n);
    const std::vector<double> new_dmat64 = testutil::pairwise_euclidean_f64(
        rest64.data(), nq, data64.data(), n, kIrisDim);
    const std::vector<float> new_dmat =
        testutil::to_f32(new_dmat64.data(), new_dmat64.size());
    const std::vector<float> emb = model.transform(new_dmat.data(), nq);
    const double trust = testutil::trustworthiness(
        iris_rest().data(), nq, kIrisDim, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.85);
}

// --- hardening additions -----------------------------------------------------
TEST_CASE("test_precomputed_sparse_transform_on_iris") {
    // distance_matrix = csr_matrix(squareform(pdist(data))) -- the zero
    // diagonal is dropped by the CSR conversion; fit takes upstream's
    // dedicated sparse precomputed branch. transform gets
    // csr_matrix(cdist(new_data, data)).
    const std::vector<double> data64 = testutil::select_rows(
        updata::iris_data, kIrisN, kIrisDim, updata::iris_selection, true);
    const std::vector<double> rest64 = testutil::select_rows(
        updata::iris_data, kIrisN, kIrisDim, updata::iris_selection, false);
    const int64_t n = subset_n(), nq = rest_n();
    const std::vector<double> dmat64 = testutil::pairwise_euclidean_f64(
        data64.data(), n, data64.data(), n, kIrisDim);
    const std::vector<float> dmat =
        testutil::to_f32(dmat64.data(), dmat64.size());
    const testutil::Csr D = testutil::dense_to_csr(dmat.data(), n, n);

    UMAPParams p = iris_params();
    p.n_epochs = 100;
    p.metric = "precomputed";
    UMAP model(p);
    model.fit(D.indptr.data(), D.indices.data(), D.data.data(), n, n);

    const std::vector<double> new_dmat64 = testutil::pairwise_euclidean_f64(
        rest64.data(), nq, data64.data(), n, kIrisDim);
    const std::vector<float> new_dmat =
        testutil::to_f32(new_dmat64.data(), new_dmat64.size());
    const testutil::Csr QD = testutil::dense_to_csr(new_dmat.data(), nq, n);
    const std::vector<float> emb =
        model.transform(QD.indptr.data(), QD.indices.data(), QD.data.data(),
                        nq);
    const double trust = testutil::trustworthiness(
        iris_rest().data(), nq, kIrisDim, emb.data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.85);
}
// -----------------------------------------------------------------------------

TEST_CASE("test_umap_clusterability_on_supervised_iris") {
    // conftest::supervised_iris_model = UMAP(n_neighbors=10, min_dist=0.01,
    // n_epochs=200, random_state=42).fit(iris.data, iris.target).
    UMAPParams p = iris_params();
    p.n_epochs = 200;
    std::vector<float> y(kIrisN);
    for (int64_t i = 0; i < kIrisN; ++i)
        y[i] = static_cast<float>(updata::iris_target[i]);
    UMAP model(p);
    model.fit(iris_f32().data(), kIrisN, kIrisDim, y.data());
    const std::vector<int> clusters = testutil::kmeans_predict(
        model.embedding().data(), kIrisN, 2, 3);
    std::vector<int> target(updata::iris_target, updata::iris_target + kIrisN);
    const double ari = testutil::adjusted_rand_score(clusters, target);
    CAPTURE(ari);
    CHECK(ari >= 0.95);
}

// --- inverse-update additions ------------------------------------------------
TEST_CASE("test_umap_inverse_transform_on_iris") {
    UMAP& fitter = iris_model();
    const std::vector<float>& emb = fitter.embedding();
    // KDTree queries ported as exact brute-force kNN (stable tie order).
    const auto knn = [](const float* data, int64_t n, int64_t dim,
                        const float* q, int k) {
        std::vector<std::pair<double, int64_t>> d(
            static_cast<std::size_t>(n));
        for (int64_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int64_t c = 0; c < dim; ++c) {
                const double diff = static_cast<double>(q[c]) -
                                    static_cast<double>(data[j * dim + c]);
                acc += diff * diff;
            }
            d[static_cast<std::size_t>(j)] = {acc, j};
        }
        std::stable_sort(d.begin(), d.end());
        std::vector<int64_t> out(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j)
            out[static_cast<std::size_t>(j)] =
                d[static_cast<std::size_t>(j)].second;
        return out;
    };
    for (int i = 1; i < 150; i += 20) {
        const float* query_point =
            emb.data() + static_cast<std::size_t>(i) * 2;
        const std::vector<int64_t> near_points =
            knn(emb.data(), kIrisN, 2, query_point, 5);
        // centroid = np.mean(embedding_[near_points], axis=0) (float32).
        float centroid[2] = {0.0f, 0.0f};
        for (int64_t np_i : near_points) {
            centroid[0] += emb[static_cast<std::size_t>(np_i) * 2 + 0];
            centroid[1] += emb[static_cast<std::size_t>(np_i) * 2 + 1];
        }
        centroid[0] /= 5.0f;
        centroid[1] /= 5.0f;
        const std::vector<float> highd_centroid =
            fitter.inverse_transform(centroid, 1);
        const std::vector<int64_t> highd_near_points = knn(
            iris_f32().data(), kIrisN, kIrisDim, highd_centroid.data(), 10);
        int common = 0;
        for (int64_t a : near_points)
            for (int64_t b : highd_near_points)
                if (a == b) ++common;
        CAPTURE(i);
        CHECK(common >= 3);
    }
}
// -----------------------------------------------------------------------------

TEST_CASE("test_precomputed_knn_on_iris") {
    const int64_t n = subset_n();
    // knn = nearest_neighbors(data, 10, "euclidean", None, False, 42): the
    // NNDescent branch returning (indices, dists, search_index).
    auto index = std::make_shared<umap::nnd::PreparedIndex>(
        umap::nnd::PreparedIndex::build(iris_subset().data(), n, kIrisDim, 10,
                                        "euclidean", 42u));
    PrecomputedKnn knn;
    knn.n_samples = n;
    knn.n_neighbors = 10;
    knn.indices = index->knn_indices();
    knn.dists = index->knn_dists();

    // force_approximation_algorithm is ignored when a precomputed knn is used.
    UMAPParams p = iris_params();
    PrecomputedKnn knn_with_index = knn;
    knn_with_index.search_index = index;
    p.precomputed_knn = knn_with_index;
    p.force_approximation_algorithm = false;
    UMAP with_pknn(p);
    with_pknn.fit(iris_subset().data(), n, kIrisDim);

    // Similarity graphs should nearly agree with the plain subset model
    // (threshold empirically chosen upstream).
    CHECK(graph_rms(with_pknn.graph(), iris_subset_model().graph()) < 0.005);

    // precomputed_knn=(knn[0], knn[1]) (no search index): upstream warns
    // "transforming new data" but the fit embedding is identical.
    UMAPParams p2 = iris_params();
    p2.precomputed_knn = knn;  // no search_index
    UMAP no_index(p2);
    no_index.fit(iris_subset().data(), n, kIrisDim);
    REQUIRE(no_index.embedding().size() == with_pknn.embedding().size());
    for (std::size_t i = 0; i < no_index.embedding().size(); ++i) {
        CAPTURE(i);
        CHECK(no_index.embedding()[i] == with_pknn.embedding()[i]);
    }

    // #848: no search index -> transform raises mentioning the search index.
    CHECK(throws_containing(
        [&] { no_index.transform(iris_rest().data(), rest_n()); },
        "search index"));

    // force_approximation_algorithm=True is likewise ignored.
    UMAPParams p3 = iris_params();
    p3.precomputed_knn = knn;
    p3.force_approximation_algorithm = true;
    UMAP force_true(p3);
    force_true.fit(iris_subset().data(), n, kIrisDim);
    REQUIRE(force_true.embedding().size() == no_index.embedding().size());
    for (std::size_t i = 0; i < force_true.embedding().size(); ++i) {
        CAPTURE(i);
        CHECK(force_true.embedding()[i] == no_index.embedding()[i]);
    }
}

}  // TEST_SUITE
