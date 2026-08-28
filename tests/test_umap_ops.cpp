// Port of reference/umap/umap/tests/test_umap_ops.py to doctest (upstream
// test names preserved). Datasets: make_blobs / disconnection data baked by
// scripts/gen_upstream_fixtures.py (upstream generates them unseeded; the
// baked copies use documented fixed seeds -- see the header comments in
// tests/upstream_data.hpp).
//
// Skips (see tests/PORTING_NOTES.md):
//   * test_umap_update / test_umap_update_large -- update() not ported.
//   * test_component_layout_options -- component_layout with
//     metric="precomputed" + linkage kwds is not ported (euclidean-only).
//   * test_disconnected_data_precomputed[sparse=True] -- sparse precomputed
//     distance matrices are not ported (the dense variant runs).
// Adaptations:
//   * warning-count assertions in test_disconnected_data are replaced by the
//     model's vertices_disconnected()/edges_removed() bookkeeping (the C++
//     port prints the same warning text to stderr).
//   * test_fast_knn_indices_matches_argsort draws its random distance matrix
//     from std::mt19937 instead of np.random.RandomState(0) -- the property
//     under test (argpartition selection == stable argsort on distinct
//     distances) is RNG-agnostic. The ties matrix of
//     test_fast_knn_indices_preserves_distances_with_ties IS the upstream
//     RandomState(1) matrix (baked as updata::tie_D).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "doctest.h"
#include "umap/spectral.hpp"
#include "umap/umap.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"

namespace {

using umap::UMAP;
using umap::UMAPParams;

// Stable (mergesort-equivalent) argsort of each row, first k columns.
std::vector<int32_t> stable_argsort_knn(const std::vector<float>& D, int64_t n,
                                        int64_t k) {
    std::vector<int32_t> out(static_cast<std::size_t>(n) * k);
    std::vector<int32_t> order(n);
    for (int64_t i = 0; i < n; ++i) {
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int32_t a, int32_t b) {
                             return D[i * n + a] < D[i * n + b];
                         });
        std::copy(order.begin(), order.begin() + k, out.begin() + i * k);
    }
    return out;
}

}  // namespace

TEST_SUITE("test_umap_ops.py") {

TEST_CASE("test_fast_knn_indices_matches_argsort") {
    // Symmetric random distances, zero diagonal, distinct off-diagonal values
    // (200 x 200, k = 15).
    const int64_t n = 200, k = 15;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    std::vector<float> D(static_cast<std::size_t>(n) * n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i; j < n; ++j) {
            const float v = (i == j) ? 0.0f : unif(rng);
            D[i * n + j] = v;
            D[j * n + i] = v;
        }
    const std::vector<int32_t> expected = stable_argsort_knn(D, n, k);
    std::vector<int32_t> result(static_cast<std::size_t>(n) * k);
    std::vector<float> result_d(static_cast<std::size_t>(n) * k);
    umap::fast_knn_from_dmat(D, n, n, k, result.data(), result_d.data());
    for (std::size_t i = 0; i < result.size(); ++i) {
        CAPTURE(i);
        CHECK(result[i] == expected[i]);
    }
}

TEST_CASE("test_fast_knn_indices_preserves_distances_with_ties") {
    const int64_t n = 150, k = 15;
    std::vector<float> D = testutil::to_f32(
        updata::tie_D, static_cast<std::size_t>(n) * n);
    const std::vector<int32_t> expected = stable_argsort_knn(D, n, k);
    std::vector<int32_t> result(static_cast<std::size_t>(n) * k);
    std::vector<float> result_d(static_cast<std::size_t>(n) * k);
    umap::fast_knn_from_dmat(D, n, n, k, result.data(), result_d.data());
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < k; ++j) {
            CAPTURE(i);
            CAPTURE(j);
            CHECK(D[i * n + expected[i * k + j]] == result_d[i * k + j]);
        }
}

TEST_CASE("test_blobs_cluster") {
    // make_blobs(500, 10, centers=5) -- upstream unseeded, baked with
    // random_state=42. UMAP(n_epochs=100), unseeded like upstream.
    const std::vector<float> data =
        testutil::to_f32(updata::blobs500, 500 * 10);
    UMAPParams p;
    p.n_epochs = 100;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(data.data(), 500, 10);
    const std::vector<int> clusters =
        testutil::kmeans_predict(emb.data(), 500, 2, 5);
    std::vector<int> labels(500);
    for (int i = 0; i < 500; ++i) labels[i] = updata::blobs500_labels[i];
    CHECK(testutil::adjusted_rand_score(labels, clusters) == 1.0);
}

TEST_CASE("test_multi_component_layout") {
    const std::vector<float> data =
        testutil::to_f32(updata::blobs100, 100 * 2);
    UMAPParams p;
    p.n_neighbors = 4;
    p.n_epochs = 100;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(data.data(), 100, 2);

    // true_centroids by the make_blobs labels; embed_centroids of the DATA
    // grouped by the embedding's kmeans labels; both l2-normalized rows.
    std::vector<double> true_c(5 * 2, 0.0), embed_c(5 * 2, 0.0);
    std::vector<int64_t> tc(5, 0), ec(5, 0);
    const std::vector<int> embed_labels =
        testutil::kmeans_predict(emb.data(), 100, 2, 5);
    for (int i = 0; i < 100; ++i) {
        const int tl = updata::blobs100_labels[i], el = embed_labels[i];
        for (int d = 0; d < 2; ++d) {
            true_c[tl * 2 + d] += updata::blobs100[i * 2 + d];
            embed_c[el * 2 + d] += updata::blobs100[i * 2 + d];
        }
        ++tc[tl];
        ++ec[el];
    }
    for (int c = 0; c < 5; ++c)
        for (int d = 0; d < 2; ++d) {
            true_c[c * 2 + d] /= std::max<int64_t>(tc[c], 1);
            embed_c[c * 2 + d] /= std::max<int64_t>(ec[c], 1);
        }
    const std::vector<double> tn = testutil::l2_normalize_rows(true_c.data(), 5, 2);
    const std::vector<double> en =
        testutil::l2_normalize_rows(embed_c.data(), 5, 2);
    double error = 0.0;
    for (int i = 0; i < 10; ++i) error += (tn[i] - en[i]) * (tn[i] - en[i]);
    CAPTURE(error);
    CHECK(error < 15.0);
}

TEST_CASE("test_multi_component_layout_precomputed") {
    const std::vector<double> dmat64 = testutil::pairwise_euclidean_f64(
        updata::blobs100, 100, updata::blobs100, 100, 2);
    const std::vector<float> dmat = testutil::to_f32(dmat64.data(), 100 * 100);
    UMAPParams p;
    p.n_neighbors = 4;
    p.n_epochs = 100;
    p.metric = "precomputed";
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(dmat.data(), 100, 100);

    std::vector<double> true_c(5 * 2, 0.0), embed_c(5 * 2, 0.0);
    std::vector<int64_t> tc(5, 0), ec(5, 0);
    const std::vector<int> embed_labels =
        testutil::kmeans_predict(emb.data(), 100, 2, 5);
    for (int i = 0; i < 100; ++i) {
        const int tl = updata::blobs100_labels[i], el = embed_labels[i];
        for (int d = 0; d < 2; ++d) {
            true_c[tl * 2 + d] += updata::blobs100[i * 2 + d];
            embed_c[el * 2 + d] += updata::blobs100[i * 2 + d];
        }
        ++tc[tl];
        ++ec[el];
    }
    for (int c = 0; c < 5; ++c)
        for (int d = 0; d < 2; ++d) {
            true_c[c * 2 + d] /= std::max<int64_t>(tc[c], 1);
            embed_c[c * 2 + d] /= std::max<int64_t>(ec[c], 1);
        }
    const std::vector<double> tn = testutil::l2_normalize_rows(true_c.data(), 5, 2);
    const std::vector<double> en =
        testutil::l2_normalize_rows(embed_c.data(), 5, 2);
    double error = 0.0;
    for (int i = 0; i < 10; ++i) error += (tn[i] - en[i]) * (tn[i] - en[i]);
    CAPTURE(error);
    CHECK(error < 15.0);
}

TEST_CASE("test_disconnected_data") {
    // Parametrized upstream over num_isolates {1,5} x metric
    // {jaccard, hellinger} x force_approximation {true, false}.
    struct Sub { int num_isolates; const std::uint8_t* raw; int64_t n, dim; };
    const Sub datasets[] = {
        {1, updata::disc1, 11, 31},
        {5, updata::disc5, 15, 35},
    };
    for (const Sub& s : datasets)
        for (const char* metric : {"jaccard", "hellinger"})
            for (const bool force_approximation : {true, false}) {
                CAPTURE(s.num_isolates);
                CAPTURE(metric);
                CAPTURE(force_approximation);
                const std::vector<float> data = testutil::to_f32(
                    s.raw, static_cast<std::size_t>(s.n) * s.dim);
                UMAPParams p;
                p.n_neighbors = 3;
                p.metric = metric;
                p.force_approximation_algorithm = force_approximation;
                UMAP model(p);
                model.fit(data.data(), s.n, s.dim);
                // Upstream counts the emitted disconnection warnings; the C++
                // port records the same event via vertices_disconnected().
                CHECK(model.vertices_disconnected() >= s.num_isolates);
                const std::vector<bool> isolated =
                    testutil::disconnected_vertices(model.graph());
                CHECK(isolated[10] == true);
                int64_t number_of_nan = 0;
                for (int64_t i = 0; i < s.n; ++i)
                    if (isolated[i])
                        for (int d = 0; d < p.n_components; ++d)
                            if (std::isnan(
                                    model.embedding()[i * p.n_components + d]))
                                ++number_of_nan;
                CHECK(number_of_nan >=
                      static_cast<int64_t>(s.num_isolates) * p.n_components);
            }
}

TEST_CASE("test_disconnected_data_precomputed") {
    // num_isolates=1; the sparse=True variant is skipped (sparse precomputed
    // not ported), the dense variant runs.
    const int64_t n = 11;
    const std::vector<float> dmat =
        testutil::to_f32(updata::discp_dmat, static_cast<std::size_t>(n) * n);
    UMAPParams p;
    p.n_neighbors = 3;
    p.metric = "precomputed";
    p.disconnection_distance = 1.0;
    UMAP model(p);
    model.fit(dmat.data(), n, n);
    const std::vector<bool> isolated =
        testutil::disconnected_vertices(model.graph());
    CHECK(isolated[10] == true);
    int64_t number_of_nan = 0;
    for (int64_t i = 0; i < n; ++i)
        if (isolated[i])
            for (int d = 0; d < p.n_components; ++d)
                if (std::isnan(model.embedding()[i * p.n_components + d]))
                    ++number_of_nan;
    CHECK(number_of_nan >= 1 * p.n_components);
}

TEST_CASE("test_bad_transform_data") {
    const float one_point[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float query[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    UMAP model{UMAPParams{}};
    model.fit(one_point, 1, 4);
    CHECK_THROWS_AS(model.transform(query, 1), std::invalid_argument);
}

TEST_CASE("test_umap_transform_embedding_stability") {
    // Part 1: iris subset model; transforming same-row-count random data must
    // not alter embedding_. (Fresh model here; conftest builds it per session.)
    const std::vector<float> iris = testutil::to_f32(updata::iris_data, 150 * 4);
    const std::vector<float> subset = testutil::select_rows(
        iris.data(), 150, 4, updata::iris_selection, true);
    const int64_t n = static_cast<int64_t>(subset.size() / 4);
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.random_state = 42;
    UMAP fitter(p);
    fitter.fit(subset.data(), n, 4);
    const std::vector<float> original = fitter.embedding();

    std::mt19937 rng(217);  // upstream: unseeded np.random.random
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    std::vector<float> new_data(subset.size());
    for (float& v : new_data) v = unif(rng);
    (void)fitter.transform(new_data.data(), n);
    REQUIRE(fitter.embedding().size() == original.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        CAPTURE(i);
        CHECK(fitter.embedding()[i] == original[i]);
    }

    // Part 2 (issue #217): same-row-count transform after an unseeded fit
    // (upstream fits a[:, :5] of a 100 x 10 array; built directly as 100 x 5).
    std::vector<float> a(100 * 5), b(100 * 5);
    for (float& v : a) v = unif(rng);
    for (float& v : b) v = unif(rng);
    UMAPParams p2;
    p2.n_epochs = 100;
    UMAP umap2(p2);
    const std::vector<float> u1 = umap2.fit_transform(a.data(), 100, 5);
    REQUIRE(umap2.embedding().size() == u1.size());
    for (std::size_t i = 0; i < u1.size(); ++i)
        CHECK(umap2.embedding()[i] == u1[i]);
    (void)umap2.transform(b.data(), 100);
    for (std::size_t i = 0; i < u1.size(); ++i) {
        CAPTURE(i);
        CHECK(umap2.embedding()[i] == u1[i]);
    }
}

TEST_CASE("test_umap_update" * doctest::skip()) {}
TEST_CASE("test_umap_update_large" * doctest::skip()) {}

TEST_CASE("test_umap_graph_layout") {
    const std::vector<float> data =
        testutil::to_f32(updata::blobs500, 500 * 10);
    UMAPParams p;
    p.n_epochs = 100;
    p.transform_mode = "graph";
    UMAP model(p);
    const umap::SparseMatrix& graph =
        model.fit_transform_graph(data.data(), 500, 10);
    std::vector<int32_t> labels;
    CHECK(umap::connected_components(graph, labels) == 5);

    // model.transform(data[:10] + normal(0, 0.1, (10, 10))).
    std::vector<float> noisy(10 * 10);
    for (int i = 0; i < 100; ++i)
        noisy[i] = static_cast<float>(updata::blobs500[i] +
                                      updata::graph_noise[i]);
    const umap::SparseMatrix new_graph = model.transform_graph(noisy.data(), 10);
    CHECK(new_graph.n_rows == 10);
}

TEST_CASE("test_component_layout_options" * doctest::skip()) {}

}  // TEST_SUITE
