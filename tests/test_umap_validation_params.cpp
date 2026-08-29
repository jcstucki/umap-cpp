// Port of reference/umap/umap/tests/test_umap_validation_params.py to doctest
// (upstream test names preserved; upstream ValueError ->
// std::invalid_argument).
//
// Skips (see tests/PORTING_NOTES.md): tests whose invalid value cannot be
// expressed against the typed C++ params surface (non-integer n_components,
// numeric metric/init objects), and tests of unported features (output_metric,
// target_n_neighbors, densmap knobs, n_jobs, callable metrics, update(),
// inverse_transform(), supervised y).
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "doctest.h"
#include "umap/umap.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"

namespace {

using umap::UMAP;
using umap::UMAPParams;

constexpr int64_t kNnN = 1002, kNnDim = 5;

const std::vector<float>& nn_data_f32() {
    static const std::vector<float> data = testutil::to_f32(
        updata::nn_data, static_cast<std::size_t>(kNnN) * kNnDim);
    return data;
}

// u = UMAP(**p); with pytest.raises(ValueError): u.fit(nn_data)
void check_fit_raises(const UMAPParams& p) {
    UMAP u(p);
    CHECK_THROWS_AS(u.fit(nn_data_f32().data(), kNnN, kNnDim),
                    std::invalid_argument);
}

}  // namespace

TEST_SUITE("test_umap_validation_params.py") {

TEST_CASE("test_umap_negative_op") {
    UMAPParams p;
    p.set_op_mix_ratio = -1.0;
    check_fit_raises(p);
}

TEST_CASE("test_umap_too_large_op") {
    UMAPParams p;
    p.set_op_mix_ratio = 1.5;
    check_fit_raises(p);
}

TEST_CASE("test_umap_bad_too_large_min_dist") {
    UMAPParams p;
    p.min_dist = 2.0;  // > spread=1.0
    check_fit_raises(p);
}

TEST_CASE("test_umap_negative_min_dist") {
    UMAPParams p;
    p.min_dist = -1.0;
    check_fit_raises(p);
}

TEST_CASE("test_umap_negative_n_components") {
    UMAPParams p;
    p.n_components = -1;
    check_fit_raises(p);
}

// n_components=1.5 etc. are unrepresentable against `int n_components`.
TEST_CASE("test_umap_non_integer_n_components" * doctest::skip()) {}

TEST_CASE("test_umap_too_small_n_neighbours") {
    UMAPParams p;
    p.n_neighbors = 0;  // upstream 0.5; the typed surface floors to int
    check_fit_raises(p);
}

TEST_CASE("test_umap_negative_n_neighbours") {
    UMAPParams p;
    p.n_neighbors = -1;
    check_fit_raises(p);
}

TEST_CASE("test_umap_bad_metric") {
    // upstream metric=45 (non-string unrepresentable); an unknown metric name
    // raises the same ValueError path.
    UMAPParams p;
    p.metric = "foobar";
    check_fit_raises(p);
}

TEST_CASE("test_umap_negative_learning_rate") {
    UMAPParams p;
    p.learning_rate = -1.5;
    check_fit_raises(p);
}

TEST_CASE("test_umap_negative_repulsion") {
    UMAPParams p;
    p.repulsion_strength = -0.5;
    check_fit_raises(p);
}

TEST_CASE("test_umap_negative_sample_rate") {
    UMAPParams p;
    p.negative_sample_rate = -1;
    check_fit_raises(p);
}

TEST_CASE("test_umap_bad_init") {
    UMAPParams p;
    p.init = "foobar";
    check_fit_raises(p);
}

// init=42 is unrepresentable against the typed init surface.
TEST_CASE("test_umap_bad_numeric_init" * doctest::skip()) {}

TEST_CASE("test_umap_bad_matrix_init") {
    UMAPParams p;
    p.init_array = std::vector<float>(2 * 3, 0.0f);  // np.zeros((2, 3))
    p.init_array_cols = 3;                           // != n_components=2
    check_fit_raises(p);
}

TEST_CASE("test_umap_negative_n_epochs") {
    UMAPParams p;
    p.n_epochs = -2;
    check_fit_raises(p);
}

// target_n_neighbors belongs to the unported supervised-fit surface.
TEST_CASE("test_umap_negative_target_n_neighbours" * doctest::skip()) {}
// output_metric is not wired into the UMAP class yet.
TEST_CASE("test_umap_bad_output_metric" * doctest::skip()) {}

TEST_CASE("test_haversine_on_highd") {
    UMAPParams p;
    p.metric = "haversine";
    check_fit_raises(p);  // nn_data is 5-dimensional
}

TEST_CASE("test_umap_haversine_embed_to_highd" * doctest::skip()) {}

TEST_CASE("test_umap_too_many_neighbors_warns") {
    UMAPParams p;
    p.a = 1.2;
    p.b = 1.75;
    p.n_neighbors = 2000;
    p.n_epochs = 11;
    p.init = "random";
    UMAP u(p);
    u.fit(nn_data_f32().data(), 100, kNnDim);  // nn_data[:100]
    CHECK(u.a() == 1.2);
    CHECK(u.b() == 1.75);
    // upstream warns and truncates _n_neighbors to X.shape[0] - 1.
    CHECK(u.effective_n_neighbors() == 99);
}

// densmap is being ported on main; its validation knobs are not present yet.
TEST_CASE("test_densmap_lambda" * doctest::skip()) {}
TEST_CASE("test_densmap_var_shift" * doctest::skip()) {}
TEST_CASE("test_densmap_frac" * doctest::skip()) {}
TEST_CASE("test_densmap_bad_output_metric" * doctest::skip()) {}

TEST_CASE("test_umap_unique_and_precomputed") {
    UMAPParams p;
    p.metric = "precomputed";
    p.unique = true;
    check_fit_raises(p);
}

// Typed n_components / metric objects are unrepresentable in C++.
TEST_CASE("test_umap_bad_n_components" * doctest::skip()) {}
TEST_CASE("test_umap_bad_metrics" * doctest::skip()) {}
// n_jobs is not part of the ported params surface.
TEST_CASE("test_umap_bad_n_jobs" * doctest::skip()) {}
// Callable metrics are not portable to C++.
TEST_CASE("test_umap_custom_distance_w_grad" * doctest::skip()) {}
TEST_CASE("test_umap_bad_output_metric_no_grad" * doctest::skip()) {}

TEST_CASE("test_umap_bad_hellinger_data") {
    // u.fit(-nn_data): the umap_.py fit guard "Metric 'hellinger' does not
    // support negative values".
    std::vector<float> neg = nn_data_f32();
    for (float& v : neg) v = -v;
    UMAPParams p;
    p.metric = "hellinger";
    UMAP u(p);
    CHECK_THROWS_AS(u.fit(neg.data(), kNnN, kNnDim), std::invalid_argument);
}

// --- inverse-update additions ------------------------------------------------
TEST_CASE("test_umap_update_bad_params") {
    // dmat = pairwise_distances(nn_data[:100]); precomputed update raises.
    {
        const std::vector<float> dmat = umap::pairwise_distances_sklearn(
            nn_data_f32().data(), 100, nn_data_f32().data(), 100, kNnDim,
            "euclidean", true);
        UMAPParams p;
        p.metric = "precomputed";
        p.n_epochs = 11;
        UMAP u(p);
        u.fit(dmat.data(), 100, 100);
        CHECK_THROWS_AS(u.update(dmat.data(), 100, 100),
                        std::invalid_argument);
    }
    // supervised fit: y = np.repeat(np.arange(5), 20); update raises.
    {
        UMAPParams p;
        p.n_epochs = 11;
        UMAP u(p);
        std::vector<float> y(100);
        for (int i = 0; i < 100; ++i)
            y[static_cast<std::size_t>(i)] = static_cast<float>(i / 20);
        u.fit(nn_data_f32().data(), 100, kNnDim, y.data());
        CHECK_THROWS_AS(
            u.update(nn_data_f32().data() + 100 * kNnDim, 100, kNnDim),
            std::invalid_argument);
    }
}
// -----------------------------------------------------------------------------
// fit(X, y) length validation belongs to the unported supervised surface.
TEST_CASE("test_umap_fit_data_and_targets_compliant" * doctest::skip()) {}

TEST_CASE("test_umap_fit_instance_returned") {
    // Data-only variant (fit(x, y) needs the supervised surface).
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    std::vector<float> x(256 * 10);
    for (float& v : x) v = unif(rng);
    UMAPParams p;
    p.n_epochs = 11;
    p.init = "random";
    UMAP u(p);
    UMAP& res = u.fit(x.data(), 256, 10);
    CHECK(&res == &u);
}

// --- inverse-update additions ------------------------------------------------
TEST_CASE("test_umap_inverse_transform_fails_expectedly") {
    // u.fit(sparse_spatial_data[:100]) -- the fixture has fewer rows, so the
    // slice keeps them all (upstream too); sparse fit -> inverse raises.
    {
        const std::vector<float> dense = testutil::to_f32(
            updata::sparse_spatial_dense,
            sizeof(updata::sparse_spatial_dense) /
                sizeof(updata::sparse_spatial_dense[0]));
        const int64_t dim = 20;
        const int64_t n = static_cast<int64_t>(dense.size()) / dim;
        const testutil::Csr sp = testutil::dense_to_csr(dense.data(), n, dim);
        UMAPParams p;
        p.n_epochs = 11;
        p.init = "random";  // spectral would raise for n <= n_components + 1
        UMAP u(p);
        u.fit(sp.indptr.data(), sp.indices.data(), sp.data.data(), n, dim);
        std::vector<float> q(static_cast<std::size_t>(n) * 2, 0.0f);
        CHECK_THROWS_AS((void)u.inverse_transform(q.data(), n),
                        std::invalid_argument);
    }
    // metric="dice" (no gradient) -> inverse raises.
    {
        UMAPParams p;
        p.metric = "dice";
        p.n_epochs = 11;
        UMAP u(p);
        u.fit(nn_data_f32().data(), 100, kNnDim);
        CHECK_THROWS_AS(
            (void)u.inverse_transform(u.embedding().data(), 10),
            std::invalid_argument);
    }
}
// -----------------------------------------------------------------------------

}  // TEST_SUITE
