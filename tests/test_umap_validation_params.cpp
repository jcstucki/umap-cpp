// Port of reference/umap/umap/tests/test_umap_validation_params.py to doctest
// (upstream test names preserved; upstream ValueError ->
// std::invalid_argument).
//
// Skips (see tests/PORTING_NOTES.md): tests whose invalid value cannot be
// expressed against the typed C++ params surface (non-integer n_components,
// numeric metric/init objects, y length mismatches against the pointer-typed
// y), n_jobs, callable metrics, and unported features (update(),
// inverse_transform()).
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

TEST_CASE("test_umap_negative_target_n_neighbours") {
    UMAPParams p;
    p.target_n_neighbors = 1;
    check_fit_raises(p);
}

TEST_CASE("test_umap_bad_output_metric") {
    UMAPParams p;
    p.output_metric = "foobar";
    check_fit_raises(p);
    p = UMAPParams();
    p.output_metric = "precomputed";
    check_fit_raises(p);
    p = UMAPParams();
    p.output_metric = "hamming";  // named distance without a gradient
    check_fit_raises(p);
}

TEST_CASE("test_haversine_on_highd") {
    UMAPParams p;
    p.metric = "haversine";
    check_fit_raises(p);  // nn_data is 5-dimensional
}

TEST_CASE("test_umap_haversine_embed_to_highd") {
    // Raised when the output metric is resolved for a 3-d embedding
    // ("haversine is only defined for 2 dimensional data"), after the kNN
    // stage, exactly like the upstream fit.
    UMAPParams p;
    p.n_components = 3;
    p.output_metric = "haversine";
    check_fit_raises(p);
}

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

TEST_CASE("test_densmap_lambda") {
    UMAPParams p;
    p.densmap = true;
    p.dens_lambda = -1.0;
    check_fit_raises(p);
}

TEST_CASE("test_densmap_var_shift") {
    UMAPParams p;
    p.densmap = true;
    p.dens_var_shift = -1.0;
    check_fit_raises(p);
}

TEST_CASE("test_densmap_frac") {
    UMAPParams p;
    p.densmap = true;
    p.dens_frac = -1.0;
    check_fit_raises(p);
    p = UMAPParams();
    p.densmap = true;
    p.dens_frac = 2.0;
    check_fit_raises(p);
}

TEST_CASE("test_densmap_bad_output_metric") {
    UMAPParams p;
    p.densmap = true;
    p.output_metric = "haversine";
    check_fit_raises(p);
}

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

// update() is not ported.
TEST_CASE("test_umap_update_bad_params" * doctest::skip()) {}
// fit(X, y) length mismatches are unrepresentable against the pointer-typed
// y surface (y carries no length; callers pass n_samples-length arrays).
TEST_CASE("test_umap_fit_data_and_targets_compliant" * doctest::skip()) {}

TEST_CASE("test_umap_fit_instance_returned") {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    std::vector<float> x(256 * 10);
    for (float& v : x) v = unif(rng);
    UMAPParams p;
    p.n_epochs = 11;
    p.init = "random";

    // Passing both data and targets.
    std::vector<float> y(256);
    std::uniform_int_distribution<int> randint(0, 9);
    for (float& v : y) v = static_cast<float>(randint(rng));
    UMAP uy(p);
    UMAP& res_y = uy.fit(x.data(), 256, 10, y.data());
    CHECK(&res_y == &uy);

    // Passing only data.
    UMAP u(p);
    UMAP& res = u.fit(x.data(), 256, 10);
    CHECK(&res == &u);
}

// inverse_transform() is not ported.
TEST_CASE("test_umap_inverse_transform_fails_expectedly" * doctest::skip()) {}

}  // TEST_SUITE
