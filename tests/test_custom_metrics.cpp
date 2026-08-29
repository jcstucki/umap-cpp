// --- hardening additions -----------------------------------------------------
// Callable ("custom") metric coverage: UMAPParams::custom_metric /
// custom_metric_grad / custom_output_metric are the C++ equivalents of
// upstream's callable metrics, and must take the exact routes a named metric
// takes. Equivalence is asserted bitwise against named twins whose kernels
// the lambdas replicate:
//  * small-data (< 4096) fit + transform vs metric="euclidean" -- both ride
//    the sklearn _pairwise_callable replication;
//  * standard branch (force_approximation_algorithm=True) fit + transform vs
//    metric="manhattan" -- manhattan has no NN-descent fast alternative,
//    non-angular (euclidean) rp-trees and epsilon 0.12, exactly the
//    treatment pynndescent gives a callable, so a kernel-faithful lambda
//    must reproduce the whole NN-descent + query pipeline bit for bit;
//  * inverse_transform via custom_metric_grad vs the named gradient
//    registry, and output_metric="custom" vs the named generic-layout path.
// Restrictions under test: sparse input raises, save() raises.
// -----------------------------------------------------------------------------
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"
#include "umap/layouts_generic.hpp"
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

// umap.distances.euclidean under numba on float32 rows (float32 squared
// diffs, float64 accumulator/sqrt) -- what a user's callable would compute
// and what the named small-data path evaluates.
float euclidean_f32(const float* x, const float* y, std::size_t n) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float d = x[i] - y[i];
        const float sq = d * d;
        result += static_cast<double>(sq);
    }
    return static_cast<float>(std::sqrt(result));
}

// pynndescent.distances.manhattan on float32 rows (float32 accumulator) --
// the kernel NN-descent runs for metric="manhattan".
float manhattan_nnd_f32(const float* x, const float* y, std::size_t n) {
    float result = 0.0f;
    for (std::size_t i = 0; i < n; ++i) result += std::fabs(x[i] - y[i]);
    return result;
}

bool same_bytes(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

TEST_SUITE("custom metrics [hardening]") {

TEST_CASE("small-data custom metric == named euclidean (fit + transform)") {
    const int64_t n = 100, nq = 20;
    const float* X = nn_data_f32().data();
    const float* Q = nn_data_f32().data() + n * kNnDim;

    UMAPParams named;
    named.metric = "euclidean";
    named.random_state = 42u;
    named.n_epochs = 11;
    UMAP u_named(named);
    u_named.fit(X, n, kNnDim);

    UMAPParams custom = named;
    custom.metric = "custom";
    custom.custom_metric = euclidean_f32;
    UMAP u_custom(custom);
    u_custom.fit(X, n, kNnDim);

    REQUIRE(same_bytes(u_named.embedding(), u_custom.embedding()));
    CHECK(same_bytes(u_named.transform(Q, nq), u_custom.transform(Q, nq)));
}

TEST_CASE("standard-branch custom metric == named manhattan "
          "(NN-descent + prepared-index query)") {
    const int64_t n = 150, nq = 15;
    const float* X = nn_data_f32().data();
    const float* Q = nn_data_f32().data() + n * kNnDim;

    UMAPParams named;
    named.metric = "manhattan";
    named.random_state = 42u;
    named.n_epochs = 11;
    named.force_approximation_algorithm = true;  // standard branch on 150 rows
    UMAP u_named(named);
    u_named.fit(X, n, kNnDim);

    UMAPParams custom = named;
    custom.metric = "custom";
    custom.custom_metric = manhattan_nnd_f32;
    UMAP u_custom(custom);
    u_custom.fit(X, n, kNnDim);

    // Same rp-trees (euclidean), same raw kernel inside NN-descent, no
    // correction, same query epsilon (0.12: callables are never angular).
    REQUIRE(u_named.knn_indices() == u_custom.knn_indices());
    REQUIRE(same_bytes(u_named.knn_dists(), u_custom.knn_dists()));
    REQUIRE(same_bytes(u_named.embedding(), u_custom.embedding()));
    CHECK(same_bytes(u_named.transform(Q, nq), u_custom.transform(Q, nq)));
}

TEST_CASE("custom_metric_grad enables inverse_transform == named manhattan") {
    const int64_t n = 150;
    const float* X = nn_data_f32().data();

    UMAPParams named;
    named.metric = "manhattan";
    named.random_state = 42u;
    named.n_epochs = 11;
    named.force_approximation_algorithm = true;
    UMAP u_named(named);
    u_named.fit(X, n, kNnDim);

    UMAPParams custom = named;
    custom.metric = "custom";
    custom.custom_metric = manhattan_nnd_f32;
    // The gradient flavour: the named registry kernel itself (what an
    // upstream callable returning (dist, grad) computes).
    custom.custom_metric_grad =
        umap::make_output_metric("manhattan", umap::MetricKwds{});
    UMAP u_custom(custom);
    u_custom.fit(X, n, kNnDim);
    REQUIRE(same_bytes(u_named.embedding(), u_custom.embedding()));

    // 5 embedding points back through the manifold.
    const std::vector<float> pts(u_named.embedding().begin(),
                                 u_named.embedding().begin() + 5 * 2);
    CHECK(same_bytes(u_named.inverse_transform(pts.data(), 5),
                     u_custom.inverse_transform(pts.data(), 5)));

    // Without the gradient flavour the inverse is unavailable (upstream:
    // "Inverse transform not available for given metric.").
    UMAPParams no_grad = custom;
    no_grad.custom_metric_grad = nullptr;
    UMAP u_no_grad(no_grad);
    u_no_grad.fit(X, n, kNnDim);
    CHECK_THROWS_AS(u_no_grad.inverse_transform(pts.data(), 5),
                    std::invalid_argument);
}

TEST_CASE("custom output metric == named non-euclidean output path") {
    const int64_t n = 100;
    const float* X = nn_data_f32().data();

    UMAPParams named;
    named.output_metric = "manhattan";
    named.random_state = 42u;
    named.n_epochs = 11;
    UMAP u_named(named);
    u_named.fit(X, n, kNnDim);

    UMAPParams custom = named;
    custom.output_metric = "custom";
    custom.custom_output_metric =
        umap::make_output_metric("manhattan", umap::MetricKwds{});
    UMAP u_custom(custom);
    u_custom.fit(X, n, kNnDim);

    CHECK(same_bytes(u_named.embedding(), u_custom.embedding()));
}

TEST_CASE("custom metric restrictions: sparse input raises, save() raises") {
    // Tiny CSR (4 x 3).
    const std::vector<int32_t> indptr = {0, 2, 4, 5, 7};
    const std::vector<int32_t> indices = {0, 2, 1, 2, 0, 1, 2};
    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 1.5f,
                                       0.5f, 2.5f, 1.0f};
    UMAPParams p;
    p.metric = "custom";
    p.custom_metric = euclidean_f32;
    p.n_neighbors = 2;
    p.random_state = 42u;
    UMAP u(p);
    CHECK_THROWS_AS(
        u.fit(indptr.data(), indices.data(), values.data(), 4, 3),
        std::invalid_argument);

    // A fitted dense custom-metric model cannot be persisted
    // (std::function is not serialisable).
    UMAPParams pd;
    pd.metric = "custom";
    pd.custom_metric = euclidean_f32;
    pd.random_state = 42u;
    pd.n_epochs = 11;
    UMAP ud(pd);
    ud.fit(nn_data_f32().data(), 100, kNnDim);
    std::ostringstream os;
    CHECK_THROWS_AS(ud.save(os), std::runtime_error);
}

}  // TEST_SUITE
