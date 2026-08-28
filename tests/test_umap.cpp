// Tests for the UMAP class (fit / fit_transform / transform), compared
// end-to-end and stage-by-stage against goldens generated from the reference
// Python UMAP (tests/golden_umap.hpp, NUMBA_NUM_THREADS=1, seeded), plus a
// trustworthiness bound (ported from sklearn.manifold.trustworthiness) and
// parameter-validation error cases.
//
// Parity levels (documented; see include/umap/umap.hpp):
//  * small-data branch + init="random" (case d): bit-exact end to end for
//    both fit and transform -- every stage (pairwise kernel, kNN, fuzzy set,
//    init draws, SGD) reproduces Python exactly.
//  * NN-descent branch (cases a, b): kNN *indices* are identical but numba's
//    fastmath kernels leave the distances ~1 ulp off (call-site-dependent
//    LLVM codegen), which perturbs the fuzzy-graph weights and hence the SGD
//    trajectory; graph structure / RandomState stream / init remain exact
//    (case a) and the result is validated at trustworthiness level.
//  * init="spectral" (cases b, c): our eigensolver is Spectra, not ARPACK --
//    eigenvectors agree only to solver precision and up to per-column sign,
//    so the init is compared up-to-sign and the embedding at quality level.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_umap.hpp"
#include "trustworthiness.hpp"
#include "umap/layouts.hpp"
#include "umap/nn_descent.hpp"
#include "umap/umap.hpp"

using umap::SparseMatrix;
using umap::UMAP;
using umap::UMAPParams;

namespace {

std::size_t count_mismatch(const std::vector<float>& a, const float* b) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++c;
    return c;
}

bool has_nan(const std::vector<float>& a) {
    for (float v : a)
        if (std::isnan(v)) return true;
    return false;
}

// The [0,10] per-column rescale simplicial_set_embedding applies to the init.
std::vector<float> rescale_init(std::vector<float> emb, int64_t n, int dim) {
    for (int d = 0; d < dim; ++d) {
        float mn = emb[d], mx = emb[d];
        for (int64_t i = 1; i < n; ++i) {
            const float v = emb[i * dim + d];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        const float span = mx - mn;
        for (int64_t i = 0; i < n; ++i) {
            float& v = emb[i * dim + d];
            v = (10.0f * (v - mn)) / span;
        }
    }
    return emb;
}

// Max |a - golden| per column allowing an overall per-column sign flip of the
// *rescaled* coordinates (x -> 10 - x), for the spectral eigenvector sign
// ambiguity.
double init_diff_up_to_sign(const std::vector<float>& ours, const float* gold,
                            int64_t n, int dim) {
    double worst = 0.0;
    for (int d = 0; d < dim; ++d) {
        double direct = 0.0, flipped = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double o = ours[i * dim + d];
            const double g = gold[i * dim + d];
            direct = std::max(direct, std::abs(o - g));
            flipped = std::max(flipped, std::abs((10.0 - o) - g));
        }
        worst = std::max(worst, std::min(direct, flipped));
    }
    return worst;
}

// Filtered-graph head/tail/epochs_per_sample as the fit SGD sees them
// (COO of the canonical CSR after epoch-based weight filtering).
struct FitEdges {
    std::vector<int32_t> head, tail;
    std::vector<double> eps;
    int n_epochs = 0;
};

FitEdges fit_edges(const UMAP& model) {
    FitEdges out;
    SparseMatrix g = model.graph();
    out.n_epochs = umap::resolve_epochs_and_filter_graph(
        g, model.params().n_epochs.has_value() ? *model.params().n_epochs
                                               : -1);
    for (int32_t row = 0; row < g.n_rows; ++row) {
        for (int32_t k = g.indptr[row]; k < g.indptr[row + 1]; ++k) {
            out.head.push_back(row);
            out.tail.push_back(g.indices[k]);
        }
    }
    out.eps = umap::make_epochs_per_sample(g.data.data(), g.data.size(),
                                           out.n_epochs);
    return out;
}

// sklearn.manifold.trustworthiness port -- factored into trustworthiness.hpp
// for the Phase 2 upstream test-suite port.
using testutil::trustworthiness;

}  // namespace

// ---------------------------------------------------------------------------
// Seeded RandomState stream parity: the draw sequence in fit
// (NN-descent seeding -> init draws -> SGD rng_state) lands on the same
// randint values as Python's, captured in the goldens.
TEST_CASE("umap: fit RandomState stream parity (NN-descent + random init)") {
    umap::nnd::NumpyRandomState rs(42);
    auto idx = umap::nnd::PreparedIndex::build(
        golden_umap::ca_X, golden_umap::ca_n, golden_umap::ca_dim, 15,
        "euclidean", rs);
    // init="random": uniform(-10, 10, size=(n, 2)).
    for (int64_t i = 0; i < golden_umap::ca_n * 2; ++i)
        (void)rs.uniform(-10.0, 10.0);
    const auto r3 = rs.rng_state3();
    for (int c = 0; c < 3; ++c)
        CHECK(r3[c] == golden_umap::ca_fit_rng_state[c]);
}

TEST_CASE("umap: fit RandomState stream parity (small data + spectral init)") {
    // Small-data branch consumes nothing before the init; spectral init draws
    // the discarded eigsh warm start normal(size=(n, dim+1)) and the
    // noisy_scale_coords normal(size=(n, dim)) -- with numpy's cached-gauss
    // semantics -- then the SGD rng_state.
    umap::nnd::NumpyRandomState rs(42);
    for (int64_t i = 0; i < golden_umap::cc_n * 3; ++i) (void)rs.gauss();
    for (int64_t i = 0; i < golden_umap::cc_n * 2; ++i)
        (void)rs.normal(0.0, 0.0001);
    const auto r3 = rs.rng_state3();
    for (int c = 0; c < 3; ++c)
        CHECK(r3[c] == golden_umap::cc_fit_rng_state[c]);
}

// ---------------------------------------------------------------------------
// Case d: small-data branch + init="random" -- fully deterministic path,
// bit-exact end to end against Python for both fit and transform.
TEST_CASE("umap: BIT-EXACT fit_transform (case d: small data, random init)") {
    UMAPParams p;
    p.init = "random";
    p.random_state = 42;
    // Explicit a/b (scipy's values): the C++ find_ab_params LM matches
    // scipy's curve_fit minimum only to ~1e-7, which would otherwise be the
    // only divergence on this path.
    p.a = golden_umap::cd_a;
    p.b = golden_umap::cd_b;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_umap::cd_X, golden_umap::cd_n, golden_umap::cd_dim);
    CHECK(model.small_data());

    // Stage parity: init (post-rescale) and SGD inputs.
    const std::vector<float> init = rescale_init(
        model.fit_init(), golden_umap::cd_n, 2);
    CHECK(count_mismatch(init, golden_umap::cd_fit_init) == 0);

    const FitEdges e = fit_edges(model);
    CHECK(e.n_epochs == golden_umap::cd_fit_n_epochs);
    REQUIRE(static_cast<int64_t>(e.head.size()) == golden_umap::cd_fit_n_edges);
    CHECK(std::memcmp(e.head.data(), golden_umap::cd_fit_head,
                      e.head.size() * sizeof(int32_t)) == 0);
    CHECK(std::memcmp(e.tail.data(), golden_umap::cd_fit_tail,
                      e.tail.size() * sizeof(int32_t)) == 0);
    double eps_diff = 0.0;
    for (std::size_t i = 0; i < e.eps.size(); ++i)
        eps_diff = std::max(eps_diff,
                            std::abs(e.eps[i] - golden_umap::cd_fit_eps[i]));
    CHECK(eps_diff == 0.0);

    // End-to-end bit-exact embedding.
    REQUIRE(emb.size() == static_cast<std::size_t>(golden_umap::cd_n) * 2);
    CHECK(count_mismatch(emb, golden_umap::cd_embedding) == 0);
}

TEST_CASE("umap: BIT-EXACT transform (case d: small data, random init)") {
    UMAPParams p;
    p.init = "random";
    p.random_state = 42;
    p.a = golden_umap::cd_a;
    p.b = golden_umap::cd_b;
    UMAP model(p);
    model.fit(golden_umap::cd_X, golden_umap::cd_n, golden_umap::cd_dim);

    // Input-hash shortcut: transforming the training data returns embedding_.
    const std::vector<float> same =
        model.transform(golden_umap::cd_X, golden_umap::cd_n);
    CHECK(count_mismatch(same, model.embedding().data()) == 0);

    const std::vector<float> temb =
        model.transform(golden_umap::cd_queries, golden_umap::cd_nq);
    REQUIRE(temb.size() == static_cast<std::size_t>(golden_umap::cd_nq) * 2);
    CHECK(count_mismatch(temb, golden_umap::cd_t_embedding) == 0);
}

// ---------------------------------------------------------------------------
// Case a: NN-descent path + init="random". kNN indices / graph structure /
// RandomState stream / init are exact; the fastmath ~1 ulp distance wobble
// makes the SGD trajectory diverge, so the embedding is quality-checked.
TEST_CASE("umap: fit parity, random init, NN-descent path (case a)") {
    UMAPParams p;
    p.init = "random";
    p.random_state = 42;
    p.force_approximation_algorithm = true;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_umap::ca_X, golden_umap::ca_n, golden_umap::ca_dim);
    CHECK(!model.small_data());

    // The random init is independent of the kNN distances: bit-exact.
    const std::vector<float> init = rescale_init(
        model.fit_init(), golden_umap::ca_n, 2);
    CHECK(count_mismatch(init, golden_umap::ca_fit_init) == 0);

    // Graph structure exact; weights (and so epochs_per_sample) only to the
    // fastmath tolerance.
    const FitEdges e = fit_edges(model);
    CHECK(e.n_epochs == golden_umap::ca_fit_n_epochs);
    REQUIRE(static_cast<int64_t>(e.head.size()) == golden_umap::ca_fit_n_edges);
    CHECK(std::memcmp(e.head.data(), golden_umap::ca_fit_head,
                      e.head.size() * sizeof(int32_t)) == 0);
    CHECK(std::memcmp(e.tail.data(), golden_umap::ca_fit_tail,
                      e.tail.size() * sizeof(int32_t)) == 0);
    double eps_rel = 0.0;
    for (std::size_t i = 0; i < e.eps.size(); ++i)
        eps_rel = std::max(
            eps_rel, std::abs(e.eps[i] - golden_umap::ca_fit_eps[i]) /
                         golden_umap::ca_fit_eps[i]);
    CHECK(eps_rel < 1e-4);

    // Embedding quality matches Python's on the same data.
    CHECK(!has_nan(emb));
    const double tw_ours = trustworthiness(
        golden_umap::ca_X, golden_umap::ca_n, golden_umap::ca_dim, emb.data(),
        2, 5);
    const double tw_py = trustworthiness(
        golden_umap::ca_X, golden_umap::ca_n, golden_umap::ca_dim,
        golden_umap::ca_embedding, 2, 5);
    CHECK(tw_ours >= tw_py - 0.02);
}

TEST_CASE("umap: transform quality on held-out points (case a)") {
    UMAPParams p;
    p.init = "random";
    p.random_state = 42;
    p.force_approximation_algorithm = true;
    UMAP model(p);
    model.fit(golden_umap::ca_X, golden_umap::ca_n, golden_umap::ca_dim);
    const std::vector<float> temb =
        model.transform(golden_umap::ca_queries, golden_umap::ca_nq);
    REQUIRE(temb.size() == static_cast<std::size_t>(golden_umap::ca_nq) * 2);
    CHECK(!has_nan(temb));
    // Same quality as Python's transform of the same queries.
    const double tw_ours = trustworthiness(
        golden_umap::ca_queries, golden_umap::ca_nq, golden_umap::ca_dim,
        temb.data(), 2, 5);
    const double tw_py = trustworthiness(
        golden_umap::ca_queries, golden_umap::ca_nq, golden_umap::ca_dim,
        golden_umap::ca_t_embedding, 2, 5);
    CHECK(tw_ours >= tw_py - 0.05);
}

// ---------------------------------------------------------------------------
// Case b: NN-descent path + init="spectral": init parity up to the
// per-column eigenvector sign, embedding at quality level.
TEST_CASE("umap: fit parity, spectral init, NN-descent path (case b)") {
    UMAPParams p;
    p.init = "spectral";
    p.random_state = 42;
    p.force_approximation_algorithm = true;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_umap::cb_X, golden_umap::cb_n, golden_umap::cb_dim);

    const FitEdges e = fit_edges(model);
    CHECK(e.n_epochs == golden_umap::cb_fit_n_epochs);
    REQUIRE(static_cast<int64_t>(e.head.size()) == golden_umap::cb_fit_n_edges);
    CHECK(std::memcmp(e.head.data(), golden_umap::cb_fit_head,
                      e.head.size() * sizeof(int32_t)) == 0);

    const std::vector<float> init = rescale_init(
        model.fit_init(), golden_umap::cb_n, 2);
    CHECK(init_diff_up_to_sign(init, golden_umap::cb_fit_init,
                               golden_umap::cb_n, 2) < 1e-2);

    CHECK(!has_nan(emb));
    const double tw_ours = trustworthiness(
        golden_umap::cb_X, golden_umap::cb_n, golden_umap::cb_dim, emb.data(),
        2, 5);
    const double tw_py = trustworthiness(
        golden_umap::cb_X, golden_umap::cb_n, golden_umap::cb_dim,
        golden_umap::cb_embedding, 2, 5);
    CHECK(tw_ours >= tw_py - 0.02);
}

// ---------------------------------------------------------------------------
// Case c: small-data branch + init="spectral": the entire graph pipeline is
// bit-exact; only the eigensolver (Spectra vs ARPACK, per-column sign)
// separates us from Python, so init is compared up-to-sign and the embedding
// at quality level.
TEST_CASE("umap: fit parity, small-data branch, spectral init (case c)") {
    UMAPParams p;
    p.init = "spectral";
    p.random_state = 42;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_umap::cc_X, golden_umap::cc_n, golden_umap::cc_dim);
    CHECK(model.small_data());

    // Graph pipeline bit-exact (weights included).
    const FitEdges e = fit_edges(model);
    CHECK(e.n_epochs == golden_umap::cc_fit_n_epochs);
    REQUIRE(static_cast<int64_t>(e.head.size()) == golden_umap::cc_fit_n_edges);
    CHECK(std::memcmp(e.head.data(), golden_umap::cc_fit_head,
                      e.head.size() * sizeof(int32_t)) == 0);
    CHECK(std::memcmp(e.tail.data(), golden_umap::cc_fit_tail,
                      e.tail.size() * sizeof(int32_t)) == 0);
    double eps_diff = 0.0;
    for (std::size_t i = 0; i < e.eps.size(); ++i)
        eps_diff = std::max(eps_diff,
                            std::abs(e.eps[i] - golden_umap::cc_fit_eps[i]));
    CHECK(eps_diff == 0.0);

    const std::vector<float> init = rescale_init(
        model.fit_init(), golden_umap::cc_n, 2);
    CHECK(init_diff_up_to_sign(init, golden_umap::cc_fit_init,
                               golden_umap::cc_n, 2) < 1e-2);

    CHECK(!has_nan(emb));
    const double tw_ours = trustworthiness(
        golden_umap::cc_X, golden_umap::cc_n, golden_umap::cc_dim, emb.data(),
        2, 5);
    const double tw_py = trustworthiness(
        golden_umap::cc_X, golden_umap::cc_n, golden_umap::cc_dim,
        golden_umap::cc_embedding, 2, 5);
    CHECK(tw_ours >= tw_py - 0.02);
}

TEST_CASE("umap: transform quality on held-out points (case c, small data)") {
    UMAPParams p;
    p.init = "spectral";
    p.random_state = 42;
    UMAP model(p);
    model.fit(golden_umap::cc_X, golden_umap::cc_n, golden_umap::cc_dim);
    const std::vector<float> temb =
        model.transform(golden_umap::cc_queries, golden_umap::cc_nq);
    REQUIRE(temb.size() == static_cast<std::size_t>(golden_umap::cc_nq) * 2);
    CHECK(!has_nan(temb));
    const double tw_ours = trustworthiness(
        golden_umap::cc_queries, golden_umap::cc_nq, golden_umap::cc_dim,
        temb.data(), 2, 5);
    const double tw_py = trustworthiness(
        golden_umap::cc_queries, golden_umap::cc_nq, golden_umap::cc_dim,
        golden_umap::cc_t_embedding, 2, 5);
    CHECK(tw_ours >= tw_py - 0.05);
}

// ---------------------------------------------------------------------------
TEST_CASE("umap: trustworthiness >= 0.97 on the iris-like dataset") {
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.random_state = 42;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_umap::tw_X, golden_umap::tw_n, golden_umap::tw_dim);
    const double tw = trustworthiness(golden_umap::tw_X, golden_umap::tw_n,
                                      golden_umap::tw_dim, emb.data(), 2, 5);
    // Python UMAP scores golden_umap::tw_python on the same data.
    CHECK(tw >= 0.97);
    CHECK(tw >= golden_umap::tw_python - 0.02);
}

TEST_CASE("umap: parameter validation") {
    auto fit_with = [](UMAPParams p) {
        UMAP model(p);
        model.fit(golden_umap::ca_X, golden_umap::ca_n, golden_umap::ca_dim);
    };
    UMAPParams p;

    p = UMAPParams();
    p.set_op_mix_ratio = 1.5;
    CHECK_THROWS_WITH_AS(fit_with(p),
                         "set_op_mix_ratio must be between 0.0 and 1.0",
                         std::invalid_argument);

    p = UMAPParams();
    p.repulsion_strength = -1.0;
    CHECK_THROWS_WITH_AS(fit_with(p), "repulsion_strength cannot be negative",
                         std::invalid_argument);

    p = UMAPParams();
    p.min_dist = 2.0;  // > spread
    CHECK_THROWS_WITH_AS(fit_with(p),
                         "min_dist must be less than or equal to spread",
                         std::invalid_argument);

    p = UMAPParams();
    p.min_dist = -0.5;
    CHECK_THROWS_WITH_AS(fit_with(p), "min_dist cannot be negative",
                         std::invalid_argument);

    p = UMAPParams();
    p.init = "bad_init";
    CHECK_THROWS_AS(fit_with(p), std::invalid_argument);

    p = UMAPParams();
    p.negative_sample_rate = -1;
    CHECK_THROWS_WITH_AS(fit_with(p), "negative sample rate must be positive",
                         std::invalid_argument);

    p = UMAPParams();
    p.learning_rate = -1.0;
    CHECK_THROWS_WITH_AS(fit_with(p), "learning_rate must be positive",
                         std::invalid_argument);

    p = UMAPParams();
    p.n_neighbors = 1;
    CHECK_THROWS_WITH_AS(fit_with(p), "n_neighbors must be greater than 1",
                         std::invalid_argument);

    p = UMAPParams();
    p.n_components = 0;
    CHECK_THROWS_WITH_AS(fit_with(p), "n_components must be greater than 0",
                         std::invalid_argument);

    p = UMAPParams();
    p.n_epochs = -5;
    CHECK_THROWS_AS(fit_with(p), std::invalid_argument);
}

TEST_CASE("umap: single-sample fit and pre-fit transform edge cases") {
    const float one_point[3] = {0.0f, 1.0f, 2.0f};
    UMAPParams p;
    p.random_state = 42;
    UMAP model(p);
    model.fit(one_point, 1, 3);
    REQUIRE(model.embedding().size() == 2);
    CHECK(model.embedding()[0] == 0.0f);
    CHECK(model.embedding()[1] == 0.0f);
    CHECK_THROWS_AS(model.transform(one_point, 1), std::invalid_argument);

    UMAP unfitted{UMAPParams()};
    CHECK_THROWS_AS(unfitted.transform(one_point, 1), std::runtime_error);
}
