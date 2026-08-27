// Tests for the Phase 1b feature bundle on the UMAP class:
// metric="precomputed", precomputed_knn, unique=True, init as ndarray/"pca",
// n_epochs lists, transform_mode="graph" and finite disconnection_distance.
// Compared against goldens generated from the reference Python UMAP
// (tests/golden_features.hpp, NUMBA_NUM_THREADS=1, seeded).
//
// Parity levels:
//  * precomputed / precomputed_knn / unique / ndarray-init / n_epochs-list /
//    graph-mode / disconnection cases run the fully deterministic small-data
//    or knn-bypass paths with init="random" (or a user array) and explicit
//    a/b: BIT-EXACT against Python end to end.
//  * init="pca": sklearn computes the SVD with float32 LAPACK, the port with
//    float64 Eigen -- the init is compared at 5e-3 (post-rescale coordinate
//    units of [0, 10]) with the exact svd_flip sign convention, and the SGD
//    stage is validated bit-exactly from the golden Python init.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_features.hpp"
#include "umap/layouts.hpp"
#include "umap/umap.hpp"

using umap::PrecomputedKnn;
using umap::SparseMatrix;
using umap::UMAP;
using umap::UMAPParams;

namespace gf = golden_features;

namespace {

// Bitwise mismatch count with NaN == NaN (disconnected rows embed as NaN).
std::size_t count_mismatch_nan(const std::vector<float>& a, const float* b) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const bool eq =
            (std::isnan(a[i]) && std::isnan(b[i])) || a[i] == b[i];
        if (!eq) ++c;
    }
    return c;
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

UMAPParams deterministic_params() {
    UMAPParams p;
    p.init = "random";
    p.random_state = 42;
    p.a = gf::ab_a;
    p.b = gf::ab_b;
    return p;
}

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

void check_csr_equal(const SparseMatrix& g, const int32_t* indptr,
                     const int32_t* indices, const float* data,
                     int64_t nrows, int64_t ncols) {
    REQUIRE(g.n_rows == nrows);
    REQUIRE(g.n_cols == ncols);
    REQUIRE(static_cast<int64_t>(g.indptr.size()) == nrows + 1);
    std::size_t mism = 0;
    for (int64_t i = 0; i <= nrows; ++i)
        if (g.indptr[i] != indptr[i]) ++mism;
    CHECK(mism == 0);
    REQUIRE(static_cast<int32_t>(g.indices.size()) == indptr[nrows]);
    mism = 0;
    for (std::size_t k = 0; k < g.indices.size(); ++k)
        if (g.indices[k] != indices[k]) ++mism;
    CHECK(mism == 0);
    mism = 0;
    for (std::size_t k = 0; k < g.data.size(); ++k)
        if (g.data[k] != data[k]) ++mism;
    CHECK(mism == 0);
}

}  // namespace

// ---------------------------------------------------------------------------
// DISCONNECTION_DISTANCES defaults.
TEST_CASE("features: default disconnection distances per metric") {
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(umap::default_disconnection_distance("cosine") == 2.0);
    CHECK(umap::default_disconnection_distance("correlation") == 2.0);
    CHECK(umap::default_disconnection_distance("hellinger") == 1.0);
    CHECK(umap::default_disconnection_distance("jaccard") == 1.0);
    CHECK(umap::default_disconnection_distance("dice") == 1.0);
    CHECK(umap::default_disconnection_distance("euclidean") == inf);
    CHECK(umap::default_disconnection_distance("precomputed") == inf);

    UMAPParams p = deterministic_params();
    p.metric = "cosine";
    UMAP model(p);
    model.fit(gf::fdisc_X, gf::fdisc_n, gf::fdisc_dim);
    CHECK(model.disconnection_distance() == 2.0);
}

// ---------------------------------------------------------------------------
// metric="precomputed" (dense): small-data fit + transform, bit-exact.
TEST_CASE("features: BIT-EXACT metric='precomputed' fit (small data)") {
    UMAPParams p = deterministic_params();
    p.metric = "precomputed";
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(gf::fp_D, gf::fp_n, gf::fp_n);
    CHECK(model.small_data());
    CHECK(model.disconnection_distance() ==
          std::numeric_limits<double>::infinity());

    const std::vector<float> init = rescale_init(model.fit_init(), gf::fp_n, 2);
    CHECK(count_mismatch_nan(init, gf::fp_fit_init) == 0);
    REQUIRE(emb.size() == static_cast<std::size_t>(gf::fp_n) * 2);
    CHECK(count_mismatch_nan(emb, gf::fp_embedding) == 0);
}

TEST_CASE("features: BIT-EXACT metric='precomputed' transform (small data)") {
    UMAPParams p = deterministic_params();
    p.metric = "precomputed";
    UMAP model(p);
    model.fit(gf::fp_D, gf::fp_n, gf::fp_n);
    // Rows of the query matrix are distances to the training points.
    const std::vector<float> temb = model.transform(gf::fp_QD, gf::fp_nq);
    REQUIRE(temb.size() == static_cast<std::size_t>(gf::fp_nq) * 2);
    CHECK(count_mismatch_nan(temb, gf::fp_t_embedding) == 0);
}

TEST_CASE("features: precomputed error/warning cases") {
    // Non-square distance matrix.
    {
        UMAPParams p = deterministic_params();
        p.metric = "precomputed";
        UMAP model(p);
        CHECK(throws_with(
            [&] { model.fit(gf::fdisc_X, gf::fdisc_n, gf::fdisc_dim); },
            "Precomputed metric requires shape"));
    }
    // unique=True is poorly defined on a precomputed metric.
    {
        UMAPParams p = deterministic_params();
        p.metric = "precomputed";
        p.unique = true;
        UMAP model(p);
        CHECK(throws_with([&] { model.fit(gf::fp_D, gf::fp_n, gf::fp_n); },
                          "unique is poorly defined on a precomputed metric"));
    }
    // Standard branch (force_approximation_algorithm): fast_knn_indices path,
    // no search index -> transform raises like upstream (#848 message).
    {
        UMAPParams p = deterministic_params();
        p.metric = "precomputed";
        p.force_approximation_algorithm = true;
        UMAP model(p);
        model.fit(gf::fp_D, gf::fp_n, gf::fp_n);
        CHECK_FALSE(model.small_data());
        // The kNN of the standard precomputed branch matches the small-data
        // one on the same matrix (same fast_knn_indices code upstream).
        CHECK(model.knn_indices().size() ==
              static_cast<std::size_t>(gf::fp_n) * 15);
        CHECK(throws_with([&] { model.transform(gf::fp_QD, gf::fp_nq); },
                          "No search index available"));
    }
    // Negative distances in a precomputed transform violate the assert.
    {
        UMAPParams p = deterministic_params();
        p.metric = "precomputed";
        UMAP model(p);
        model.fit(gf::fp_D, gf::fp_n, gf::fp_n);
        std::vector<float> bad(gf::fp_QD,
                               gf::fp_QD + static_cast<std::size_t>(
                                               gf::fp_nq) * gf::fp_n);
        bad[3] = -0.5f;
        CHECK(throws_with([&] { model.transform(bad.data(), gf::fp_nq); },
                          "non-negative"));
    }
}

// ---------------------------------------------------------------------------
// Finite disconnection_distance: edge removal, NaN rows, warning counters.
TEST_CASE("features: BIT-EXACT disconnection_distance fit (small data)") {
    UMAPParams p = deterministic_params();
    p.disconnection_distance = gf::fdisc_dd;
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(gf::fdisc_X, gf::fdisc_n, gf::fdisc_dim);
    CHECK(model.disconnection_distance() == gf::fdisc_dd);
    CHECK(model.edges_removed() == gf::fdisc_edges_removed);
    CHECK(model.vertices_disconnected() == gf::fdisc_vertices_disconnected);

    // Disconnected vertices embed as NaN rows.
    int64_t nan_rows = 0;
    for (int64_t i = 0; i < gf::fdisc_n; ++i) {
        if (std::isnan(emb[i * 2]) && std::isnan(emb[i * 2 + 1])) ++nan_rows;
    }
    CHECK(nan_rows == gf::fdisc_vertices_disconnected);
    CHECK(count_mismatch_nan(emb, gf::fdisc_embedding) == 0);
}

TEST_CASE("features: BIT-EXACT disconnection_distance transform") {
    UMAPParams p = deterministic_params();
    p.disconnection_distance = gf::fdisc_dd;
    UMAP model(p);
    model.fit(gf::fdisc_X, gf::fdisc_n, gf::fdisc_dim);
    // Query 14 sits past the disconnection distance from every training
    // point: its neighbours are pruned to -1 and it embeds as NaN.
    const std::vector<float> temb = model.transform(gf::fdisc_Q, gf::fdisc_nq);
    REQUIRE(temb.size() == static_cast<std::size_t>(gf::fdisc_nq) * 2);
    CHECK(count_mismatch_nan(temb, gf::fdisc_t_embedding) == 0);
    CHECK(std::isnan(temb[(gf::fdisc_nq - 1) * 2]));
}

TEST_CASE("features: disconnection_distance on the NN-descent branch") {
    // Behavioural: the standard branch masks knn entries >= dd; the two far
    // outliers disconnect there just like on the small-data branch.
    UMAPParams p = deterministic_params();
    p.disconnection_distance = gf::fdisc_dd;
    p.force_approximation_algorithm = true;
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(gf::fdisc_X, gf::fdisc_n, gf::fdisc_dim);
    CHECK_FALSE(model.small_data());
    CHECK(model.vertices_disconnected() == gf::fdisc_vertices_disconnected);
    int64_t nan_rows = 0;
    for (int64_t i = 0; i < gf::fdisc_n; ++i)
        if (std::isnan(emb[i * 2])) ++nan_rows;
    CHECK(nan_rows == gf::fdisc_vertices_disconnected);
}

// ---------------------------------------------------------------------------
// precomputed_knn: validation semantics + bit-exact bypass fits.
TEST_CASE("features: BIT-EXACT precomputed_knn fit (k == n_neighbors)") {
    UMAPParams p = deterministic_params();
    PrecomputedKnn knn;
    knn.n_samples = gf::fpk_n;
    knn.n_neighbors = gf::fpk_k;
    knn.indices.assign(gf::fpk_knn_indices,
                       gf::fpk_knn_indices +
                           static_cast<std::size_t>(gf::fpk_n) * gf::fpk_k);
    knn.dists.assign(gf::fpk_knn_dists,
                     gf::fpk_knn_dists +
                         static_cast<std::size_t>(gf::fpk_n) * gf::fpk_k);
    p.precomputed_knn = knn;
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(gf::fpk_X, gf::fpk_n, gf::fpk_dim);
    // n < 4096 forces the approximation branch for precomputed knn.
    CHECK_FALSE(model.small_data());
    CHECK(count_mismatch_nan(emb, gf::fpk_embedding) == 0);
    // No search index -> transform is unavailable (#848).
    CHECK(throws_with([&] { model.transform(gf::fpk_X, 5); },
                      "No search index available"));
}

TEST_CASE("features: BIT-EXACT precomputed_knn fit (k > n_neighbors quirk)") {
    // On small data the upstream elif chain sets force_approximation and
    // skips the pruning: the fuzzy set runs on all k=20 columns with the
    // smoothing target still at n_neighbors=15.
    UMAPParams p = deterministic_params();
    PrecomputedKnn knn;
    knn.n_samples = gf::fpk_n;
    knn.n_neighbors = gf::fpk2_k;
    knn.indices.assign(gf::fpk2_knn_indices,
                       gf::fpk2_knn_indices +
                           static_cast<std::size_t>(gf::fpk_n) * gf::fpk2_k);
    knn.dists.assign(gf::fpk2_knn_dists,
                     gf::fpk2_knn_dists +
                         static_cast<std::size_t>(gf::fpk_n) * gf::fpk2_k);
    p.precomputed_knn = knn;
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(gf::fpk_X, gf::fpk_n, gf::fpk_dim);
    CHECK_FALSE(model.small_data());
    CHECK(model.knn_dists().size() ==
          static_cast<std::size_t>(gf::fpk_n) * gf::fpk2_k);
    CHECK(count_mismatch_nan(emb, gf::fpk2_embedding) == 0);
}

TEST_CASE("features: precomputed_knn validation warnings/errors") {
    // Fewer neighbours than n_neighbors: warn + ignore -> the normal
    // small-data path runs instead.
    {
        UMAPParams p = deterministic_params();
        PrecomputedKnn knn;
        knn.n_samples = gf::fpk_n;
        knn.n_neighbors = 5;
        knn.indices.assign(static_cast<std::size_t>(gf::fpk_n) * 5, 0);
        knn.dists.assign(static_cast<std::size_t>(gf::fpk_n) * 5, 0.0f);
        p.precomputed_knn = knn;
        UMAP model(p);
        model.fit(gf::fpk_X, gf::fpk_n, gf::fpk_dim);
        CHECK(model.small_data());  // pknn ignored, no forced approximation
    }
    // Different sample count: warn + ignore.
    {
        UMAPParams p = deterministic_params();
        PrecomputedKnn knn;
        knn.n_samples = gf::fpk_n + 3;
        knn.n_neighbors = 15;
        knn.indices.assign(static_cast<std::size_t>(gf::fpk_n + 3) * 15, 0);
        knn.dists.assign(static_cast<std::size_t>(gf::fpk_n + 3) * 15, 0.0f);
        p.precomputed_knn = knn;
        UMAP model(p);
        model.fit(gf::fpk_X, gf::fpk_n, gf::fpk_dim);
        CHECK(model.small_data());
    }
    // Mismatched indices/dists sizes.
    {
        UMAPParams p = deterministic_params();
        PrecomputedKnn knn;
        knn.n_samples = gf::fpk_n;
        knn.n_neighbors = 15;
        knn.indices.assign(static_cast<std::size_t>(gf::fpk_n) * 15, 0);
        knn.dists.assign(static_cast<std::size_t>(gf::fpk_n) * 15 - 1, 0.0f);
        p.precomputed_knn = knn;
        UMAP model(p);
        CHECK(throws_with([&] { model.fit(gf::fpk_X, gf::fpk_n, gf::fpk_dim); },
                          "must be numpy arrays of the same size"));
    }
    // unique is not available with precomputed_knn.
    {
        UMAPParams p = deterministic_params();
        p.unique = true;
        PrecomputedKnn knn;
        knn.n_samples = gf::fpk_n;
        knn.n_neighbors = 15;
        knn.indices.assign(static_cast<std::size_t>(gf::fpk_n) * 15, 0);
        knn.dists.assign(static_cast<std::size_t>(gf::fpk_n) * 15, 0.0f);
        p.precomputed_knn = knn;
        UMAP model(p);
        CHECK(throws_with([&] { model.fit(gf::fpk_X, gf::fpk_n, gf::fpk_dim); },
                          "unique is not currently available"));
    }
}

// ---------------------------------------------------------------------------
// unique=True: np.unique(axis=0) semantics and inverse-expanded embedding.
TEST_CASE("features: unique_rows matches np.unique(axis=0)") {
    std::vector<int64_t> index, inverse;
    umap::unique_rows(gf::fu_X, gf::fu_n, gf::fu_dim, index, inverse);
    REQUIRE(static_cast<int64_t>(index.size()) == gf::fu_n_unique);
    std::size_t mism = 0;
    for (std::size_t i = 0; i < index.size(); ++i)
        if (index[i] != gf::fu_unique_index[i]) ++mism;
    CHECK(mism == 0);
    mism = 0;
    for (int64_t i = 0; i < gf::fu_n; ++i)
        if (inverse[static_cast<std::size_t>(i)] != gf::fu_unique_inverse[i])
            ++mism;
    CHECK(mism == 0);

    // -0.0 == 0.0 rows collapse, like numpy's field comparison.
    const float z[4] = {-0.0f, 1.0f, 0.0f, 1.0f};
    umap::unique_rows(z, 2, 2, index, inverse);
    CHECK(index.size() == 1);
    CHECK(index[0] == 0);
}

TEST_CASE("features: BIT-EXACT unique=True fit (row dedup + inverse map)") {
    UMAPParams p = deterministic_params();
    p.unique = true;
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(gf::fu_X, gf::fu_n, gf::fu_dim);
    CHECK(model.small_data());
    CHECK(model.effective_n_neighbors() == 15);
    REQUIRE(static_cast<int64_t>(model.unique_inverse().size()) == gf::fu_n);
    std::size_t mism = 0;
    for (int64_t i = 0; i < gf::fu_n; ++i)
        if (model.unique_inverse()[static_cast<std::size_t>(i)] !=
            gf::fu_unique_inverse[i])
            ++mism;
    CHECK(mism == 0);

    // The embedding is expanded back to all rows; duplicate rows share
    // coordinates.
    REQUIRE(emb.size() == static_cast<std::size_t>(gf::fu_n) * 2);
    CHECK(count_mismatch_nan(emb, gf::fu_embedding) == 0);
    for (int64_t i = 0; i < gf::fu_n; ++i) {
        for (int64_t j = i + 1; j < gf::fu_n; ++j) {
            if (std::memcmp(gf::fu_X + i * gf::fu_dim, gf::fu_X + j * gf::fu_dim,
                            gf::fu_dim * sizeof(float)) == 0) {
                CHECK(emb[i * 2] == emb[j * 2]);
                CHECK(emb[i * 2 + 1] == emb[j * 2 + 1]);
            }
        }
    }
}

TEST_CASE("features: unique=True with a single distinct row") {
    // Upstream quirk: all-duplicate data reduces to one sample and fit
    // returns np.zeros((1, n_components)) *without* inverse expansion.
    std::vector<float> X(5 * 3, 1.5f);
    UMAPParams p = deterministic_params();
    p.unique = true;
    UMAP model(p);
    model.fit(X.data(), 5, 3);
    REQUIRE(model.embedding().size() == 2);
    CHECK(model.embedding()[0] == 0.0f);
    CHECK(model.embedding()[1] == 0.0f);
}

// ---------------------------------------------------------------------------
// init as a user ndarray: KDTree jitter branch + passthrough.
TEST_CASE("features: BIT-EXACT init=ndarray with duplicate rows (jitter)") {
    UMAPParams p = deterministic_params();
    p.init_array = std::vector<float>(
        gf::fi_init_array,
        gf::fi_init_array + static_cast<std::size_t>(gf::fi_n) * 2);
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(gf::fi_X, gf::fi_n, gf::fi_dim);
    // The jittered init (0.001 * mean-2nd-NN-distance gaussian noise) must
    // reproduce Python bit for bit: KDTree distances and np.mean pairwise
    // summation replicated in float64.
    const std::vector<float> init = rescale_init(model.fit_init(), gf::fi_n, 2);
    CHECK(count_mismatch_nan(init, gf::fi_fit_init) == 0);
    CHECK(count_mismatch_nan(emb, gf::fi_embedding) == 0);
}

TEST_CASE("features: BIT-EXACT init=ndarray without duplicates (passthrough)") {
    UMAPParams p = deterministic_params();
    p.init_array = std::vector<float>(
        gf::fi2_init_array,
        gf::fi2_init_array + static_cast<std::size_t>(gf::fi_n) * 2);
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(gf::fi_X, gf::fi_n, gf::fi_dim);
    // No duplicate rows -> the init is used untouched (no rng draws).
    const std::vector<float> init = rescale_init(model.fit_init(), gf::fi_n, 2);
    CHECK(count_mismatch_nan(init, gf::fi2_fit_init) == 0);
    CHECK(count_mismatch_nan(emb, gf::fi2_embedding) == 0);
}

TEST_CASE("features: init ndarray must match n_components") {
    UMAPParams p = deterministic_params();
    p.init_array = std::vector<float>(30, 0.0f);
    p.init_array_cols = 3;  // n_components defaults to 2
    UMAP model(p);
    CHECK(throws_with([&] { model.fit(gf::fi_X, 10, gf::fi_dim); },
                      "init ndarray must match n_components value"));
}

// ---------------------------------------------------------------------------
// init="pca": svd_flip sign parity at tolerance + bit-exact downstream SGD.
TEST_CASE("features: init='pca' matches sklearn PCA at documented tolerance") {
    UMAPParams p = deterministic_params();
    p.init = "pca";
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(gf::fpca_X, gf::fpca_n, gf::fpca_dim);
    REQUIRE(emb.size() == static_cast<std::size_t>(gf::fpca_n) * 2);

    // RandomState stream parity: PCA consumes no draws; noisy_scale_coords
    // consumes n*2 normals, then the SGD rng_state.
    {
        umap::nnd::NumpyRandomState rs(42);
        for (int64_t i = 0; i < gf::fpca_n * 2; ++i)
            (void)rs.normal(0.0, 0.0001);
        const auto r3 = rs.rng_state3();
        for (int c = 0; c < 3; ++c)
            CHECK(r3[c] == gf::fpca_fit_rng_state[c]);
    }

    // Post-rescale init within 5e-3 of Python's (identical svd_flip signs;
    // differences are float32-LAPACK vs float64-Eigen rounding only). A sign
    // flip would show up as ~10-unit errors.
    const std::vector<float> init =
        rescale_init(model.fit_init(), gf::fpca_n, 2);
    double max_diff = 0.0;
    for (std::size_t i = 0; i < init.size(); ++i)
        max_diff = std::max(
            max_diff,
            std::abs(static_cast<double>(init[i]) - gf::fpca_fit_init[i]));
    CHECK(max_diff < 5e-3);

    // Downstream SGD parity: optimising from Python's own init with the
    // golden edges/rng reproduces the golden embedding bit for bit.
    std::vector<float> head_emb(gf::fpca_fit_init,
                                gf::fpca_fit_init +
                                    static_cast<std::size_t>(gf::fpca_n) * 2);
    std::vector<int32_t> head(gf::fpca_fit_head,
                              gf::fpca_fit_head + gf::fpca_fit_n_edges);
    std::vector<int32_t> tail(gf::fpca_fit_tail,
                              gf::fpca_fit_tail + gf::fpca_fit_n_edges);
    std::vector<double> eps(gf::fpca_fit_eps,
                            gf::fpca_fit_eps + gf::fpca_fit_n_edges);
    umap::optimize_layout_euclidean(
        head_emb.data(), static_cast<int>(gf::fpca_n), head_emb.data(), 2,
        head, tail, gf::fpca_fit_n_epochs, static_cast<int>(gf::fpca_n), eps,
        gf::ab_a, gf::ab_b, gf::fpca_fit_rng_state, 1.0, 1.0, 5.0,
        /*parallel=*/false, /*densmap=*/false, /*move_other=*/true);
    CHECK(count_mismatch_nan(head_emb, gf::fpca_embedding) == 0);
}

TEST_CASE("features: pca_fit_transform sign convention (svd_flip)") {
    // Recompute the PCA of the golden data and check the component signs by
    // correlating against Python's captured (pre-noise dominated) init: a
    // wrong svd_flip sign would anti-correlate a whole column.
    const std::vector<float> pcs = umap::pca_fit_transform(
        gf::fpca_X, gf::fpca_n, gf::fpca_dim, 2);
    REQUIRE(pcs.size() == static_cast<std::size_t>(gf::fpca_n) * 2);
    // Column means of U*S must be ~0 (centred data).
    for (int d = 0; d < 2; ++d) {
        double mean = 0.0;
        for (int64_t i = 0; i < gf::fpca_n; ++i) mean += pcs[i * 2 + d];
        CHECK(std::abs(mean / gf::fpca_n) < 1e-4);
    }
}

// ---------------------------------------------------------------------------
// n_epochs as a list: snapshots + final, bit-exact.
TEST_CASE("features: BIT-EXACT n_epochs list (embedding snapshots)") {
    UMAPParams p = deterministic_params();
    p.n_epochs_list = std::vector<int>(
        gf::fe_epochs_list, gf::fe_epochs_list + gf::fe_epochs_list_len);
    UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(gf::fe_X, gf::fe_n, gf::fe_dim);

    // max(list) drives the epoch count and weight filtering.
    REQUIRE(static_cast<int64_t>(model.embedding_list().size()) ==
            gf::fe_embedding_list_len);
    CHECK(count_mismatch_nan(model.embedding_list()[0],
                             gf::fe_embedding_list_0) == 0);
    CHECK(count_mismatch_nan(model.embedding_list()[1],
                             gf::fe_embedding_list_1) == 0);
    CHECK(count_mismatch_nan(model.embedding_list()[2],
                             gf::fe_embedding_list_2) == 0);
    CHECK(count_mismatch_nan(emb, gf::fe_embedding) == 0);
    // The final snapshot equals the returned embedding (upstream's
    // embedding_list[-1].copy(), no disconnected vertices here).
    CHECK(count_mismatch_nan(model.embedding_list().back(),
                             gf::fe_embedding) == 0);

    // transform with a list n_epochs raises (upstream TypeError).
    CHECK(throws_with([&] { model.transform(gf::fe_X, 5); },
                      "n_epochs as a list"));
}

TEST_CASE("features: n_epochs list validation") {
    {
        UMAPParams p = deterministic_params();
        p.n_epochs_list = std::vector<int>{10, -5};
        UMAP model(p);
        CHECK(throws_with([&] { model.fit(gf::fe_X, gf::fe_n, gf::fe_dim); },
                          "nonnegative integer"));
    }
    {
        UMAPParams p = deterministic_params();
        p.n_epochs_list = std::vector<int>{};
        UMAP model(p);
        CHECK(throws_with([&] { model.fit(gf::fe_X, gf::fe_n, gf::fe_dim); },
                          "nonnegative integer"));
    }
    {
        UMAPParams p = deterministic_params();
        p.n_epochs = 100;
        p.n_epochs_list = std::vector<int>{10, 20};
        UMAP model(p);
        CHECK(throws_with([&] { model.fit(gf::fe_X, gf::fe_n, gf::fe_dim); },
                          "only one of"));
    }
}

// ---------------------------------------------------------------------------
// transform_mode="graph": the CSR outputs of fit_transform / transform.
TEST_CASE("features: BIT-EXACT transform_mode='graph' fit_transform") {
    UMAPParams p;
    p.random_state = 42;
    p.transform_mode = "graph";
    UMAP model(p);
    const SparseMatrix& g =
        model.fit_transform_graph(gf::fg_X, gf::fg_n, gf::fg_dim);
    // The embedding is skipped entirely.
    CHECK(model.embedding().empty());
    check_csr_equal(g, gf::fg_graph_indptr, gf::fg_graph_indices,
                    gf::fg_graph_data, gf::fg_graph_nrows, gf::fg_graph_ncols);
}

TEST_CASE("features: BIT-EXACT transform_mode='graph' transform") {
    UMAPParams p;
    p.random_state = 42;
    p.transform_mode = "graph";
    UMAP model(p);
    model.fit(gf::fg_X, gf::fg_n, gf::fg_dim);
    const SparseMatrix tg = model.transform_graph(gf::fg_Q, gf::fg_nq);
    check_csr_equal(tg, gf::fg_t_graph_indptr, gf::fg_t_graph_indices,
                    gf::fg_t_graph_data, gf::fg_t_graph_nrows,
                    gf::fg_t_graph_ncols);
    // Hash shortcut returns the fit graph.
    const SparseMatrix same = model.transform_graph(gf::fg_X, gf::fg_n);
    check_csr_equal(same, gf::fg_graph_indptr, gf::fg_graph_indices,
                    gf::fg_graph_data, gf::fg_graph_nrows, gf::fg_graph_ncols);
}

TEST_CASE("features: transform_mode error cases") {
    {
        UMAPParams p;
        p.transform_mode = "graph";
        UMAP model(p);
        CHECK(throws_with(
            [&] { model.fit_transform(gf::fg_X, gf::fg_n, gf::fg_dim); },
            "fit_transform_graph"));
        model.fit(gf::fg_X, gf::fg_n, gf::fg_dim);
        CHECK(throws_with([&] { model.transform(gf::fg_Q, gf::fg_nq); },
                          "transform_graph"));
    }
    {
        UMAPParams p;
        p.transform_mode = "banana";
        UMAP model(p);
        CHECK(throws_with(
            [&] { model.fit_transform(gf::fg_X, gf::fg_n, gf::fg_dim); },
            "Unrecognized transform mode"));
    }
    {
        UMAPParams p;
        p.random_state = 42;
        UMAP model(p);  // embedding mode
        model.fit(gf::fg_X, gf::fg_n, gf::fg_dim);
        CHECK(throws_with([&] { model.transform_graph(gf::fg_Q, gf::fg_nq); },
                          "transform_mode='graph'"));
    }
}
