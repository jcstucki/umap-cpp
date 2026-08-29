// update() tests: seeded parity against the reference implementation
// (goldens from scripts/gen_golden_update.py) plus error/quirk coverage.
//
// Parity notes (all cases init="random", explicit a/b):
//   * u1 (euclidean small -> small) is BIT-EXACT vs Python end to end: fit
//     embedding, post-update graph_ (including the epoch-filter zeros written
//     through scipy's shared tocoo() buffer), sigmas/rhos and the final
//     embedding.
//   * u2 (minkowski p=3 standard branch: NNDescent.update + re-prepare) and
//     u3 (small -> large transition) are BIT-EXACT too -- the minkowski
//     pow-chain kernel is bit-reproducible under numba fastmath, unlike
//     euclidean (whose kNN *distances* vary in the last ulp between numba
//     compilations; indices still match).
//   * u4 (sparse euclidean small -> small) is BIT-EXACT.
//   * u5/u6/u7 chain a second update onto u1/u2/u3, pinning the RandomState
//     bookkeeping: each update draws a fresh check_random_state(
//     transform_seed) stream for the embedding stage, while NNDescent.update
//     continues the RandomState *instance* the index holds (fit's stream for
//     u6; the first update's transform stream for u7) plus the persisted tau
//     rng_state. All BIT-EXACT.
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_update.hpp"
#include "umap/umap.hpp"

namespace {

using umap::UMAP;
using umap::UMAPParams;

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

void check_graph(const umap::SparseMatrix& g, const std::int32_t* indptr,
                 std::size_t indptr_len, const std::int32_t* indices,
                 const float* data, std::size_t nnz) {
    REQUIRE(static_cast<std::size_t>(g.n_rows) + 1 == indptr_len);
    for (std::size_t i = 0; i < indptr_len; ++i)
        CHECK(g.indptr[i] == indptr[i]);
    REQUIRE(g.indices.size() == nnz);
    for (std::size_t i = 0; i < nnz; ++i) {
        CHECK(g.indices[i] == indices[i]);
        CHECK(g.data[i] == data[i]);
    }
}

}  // namespace

TEST_SUITE("update") {

TEST_CASE("update u1 (euclidean, small -> small) bit-exact") {
    using namespace golden_update;
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.n_epochs = 100;
    p.init = "random";
    p.random_state = 42;
    p.a = ab_a;
    p.b = ab_b;
    UMAP m(p);
    m.fit(u1_X, u1_n, u1_dim);
    for (int i = 0; i < u1_n * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] ==
              u1_fit_embedding[i]);

    m.update(u1_X_new, u1_n_new, u1_dim);
    const int n_total = u1_n + u1_n_new;
    CHECK(m.n_samples() == n_total);
    CHECK(m.small_data());

    // graph_ after the update's embedding stage (incl. the shared-buffer
    // epoch-filter zeros, canonical CSR order like the golden).
    check_graph(m.graph(), u1_graph_indptr,
                sizeof(u1_graph_indptr) / sizeof(u1_graph_indptr[0]),
                u1_graph_indices, u1_graph_data,
                sizeof(u1_graph_indices) / sizeof(u1_graph_indices[0]));
    REQUIRE(m.sigmas().size() == static_cast<std::size_t>(n_total));
    for (int i = 0; i < n_total; ++i) {
        CHECK(m.sigmas()[static_cast<std::size_t>(i)] == u1_sigmas[i]);
        CHECK(m.rhos()[static_cast<std::size_t>(i)] == u1_rhos[i]);
    }
    REQUIRE(m.embedding().size() == static_cast<std::size_t>(n_total) * 2);
    for (int i = 0; i < n_total * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] == u1_embedding[i]);

    // Upstream quirk preserved: _input_hash still refers to the *fit* data,
    // so transforming the original 90 rows returns the full updated
    // embedding (now 115 rows).
    const std::vector<float> shortcut = m.transform(u1_X, u1_n);
    CHECK(shortcut.size() == static_cast<std::size_t>(n_total) * 2);
    CHECK(shortcut == m.embedding());

    // u5: a second (chained) small update -- each update draws a fresh
    // check_random_state(transform_seed) stream.
    m.update(u5_X_new, u5_n_new, u1_dim);
    const int n_total2 = n_total + u5_n_new;
    CHECK(m.n_samples() == n_total2);
    check_graph(m.graph(), u5_graph_indptr,
                sizeof(u5_graph_indptr) / sizeof(u5_graph_indptr[0]),
                u5_graph_indices, u5_graph_data,
                sizeof(u5_graph_indices) / sizeof(u5_graph_indices[0]));
    REQUIRE(m.embedding().size() == static_cast<std::size_t>(n_total2) * 2);
    for (int i = 0; i < n_total2 * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] == u5_embedding[i]);
}

TEST_CASE("update u2 (minkowski p=3, standard branch) bit-exact") {
    using namespace golden_update;
    UMAPParams p;
    p.n_neighbors = 10;
    p.metric = "minkowski";
    p.metric_kwds.p = 3.0;
    p.n_epochs = 200;
    p.init = "random";
    p.random_state = 42;
    p.a = ab_a;
    p.b = ab_b;
    UMAP m(p);
    m.fit(u2_X, u2_n, u2_dim);
    CHECK_FALSE(m.small_data());
    // fit anchor: the last 40 embedding rows.
    for (int i = 0; i < 40 * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>((u2_n - 40) * 2 + i)] ==
              u2_fit_embedding_tail[i]);

    m.update(u2_X_new, u2_n_new, u2_dim);
    const int n_total = u2_n + u2_n_new;
    CHECK(m.n_samples() == n_total);

    // kNN rows of the appended points after NNDescent.update (indices AND
    // corrected distances -- the pow-chain kernel is bit-reproducible).
    for (int i = 0; i < u2_n_new * 10; ++i) {
        CHECK(m.knn_indices()[static_cast<std::size_t>(u2_n * 10 + i)] ==
              u2_knn_indices_new[i]);
        CHECK(m.knn_dists()[static_cast<std::size_t>(u2_n * 10 + i)] ==
              u2_knn_dists_new[i]);
    }
    REQUIRE(m.embedding().size() == static_cast<std::size_t>(n_total) * 2);
    for (int i = 0; i < n_total * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] == u2_embedding[i]);

    // u6: a second (chained) standard update -- NNDescent.update continues
    // the tau rng_state and the fit RandomState exactly where the first
    // update left them.
    m.update(u6_X_new, u6_n_new, u2_dim);
    const int n_total2 = n_total + u6_n_new;
    CHECK(m.n_samples() == n_total2);
    for (int i = 0; i < u6_n_new * 10; ++i) {
        CHECK(m.knn_indices()[static_cast<std::size_t>(n_total * 10 + i)] ==
              u6_knn_indices_new[i]);
        CHECK(m.knn_dists()[static_cast<std::size_t>(n_total * 10 + i)] ==
              u6_knn_dists_new[i]);
    }
    REQUIRE(m.embedding().size() == static_cast<std::size_t>(n_total2) * 2);
    for (int i = 0; i < 60 * 2; ++i)  // golden: the last 60 embedding rows
        CHECK(m.embedding()[static_cast<std::size_t>((n_total2 - 60) * 2 +
                                                     i)] ==
              u6_embedding_tail[i]);
}

TEST_CASE("update u3 (minkowski p=3, small -> large transition) bit-exact") {
    using namespace golden_update;
    UMAPParams p;
    p.n_neighbors = 10;
    p.metric = "minkowski";
    p.metric_kwds.p = 3.0;
    p.n_epochs = 200;
    p.init = "random";
    p.random_state = 42;
    p.a = ab_a;
    p.b = ab_b;
    UMAP m(p);
    m.fit(u3_X, u3_n, u3_dim);
    CHECK(m.small_data());
    for (int i = 0; i < 40 * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>((u3_n - 40) * 2 + i)] ==
              u3_fit_embedding_tail[i]);

    m.update(u3_X_new, u3_n_new, u3_dim);
    const int n_total = u3_n + u3_n_new;
    CHECK(m.n_samples() == n_total);
    CHECK_FALSE(m.small_data());  // crossed the 4096 threshold

    for (int i = 0; i < u3_n_new * 10; ++i) {
        CHECK(m.knn_indices()[static_cast<std::size_t>(u3_n * 10 + i)] ==
              u3_knn_indices_new[i]);
        CHECK(m.knn_dists()[static_cast<std::size_t>(u3_n * 10 + i)] ==
              u3_knn_dists_new[i]);
    }
    REQUIRE(m.embedding().size() == static_cast<std::size_t>(n_total) * 2);
    for (int i = 0; i < n_total * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] == u3_embedding[i]);

    // u7: a second update after the transition -- the search index was built
    // *during* update 1 from that update's transform_seed stream, and
    // NNDescent.update now continues that instance.
    m.update(u7_X_new, u7_n_new, u3_dim);
    const int n_total2 = n_total + u7_n_new;
    CHECK(m.n_samples() == n_total2);
    for (int i = 0; i < u7_n_new * 10; ++i) {
        CHECK(m.knn_indices()[static_cast<std::size_t>(n_total * 10 + i)] ==
              u7_knn_indices_new[i]);
        CHECK(m.knn_dists()[static_cast<std::size_t>(n_total * 10 + i)] ==
              u7_knn_dists_new[i]);
    }
    REQUIRE(m.embedding().size() == static_cast<std::size_t>(n_total2) * 2);
    for (int i = 0; i < 60 * 2; ++i)  // golden: the last 60 embedding rows
        CHECK(m.embedding()[static_cast<std::size_t>((n_total2 - 60) * 2 +
                                                     i)] ==
              u7_embedding_tail[i]);
}

TEST_CASE("update u4 (sparse euclidean, small -> small) bit-exact") {
    using namespace golden_update;
    UMAPParams p;
    p.n_neighbors = 8;
    p.n_epochs = 80;
    p.init = "random";
    p.random_state = 42;
    p.a = ab_a;
    p.b = ab_b;
    UMAP m(p);
    m.fit(u4_indptr, u4_indices, u4_data, u4_n, u4_dim);
    for (int i = 0; i < u4_n * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] ==
              u4_fit_embedding[i]);

    m.update(u4_new_indptr, u4_new_indices, u4_new_data, u4_n_new, u4_dim);
    const int n_total = u4_n + u4_n_new;
    CHECK(m.n_samples() == n_total);
    check_graph(m.graph(), u4_graph_indptr,
                sizeof(u4_graph_indptr) / sizeof(u4_graph_indptr[0]),
                u4_graph_indices, u4_graph_data,
                sizeof(u4_graph_indices) / sizeof(u4_graph_indices[0]));
    REQUIRE(m.embedding().size() == static_cast<std::size_t>(n_total) * 2);
    for (int i = 0; i < n_total * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] == u4_embedding[i]);
}

TEST_CASE("update u8 (DEFAULT n_epochs runs ZERO SGD epochs) bit-exact") {
    // Upstream update() resolves self.n_epochs=None to n_epochs=0, which
    // survives simplicial_set_embedding's None-only remap: every
    // epochs_per_sample entry is -1 and the SGD loop is range(0). The
    // update embedding is exactly the [0,10]-rescaled init (old embedding +
    // init_update mean rows), NOT a re-optimized layout. Audit 2026-08-29
    // (c01): the port used to map unset -> the 500-epoch default here.
    using namespace golden_update;
    UMAPParams p;
    p.n_neighbors = 11;
    p.min_dist = 0.01;
    // n_epochs deliberately UNSET (upstream None).
    p.init = "random";
    p.random_state = 42;
    p.a = ab_a;
    p.b = ab_b;
    UMAP m(p);
    m.fit(u8_X, u8_n, u8_dim);
    for (int i = 0; i < u8_n * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] ==
              u8_fit_embedding[i]);

    static_assert(u8_update_n_epochs == 0,
                  "upstream runs zero epochs for default-n_epochs update()");
    m.update(u8_X_new, u8_n_new, u8_dim);
    const int n_total = u8_n + u8_n_new;
    CHECK(m.n_samples() == n_total);
    REQUIRE(m.embedding().size() == static_cast<std::size_t>(n_total) * 2);
    for (int i = 0; i < n_total * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] ==
              u8_upd1_embedding[i]);

    // Second chained default update: another rescale + init_update on top.
    m.update(u8_X_new2, u8_n_new2, u8_dim);
    const int n_total2 = n_total + u8_n_new2;
    CHECK(m.n_samples() == n_total2);
    check_graph(m.graph(), u8_graph_indptr,
                sizeof(u8_graph_indptr) / sizeof(u8_graph_indptr[0]),
                u8_graph_indices, u8_graph_data,
                sizeof(u8_graph_indices) / sizeof(u8_graph_indices[0]));
    REQUIRE(m.embedding().size() == static_cast<std::size_t>(n_total2) * 2);
    for (int i = 0; i < n_total2 * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] == u8_embedding[i]);
}

TEST_CASE("update error cases match upstream") {
    using namespace golden_update;
    // precomputed metric -> ValueError.
    {
        UMAPParams p;
        p.metric = "precomputed";
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        std::vector<float> dmat = umap::pairwise_distances_sklearn(
            u1_X, u1_n, u1_X, u1_n, u1_dim, "euclidean", true);
        m.fit(dmat.data(), u1_n, u1_n);
        CHECK(throws_with(
            [&] { m.update(dmat.data(), u1_n, u1_n); },
            "Update does not currently support precomputed metrics"));
    }
    // supervised model -> ValueError.
    {
        UMAPParams p;
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        std::vector<float> y(static_cast<std::size_t>(u1_n));
        for (int i = 0; i < u1_n; ++i)
            y[static_cast<std::size_t>(i)] = static_cast<float>(i % 3);
        m.fit(u1_X, u1_n, u1_dim, y.data());
        CHECK(throws_with(
            [&] { m.update(u1_X_new, u1_n_new, u1_dim); },
            "Updating supervised models is not currently supported"));
    }
    // densmap -> unsupported (upstream reads a stale graph_dists: UB).
    {
        UMAPParams p;
        p.densmap = true;
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        m.fit(u1_X, u1_n, u1_dim);
        CHECK_THROWS_AS(m.update(u1_X_new, u1_n_new, u1_dim),
                        std::runtime_error);
    }
    // sparse standard branch -> pynndescent's NotImplementedError text.
    {
        UMAPParams p;
        p.n_neighbors = 8;
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        p.force_approximation_algorithm = true;
        UMAP m(p);
        m.fit(u4_indptr, u4_indices, u4_data, u4_n, u4_dim);
        CHECK(throws_with(
            [&] {
                m.update(u4_new_indptr, u4_new_indices, u4_new_data, u4_n_new,
                         u4_dim);
            },
            "Sparse update not complete yet"));
    }
    // feature-width mismatch.
    {
        UMAPParams p;
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        m.fit(u1_X, u1_n, u1_dim);
        CHECK_THROWS_AS(m.update(u1_X_new, u1_n_new, u1_dim - 1),
                        std::invalid_argument);
    }
    // A tight far-away batch of new points whose kNN contains no original
    // rows: upstream's numba init_update raises ZeroDivisionError.
    {
        UMAPParams p;
        p.n_neighbors = 10;
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        m.fit(u1_X, u1_n, u1_dim);
        std::vector<float> far(static_cast<std::size_t>(15) * u1_dim);
        for (std::size_t i = 0; i < far.size(); ++i)
            far[i] = 500.0f + 0.01f * static_cast<float>(i);
        CHECK(throws_with([&] { m.update(far.data(), 15, u1_dim); },
                          "division by zero"));
    }
    // dense/sparse overload mixups.
    {
        UMAPParams p;
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        m.fit(u1_X, u1_n, u1_dim);
        CHECK_THROWS_AS(m.update(u4_new_indptr, u4_new_indices, u4_new_data,
                                 u4_n_new, u4_dim),
                        std::invalid_argument);
    }
}

}  // TEST_SUITE
