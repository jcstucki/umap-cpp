// inverse_transform tests: the vendored-qhull Delaunay port against scipy
// 1.18.1 goldens, and UMAP::inverse_transform seeded parity against the
// reference implementation (goldens from scripts/gen_golden_inverse.py).
//
// Parity notes:
//   * Delaunay(points, incremental=True, qhull_options="QJ") -- the exact
//     construction inverse_transform performs -- reproduces scipy bitwise:
//     simplices, neighbors, equations, paraboloid scale/shift and every
//     find_simplex result match. The QJ joggle is deterministic (qhull seeds
//     its internal PRNG with 1 unless QRn is given), and the vendored
//     third_party/qhull_r sources are byte-identical to scipy's
//     subprojects/qhull_r. Only the lazily-computed barycentric `transform`
//     arrays differ from scipy (~1e-15: our reference-LAPACK dgetf2/dgetrs
//     arithmetic vs OpenBLAS's fma kernels; find_simplex, their only
//     consumer, is verified identical) -- so `transform` values are not
//     asserted bitwise here.
//   * iv1/iv2 (euclidean) and iv3 (manhattan) run the fully deterministic
//     small-data path: fit embedding, find_simplex of the queries and the
//     final inverse_transform points are all BIT-EXACT vs Python.
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_inverse.hpp"
#include "umap/delaunay.hpp"
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

void check_delaunay_case(const double* points, int n, int d, int nsimplex,
                         const std::int32_t* simplices,
                         const std::int32_t* neighbors,
                         const double* equations, double pscale, double pshift,
                         const double* queries, int nq,
                         const std::int32_t* find_simplex_golden) {
    const umap::Delaunay tri = umap::delaunay_qj(points, n, d);
    REQUIRE(tri.nsimplex == nsimplex);
    const int k = d + 1;
    for (int i = 0; i < nsimplex * k; ++i) {
        CHECK(tri.simplices[static_cast<std::size_t>(i)] == simplices[i]);
        CHECK(tri.neighbors[static_cast<std::size_t>(i)] == neighbors[i]);
    }
    // equations bitwise (they come straight out of qhull).
    for (int i = 0; i < nsimplex * (d + 2); ++i)
        CHECK(tri.equations[static_cast<std::size_t>(i)] == equations[i]);
    CHECK(tri.paraboloid_scale == pscale);
    CHECK(tri.paraboloid_shift == pshift);
    std::vector<std::int32_t> fs(static_cast<std::size_t>(nq));
    tri.find_simplex(queries, nq, fs.data());
    for (int i = 0; i < nq; ++i)
        CHECK(fs[static_cast<std::size_t>(i)] == find_simplex_golden[i]);
}

}  // namespace

TEST_SUITE("inverse_transform") {

TEST_CASE("delaunay QJ reproduces scipy bitwise (2d, n=40)") {
    using namespace golden_inverse;
    check_delaunay_case(dt1_points, dt1_n, dt1_d, dt1_nsimplex, dt1_simplices,
                        dt1_neighbors, dt1_equations, dt1_pscale, dt1_pshift,
                        dt1_queries, dt1_nq, dt1_find_simplex);
}

TEST_CASE("delaunay QJ reproduces scipy bitwise (2d, n=120)") {
    using namespace golden_inverse;
    check_delaunay_case(dt2_points, dt2_n, dt2_d, dt2_nsimplex, dt2_simplices,
                        dt2_neighbors, dt2_equations, dt2_pscale, dt2_pshift,
                        dt2_queries, dt2_nq, dt2_find_simplex);
}

TEST_CASE("delaunay QJ reproduces scipy bitwise (3d, n=35)") {
    using namespace golden_inverse;
    check_delaunay_case(dt3_points, dt3_n, dt3_d, dt3_nsimplex, dt3_simplices,
                        dt3_neighbors, dt3_equations, dt3_pscale, dt3_pshift,
                        dt3_queries, dt3_nq, dt3_find_simplex);
}

TEST_CASE("delaunay input validation matches scipy") {
    const double one_pt[] = {0.0, 0.0};
    CHECK_THROWS_WITH_AS(umap::delaunay_qj(one_pt, 0, 2), "No points given",
                         std::runtime_error);
    const double line[] = {0.0, 1.0, 2.0};
    CHECK_THROWS_WITH_AS(umap::delaunay_qj(line, 3, 1),
                         "Need at least 2-D data", std::runtime_error);
    const double with_nan[] = {0.0, 0.0, 1.0,
                               std::numeric_limits<double>::quiet_NaN()};
    CHECK_THROWS_WITH_AS(umap::delaunay_qj(with_nan, 2, 2),
                         "Points cannot contain NaN", std::runtime_error);
}

TEST_CASE("inverse_transform iv1 (euclidean, n_epochs=100) bit-exact") {
    using namespace golden_inverse;
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.n_epochs = 100;
    p.init = "random";
    p.random_state = 42;
    p.a = ab_a;
    p.b = ab_b;
    UMAP m(p);
    m.fit(iv1_X, iv1_n, iv1_dim);
    // fit parity anchor
    for (int i = 0; i < iv1_n * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] ==
              iv1_embedding[i]);
    // find_simplex over the queries, via the same public Delaunay surface
    // inverse_transform uses internally (note two -1 entries: the far-outside
    // queries; upstream then indexes simplices[-1]).
    {
        std::vector<double> pts(m.embedding().begin(), m.embedding().end());
        const umap::Delaunay tri = umap::delaunay_qj(pts.data(), iv1_n, 2);
        std::vector<double> q(static_cast<std::size_t>(iv1_nq) * 2);
        for (std::size_t i = 0; i < q.size(); ++i)
            q[i] = static_cast<double>(iv1_queries[i]);
        std::vector<std::int32_t> fs(static_cast<std::size_t>(iv1_nq));
        tri.find_simplex(q.data(), iv1_nq, fs.data());
        for (int i = 0; i < iv1_nq; ++i)
            CHECK(fs[static_cast<std::size_t>(i)] == iv1_find_simplex[i]);
    }
    const std::vector<float> inv = m.inverse_transform(iv1_queries, iv1_nq);
    REQUIRE(inv.size() == static_cast<std::size_t>(iv1_nq) * iv1_dim);
    for (int i = 0; i < iv1_nq * iv1_dim; ++i)
        CHECK(inv[static_cast<std::size_t>(i)] == iv1_inv[i]);
}

TEST_CASE("inverse_transform iv2 (euclidean, default n_epochs) bit-exact") {
    using namespace golden_inverse;
    UMAPParams p;
    p.n_neighbors = 12;
    p.min_dist = 0.05;
    p.init = "random";
    p.random_state = 17;
    p.a = ab_a;
    p.b = ab_b;
    UMAP m(p);
    m.fit(iv2_X, iv2_n, iv2_dim);
    for (int i = 0; i < iv2_n * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] ==
              iv2_embedding[i]);
    const std::vector<float> inv = m.inverse_transform(iv2_queries, iv2_nq);
    for (int i = 0; i < iv2_nq * iv2_dim; ++i)
        CHECK(inv[static_cast<std::size_t>(i)] == iv2_inv[i]);
}

TEST_CASE("inverse_transform iv3 (manhattan gradient) bit-exact") {
    using namespace golden_inverse;
    UMAPParams p;
    p.n_neighbors = 8;
    p.metric = "manhattan";
    p.n_epochs = 90;
    p.init = "random";
    p.random_state = 7;
    p.a = ab_a;
    p.b = ab_b;
    UMAP m(p);
    m.fit(iv3_X, iv3_n, iv3_dim);
    for (int i = 0; i < iv3_n * 2; ++i)
        CHECK(m.embedding()[static_cast<std::size_t>(i)] ==
              iv3_embedding[i]);
    const std::vector<float> inv = m.inverse_transform(iv3_queries, iv3_nq);
    for (int i = 0; i < iv3_nq * iv3_dim; ++i)
        CHECK(inv[static_cast<std::size_t>(i)] == iv3_inv[i]);
}

TEST_CASE("inverse_transform restriction chain matches upstream") {
    using namespace golden_inverse;
    // metric without a gradient -> "given metric."
    {
        UMAPParams p;
        p.metric = "dice";
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        m.fit(iv1_X, iv1_n, iv1_dim);
        CHECK(throws_with(
            [&] { (void)m.inverse_transform(iv1_queries, 2); },
            "Inverse transform not available for given metric."));
    }
    // precomputed metric -> same message (upstream sets
    // _inverse_distance_func = None with a warning at fit time).
    {
        UMAPParams p;
        p.metric = "precomputed";
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        std::vector<float> dmat = umap::pairwise_distances_sklearn(
            iv1_X, iv1_n, iv1_X, iv1_n, iv1_dim, "euclidean", true);
        m.fit(dmat.data(), iv1_n, iv1_n);
        CHECK(throws_with(
            [&] { (void)m.inverse_transform(iv1_queries, 2); },
            "Inverse transform not available for given metric."));
    }
    // densMAP -> "for densMAP."
    {
        UMAPParams p;
        p.densmap = true;
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        m.fit(iv1_X, iv1_n, iv1_dim);
        CHECK(throws_with(
            [&] { (void)m.inverse_transform(iv1_queries, 2); },
            "Inverse transform not available for densMAP."));
    }
    // transform_mode="graph" -> its dedicated message...
    {
        UMAPParams p;
        p.transform_mode = "graph";
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        m.fit(iv1_X, iv1_n, iv1_dim);
        CHECK(throws_with(
            [&] { (void)m.inverse_transform(iv1_queries, 2); },
            "Inverse transform not available for transform_mode = 'graph'"));
    }
    // ...but n_components >= 8 short-circuits the elif chain with a warning,
    // so a graph-mode model with n_components=8 skips the graph-mode
    // ValueError and dies later on the missing embedding (upstream:
    // AttributeError at the Delaunay construction).
    {
        UMAPParams p;
        p.transform_mode = "graph";
        p.n_components = 8;
        p.n_epochs = 11;
        p.init = "random";
        p.random_state = 42;
        UMAP m(p);
        m.fit(iv1_X, iv1_n, iv1_dim);
        std::vector<float> q8(16, 0.0f);
        CHECK(throws_with([&] { (void)m.inverse_transform(q8.data(), 2); },
                          "requires an embedding"));
    }
}

}  // TEST_SUITE
