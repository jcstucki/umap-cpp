// Port of the two reference/umap/umap/tests/test_spectral.py cases (both
// target tswspectral_layout) plus C++-side coverage of the LOBPCG solver,
// the multi-component route and fixed-random_state determinism.
//
// Upstream-fidelity notes:
//   * test_tsw_spectral_init is skipped upstream for scipy >= 1.15 because
//     eigsh's regular-mode which="SM" at tol=1e-4 misconverges there (venv
//     scipy 1.18: spectral_layout lands ~0.63 from the true eigenvectors
//     while tsw at tol=1e-8 matches dense eigh to ~1e-15). The C++ solvers
//     converge tightly, i.e. behave like the scipy in [1.10, 1.15) window
//     where the test is active, so the upstream rmsd < 1e-6 threshold is
//     asserted verbatim -- with per-column sign alignment, since eigenvector
//     signs are solver-arbitrary (upstream gets sign agreement for free from
//     ARPACK determinism; goldens anchor both layouts to dense eigh truth).
//   * The n=20 fixture graph is np.random.default_rng(42) (PCG64) output and
//     is therefore baked in tests/golden_tsw.hpp (scripts/gen_golden_tsw.py).
//   * test_ensure_fallback_to_random_on_spectral_failure draws
//     u = np.random.random((1000, 100)) UNSEEDED upstream; per the
//     tests-port convention it is regenerated here from the documented bake
//     seed np.random.seed(777) via the bit-exact NumpyRandomState stream
//     (spot values checked against the golden header). The
//     pytest.warns(UserWarning, match="Spectral initialisation failed!")
//     assertion maps to tswspectral_layout's used_random_fallback out-param.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "doctest.h"
#include "golden_spectral.hpp"
#include "golden_tsw.hpp"
#include "trustworthiness.hpp"
#include "umap/layouts.hpp"
#include "umap/nn_descent.hpp"  // nnd::NumpyRandomState (u regeneration)
#include "umap/rng.hpp"
#include "umap/sparse_matrix.hpp"
#include "umap/spectral.hpp"
#include "umap/umap.hpp"

namespace {

// CSR from a dense row-major matrix (all entries kept; the fixtures are
// strictly positive dense affinity matrices).
umap::SparseMatrix dense_to_csr(int n, const double* vals) {
    std::vector<int32_t> rows, cols;
    std::vector<float> data;
    rows.reserve(static_cast<std::size_t>(n) * n);
    cols.reserve(static_cast<std::size_t>(n) * n);
    data.reserve(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            rows.push_back(i);
            cols.push_back(j);
            data.push_back(static_cast<float>(vals[i * n + j]));
        }
    return umap::SparseMatrix::from_coo(n, n, rows, cols, data);
}

// Symmetric CSR from upper-triangle COO triplets (as in test_spectral.cpp).
umap::SparseMatrix sym_graph(int n, const int32_t* rows, const int32_t* cols,
                             const float* vals, std::size_t nnz) {
    std::vector<int32_t> r(rows, rows + nnz);
    std::vector<int32_t> c(cols, cols + nnz);
    std::vector<float> v(vals, vals + nnz);
    r.insert(r.end(), cols, cols + nnz);
    c.insert(c.end(), rows, rows + nnz);
    v.insert(v.end(), vals, vals + nnz);
    return umap::SparseMatrix::from_coo(n, n, r, c, v);
}

// Align the per-column sign of `a` to `b` (dot-product sign), then return
// np.mean(np.sum((a - b) ** 2, axis=1)) -- the upstream rmsd -- over the
// (n x dim) row-major layouts.
double signed_rmsd(int n, int dim, const std::vector<float>& a,
                   const std::vector<float>& b) {
    std::vector<double> sign(static_cast<std::size_t>(dim), 1.0);
    for (int j = 0; j < dim; ++j) {
        double dot = 0.0;
        for (int i = 0; i < n; ++i)
            dot += static_cast<double>(a[i * dim + j]) *
                   static_cast<double>(b[i * dim + j]);
        if (dot < 0.0) sign[static_cast<std::size_t>(j)] = -1.0;
    }
    double acc = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < dim; ++j) {
            const double diff = sign[static_cast<std::size_t>(j)] *
                                    static_cast<double>(a[i * dim + j]) -
                                static_cast<double>(b[i * dim + j]);
            acc += diff * diff;
        }
    return acc / n;
}

// max_i |a[i] - b[i]| minimized over the sign of a (per-column freedom).
template <typename GetA, typename GetB>
double up_to_sign_err(int n, GetA a, GetB b) {
    double err_pos = 0.0, err_neg = 0.0;
    for (int i = 0; i < n; ++i) {
        err_pos = std::max(err_pos, std::abs(a(i) - b(i)));
        err_neg = std::max(err_neg, std::abs(a(i) + b(i)));
    }
    return std::min(err_pos, err_neg);
}

bool all_finite(const std::vector<float>& v) {
    return std::all_of(v.begin(), v.end(),
                       [](float x) { return std::isfinite(x); });
}

}  // namespace

TEST_SUITE("test_spectral.py") {

TEST_CASE("test_tsw_spectral_init") {
    using namespace golden_tsw;
    const umap::SparseMatrix g = dense_to_csr(t1_n, t1_graph);

    // spec = spectral_layout(None, graph, 2, random_state=seed**2)
    int64_t rng_spec[3] = {1764, 1765, 1766};
    const std::vector<float> spec = umap::spectral_layout(g, 2, rng_spec);
    // tsw_spec = tswspectral_layout(None, graph, 2, random_state=seed**2,
    //                               tol=1e-8)  (method=None -> "eigsh")
    int64_t rng_tsw[3] = {1764, 1765, 1766};
    bool fallback = true;
    const std::vector<float> tsw = umap::tswspectral_layout(
        g, 2, rng_tsw, std::string(), 1e-8, 0, &fallback);
    CHECK_FALSE(fallback);
    REQUIRE(spec.size() == static_cast<std::size_t>(t1_n) * 2);
    REQUIRE(tsw.size() == spec.size());
    CHECK(all_finite(spec));
    CHECK(all_finite(tsw));

    // "Make sure the two methods produce similar embeddings."
    // rmsd = np.mean(np.sum((spec - tsw_spec) ** 2, axis=1)) < 1e-6
    // (upstream threshold verbatim; per-column sign aligned, see file note)
    CHECK(signed_rmsd(t1_n, 2, tsw, spec) < 1e-6);

    // Anchor both layouts to the dense-eigh truth (== the upstream
    // tswspectral output at tol=1e-8, t1_tsw_upstream, to ~1.4e-15).
    // spec runs at the upstream default tol=1e-4 (perf pass 2026-08-30: the
    // solver now mirrors eigsh's matvec-only solve at upstream's tolerance
    // rather than the original shift-invert at tol=1e-10, so its anchor is
    // the tolerance contract, not machine precision; measured err ~8e-5 on
    // this fixture). tsw explicitly passes tol=1e-8 and stays tight.
    for (int j = 0; j < 2; ++j) {
        const double err_spec = up_to_sign_err(
            t1_n,
            [&](int i) { return static_cast<double>(spec[i * 2 + j]); },
            [&](int i) { return t1_embed_eigh[i * 2 + j]; });
        const double err_tsw = up_to_sign_err(
            t1_n,
            [&](int i) { return static_cast<double>(tsw[i * 2 + j]); },
            [&](int i) { return t1_tsw_upstream[i * 2 + j]; });
        CHECK(err_spec < 2e-4);
        CHECK(err_tsw < 1e-5);
    }

    // method="lobpcg" exercises the warm-started LOBPCG solver on the same
    // fixture (scipy's own lobpcg-vs-eigsh agreement here is ~4e-9); the
    // upstream rmsd threshold must hold across the two C++ solvers too.
    int64_t rng_lob[3] = {1764, 1765, 1766};
    bool lob_fallback = true;
    const std::vector<float> tsw_lob = umap::tswspectral_layout(
        g, 2, rng_lob, "lobpcg", 1e-8, 0, &lob_fallback);
    CHECK_FALSE(lob_fallback);
    CHECK(signed_rmsd(t1_n, 2, tsw_lob, spec) < 1e-6);
    for (int j = 0; j < 2; ++j) {
        const double err = up_to_sign_err(
            t1_n,
            [&](int i) { return static_cast<double>(tsw_lob[i * 2 + j]); },
            [&](int i) { return t1_embed_eigh[i * 2 + j]; });
        CHECK(err < 1e-5);
    }
}

TEST_CASE("test_ensure_fallback_to_random_on_spectral_failure") {
    using namespace golden_tsw;
    // dim = 1000; k = 10; assert k >= 10; assert dim // 10 > k
    const int dim = t2_dim;
    const int k = 10;
    REQUIRE(k >= 10);
    REQUIRE(dim / 10 > k);

    // u = np.random.random((dim, dim // 10)) -- unseeded upstream; baked
    // seed 777 (numpy legacy RandomState / rk_double, row-major fill).
    const int ucols = t2_ucols;
    umap::nnd::NumpyRandomState rs(777);
    Eigen::MatrixXd u(dim, ucols);
    std::vector<float> u_f32(static_cast<std::size_t>(dim) * ucols);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < ucols; ++j) {
            u(i, j) = rs.random_sample();
            u_f32[static_cast<std::size_t>(i) * ucols + j] =
                static_cast<float>(u(i, j));
        }
    // Spot-check the bit-exact replication of the baked numpy draws.
    for (int t = 0; t < 5; ++t) CHECK(u(0, t) == t2_u_head[t]);
    for (int t = 0; t < 3; ++t)
        CHECK(u(dim - 1, ucols - 3 + t) == t2_u_tail[t]);
    CHECK(std::abs(u.mean() - t2_u_mean) < 1e-12);

    // graph = y + y.T + u @ u.T  with y = np.eye(dim, k=1)
    Eigen::MatrixXd graph = u * u.transpose();
    for (int i = 0; i + 1 < dim; ++i) {
        graph(i, i + 1) += 1.0;
        graph(i + 1, i) += 1.0;
    }
    std::vector<int32_t> rows, cols;
    std::vector<float> vals;
    rows.reserve(static_cast<std::size_t>(dim) * dim);
    cols.reserve(static_cast<std::size_t>(dim) * dim);
    vals.reserve(static_cast<std::size_t>(dim) * dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(static_cast<float>(graph(i, j)));
        }
    const umap::SparseMatrix g =
        umap::SparseMatrix::from_coo(dim, dim, rows, cols, vals);

    // with pytest.warns(UserWarning, match="Spectral initialisation failed!"):
    //     tswspectral_layout(u, graph, k, random_state=42, maxiter=2,
    //                        method="lobpcg")
    int64_t rng[3] = {42, 43, 44};
    bool fallback = false;
    const std::vector<float> out = umap::tswspectral_layout(
        u_f32.data(), ucols, g, k, rng, "lobpcg", 0.0, 2, &fallback);
    CHECK(fallback);  // the escalated convergence warning -> random fallback
    REQUIRE(out.size() == static_cast<std::size_t>(dim) * k);
    CHECK(all_finite(out));
    // gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))
    for (const float v : out) {
        CHECK(v >= -10.0f);
        CHECK(v <= 10.0f);
    }
    // and it is a non-degenerate random draw
    CHECK(out[0] != out[1]);

    // Fixed random_state determinism across the whole failing path.
    int64_t rng2[3] = {42, 43, 44};
    bool fallback2 = false;
    const std::vector<float> out2 = umap::tswspectral_layout(
        u_f32.data(), ucols, g, k, rng2, "lobpcg", 0.0, 2, &fallback2);
    CHECK(fallback2);
    CHECK(out == out2);
}

TEST_CASE("tswspectral multi-component graphs route through "
          "multi_component_layout (bitwise = spectral_layout)") {
    using namespace golden_spectral;
    const std::size_t nnz = sizeof(b_vals) / sizeof(b_vals[0]);
    const umap::SparseMatrix g = sym_graph(b_n, b_rows, b_cols, b_vals, nnz);

    std::vector<int32_t> labels;
    REQUIRE(umap::connected_components(g, labels) == 2);

    // The component check precedes anything tsvd-related upstream, so with
    // equal rng states the two entry points consume identical draws and the
    // layouts are bit-identical.
    int64_t rng_spec[3] = {42, 43, 44};
    const std::vector<float> spec = umap::spectral_layout(g, 2, rng_spec);
    int64_t rng_tsw[3] = {42, 43, 44};
    bool fallback = true;
    const std::vector<float> tsw =
        umap::tswspectral_layout(g, 2, rng_tsw, std::string(), 0.0, 0,
                                 &fallback);
    CHECK_FALSE(fallback);
    REQUIRE(tsw.size() == spec.size());
    CHECK(tsw == spec);
    CHECK(rng_tsw[0] == rng_spec[0]);  // identical rng consumption
    CHECK(rng_tsw[1] == rng_spec[1]);
    CHECK(rng_tsw[2] == rng_spec[2]);
}

TEST_CASE("tswspectral determinism for a fixed random_state") {
    using namespace golden_tsw;
    const umap::SparseMatrix g = dense_to_csr(t1_n, t1_graph);

    for (const char* method : {"", "eigsh", "lobpcg"}) {
        CAPTURE(method);
        int64_t rng1[3] = {7, 8, 9};
        int64_t rng2[3] = {7, 8, 9};
        const std::vector<float> a =
            umap::tswspectral_layout(g, 2, rng1, method, 1e-8, 0, nullptr);
        const std::vector<float> b =
            umap::tswspectral_layout(g, 2, rng2, method, 1e-8, 0, nullptr);
        CHECK(a == b);  // bitwise
    }
}

TEST_CASE("tswspectral method validation and tiny-graph guard") {
    using namespace golden_tsw;
    const umap::SparseMatrix g = dense_to_csr(t1_n, t1_graph);
    int64_t rng[3] = {1, 2, 3};
    // "Method should either be None, 'eigsh' or 'lobpcg'"
    CHECK_THROWS_AS(umap::tswspectral_layout(g, 2, rng, "arpack"),
                    std::invalid_argument);

    // n <= dim + 1 raises like spectral_layout (upstream uncaught scipy error)
    std::vector<int32_t> r{0, 0, 1}, c{1, 2, 2};
    std::vector<float> v{0.5f, 0.6f, 0.7f};
    const umap::SparseMatrix tiny = sym_graph(3, r.data(), c.data(), v.data(), 3);
    CHECK_THROWS_AS(umap::tswspectral_layout(tiny, 2, rng),
                    std::invalid_argument);
}

TEST_CASE("normalized_laplacian_lobpcg eigenpair parity with the Spectra "
          "solver (graph A) and the dense small-problem branch (graph C)") {
    using namespace golden_spectral;
    // Graph A: n=16, m=3 -> 16 >= 5*3, the iterative LOBPCG path.
    {
        const std::size_t nnz = sizeof(a_vals) / sizeof(a_vals[0]);
        const umap::SparseMatrix g = sym_graph(a_n, a_rows, a_cols, a_vals, nnz);
        const int m = 3;
        int64_t rng[3] = {11, 22, 33};
        std::vector<double> X0(static_cast<std::size_t>(a_n) * m);
        for (auto& x : X0)
            x = 2.0 * static_cast<double>(umap::tau_rand(rng)) - 1.0;
        std::vector<double> evals, evecs;
        const bool conv = umap::normalized_laplacian_lobpcg(
            g, X0.data(), m, 1e-8, 5 * a_n, evals, evecs);
        CHECK(conv);
        REQUIRE(evals.size() == static_cast<std::size_t>(m));
        for (int j = 0; j < m; ++j)
            CHECK(std::abs(evals[j] - a_evals[j]) < 1e-6);
        for (int j = 1; j < m; ++j) {
            const double err = up_to_sign_err(
                a_n,
                [&](int i) {
                    return evecs[static_cast<std::size_t>(i) * m + j];
                },
                [&](int i) { return a_evecs[i * 2 + (j - 1)]; });
            CHECK(err < 1e-4);
        }
    }
    // Graph C: n=4 < 5*3 -> scipy's dense-eigensolver branch (always
    // "converged"; that warning is not escalated upstream).
    {
        const std::size_t nnz = sizeof(c_vals) / sizeof(c_vals[0]);
        const umap::SparseMatrix g = sym_graph(c_n, c_rows, c_cols, c_vals, nnz);
        const int m = 3;
        std::vector<double> X0(static_cast<std::size_t>(c_n) * m, 0.5);
        std::vector<double> evals, evecs;
        const bool conv = umap::normalized_laplacian_lobpcg(
            g, X0.data(), m, 1e-4, 2, evals, evecs);
        CHECK(conv);
        for (int j = 0; j < m; ++j)
            CHECK(std::abs(evals[j] - c_evals[j]) < 1e-8);
        for (int j = 1; j < m; ++j) {
            const double err = up_to_sign_err(
                c_n,
                [&](int i) {
                    return evecs[static_cast<std::size_t>(i) * m + j];
                },
                [&](int i) { return c_evecs[i * 2 + (j - 1)]; });
            CHECK(err < 1e-6);
        }
    }
}

TEST_CASE("initialize_embedding dispatch: tswspectral") {
    using namespace golden_tsw;
    const umap::SparseMatrix g = dense_to_csr(t1_n, t1_graph);

    // init='tswspectral': tswspectral_layout + noisy_scale_coords(10, 1e-4),
    // exactly the shape of the init='spectral' branch (umap_.py:1172).
    int64_t rng_state[3] = {101, 202, 303};
    const std::vector<float> emb = umap::initialize_embedding(
        nullptr, 0, g, 2, umap::EmbeddingInit::TSWSpectral, rng_state);
    REQUIRE(emb.size() == static_cast<std::size_t>(t1_n) * 2);
    CHECK(all_finite(emb));
    float max_abs = 0.0f;
    for (const float v : emb) max_abs = std::max(max_abs, std::abs(v));
    CHECK(std::abs(max_abs - 10.0f) < 0.01f);

    // matches the raw tsw layout after rescaling, up to sign + 1e-4 noise
    int64_t rng_raw[3] = {101, 202, 303};
    const std::vector<float> raw = umap::tswspectral_layout(g, 2, rng_raw);
    float raw_max = 0.0f;
    for (const float v : raw) raw_max = std::max(raw_max, std::abs(v));
    for (int j = 0; j < 2; ++j) {
        const double err = up_to_sign_err(
            t1_n,
            [&](int i) { return static_cast<double>(emb[i * 2 + j]); },
            [&](int i) {
                return static_cast<double>(raw[i * 2 + j]) * 10.0 / raw_max;
            });
        CHECK(err < 0.01);
    }
}

// --- UMAP-class init="tswspectral" wiring (goldens: gen_golden_tsw.py) ------
// Seeded parity vs a real Python UMAP(init="tswspectral", random_state=42)
// fit on the small-data branch. The graph pipeline and the SGD rng_state are
// bit-exact; the init is compared up to the per-column eigenvector sign
// (Spectra vs ARPACK) and the embedding at quality level, exactly like the
// init="spectral" cases in test_umap.cpp.

TEST_CASE("umap class: fit RandomState stream parity (tswspectral init)") {
    using namespace golden_tsw;
    // Small-data branch consumes nothing before the init; tswspectral init
    // draws the discarded TruncatedSVD warm start normal(size=(n, dim+1+10))
    // -- n_oversamples=10, i.e. n*(dim+11) gaussians instead of spectral's
    // n*(dim+1) -- then noisy_scale_coords' normal(size=(n, dim)), then the
    // SGD rng_state. The golden was captured from the reference fit itself.
    umap::nnd::NumpyRandomState rs(42);
    for (int64_t i = 0; i < tswc_n * (2 + 11); ++i) (void)rs.gauss();
    for (int64_t i = 0; i < tswc_n * 2; ++i) (void)rs.normal(0.0, 0.0001);
    const auto r3 = rs.rng_state3();
    for (int c = 0; c < 3; ++c) CHECK(r3[c] == tswc_fit_rng_state[c]);
}

TEST_CASE("umap class: golden fit parity, init=\"tswspectral\" "
          "(small-data branch)") {
    using namespace golden_tsw;
    umap::UMAPParams p;
    p.init = "tswspectral";
    p.random_state = 42;
    umap::UMAP model(p);
    const std::vector<float> emb =
        model.fit_transform(tswc_X, tswc_n, tswc_dim);
    CHECK(model.small_data());

    // Graph pipeline bit-exact: n_epochs default resolution, filtered
    // head/tail edge lists, epochs_per_sample (a/b-independent).
    umap::SparseMatrix g = model.graph();
    const int n_epochs = umap::resolve_epochs_and_filter_graph(g, -1);
    CHECK(n_epochs == tswc_fit_n_epochs);
    std::vector<int32_t> head, tail;
    for (int32_t row = 0; row < g.n_rows; ++row) {
        for (int32_t k = g.indptr[row]; k < g.indptr[row + 1]; ++k) {
            head.push_back(row);
            tail.push_back(g.indices[k]);
        }
    }
    REQUIRE(static_cast<int64_t>(head.size()) == tswc_fit_n_edges);
    CHECK(std::memcmp(head.data(), tswc_fit_head,
                      head.size() * sizeof(int32_t)) == 0);
    CHECK(std::memcmp(tail.data(), tswc_fit_tail,
                      tail.size() * sizeof(int32_t)) == 0);
    const std::vector<double> eps = umap::make_epochs_per_sample(
        g.data.data(), g.data.size(), n_epochs);
    double eps_diff = 0.0;
    for (std::size_t i = 0; i < eps.size(); ++i)
        eps_diff = std::max(eps_diff, std::abs(eps[i] - tswc_fit_eps[i]));
    CHECK(eps_diff == 0.0);

    // Post-rescale init up to the per-column sign (in the rescaled [0,10]
    // space a pre-rescale sign flip is x -> 10 - x). The gen script verified
    // the Python side converged to the dense-eigh truth (~2e-6), so 1e-2
    // bounds the Spectra-vs-ARPACK gap + the 1e-4 noise.
    std::vector<float> init = model.fit_init();
    for (int d = 0; d < 2; ++d) {
        float mn = init[d], mx = init[d];
        for (int64_t i = 1; i < tswc_n; ++i) {
            mn = std::min(mn, init[i * 2 + d]);
            mx = std::max(mx, init[i * 2 + d]);
        }
        for (int64_t i = 0; i < tswc_n; ++i) {
            float& v = init[i * 2 + d];
            v = (10.0f * (v - mn)) / (mx - mn);
        }
    }
    double init_err = 0.0;
    for (int d = 0; d < 2; ++d) {
        double direct = 0.0, flipped = 0.0;
        for (int64_t i = 0; i < tswc_n; ++i) {
            const double o = init[i * 2 + d];
            const double gv = tswc_fit_init[i * 2 + d];
            direct = std::max(direct, std::abs(o - gv));
            flipped = std::max(flipped, std::abs((10.0 - o) - gv));
        }
        init_err = std::max(init_err, std::min(direct, flipped));
    }
    CHECK(init_err < 1e-2);

    // Embedding quality matches Python's on the same data.
    CHECK(all_finite(emb));
    const double tw_ours = testutil::trustworthiness(
        tswc_X, tswc_n, tswc_dim, emb.data(), 2, 5);
    const double tw_py = testutil::trustworthiness(
        tswc_X, tswc_n, tswc_dim, tswc_embedding, 2, 5);
    CHECK(tw_ours >= tw_py - 0.02);
    CHECK(tw_ours >= tswc_trust_py - 0.02);
}
// ----------------------------------------------------------------------------

}  // TEST_SUITE
