// Tests for umap/spectral.hpp against goldens from the reference Python UMAP
// (scripts/gen_golden_spectral.py) plus the applicable upstream assertions
// from umap/tests/test_spectral.py and test_umap_ops.py.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "doctest.h"
#include "golden_spectral.hpp"
#include "umap/sparse_matrix.hpp"
#include "umap/spectral.hpp"

namespace {

// Symmetric CSR from upper-triangle COO triplets (mirrors sym_coo in the
// golden generator).
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

// max_i |a[i] - b[i]| minimized over the sign of a (eigenvector sign freedom).
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

TEST_CASE("connected_components matches scipy weak-connectivity semantics") {
    // scipy: connected_components on edges {3-4, 0-1, 2-5} -> (3, [0,0,1,2,2,1])
    {
        std::vector<int32_t> r{3, 0, 2}, c{4, 1, 5};
        std::vector<float> v{1.0f, 1.0f, 1.0f};
        // directed edges only (weak connectivity must still join them)
        umap::SparseMatrix g = umap::SparseMatrix::from_coo(6, 6, r, c, v);
        std::vector<int32_t> labels;
        CHECK(umap::connected_components(g, labels) == 3);
        CHECK(labels == std::vector<int32_t>{0, 0, 1, 2, 2, 1});
    }
    // scipy: single edge 4-5 in a 6-vertex graph -> (5, [0,1,2,3,4,4])
    {
        std::vector<int32_t> r{5, 4}, c{4, 5};
        std::vector<float> v{1.0f, 1.0f};
        umap::SparseMatrix g = umap::SparseMatrix::from_coo(6, 6, r, c, v);
        std::vector<int32_t> labels;
        CHECK(umap::connected_components(g, labels) == 5);
        CHECK(labels == std::vector<int32_t>{0, 1, 2, 3, 4, 4});
    }
}

TEST_CASE("normalized Laplacian eigenpairs match scipy (graph A, connected)") {
    using namespace golden_spectral;
    const std::size_t nnz = sizeof(a_vals) / sizeof(a_vals[0]);
    umap::SparseMatrix g = sym_graph(a_n, a_rows, a_cols, a_vals, nnz);

    const int k = 3;  // dim + 1
    std::vector<double> evals, evecs;
    REQUIRE(umap::normalized_laplacian_eigs(g, k, evals, evecs));
    REQUIRE(evals.size() == static_cast<std::size_t>(k));
    REQUIRE(evecs.size() == static_cast<std::size_t>(a_n) * k);

    // eigenvalues ascending, within 1e-5 of dense-eigh goldens; the trivial
    // first eigenvalue of a connected graph is ~0
    CHECK(std::abs(evals[0]) < 1e-8);
    for (int j = 0; j < k; ++j)
        CHECK(std::abs(evals[j] - a_evals[j]) < 1e-5);
    CHECK(evals[0] <= evals[1]);
    CHECK(evals[1] <= evals[2]);

    // eigenvectors (unit norm) match up to sign within 1e-4
    for (int j = 1; j < k; ++j) {
        double norm = 0.0;
        for (int i = 0; i < a_n; ++i) {
            const double e = evecs[static_cast<std::size_t>(i) * k + j];
            norm += e * e;
        }
        CHECK(std::abs(std::sqrt(norm) - 1.0) < 1e-8);
        const double err = up_to_sign_err(
            a_n,
            [&](int i) { return evecs[static_cast<std::size_t>(i) * k + j]; },
            [&](int i) { return a_evecs[i * 2 + (j - 1)]; });
        CHECK(err < 1e-4);
    }

    // spectral_layout drops the trivial eigenvector and keeps columns 1..dim
    int64_t rng_state[3] = {123, 456, 789};
    std::vector<float> layout = umap::spectral_layout(g, 2, rng_state);
    REQUIRE(layout.size() == static_cast<std::size_t>(a_n) * 2);
    CHECK(all_finite(layout));
    for (int j = 0; j < 2; ++j) {
        const double err = up_to_sign_err(
            a_n,
            [&](int i) { return static_cast<double>(layout[i * 2 + j]); },
            [&](int i) { return a_evecs[i * 2 + j]; });
        CHECK(err < 1e-4);
    }
}

TEST_CASE("tiny connected graph n = dim + 2 works; n <= dim + 1 throws") {
    using namespace golden_spectral;
    const std::size_t nnz = sizeof(c_vals) / sizeof(c_vals[0]);
    umap::SparseMatrix g = sym_graph(c_n, c_rows, c_cols, c_vals, nnz);

    std::vector<double> evals, evecs;
    REQUIRE(umap::normalized_laplacian_eigs(g, 3, evals, evecs));
    for (int j = 0; j < 3; ++j)
        CHECK(std::abs(evals[j] - c_evals[j]) < 1e-5);
    for (int j = 1; j < 3; ++j) {
        const double err = up_to_sign_err(
            c_n,
            [&](int i) { return evecs[static_cast<std::size_t>(i) * 3 + j]; },
            [&](int i) { return c_evecs[i * 2 + (j - 1)]; });
        CHECK(err < 1e-4);
    }

    // n = 3, dim = 2 -> k = 3 >= n: upstream raises (uncaught scipy error)
    std::vector<int32_t> r{0, 0, 1}, c{1, 2, 2};
    std::vector<float> v{0.5f, 0.6f, 0.7f};
    umap::SparseMatrix tiny = sym_graph(3, r.data(), c.data(), v.data(), 3);
    int64_t rng_state[3] = {1, 2, 3};
    CHECK_THROWS_AS(umap::spectral_layout(tiny, 2, rng_state),
                    std::invalid_argument);
}

TEST_CASE("multi-component layout: two cliques (graph B) match upstream") {
    using namespace golden_spectral;
    const std::size_t nnz = sizeof(b_vals) / sizeof(b_vals[0]);
    umap::SparseMatrix g = sym_graph(b_n, b_rows, b_cols, b_vals, nnz);

    std::vector<int32_t> labels;
    REQUIRE(umap::connected_components(g, labels) == 2);

    int64_t rng_state[3] = {42, 43, 44};
    std::vector<float> res = umap::spectral_layout(g, 2, rng_state);
    REQUIRE(res.size() == static_cast<std::size_t>(b_n) * 2);
    CHECK(all_finite(res));

    // n_components = 2 <= 2*dim: deterministic meta [[1,0],[-1,0]], and the
    // minimum positive meta distance is 2, so data_range = 1 for both.
    const double meta[2][2] = {{1.0, 0.0}, {-1.0, 0.0}};
    const double data_range = 1.0;
    for (int i = 0; i < b_n; ++i)
        for (int j = 0; j < 2; ++j)
            CHECK(std::abs(res[i * 2 + j] - meta[labels[i]][j]) <=
                  data_range + 1e-4);

    // components separated in the layout (meta centers at x = +/-1)
    double mean_x0 = 0.0, mean_x1 = 0.0;
    for (int i = 0; i < b_sizes[0]; ++i) mean_x0 += res[i * 2];
    for (int i = b_sizes[0]; i < b_n; ++i) mean_x1 += res[i * 2];
    mean_x0 /= b_sizes[0];
    mean_x1 /= b_sizes[1];
    CHECK(mean_x0 > 0.0);
    CHECK(mean_x1 < 0.0);
    CHECK(mean_x0 - mean_x1 > 0.3);

    // per-component spectral parts match the upstream result up to the
    // per-column eigenvector sign
    int start = 0;
    for (int c = 0; c < 2; ++c) {
        for (int j = 0; j < 2; ++j) {
            const double err = up_to_sign_err(
                b_sizes[c],
                [&](int i) {
                    return static_cast<double>(res[(start + i) * 2 + j]) -
                           meta[c][j];
                },
                [&](int i) {
                    return static_cast<double>(b_result[(start + i) * 2 + j]) -
                           meta[c][j];
                });
            CHECK(err < 2e-3);
        }
        start += b_sizes[c];
    }
}

TEST_CASE("multi-component layout: tiny components take the random branch") {
    // component 0: triangle (size 3 <= dim + 1), component 1: singleton
    std::vector<int32_t> r{0, 0, 1}, c{1, 2, 2};
    std::vector<float> v{0.5f, 0.6f, 0.7f};
    std::vector<int32_t> rr(r), cc(c);
    std::vector<float> vv(v);
    rr.insert(rr.end(), c.begin(), c.end());
    cc.insert(cc.end(), r.begin(), r.end());
    vv.insert(vv.end(), v.begin(), v.end());
    umap::SparseMatrix g = umap::SparseMatrix::from_coo(4, 4, rr, cc, vv);

    std::vector<int32_t> labels;
    REQUIRE(umap::connected_components(g, labels) == 2);
    CHECK(labels == std::vector<int32_t>{0, 0, 0, 1});

    int64_t rng_state[3] = {7, 8, 9};
    std::vector<float> res = umap::spectral_layout(g, 2, rng_state);
    REQUIRE(res.size() == 8);
    CHECK(all_finite(res));

    // meta [[1,0],[-1,0]], data_range = 1: uniform placement in the box
    // around each component's meta position
    const double meta[2][2] = {{1.0, 0.0}, {-1.0, 0.0}};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            CHECK(std::abs(res[i * 2 + j] - meta[labels[i]][j]) <= 1.0 + 1e-6);
    // random draws should not be degenerate (all identical)
    CHECK(res[0] != res[2]);
}

TEST_CASE("component_layout + five-component layout with data (graph D)") {
    using namespace golden_spectral;
    const std::size_t nnz = sizeof(d_vals) / sizeof(d_vals[0]);
    umap::SparseMatrix g = sym_graph(d_n, d_rows, d_cols, d_vals, nnz);

    std::vector<int32_t> labels;
    REQUIRE(umap::connected_components(g, labels) == d_n_components);
    for (int i = 0; i < d_n; ++i) CHECK(labels[i] == d_labels[i]);

    // component_layout: meta embedding of centroids matches sklearn's
    // SpectralEmbedding-based result exactly (deterministic sign flip)
    int64_t rng_state[3] = {11, 22, 33};
    std::vector<double> meta = umap::component_layout(
        d_data, d_n, d_n_features, d_n_components, labels, 2, rng_state);
    REQUIRE(meta.size() == static_cast<std::size_t>(d_n_components) * 2);
    for (int c = 0; c < d_n_components; ++c)
        for (int j = 0; j < 2; ++j)
            CHECK(std::abs(meta[c * 2 + j] - d_meta[c * 2 + j]) < 1e-4);

    // full multi-component layout vs upstream, per component per column up to
    // eigenvector sign (relative to the shared meta position)
    std::vector<float> res =
        umap::spectral_layout(d_data, d_n_features, g, 2, rng_state);
    REQUIRE(res.size() == static_cast<std::size_t>(d_n) * 2);
    CHECK(all_finite(res));
    const int comp_size = d_n / d_n_components;
    for (int c = 0; c < d_n_components; ++c) {
        const int start = c * comp_size;
        for (int j = 0; j < 2; ++j) {
            const double err = up_to_sign_err(
                comp_size,
                [&](int i) {
                    return static_cast<double>(res[(start + i) * 2 + j]) -
                           d_meta[c * 2 + j];
                },
                [&](int i) {
                    return static_cast<double>(d_result[(start + i) * 2 + j]) -
                           d_meta[c * 2 + j];
                });
            CHECK(err < 2e-3);
        }
    }

    // component_layout without data: upstream "just guess" branch, uniform
    // in [0, 10)
    std::vector<double> guess = umap::component_layout(
        nullptr, d_n, 0, d_n_components, labels, 2, rng_state);
    REQUIRE(guess.size() == static_cast<std::size_t>(d_n_components) * 2);
    for (const double x : guess) {
        CHECK(x >= 0.0);
        CHECK(x <= 10.0);
    }
}

TEST_CASE("noisy_scale_coords expands to |max| = max_coord and jitters") {
    std::vector<float> coords{0.5f, -2.5f, 1.25f, 0.0f, -0.75f, 2.0f};
    int64_t rng_state[3] = {314, 159, 265};
    std::vector<float> out =
        umap::noisy_scale_coords(coords, rng_state, 10.0f, 0.0001f);
    REQUIRE(out.size() == coords.size());
    CHECK(all_finite(out));
    float max_abs = 0.0f;
    for (float v : out) max_abs = std::max(max_abs, std::abs(v));
    // expansion maps max |coord| (2.5) to 10; noise sigma 1e-4
    CHECK(std::abs(max_abs - 10.0f) < 0.01f);
    for (std::size_t i = 0; i < out.size(); ++i)
        CHECK(std::abs(out[i] - coords[i] * 4.0f) < 0.01f);
    // noise applied (not the pure scaling)
    bool any_diff = false;
    for (std::size_t i = 0; i < out.size(); ++i)
        if (out[i] != coords[i] * 4.0f) any_diff = true;
    CHECK(any_diff);
}

TEST_CASE("initialize_embedding dispatch: random, spectral, pca") {
    using namespace golden_spectral;
    const std::size_t nnz = sizeof(a_vals) / sizeof(a_vals[0]);
    umap::SparseMatrix g = sym_graph(a_n, a_rows, a_cols, a_vals, nnz);

    int64_t rng_state[3] = {101, 202, 303};
    // init='random': uniform in [-10, 10]
    std::vector<float> rnd = umap::initialize_embedding(
        nullptr, 0, g, 2, umap::EmbeddingInit::Random, rng_state);
    REQUIRE(rnd.size() == static_cast<std::size_t>(a_n) * 2);
    CHECK(all_finite(rnd));
    for (const float v : rnd) {
        CHECK(v >= -10.0f);
        CHECK(v <= 10.0f);
    }
    CHECK(rnd[0] != rnd[1]);

    // init='spectral': spectral_layout + noisy_scale_coords(10, 1e-4)
    std::vector<float> spec = umap::initialize_embedding(
        nullptr, 0, g, 2, umap::EmbeddingInit::Spectral, rng_state);
    REQUIRE(spec.size() == static_cast<std::size_t>(a_n) * 2);
    CHECK(all_finite(spec));
    float max_abs = 0.0f;
    for (float v : spec) max_abs = std::max(max_abs, std::abs(v));
    CHECK(std::abs(max_abs - 10.0f) < 0.01f);
    // matches the raw spectral layout after rescaling, up to sign + noise
    std::vector<float> raw = umap::spectral_layout(g, 2, rng_state);
    float raw_max = 0.0f;
    for (float v : raw) raw_max = std::max(raw_max, std::abs(v));
    for (int j = 0; j < 2; ++j) {
        const double err = up_to_sign_err(
            a_n,
            [&](int i) { return static_cast<double>(spec[i * 2 + j]); },
            [&](int i) {
                return static_cast<double>(raw[i * 2 + j]) * 10.0 / raw_max;
            });
        CHECK(err < 0.01);
    }

    // init='pca': unimplemented (clear error)
    CHECK_THROWS_AS(umap::initialize_embedding(nullptr, 0, g, 2,
                                               umap::EmbeddingInit::PCA,
                                               rng_state),
                    std::runtime_error);
}
