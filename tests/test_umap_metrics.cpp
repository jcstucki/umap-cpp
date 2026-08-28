// Port of reference/umap/umap/tests/test_umap_metrics.py to doctest.
// One TEST_CASE per upstream test function (names preserved). Golden distance
// matrices were computed with sklearn/scipy/BallTree on the conftest datasets
// (np.random.seed(189212)) by scripts/gen_upstream_fixtures.py, with the
// upstream tests' non-finite corrections already applied
// (tests/upstream_metric_goldens.hpp).
//
// Upstream-skip parity (see tests/PORTING_NOTES.md):
//   * test_kulsinski / test_sparse_kulsinski: skipped upstream for
//     scipy >= 1.9 (venv: 1.18) -- goldens cannot be generated; mirrored as
//     doctest skips.
//   * test_sokalmichener / test_sparse_sokalmichener / test_sparse_sokalsneath:
//     skipped upstream for sklearn >= 1.8 (venv: 1.9); mirrored.
//   * sparse chebyshev/minkowski/hamming/canberra/braycurtis/correlation and
//     the sparse binary metrics + sparse_hellinger: the C++ sparse registry
//     ports only euclidean/manhattan/cosine so far (Phase 1b TODO); skipped.
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "doctest.h"
#include "umap/distances.hpp"
#include "umap/sparse_distances.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"
#include "upstream_metric_goldens.hpp"

namespace {

using DistFn = std::function<double(const double*, const double*, std::size_t)>;

constexpr int kSpatialN = 12, kSpatialDim = 20;

// numpy.testing.assert_array_almost_equal(decimal=d): abs diff < 1.5e-d,
// with NaN == NaN counting as equal (numpy semantics; the mahalanobis grad
// golden contains NaNs from upstream's non-PSD VI draw).
bool almost_equal(double a, double b, double tol) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::abs(a - b) < tol;
}

void check_matrix(const DistFn& fn, const double* data, int n, int dim,
                  const double* golden, int decimal = 6) {
    const double tol = 1.5 * std::pow(10.0, -decimal);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const double d = fn(data + static_cast<std::size_t>(i) * dim,
                                data + static_cast<std::size_t>(j) * dim, dim);
            CAPTURE(i);
            CAPTURE(j);
            CHECK(almost_equal(d, golden[i * n + j], tol));
        }
}

DistFn named(const std::string& name) {
    return umap::named_distances().at(name);
}

// run_test_metric with with_grad=True: evaluate the gradient variant's
// distance component against the same golden.
void check_matrix_grad(const umap::GradientFn& fn, const double* data, int n,
                       int dim, const double* golden, int decimal = 6) {
    const double tol = 1.5 * std::pow(10.0, -decimal);
    std::vector<double> x(dim), y(dim), grad(std::max(dim, 6));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            x.assign(data + static_cast<std::size_t>(i) * dim,
                     data + static_cast<std::size_t>(i + 1) * dim);
            y.assign(data + static_cast<std::size_t>(j) * dim,
                     data + static_cast<std::size_t>(j + 1) * dim);
            const double d = fn(x.data(), y.data(), dim, grad.data());
            CAPTURE(i);
            CAPTURE(j);
            CHECK(almost_equal(d, golden[i * n + j], tol));
        }
}

std::vector<double> binary_f64() {
    return testutil::to_f64(updata::binary_data,
                            static_cast<std::size_t>(kSpatialN) * kSpatialDim);
}

// CSR view of conftest's sparse_spatial_data (= csr(spatial_data*binary_data))
// with float32 values like the reference kernels consume.
struct SparseRows {
    std::vector<int32_t> indptr, indices;
    std::vector<float> data;
};

SparseRows sparse_spatial_csr() {
    const std::vector<float> dense = testutil::to_f32(
        updata::sparse_spatial_dense,
        static_cast<std::size_t>(kSpatialN) * kSpatialDim);
    const testutil::Csr c = testutil::dense_to_csr(dense.data(), kSpatialN,
                                                   kSpatialDim);
    return SparseRows{c.indptr, c.indices, c.data};
}

using SparseFn = float (*)(const int32_t*, const float*, int64_t,
                           const int32_t*, const float*, int64_t);

void check_sparse_matrix(SparseFn fn, const double* golden, int decimal = 6) {
    const SparseRows s = sparse_spatial_csr();
    const double tol = 1.5 * std::pow(10.0, -decimal);
    for (int i = 0; i < kSpatialN; ++i)
        for (int j = 0; j < kSpatialN; ++j) {
            const float d = fn(s.indices.data() + s.indptr[i],
                               s.data.data() + s.indptr[i],
                               s.indptr[i + 1] - s.indptr[i],
                               s.indices.data() + s.indptr[j],
                               s.data.data() + s.indptr[j],
                               s.indptr[j + 1] - s.indptr[j]);
            CAPTURE(i);
            CAPTURE(j);
            CHECK(std::abs(static_cast<double>(d) -
                           golden[i * kSpatialN + j]) < tol);
        }
}

}  // namespace

TEST_SUITE("test_umap_metrics.py") {

// --------------------------------------------------- spatial metric tests

TEST_CASE("test_euclidean") {
    check_matrix(named("euclidean"), updata::spatial_data, kSpatialN,
                 kSpatialDim, umgold::spatial_euclidean);
}

TEST_CASE("test_manhattan") {
    check_matrix(named("manhattan"), updata::spatial_data, kSpatialN,
                 kSpatialDim, umgold::spatial_manhattan);
}

TEST_CASE("test_chebyshev") {
    check_matrix(named("chebyshev"), updata::spatial_data, kSpatialN,
                 kSpatialDim, umgold::spatial_chebyshev);
}

TEST_CASE("test_minkowski") {
    check_matrix(named("minkowski"), updata::spatial_data, kSpatialN,
                 kSpatialDim, umgold::spatial_minkowski);
}

TEST_CASE("test_hamming") {
    check_matrix(named("hamming"), updata::spatial_data, kSpatialN,
                 kSpatialDim, umgold::spatial_hamming);
}

TEST_CASE("test_canberra") {
    check_matrix(named("canberra"), updata::spatial_data, kSpatialN,
                 kSpatialDim, umgold::spatial_canberra);
}

TEST_CASE("test_braycurtis") {
    check_matrix(named("braycurtis"), updata::spatial_data, kSpatialN,
                 kSpatialDim, umgold::spatial_braycurtis);
}

TEST_CASE("test_cosine") {
    check_matrix(named("cosine"), updata::spatial_data, kSpatialN,
                 kSpatialDim, umgold::spatial_cosine);
}

TEST_CASE("test_correlation") {
    check_matrix(named("correlation"), updata::spatial_data, kSpatialN,
                 kSpatialDim, umgold::spatial_correlation);
}

// ---------------------------------------------------- binary metric tests

TEST_CASE("test_jaccard") {
    check_matrix(named("jaccard"), binary_f64().data(), kSpatialN, kSpatialDim,
                 umgold::binary_jaccard);
}

TEST_CASE("test_matching") {
    check_matrix(named("matching"), binary_f64().data(), kSpatialN,
                 kSpatialDim, umgold::binary_matching);
}

TEST_CASE("test_dice") {
    check_matrix(named("dice"), binary_f64().data(), kSpatialN, kSpatialDim,
                 umgold::binary_dice);
}

// Upstream: skipped for scipy >= 1.9 (metric removed; venv scipy 1.18).
TEST_CASE("test_kulsinski" * doctest::skip()) {}

TEST_CASE("test_rogerstanimoto") {
    check_matrix(named("rogerstanimoto"), binary_f64().data(), kSpatialN,
                 kSpatialDim, umgold::binary_rogerstanimoto);
}

TEST_CASE("test_russellrao") {
    check_matrix(named("russellrao"), binary_f64().data(), kSpatialN,
                 kSpatialDim, umgold::binary_russellrao);
}

// Upstream: skipped for sklearn >= 1.8 (metric removed; venv sklearn 1.9).
TEST_CASE("test_sokalmichener" * doctest::skip()) {}

TEST_CASE("test_sokalsneath") {
    check_matrix(named("sokalsneath"), binary_f64().data(), kSpatialN,
                 kSpatialDim, umgold::binary_sokalsneath);
}

TEST_CASE("test_yule") {
    check_matrix(named("yule"), binary_f64().data(), kSpatialN, kSpatialDim,
                 umgold::binary_yule);
}

// -------------------------------------------- sparse spatial metric tests

TEST_CASE("test_sparse_euclidean") {
    check_sparse_matrix(&umap::spdist::sparse_euclidean,
                        umgold::sparse_euclidean);
}

TEST_CASE("test_sparse_manhattan") {
    check_sparse_matrix(&umap::spdist::sparse_manhattan,
                        umgold::sparse_manhattan);
}

// Sparse metrics beyond euclidean/manhattan/cosine are not in the C++ sparse
// registry yet (Phase 1b "remaining sparse_named_distances metrics" TODO).
TEST_CASE("test_sparse_chebyshev" * doctest::skip()) {}
TEST_CASE("test_sparse_minkowski" * doctest::skip()) {}
TEST_CASE("test_sparse_hamming" * doctest::skip()) {}
TEST_CASE("test_sparse_canberra" * doctest::skip()) {}

TEST_CASE("test_sparse_cosine") {
    check_sparse_matrix(&umap::spdist::sparse_cosine, umgold::sparse_cosine);
}

TEST_CASE("test_sparse_correlation" * doctest::skip()) {}
TEST_CASE("test_sparse_braycurtis" * doctest::skip()) {}

// --------------------------------------------- sparse binary metric tests
// (none of the binary metrics are in the C++ sparse registry yet)

TEST_CASE("test_sparse_jaccard" * doctest::skip()) {}
TEST_CASE("test_sparse_matching" * doctest::skip()) {}
TEST_CASE("test_sparse_dice" * doctest::skip()) {}
TEST_CASE("test_sparse_kulsinski" * doctest::skip()) {}     // + upstream skip
TEST_CASE("test_sparse_rogerstanimoto" * doctest::skip()) {}
TEST_CASE("test_sparse_russellrao" * doctest::skip()) {}
TEST_CASE("test_sparse_sokalmichener" * doctest::skip()) {}  // upstream skip
TEST_CASE("test_sparse_sokalsneath" * doctest::skip()) {}    // upstream skip

// ------------------------------------- standardised / weighted distances

TEST_CASE("test_seuclidean") {
    umap::MetricKwds kwds;
    kwds.v.assign(umgold::seuclidean_v, umgold::seuclidean_v + kSpatialDim);
    check_matrix(umap::make_metric("seuclidean", kwds), updata::spatial_data,
                 kSpatialN, kSpatialDim, umgold::spatial_seuclidean);
}

TEST_CASE("test_weighted_minkowski") {
    // Upstream: skipped for scipy < 1.8 (venv scipy 1.18 -> runs).
    umap::MetricKwds kwds;
    kwds.p = 3.0;
    kwds.v.assign(umgold::wminkowski_v, umgold::wminkowski_v + kSpatialDim);
    check_matrix(umap::make_metric("wminkowski", kwds), updata::spatial_data,
                 kSpatialN, kSpatialDim, umgold::spatial_wminkowski);
}

TEST_CASE("test_mahalanobis") {
    umap::MetricKwds kwds;
    kwds.vinv.assign(umgold::mahal_V,
                     umgold::mahal_V + kSpatialDim * kSpatialDim);
    check_matrix(umap::make_metric("mahalanobis", kwds), updata::spatial_data,
                 kSpatialN, kSpatialDim, umgold::spatial_mahalanobis);
}

TEST_CASE("test_haversine") {
    // BallTree returns each row's distances sorted ascending; replicate the
    // upstream row-sort of the test matrix before comparing.
    const double tol = 1.5e-6;
    for (int i = 0; i < kSpatialN; ++i) {
        std::vector<double> row(kSpatialN);
        for (int j = 0; j < kSpatialN; ++j)
            row[j] = umap::haversine(
                updata::spatial_data + static_cast<std::size_t>(i) * kSpatialDim,
                updata::spatial_data + static_cast<std::size_t>(j) * kSpatialDim,
                2);
        std::sort(row.begin(), row.end());
        for (int j = 0; j < kSpatialN; ++j) {
            CAPTURE(i);
            CAPTURE(j);
            CHECK(std::abs(row[j] -
                           umgold::haversine_sorted[i * kSpatialN + j]) < tol);
        }
    }
}

TEST_CASE("test_hellinger") {
    // pairwise_special_metric(|spatial_data[:-2]|) vs the closed-form golden.
    const int n = kSpatialN - 2;
    std::vector<double> absd(static_cast<std::size_t>(n) * kSpatialDim);
    for (std::size_t i = 0; i < absd.size(); ++i)
        absd[i] = std::abs(updata::spatial_data[i]);
    check_matrix(named("hellinger"), absd.data(), n, kSpatialDim,
                 umgold::hellinger_golden);

    // "Ensure ll_dirichlet runs" -- evaluate the full matrix and require
    // real (non-NaN) values.
    const DistFn lld = named("ll_dirichlet");
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const double d =
                lld(absd.data() + static_cast<std::size_t>(i) * kSpatialDim,
                    absd.data() + static_cast<std::size_t>(j) * kSpatialDim,
                    kSpatialDim);
            CHECK_FALSE(std::isnan(d));
        }
}

// sparse_hellinger / sparse_ll_dirichlet are not ported yet.
TEST_CASE("test_sparse_hellinger" * doctest::skip()) {}

TEST_CASE("test_grad_metrics_match_metrics") {
    // named_distances_with_gradients ∩ spatial_distances, like upstream.
    const auto& reg = umap::named_distances_with_gradients();
    for (const char* metric : {"euclidean", "manhattan", "chebyshev",
                               "minkowski", "canberra", "braycurtis", "cosine",
                               "correlation"}) {
        CAPTURE(metric);
        const double* golden = nullptr;
        if (std::string(metric) == "euclidean") golden = umgold::spatial_euclidean;
        else if (std::string(metric) == "manhattan") golden = umgold::spatial_manhattan;
        else if (std::string(metric) == "chebyshev") golden = umgold::spatial_chebyshev;
        else if (std::string(metric) == "minkowski") golden = umgold::spatial_minkowski;
        else if (std::string(metric) == "canberra") golden = umgold::spatial_canberra;
        else if (std::string(metric) == "braycurtis") golden = umgold::spatial_braycurtis;
        else if (std::string(metric) == "cosine") golden = umgold::spatial_cosine;
        else golden = umgold::spatial_correlation;
        check_matrix_grad(reg.at(metric), updata::spatial_data, kSpatialN,
                          kSpatialDim, golden);
    }

    // SEuclidean.
    umap::MetricKwds skwds;
    skwds.v.assign(umgold::seuclidean_v, umgold::seuclidean_v + kSpatialDim);
    check_matrix_grad(umap::make_metric_grad("seuclidean", skwds),
                      updata::spatial_data, kSpatialN, kSpatialDim,
                      umgold::spatial_seuclidean);

    // Weighted minkowski (scipy >= 1.8 branch; venv 1.18).
    umap::MetricKwds wkwds;
    wkwds.p = 3.0;
    wkwds.v.assign(umgold::wminkowski_v, umgold::wminkowski_v + kSpatialDim);
    check_matrix_grad(umap::make_metric_grad("wminkowski", wkwds),
                      updata::spatial_data, kSpatialN, kSpatialDim,
                      umgold::spatial_wminkowski);

    // Mahalanobis (upstream decimal=5).
    umap::MetricKwds mkwds;
    mkwds.vinv.assign(umgold::mahal_grad_V,
                      umgold::mahal_grad_V + kSpatialDim * kSpatialDim);
    check_matrix_grad(umap::make_metric_grad("mahalanobis", mkwds),
                      updata::spatial_data, kSpatialN, kSpatialDim,
                      umgold::spatial_mahalanobis_grad_golden, 5);

    // Hellinger: hellinger_grad distances vs pairwise_special_metric(X, X).
    const int n = kSpatialN - 2;
    std::vector<double> absd(static_cast<std::size_t>(n) * kSpatialDim);
    for (std::size_t i = 0; i < absd.size(); ++i)
        absd[i] = std::abs(updata::spatial_data[i]);
    check_matrix_grad(reg.at("hellinger"), absd.data(), n, kSpatialDim,
                      umgold::hellinger_xy_golden);
}

// ------------------------------------------ string metric (levenshtein)

TEST_CASE("test_core_distances") {
    struct Case { const char* x; const char* y; double d; };
    const Case cases[] = {
        {"", "", 0},        {"a", "a", 0},       {"abc", "abc", 0},
        {"", "a", 1},       {"a", "", 1},        {"", "abc", 3},
        {"a", "b", 1},      {"ab", "a", 1},      {"a", "ab", 1},
        {"kitten", "sitting", 3}, {"flaw", "lawn", 2}, {"gumbo", "gambol", 2},
        {"aaaa", "aaa", 1}, {"aaaa", "bbbb", 4},
        {"abc", "zabc", 1}, {"abc", "abcz", 1},  {"abc", "zabcz", 2},
        {"abcdef", "azced", 3},
        {"001122334455667788990a1b2c3d4e5f6g7h8i9j0k",
         "0112x231455667y78390a1b2uc3d4e5f6gvh4i9j0k", 10},
    };
    for (const auto& c : cases) {
        CAPTURE(c.x);
        CAPTURE(c.y);
        CHECK(umap::levenshtein(c.x, c.y) == c.d);
        CHECK(umap::levenshtein(c.y, c.x) == c.d);
        CHECK(umap::levenshtein_myers_ascii(c.x, c.y) == c.d);
        CHECK(umap::levenshtein_myers_ascii(c.y, c.x) == c.d);
    }
}

TEST_CASE("test_ascii_boundaries") {
    struct Case { std::string x; std::string y; double d; };
    const Case cases[] = {
        {std::string(1, '\x00'), std::string(1, '\x00'), 0},
        {"\x7f", "\x7f", 0},
        {std::string(1, '\x00'), "\x7f", 1},
        {"ABC", "abc", 3},
    };
    for (const auto& c : cases) {
        CHECK(umap::levenshtein(c.x, c.y) == c.d);
        CHECK(umap::levenshtein_myers_ascii(c.x, c.y) == c.d);
    }
}

TEST_CASE("test_length_difference_guard") {
    const std::string x(30, 'a'), y = "a";
    CHECK(umap::levenshtein(x, y, 1.0, 20) == 20.0);
    CHECK(umap::levenshtein_myers_ascii(x, y, 1.0, 20) == 20.0);
}

TEST_CASE("test_length_difference_guard_normalised") {
    const std::string x(30, 'a'), y = "a";
    CHECK(umap::levenshtein(x, y, 10.0, 20) == 2.0);
    CHECK(umap::levenshtein_myers_ascii(x, y, 10.0, 20) == 2.0);
}

TEST_CASE("test_max_dist_guard") {
    const std::string x(25, 'a'), y(25, 'b');
    CHECK(umap::levenshtein(x, y, 1.0, 10) == 10.0);
    CHECK(umap::levenshtein_myers_ascii(x, y, 1.0, 10) == 10.0);
}

TEST_CASE("test_max_dist_guard_normalised") {
    const std::string x(25, 'a'), y(25, 'b');
    CHECK(umap::levenshtein(x, y, 5.0, 10) == 2.0);
    CHECK(umap::levenshtein_myers_ascii(x, y, 5.0, 10) == 2.0);
}

TEST_CASE("test_fallback_path") {
    // len > 63 forces the Myers variant onto its DP fallback.
    const std::string x(64, 'a'), y(64, 'a');
    CHECK(umap::levenshtein(x, y) == 0.0);
    CHECK(umap::levenshtein_myers_ascii(x, y) == 0.0);
}

}  // TEST_SUITE
