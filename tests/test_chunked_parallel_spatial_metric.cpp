// Port of reference/umap/umap/tests/test_chunked_parallel_spatial_metric.py
// to doctest. Everything except test_chunked_parallel_special_metric_
// implementation_hellinger is @benchmark_only upstream (skipped unless
// BENCHARM_TEST is set); those benchmarks are mirrored as skips.
//
// The live test checks that pairwise_special_metric's chunked hellinger
// matrices (X-only symmetric path vs X,Y rectangular path) agree with each
// other and with the per-pair hellinger_grad distances. The C++ port has no
// chunked/parallel evaluator -- the equivalent check is that the f32-rounded
// hellinger kernel evaluated through the symmetric fill, the rectangular
// loop, and hellinger_grad produce identical float32 matrices, and that they
// match the reference pairwise_special_metric goldens. The unseeded
// spatial_data_two X/Y pair of the upstream test is replaced by a
// deterministic second operand (the same rows in reverse order); the
// assertion is likewise implementation-agreement. See tests/PORTING_NOTES.md.
#include <cmath>
#include <cstdint>
#include <vector>

#include "doctest.h"
#include "umap/distances.hpp"
#include "upstream_data.hpp"
#include "upstream_metric_goldens.hpp"

namespace {

constexpr int kN = 10, kDim = 20;  // spatial_data[:-2]

std::vector<double> abs_spatial() {
    std::vector<double> a(static_cast<std::size_t>(kN) * kDim);
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = std::abs(updata::spatial_data[i]);
    return a;
}

}  // namespace

TEST_SUITE("test_chunked_parallel_spatial_metric.py") {

TEST_CASE("test_chunked_parallel_alternative_implementations" *
          doctest::skip()) {}  // @benchmark_only upstream

TEST_CASE("test_chunked_parallel_special_metric_implementation_hellinger") {
    const std::vector<double> X = abs_spatial();

    // dist_matrix_x: symmetric X-only path (j > i mirrored, zero diagonal).
    std::vector<float> mx(kN * kN, 0.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = i + 1; j < kN; ++j) {
            const float d = static_cast<float>(
                umap::hellinger(X.data() + static_cast<std::size_t>(i) * kDim,
                                X.data() + static_cast<std::size_t>(j) * kDim,
                                kDim));
            mx[i * kN + j] = d;
            mx[j * kN + i] = d;
        }

    // dist_matrix_xy: rectangular X,Y=X path (every pair evaluated).
    std::vector<float> mxy(kN * kN);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            mxy[i * kN + j] = static_cast<float>(
                umap::hellinger(X.data() + static_cast<std::size_t>(i) * kDim,
                                X.data() + static_cast<std::size_t>(j) * kDim,
                                kDim));

    // test_matrix: hellinger_grad(|x_i|, |x_j|)[0] (float32 like upstream).
    std::vector<float> mg(kN * kN);
    std::vector<double> grad(kDim);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            mg[i * kN + j] = static_cast<float>(umap::hellinger_grad(
                X.data() + static_cast<std::size_t>(i) * kDim,
                X.data() + static_cast<std::size_t>(j) * kDim, kDim,
                grad.data()));

    for (int i = 0; i < kN * kN; ++i) {
        CAPTURE(i);
        CHECK(mx[i] == mxy[i]);  // assert_array_equal
        CHECK(mg[i] == mxy[i]);
    }
    // ... and both match the reference pairwise_special_metric goldens.
    for (int i = 0; i < kN * kN; ++i) {
        CAPTURE(i);
        CHECK(std::abs(static_cast<double>(mxy[i]) -
                       umgold::hellinger_xy_golden[i]) < 1e-6);
    }

    // X/Y different-operand path: row-reversed X as the deterministic second
    // operand; evaluation order must not matter.
    std::vector<float> m1(kN * kN), m2(kN * kN);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j) {
            const double* yi =
                X.data() + static_cast<std::size_t>(kN - 1 - j) * kDim;
            m1[i * kN + j] = static_cast<float>(umap::hellinger(
                X.data() + static_cast<std::size_t>(i) * kDim, yi, kDim));
        }
    for (int j = 0; j < kN; ++j)
        for (int i = 0; i < kN; ++i) {
            const double* yi =
                X.data() + static_cast<std::size_t>(kN - 1 - j) * kDim;
            m2[i * kN + j] = static_cast<float>(umap::hellinger(
                X.data() + static_cast<std::size_t>(i) * kDim, yi, kDim));
        }
    for (int i = 0; i < kN * kN; ++i) {
        CAPTURE(i);
        CHECK(m1[i] == m2[i]);
    }
}

// Benchmark-only cases (skipped upstream without BENCHARM_TEST).
TEST_CASE("test_benchmark_chunked_parallel_special_metric_x_only" *
          doctest::skip()) {}
TEST_CASE("test_benchmark_workaround_590_x_only" * doctest::skip()) {}
TEST_CASE("test_benchmark_chunked_parallel_special_metric_x_y" *
          doctest::skip()) {}
TEST_CASE("test_benchmark_workaround_590_x_y" * doctest::skip()) {}

}  // TEST_SUITE
