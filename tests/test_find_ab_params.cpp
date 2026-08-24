// find_ab_params vs golden (a, b) pairs from scipy.optimize.curve_fit in the
// reference implementation (scripts/gen_golden.py).
#include <cmath>

#include "doctest.h"
#include "golden_data.hpp"
#include "umap/find_ab_params.hpp"

TEST_CASE("find_ab_params matches scipy curve_fit goldens") {
    for (int i = 0; i < golden::n_ab_cases; ++i) {
        const double spread = golden::ab_cases[i][0];
        const double min_dist = golden::ab_cases[i][1];
        const double exp_a = golden::ab_cases[i][2];
        const double exp_b = golden::ab_cases[i][3];
        CAPTURE(spread);
        CAPTURE(min_dist);
        auto ab = umap::find_ab_params(spread, min_dist);
        CHECK(ab.a == doctest::Approx(exp_a).epsilon(1e-4));
        CHECK(ab.b == doctest::Approx(exp_b).epsilon(1e-4));
    }
}

TEST_CASE("find_ab_params canonical defaults (spread=1, min_dist=0.1)") {
    auto ab = umap::find_ab_params(1.0, 0.1);
    // Widely-quoted reference values for UMAP defaults.
    CHECK(ab.a == doctest::Approx(1.577).epsilon(1e-3));
    CHECK(ab.b == doctest::Approx(0.8951).epsilon(1e-3));
}

TEST_CASE("find_ab_params fit is self-consistent (residual sanity)") {
    // The fitted curve must approximate the target piecewise curve well.
    const double spread = 1.0, min_dist = 0.1;
    auto ab = umap::find_ab_params(spread, min_dist);
    double ss = 0.0;
    const int N = 300;
    for (int i = 0; i < N; ++i) {
        double x = spread * 3.0 * i / (N - 1);
        double y = x < min_dist ? 1.0 : std::exp(-(x - min_dist) / spread);
        double t = x > 0.0 ? std::pow(x, 2.0 * ab.b) : 0.0;
        double f = 1.0 / (1.0 + ab.a * t);
        ss += (f - y) * (f - y);
    }
    CHECK(ss / N < 1e-3);  // small mean squared residual
}
