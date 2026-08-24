// Distance metrics vs golden values computed by the reference Python UMAP
// (scripts/gen_golden.py).
#include <cmath>
#include <cstddef>

#include "doctest.h"
#include "golden_data.hpp"
#include "umap/distances.hpp"

namespace {
constexpr std::size_t N1 = 4;  // x1 / y1
constexpr std::size_t N2 = 5;  // x2 / y2
constexpr std::size_t NB = 6;  // xb / yb
constexpr double TOL = 1e-12;  // metrics are exact double math on both sides
}  // namespace

TEST_CASE("euclidean") {
    CHECK(umap::euclidean(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_euclidean_x1_y1).epsilon(TOL));
    CHECK(umap::euclidean(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_euclidean_x2_y2).epsilon(TOL));
    CHECK(umap::euclidean(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_euclidean_xb_yb).epsilon(TOL));
    CHECK(umap::euclidean(golden::dist_x1, golden::dist_x1, N1) == 0.0);
}

TEST_CASE("squared euclidean") {
    CHECK(umap::squared_euclidean(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_sqeuclidean_x1_y1).epsilon(TOL));
    CHECK(umap::squared_euclidean(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_sqeuclidean_x2_y2).epsilon(TOL));
    CHECK(umap::squared_euclidean(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_sqeuclidean_xb_yb).epsilon(TOL));
}

TEST_CASE("manhattan") {
    CHECK(umap::manhattan(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_manhattan_x1_y1).epsilon(TOL));
    CHECK(umap::manhattan(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_manhattan_x2_y2).epsilon(TOL));
    CHECK(umap::manhattan(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_manhattan_xb_yb).epsilon(TOL));
}

TEST_CASE("chebyshev") {
    CHECK(umap::chebyshev(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_chebyshev_x1_y1).epsilon(TOL));
    CHECK(umap::chebyshev(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_chebyshev_x2_y2).epsilon(TOL));
    CHECK(umap::chebyshev(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_chebyshev_xb_yb).epsilon(TOL));
}

TEST_CASE("minkowski") {
    CHECK(umap::minkowski(golden::dist_x1, golden::dist_y1, N1, 3.0) ==
          doctest::Approx(golden::d_minkowski_p3_x1_y1).epsilon(TOL));
    CHECK(umap::minkowski(golden::dist_x2, golden::dist_y2, N2, 3.0) ==
          doctest::Approx(golden::d_minkowski_p3_x2_y2).epsilon(TOL));
    CHECK(umap::minkowski(golden::dist_xb, golden::dist_yb, NB, 3.0) ==
          doctest::Approx(golden::d_minkowski_p3_xb_yb).epsilon(TOL));
    // p=2 must agree with euclidean
    CHECK(umap::minkowski(golden::dist_x1, golden::dist_y1, N1, 2.0) ==
          doctest::Approx(umap::euclidean(golden::dist_x1, golden::dist_y1, N1))
              .epsilon(TOL));
}

TEST_CASE("cosine incl. zero-norm edge cases") {
    CHECK(umap::cosine(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_cosine_x1_y1).epsilon(TOL));
    CHECK(umap::cosine(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_cosine_x2_y2).epsilon(TOL));
    CHECK(umap::cosine(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_cosine_xb_yb).epsilon(TOL));
    const double z[4] = {0.0, 0.0, 0.0, 0.0};
    CHECK(umap::cosine(z, z, N1) == golden::d_cosine_zz);       // both zero -> 0
    CHECK(umap::cosine(z, golden::dist_x1, N1) == golden::d_cosine_zx);  // one zero -> 1
}

TEST_CASE("correlation incl. degenerate edge cases") {
    CHECK(umap::correlation(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_correlation_x1_y1).epsilon(TOL));
    CHECK(umap::correlation(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_correlation_x2_y2).epsilon(TOL));
    CHECK(umap::correlation(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_correlation_xb_yb).epsilon(TOL));
    const double z[4] = {0.0, 0.0, 0.0, 0.0};
    const double c[4] = {2.5, 2.5, 2.5, 2.5};
    CHECK(umap::correlation(z, z, N1) == golden::d_correlation_zz);  // -> 0
    // constant vector => zero dot product -> 1 (upstream's elif ordering)
    CHECK(umap::correlation(c, golden::dist_x1, N1) == golden::d_correlation_cx);
}

TEST_CASE("hamming") {
    CHECK(umap::hamming(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_hamming_x1_y1).epsilon(TOL));
    CHECK(umap::hamming(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_hamming_x2_y2).epsilon(TOL));
    CHECK(umap::hamming(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_hamming_xb_yb).epsilon(TOL));
}

TEST_CASE("jaccard") {
    CHECK(umap::jaccard(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_jaccard_x1_y1).epsilon(TOL));
    CHECK(umap::jaccard(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_jaccard_x2_y2).epsilon(TOL));
    CHECK(umap::jaccard(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_jaccard_xb_yb).epsilon(TOL));
    const double z[4] = {0.0, 0.0, 0.0, 0.0};
    CHECK(umap::jaccard(z, z, N1) == 0.0);  // no non-zeros -> 0
}

TEST_CASE("dice") {
    CHECK(umap::dice(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_dice_x1_y1).epsilon(TOL));
    CHECK(umap::dice(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_dice_x2_y2).epsilon(TOL));
    CHECK(umap::dice(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_dice_xb_yb).epsilon(TOL));
}

TEST_CASE("hellinger incl. zero-norm edge cases") {
    CHECK(umap::hellinger(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_hellinger_x2_y2).epsilon(TOL));
    CHECK(umap::hellinger(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_hellinger_xb_yb).epsilon(TOL));
    const double z[4] = {0.0, 0.0, 0.0, 0.0};
    CHECK(umap::hellinger(z, z, N1) == golden::d_hellinger_zz);  // -> 0
    CHECK(umap::hellinger(z, golden::dist_x1, N1) == golden::d_hellinger_zx);  // -> 1
}

TEST_CASE("canberra") {
    CHECK(umap::canberra(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_canberra_x1_y1).epsilon(TOL));
    CHECK(umap::canberra(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_canberra_x2_y2).epsilon(TOL));
    CHECK(umap::canberra(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_canberra_xb_yb).epsilon(TOL));
}

TEST_CASE("braycurtis") {
    CHECK(umap::bray_curtis(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_braycurtis_x1_y1).epsilon(TOL));
    CHECK(umap::bray_curtis(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_braycurtis_x2_y2).epsilon(TOL));
    CHECK(umap::bray_curtis(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_braycurtis_xb_yb).epsilon(TOL));
    const double z[4] = {0.0, 0.0, 0.0, 0.0};
    CHECK(umap::bray_curtis(z, z, N1) == 0.0);  // zero denominator -> 0
}

TEST_CASE("named_distances registry") {
    const auto& reg = umap::named_distances();
    // aliases resolve to the same metric results
    for (const char* name : {"euclidean", "l2"}) {
        CHECK(reg.at(name)(golden::dist_x1, golden::dist_y1, N1) ==
              doctest::Approx(golden::d_euclidean_x1_y1).epsilon(TOL));
    }
    for (const char* name : {"manhattan", "taxicab", "l1", "cityblock"}) {
        CHECK(reg.at(name)(golden::dist_x1, golden::dist_y1, N1) ==
              doctest::Approx(golden::d_manhattan_x1_y1).epsilon(TOL));
    }
    for (const char* name : {"chebyshev", "linfinity", "linfty", "linf"}) {
        CHECK(reg.at(name)(golden::dist_x1, golden::dist_y1, N1) ==
              doctest::Approx(golden::d_chebyshev_x1_y1).epsilon(TOL));
    }
    // minkowski default p=2 == euclidean
    CHECK(reg.at("minkowski")(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_euclidean_x1_y1).epsilon(TOL));
    CHECK(reg.at("sqeuclidean")(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_sqeuclidean_x1_y1).epsilon(TOL));
    CHECK(reg.at("cosine")(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_cosine_x1_y1).epsilon(TOL));
    CHECK(reg.at("correlation")(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_correlation_x1_y1).epsilon(TOL));
    CHECK(reg.at("hamming")(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_hamming_xb_yb).epsilon(TOL));
    CHECK(reg.at("jaccard")(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_jaccard_xb_yb).epsilon(TOL));
    CHECK(reg.at("dice")(golden::dist_xb, golden::dist_yb, NB) ==
          doctest::Approx(golden::d_dice_xb_yb).epsilon(TOL));
    CHECK(reg.at("hellinger")(golden::dist_x2, golden::dist_y2, N2) ==
          doctest::Approx(golden::d_hellinger_x2_y2).epsilon(TOL));
    CHECK(reg.at("canberra")(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_canberra_x1_y1).epsilon(TOL));
    CHECK(reg.at("braycurtis")(golden::dist_x1, golden::dist_y1, N1) ==
          doctest::Approx(golden::d_braycurtis_x1_y1).epsilon(TOL));
    CHECK(reg.count("no_such_metric") == 0);
}
