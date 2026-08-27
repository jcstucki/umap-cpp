// Full metric surface (remaining distances.py metrics + _grad variants +
// kwds registry) vs golden values from the reference Python UMAP
// (scripts/gen_golden_metrics_full.py).
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_metrics_full.hpp"
#include "umap/distances.hpp"

namespace {
// Non-fastmath numba kernels: exact double math on both sides.
constexpr double TOL = 1e-12;
// fastmath=True numba kernels (LLVM may reassociate): 1e-6 relative bound
// required by spec; observed differences are ~1e-15 in practice.
constexpr double FTOL = 1e-6;

template <std::size_t N>
void check_grad(const double (&got)[N], const double* want, std::size_t n,
                double tol) {
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(got[i] == doctest::Approx(want[i]).epsilon(tol));
    }
}
}  // namespace

TEST_CASE("standardised euclidean (seuclidean)") {
    CHECK(umap::standardised_euclidean(golden::mf_a6, golden::mf_b6, 6,
                                       golden::mf_s6) ==
          doctest::Approx(golden::d_seuclidean_a6_b6).epsilon(TOL));
    CHECK(umap::standardised_euclidean(golden::mf_p6, golden::mf_q6, 6,
                                       golden::mf_s6) ==
          doctest::Approx(golden::d_seuclidean_p6_q6).epsilon(TOL));
    CHECK(umap::standardised_euclidean(golden::mf_a6, golden::mf_a6, 6,
                                       golden::mf_s6) ==
          golden::d_seuclidean_same);
}

TEST_CASE("weighted minkowski (wminkowski)") {
    CHECK(umap::weighted_minkowski(golden::mf_a6, golden::mf_b6, 6,
                                   golden::mf_w6, 3.0) ==
          doctest::Approx(golden::d_wminkowski_p3_a6_b6).epsilon(TOL));
    CHECK(umap::weighted_minkowski(golden::mf_a6, golden::mf_b6, 6,
                                   golden::mf_w6, 2.5) ==
          doctest::Approx(golden::d_wminkowski_p25_a6_b6).epsilon(TOL));
}

TEST_CASE("mahalanobis") {
    CHECK(umap::mahalanobis(golden::mf_a6, golden::mf_b6, 6,
                            golden::mf_vinv6) ==
          doctest::Approx(golden::d_mahalanobis_a6_b6).epsilon(TOL));
    CHECK(umap::mahalanobis(golden::mf_a6, golden::mf_a6, 6,
                            golden::mf_vinv6) == golden::d_mahalanobis_same);
}

TEST_CASE("binary metrics: matching / kulsinski / rogerstanimoto") {
    CHECK(umap::matching(golden::mf_bin1, golden::mf_bin2, 10) ==
          doctest::Approx(golden::d_matching_b1_b2).epsilon(TOL));
    CHECK(umap::matching(golden::mf_bin1, golden::mf_bin1, 10) ==
          golden::d_matching_same);
    const double z10[10] = {0};
    double o10[10];
    for (int i = 0; i < 10; ++i) o10[i] = 1.0;
    CHECK(umap::matching(z10, o10, 10) == golden::d_matching_z_o);

    CHECK(umap::kulsinski(golden::mf_bin1, golden::mf_bin2, 10) ==
          doctest::Approx(golden::d_kulsinski_b1_b2).epsilon(TOL));
    CHECK(umap::kulsinski(golden::mf_bin1, golden::mf_bin1, 10) ==
          golden::d_kulsinski_same);
    CHECK(umap::kulsinski(z10, o10, 10) ==
          doctest::Approx(golden::d_kulsinski_z_o).epsilon(TOL));

    CHECK(umap::rogers_tanimoto(golden::mf_bin1, golden::mf_bin2, 10) ==
          doctest::Approx(golden::d_rogerstanimoto_b1_b2).epsilon(TOL));
    CHECK(umap::rogers_tanimoto(golden::mf_bin1, golden::mf_bin1, 10) ==
          golden::d_rogerstanimoto_same);
    CHECK(umap::rogers_tanimoto(z10, o10, 10) ==
          doctest::Approx(golden::d_rogerstanimoto_z_o).epsilon(TOL));
}

TEST_CASE("binary metrics: russellrao / sokalmichener / sokalsneath / yule") {
    const double z10[10] = {0};
    double o10[10];
    for (int i = 0; i < 10; ++i) o10[i] = 1.0;

    CHECK(umap::russellrao(golden::mf_bin1, golden::mf_bin2, 10) ==
          doctest::Approx(golden::d_russellrao_b1_b2).epsilon(TOL));
    CHECK(umap::russellrao(golden::mf_bin1, golden::mf_bin1, 10) ==
          golden::d_russellrao_same);
    CHECK(umap::russellrao(o10, o10, 10) == golden::d_russellrao_ones);
    CHECK(umap::russellrao(z10, o10, 10) ==
          doctest::Approx(golden::d_russellrao_z_o).epsilon(TOL));

    CHECK(umap::sokal_michener(golden::mf_bin1, golden::mf_bin2, 10) ==
          doctest::Approx(golden::d_sokalmichener_b1_b2).epsilon(TOL));
    CHECK(umap::sokal_michener(golden::mf_bin1, golden::mf_bin1, 10) ==
          golden::d_sokalmichener_same);

    CHECK(umap::sokal_sneath(golden::mf_bin1, golden::mf_bin2, 10) ==
          doctest::Approx(golden::d_sokalsneath_b1_b2).epsilon(TOL));
    CHECK(umap::sokal_sneath(golden::mf_bin1, golden::mf_bin1, 10) ==
          golden::d_sokalsneath_same);
    CHECK(umap::sokal_sneath(z10, z10, 10) == golden::d_sokalsneath_z_z);

    CHECK(umap::yule(golden::mf_bin1, golden::mf_bin2, 10) ==
          doctest::Approx(golden::d_yule_b1_b2).epsilon(TOL));
    CHECK(umap::yule(golden::mf_bin1, golden::mf_bin1, 10) ==
          golden::d_yule_same);
    CHECK(umap::yule(z10, o10, 10) == golden::d_yule_z_o);
}

TEST_CASE("haversine") {
    CHECK(umap::haversine(golden::mf_h1, golden::mf_h2, 2) ==
          doctest::Approx(golden::d_haversine_h1_h2).epsilon(TOL));
    CHECK(umap::haversine(golden::mf_h1, golden::mf_h1, 2) ==
          doctest::Approx(golden::d_haversine_same).epsilon(TOL));
    CHECK_THROWS_AS(umap::haversine(golden::mf_a6, golden::mf_b6, 6),
                    std::invalid_argument);
}

TEST_CASE("poincare") {
    CHECK(umap::poincare(golden::mf_u3, golden::mf_v3, 3) ==
          doctest::Approx(golden::d_poincare_u3_v3).epsilon(TOL));
}

TEST_CASE("ll_dirichlet + helpers") {
    CHECK(umap::approx_log_Gamma(7.3) ==
          doctest::Approx(golden::d_approx_log_gamma_7_3).epsilon(TOL));
    CHECK(umap::approx_log_Gamma(1.0) == 0.0);
    CHECK(umap::log_beta(2.0, 3.0) ==
          doctest::Approx(golden::d_log_beta_2_3).epsilon(TOL));
    CHECK(umap::log_beta(7.0, 11.0) ==
          doctest::Approx(golden::d_log_beta_7_11).epsilon(TOL));
    CHECK(umap::log_single_beta(4.2) ==
          doctest::Approx(golden::d_log_single_beta_4_2).epsilon(TOL));

    CHECK(umap::ll_dirichlet(golden::mf_counts1, golden::mf_counts2, 6) ==
          doctest::Approx(golden::d_ll_dirichlet_c1_c2).epsilon(TOL));
    CHECK(umap::ll_dirichlet(golden::mf_counts2, golden::mf_counts1, 6) ==
          doctest::Approx(golden::d_ll_dirichlet_c2_c1).epsilon(TOL));
}

TEST_CASE("symmetric_kl") {
    CHECK(umap::symmetric_kl(golden::mf_p6, golden::mf_q6, 6) ==
          doctest::Approx(golden::d_symmetric_kl_p6_q6).epsilon(FTOL));
    CHECK(umap::symmetric_kl(golden::mf_counts1, golden::mf_counts2, 6) ==
          doctest::Approx(golden::d_symmetric_kl_c1_c2).epsilon(FTOL));
    CHECK(umap::symmetric_kl(golden::mf_p6, golden::mf_q6, 6, 0.01) ==
          doctest::Approx(golden::d_symmetric_kl_z01_p6_q6).epsilon(FTOL));
}

TEST_CASE("softmax_hellinger") {
    CHECK(umap::softmax_hellinger(golden::mf_a6, golden::mf_b6, 6) ==
          doctest::Approx(golden::d_softmax_hellinger_a6_b6).epsilon(TOL));
    CHECK(umap::softmax_hellinger(golden::mf_a6, golden::mf_a6, 6) ==
          golden::d_softmax_hellinger_same);
}

TEST_CASE("special discrete metrics") {
    CHECK(umap::categorical_distance(3.0, 3.0) == golden::d_categorical_eq);
    CHECK(umap::categorical_distance(3.0, 5.0) == golden::d_categorical_ne);
    CHECK(umap::ordinal_distance(2.0, 7.0, 4.5) ==
          doctest::Approx(golden::d_ordinal_2_7).epsilon(TOL));
    CHECK(umap::count_distance(3.0, 9.0, 2.5, 1.7) ==
          doctest::Approx(golden::d_count_3_9).epsilon(TOL));
    CHECK(umap::count_distance(11.0, 15.0, 2.5, 1.7) ==
          doctest::Approx(golden::d_count_11_15).epsilon(TOL));
    CHECK(umap::count_distance(1.0, 4.0, 1.0, 1.0) ==
          doctest::Approx(golden::d_count_1_4).epsilon(TOL));
    // lo = 0 hits log(0) upstream: both sides yield +inf.
    CHECK(std::isinf(umap::count_distance(0.0, 4.0, 1.0, 1.0)));

    const std::vector<std::unordered_map<double, double>> hier = {
        {{1.0, 0.0}, {2.0, 0.0}, {3.0, 1.0}},
        {{1.0, 0.0}, {2.0, 1.0}, {3.0, 2.0}}};
    CHECK(umap::hierarchical_categorical_distance(1.0, 2.0, hier) ==
          doctest::Approx(golden::d_hier_1_2).epsilon(TOL));
    CHECK(umap::hierarchical_categorical_distance(1.0, 3.0, hier) ==
          doctest::Approx(golden::d_hier_1_3).epsilon(TOL));
    CHECK(umap::hierarchical_categorical_distance(1.0, 1.0, hier) ==
          golden::d_hier_1_1);

    CHECK(umap::levenshtein("kitten", "sitting", 1.0, 20) ==
          golden::d_lev_kitten);
    CHECK(umap::levenshtein("abc", "abc", 1.0, 20) == golden::d_lev_same);
    CHECK(umap::levenshtein("", "abc", 1.0, 20) == golden::d_lev_empty);
    CHECK(umap::levenshtein("levenshtein", "meilenstein", 2.0, 20) ==
          doctest::Approx(golden::d_lev_norm).epsilon(TOL));
    CHECK(umap::levenshtein(std::string(30, 'a'), "abc", 1.5, 20) ==
          doctest::Approx(golden::d_lev_maxdist).epsilon(TOL));

    CHECK(umap::levenshtein_myers_ascii("kitten", "sitting", 1.0, 20) ==
          golden::d_myers_kitten);
    CHECK(umap::levenshtein_myers_ascii("abc", "abc", 1.0, 20) ==
          golden::d_myers_same);
    CHECK(umap::levenshtein_myers_ascii("levenshtein", "meilenstein", 2.0,
                                        20) ==
          doctest::Approx(golden::d_myers_norm).epsilon(TOL));
    // > 63 chars falls back to the DP levenshtein.
    CHECK(umap::levenshtein_myers_ascii(std::string(70, 'x'),
                                        std::string(60, 'x') + "abcdefghij",
                                        1.0, 20) == golden::d_myers_long);
}

// ------------------------------------------------------------------ gradients

TEST_CASE("euclidean_grad") {
    double g[6];
    CHECK(umap::euclidean_grad(golden::mf_a6, golden::mf_b6, 6, g) ==
          doctest::Approx(golden::gd_euclidean_a6_b6).epsilon(FTOL));
    check_grad(g, golden::gg_euclidean_a6_b6, 6, FTOL);
    const double z6[6] = {0};
    CHECK(umap::euclidean_grad(z6, z6, 6, g) == golden::gd_euclidean_z6_z6);
    check_grad(g, golden::gg_euclidean_z6_z6, 6, FTOL);
}

TEST_CASE("standardised_euclidean_grad") {
    double g[6];
    CHECK(umap::standardised_euclidean_grad(golden::mf_a6, golden::mf_b6, 6,
                                            golden::mf_s6, g) ==
          doctest::Approx(golden::gd_seuclidean_a6_b6).epsilon(FTOL));
    check_grad(g, golden::gg_seuclidean_a6_b6, 6, FTOL);
}

TEST_CASE("manhattan_grad / chebyshev_grad") {
    double g[6];
    CHECK(umap::manhattan_grad(golden::mf_a6, golden::mf_b6, 6, g) ==
          doctest::Approx(golden::gd_manhattan_a6_b6).epsilon(TOL));
    check_grad(g, golden::gg_manhattan_a6_b6, 6, TOL);
    CHECK(umap::chebyshev_grad(golden::mf_a6, golden::mf_b6, 6, g) ==
          doctest::Approx(golden::gd_chebyshev_a6_b6).epsilon(TOL));
    check_grad(g, golden::gg_chebyshev_a6_b6, 6, TOL);
}

TEST_CASE("minkowski_grad / weighted_minkowski_grad") {
    double g[6];
    CHECK(umap::minkowski_grad(golden::mf_a6, golden::mf_b6, 6, 3.0, g) ==
          doctest::Approx(golden::gd_minkowski_p3_a6_b6).epsilon(TOL));
    check_grad(g, golden::gg_minkowski_p3_a6_b6, 6, TOL);
    // S == 0: zero gradient.
    CHECK(umap::minkowski_grad(golden::mf_a6, golden::mf_a6, 6, 3.0, g) ==
          golden::gd_minkowski_p3_same);
    check_grad(g, golden::gg_minkowski_p3_same, 6, TOL);

    CHECK(umap::weighted_minkowski_grad(golden::mf_a6, golden::mf_b6, 6,
                                        golden::mf_w6, 3.0, g) ==
          doctest::Approx(golden::gd_wminkowski_p3_a6_b6).epsilon(TOL));
    check_grad(g, golden::gg_wminkowski_p3_a6_b6, 6, TOL);
}

TEST_CASE("mahalanobis_grad") {
    double g[6];
    CHECK(umap::mahalanobis_grad(golden::mf_a6, golden::mf_b6, 6,
                                 golden::mf_vinv6, g) ==
          doctest::Approx(golden::gd_mahalanobis_a6_b6).epsilon(TOL));
    check_grad(g, golden::gg_mahalanobis_a6_b6, 6, TOL);
}

TEST_CASE("canberra_grad / bray_curtis_grad") {
    double g[6];
    const double z6[6] = {0};
    CHECK(umap::canberra_grad(golden::mf_a6, golden::mf_b6, 6, g) ==
          doctest::Approx(golden::gd_canberra_a6_b6).epsilon(TOL));
    check_grad(g, golden::gg_canberra_a6_b6, 6, TOL);
    CHECK(umap::canberra_grad(z6, z6, 6, g) == golden::gd_canberra_z6_z6);
    check_grad(g, golden::gg_canberra_z6_z6, 6, TOL);

    CHECK(umap::bray_curtis_grad(golden::mf_a6, golden::mf_b6, 6, g) ==
          doctest::Approx(golden::gd_braycurtis_a6_b6).epsilon(TOL));
    check_grad(g, golden::gg_braycurtis_a6_b6, 6, TOL);
    CHECK(umap::bray_curtis_grad(z6, z6, 6, g) ==
          golden::gd_braycurtis_z6_z6);
    check_grad(g, golden::gg_braycurtis_z6_z6, 6, TOL);
}

TEST_CASE("cosine_grad / correlation_grad") {
    double g[6];
    const double z6[6] = {0};
    CHECK(umap::cosine_grad(golden::mf_a6, golden::mf_b6, 6, g) ==
          doctest::Approx(golden::gd_cosine_a6_b6).epsilon(FTOL));
    check_grad(g, golden::gg_cosine_a6_b6, 6, FTOL);
    CHECK(umap::cosine_grad(z6, golden::mf_a6, 6, g) ==
          golden::gd_cosine_z6_a6);
    check_grad(g, golden::gg_cosine_z6_a6, 6, TOL);
    CHECK(umap::cosine_grad(z6, z6, 6, g) == golden::gd_cosine_z6_z6);
    check_grad(g, golden::gg_cosine_z6_z6, 6, TOL);

    CHECK(umap::correlation_grad(golden::mf_a6, golden::mf_b6, 6, g) ==
          doctest::Approx(golden::gd_correlation_a6_b6).epsilon(FTOL));
    check_grad(g, golden::gg_correlation_a6_b6, 6, FTOL);
    CHECK(umap::correlation_grad(golden::mf_const6, golden::mf_a6, 6, g) ==
          golden::gd_correlation_const_a6);
    check_grad(g, golden::gg_correlation_const_a6, 6, TOL);
}

TEST_CASE("hellinger_grad / softmax_hellinger_grad") {
    double g[6];
    CHECK(umap::hellinger_grad(golden::mf_p6, golden::mf_q6, 6, g) ==
          doctest::Approx(golden::gd_hellinger_p6_q6).epsilon(TOL));
    check_grad(g, golden::gg_hellinger_p6_q6, 6, TOL);
    CHECK(umap::hellinger_grad(golden::mf_p6, golden::mf_p6, 6, g) ==
          golden::gd_hellinger_same);
    check_grad(g, golden::gg_hellinger_same, 6, TOL);

    CHECK(umap::softmax_hellinger_grad(golden::mf_a6, golden::mf_b6, 6, g) ==
          doctest::Approx(golden::gd_softmax_hellinger_a6_b6).epsilon(TOL));
    check_grad(g, golden::gg_softmax_hellinger_a6_b6, 6, TOL);
}

TEST_CASE("haversine_grad") {
    double g[2];
    CHECK(umap::haversine_grad(golden::mf_h1, golden::mf_h2, 2, g) ==
          doctest::Approx(golden::gd_haversine_h1_h2).epsilon(TOL));
    check_grad(g, golden::gg_haversine_h1_h2, 2, TOL);
    CHECK(umap::haversine_grad(golden::mf_h1, golden::mf_h1, 2, g) ==
          doctest::Approx(golden::gd_haversine_same).epsilon(TOL));
    check_grad(g, golden::gg_haversine_same, 2, TOL);
    CHECK_THROWS_AS(umap::haversine_grad(golden::mf_a6, golden::mf_b6, 6, g),
                    std::invalid_argument);
}

TEST_CASE("symmetric_kl_grad (mutates inputs like upstream)") {
    double x[6], y[6], g[6];
    for (int i = 0; i < 6; ++i) {
        x[i] = golden::mf_p6[i];
        y[i] = golden::mf_q6[i];
    }
    CHECK(umap::symmetric_kl_grad(x, y, 6, g) ==
          doctest::Approx(golden::gd_symmetric_kl_p6_q6).epsilon(FTOL));
    check_grad(g, golden::gg_symmetric_kl_p6_q6, 6, FTOL);
    // In-place normalisation semantics.
    check_grad(x, golden::gmut_symmetric_kl_x, 6, FTOL);
    check_grad(y, golden::gmut_symmetric_kl_y, 6, FTOL);
}

TEST_CASE("gaussian energy gradients") {
    double g[6];
    CHECK(umap::spherical_gaussian_energy_grad(golden::mf_sg1, golden::mf_sg2,
                                               3, g) ==
          doctest::Approx(golden::gd_sge_sg1_sg2).epsilon(FTOL));
    check_grad(g, golden::gg_sge_sg1_sg2, 3, FTOL);

    CHECK(umap::diagonal_gaussian_energy_grad(golden::mf_dg1, golden::mf_dg2,
                                              4, g) ==
          doctest::Approx(golden::gd_dge_dg1_dg2).epsilon(FTOL));
    check_grad(g, golden::gg_dge_dg1_dg2, 4, FTOL);
    // det == 0 branch.
    CHECK(umap::diagonal_gaussian_energy_grad(golden::mf_dg_zero1,
                                              golden::mf_dg_zero2, 4, g) ==
          doctest::Approx(golden::gd_dge_zero).epsilon(FTOL));
    check_grad(g, golden::gg_dge_zero, 4, FTOL);

    double x[5], y[5];
    for (int i = 0; i < 5; ++i) {
        x[i] = golden::mf_ge1[i];
        y[i] = golden::mf_ge2[i];
    }
    double g5[5];
    CHECK(umap::gaussian_energy_grad(x, y, 5, g5) ==
          doctest::Approx(golden::gd_ge_ge1_ge2).epsilon(FTOL));
    check_grad(g5, golden::gg_ge_ge1_ge2, 5, FTOL);
    // In-place clamping semantics (abs widths/heights, wrapped angle).
    check_grad(x, golden::gmut_ge_x, 5, FTOL);
    check_grad(y, golden::gmut_ge_y, 5, FTOL);
}

TEST_CASE("hyperboloid_grad") {
    double g[4];
    CHECK(umap::hyperboloid_grad(golden::mf_hyp4a, golden::mf_hyp4b, 4, g) ==
          doctest::Approx(golden::gd_hyperboloid_h4).epsilon(TOL));
    check_grad(g, golden::gg_hyperboloid_h4, 4, TOL);
}

// ---------------------------------------------------------- kwds + registries

TEST_CASE("make_metric binds kwds like upstream metric_kwds") {
    umap::MetricKwds kw;
    kw.p = 3.0;
    auto mink3 = umap::make_metric("minkowski", kw);
    CHECK(mink3(golden::mf_a6, golden::mf_b6, 6) ==
          umap::minkowski(golden::mf_a6, golden::mf_b6, 6, 3.0));

    umap::MetricKwds kw_se;
    kw_se.v.assign(golden::mf_s6, golden::mf_s6 + 6);
    CHECK(umap::make_metric("seuclidean", kw_se)(golden::mf_a6, golden::mf_b6,
                                                 6) ==
          doctest::Approx(golden::d_seuclidean_a6_b6).epsilon(TOL));

    umap::MetricKwds kw_wm;
    kw_wm.v.assign(golden::mf_w6, golden::mf_w6 + 6);
    kw_wm.p = 3.0;
    CHECK(umap::make_metric("wminkowski", kw_wm)(golden::mf_a6, golden::mf_b6,
                                                 6) ==
          doctest::Approx(golden::d_wminkowski_p3_a6_b6).epsilon(TOL));

    umap::MetricKwds kw_mh;
    kw_mh.vinv.assign(golden::mf_vinv6, golden::mf_vinv6 + 36);
    CHECK(umap::make_metric("mahalanobis", kw_mh)(golden::mf_a6,
                                                  golden::mf_b6, 6) ==
          doctest::Approx(golden::d_mahalanobis_a6_b6).epsilon(TOL));

    umap::MetricKwds kw_kl;
    kw_kl.z = 0.01;
    CHECK(umap::make_metric("symmetric_kl", kw_kl)(golden::mf_p6,
                                                   golden::mf_q6, 6) ==
          doctest::Approx(golden::d_symmetric_kl_z01_p6_q6).epsilon(FTOL));

    // Parameterless metric resolves to the registry entry.
    CHECK(umap::make_metric("haversine")(golden::mf_h1, golden::mf_h2, 2) ==
          doctest::Approx(golden::d_haversine_h1_h2).epsilon(TOL));

    CHECK_THROWS_AS(umap::make_metric("no_such_metric"),
                    std::invalid_argument);
    umap::MetricKwds bad;
    bad.v = {1.0, 2.0};  // wrong length for 6-d input
    CHECK_THROWS_AS(
        umap::make_metric("seuclidean", bad)(golden::mf_a6, golden::mf_b6, 6),
        std::invalid_argument);
}

TEST_CASE("make_metric_grad binds kwds") {
    double g1[6], g2[6];
    umap::MetricKwds kw;
    kw.p = 3.0;
    kw.v.assign(golden::mf_w6, golden::mf_w6 + 6);
    auto wm = umap::make_metric_grad("weighted_minkowski", kw);
    std::vector<double> x(golden::mf_a6, golden::mf_a6 + 6);
    std::vector<double> y(golden::mf_b6, golden::mf_b6 + 6);
    const double d1 = wm(x.data(), y.data(), 6, g1);
    const double d2 = umap::weighted_minkowski_grad(golden::mf_a6,
                                                    golden::mf_b6, 6,
                                                    golden::mf_w6, 3.0, g2);
    CHECK(d1 == d2);
    for (int i = 0; i < 6; ++i) CHECK(g1[i] == g2[i]);
    CHECK_THROWS_AS(umap::make_metric_grad("no_such_metric"),
                    std::invalid_argument);
}

TEST_CASE("registries mirror upstream named_distances(_with_gradients)") {
    const auto& nd = umap::named_distances();
    for (const char* key :
         {"euclidean", "l2", "manhattan", "taxicab", "l1", "chebyshev",
          "linfinity", "linfty", "linf", "minkowski", "poincare", "seuclidean",
          "standardised_euclidean", "wminkowski", "weighted_minkowski",
          "mahalanobis", "canberra", "cosine", "correlation", "hellinger",
          "softmax_hellinger", "haversine", "braycurtis", "ll_dirichlet",
          "symmetric_kl", "hamming", "jaccard", "dice", "matching",
          "kulsinski", "rogerstanimoto", "russellrao", "sokalsneath",
          "sokalmichener", "yule"}) {
        CAPTURE(key);
        CHECK(nd.count(key) == 1);
    }

    const auto& ng = umap::named_distances_with_gradients();
    for (const char* key :
         {"euclidean", "l2", "manhattan", "taxicab", "l1", "chebyshev",
          "linfinity", "linfty", "linf", "minkowski", "seuclidean",
          "standardised_euclidean", "wminkowski", "weighted_minkowski",
          "mahalanobis", "canberra", "cosine", "correlation", "hellinger",
          "softmax_hellinger", "haversine", "braycurtis", "symmetric_kl",
          "spherical_gaussian_energy", "diagonal_gaussian_energy",
          "gaussian_energy", "hyperboloid"}) {
        CAPTURE(key);
        CHECK(ng.count(key) == 1);
    }

    // Alias entries share behaviour.
    CHECK(nd.at("rogerstanimoto")(golden::mf_bin1, golden::mf_bin2, 10) ==
          doctest::Approx(golden::d_rogerstanimoto_b1_b2).epsilon(TOL));
    std::vector<double> x(golden::mf_h1, golden::mf_h1 + 2);
    std::vector<double> y(golden::mf_h2, golden::mf_h2 + 2);
    double g[2];
    CHECK(ng.at("haversine")(x.data(), y.data(), 2, g) ==
          doctest::Approx(golden::gd_haversine_h1_h2).epsilon(TOL));
}
