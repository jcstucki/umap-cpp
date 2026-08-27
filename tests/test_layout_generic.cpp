// optimize_layout_generic / optimize_layout_inverse vs the sequential numba
// reference (scripts/gen_golden_metrics_full.py), with the canonical
// non-euclidean output metric "haversine" -- bit-exact, like the euclidean
// layout port.
#include <cmath>
#include <cstddef>
#include <vector>

#include "doctest.h"
#include "golden_metrics_full.hpp"
#include "umap/layouts_generic.hpp"

namespace {

template <class T, std::size_t N>
constexpr std::size_t countof(const T (&)[N]) {
    return N;
}

std::vector<float> run_generic(int n_epochs) {
    const std::size_t n_edges = countof(golden::gen_eps);
    std::vector<int32_t> head(golden::gen_head, golden::gen_head + n_edges);
    std::vector<int32_t> tail(golden::gen_tail, golden::gen_tail + n_edges);
    std::vector<double> eps(golden::gen_eps, golden::gen_eps + n_edges);
    std::vector<float> emb(golden::gen_init,
                           golden::gen_init + golden::gen_n * 2);
    umap::optimize_layout_generic(
        emb.data(), golden::gen_n, emb.data(), 2, head, tail, n_epochs,
        golden::gen_n, eps, golden::gen_a, golden::gen_b,
        golden::gen_rng_state, umap::make_output_metric("haversine"),
        /*gamma=*/1.0, /*initial_alpha=*/1.0, /*negative_sample_rate=*/5.0,
        /*move_other=*/true);
    return emb;
}

std::vector<float> run_inverse(const umap::OutputMetricFn& metric) {
    const std::size_t n_edges = countof(golden::inv_eps);
    std::vector<int32_t> head(golden::inv_head, golden::inv_head + n_edges);
    std::vector<int32_t> tail(golden::inv_tail, golden::inv_tail + n_edges);
    std::vector<double> eps(golden::inv_eps, golden::inv_eps + n_edges);
    std::vector<float> emb(golden::inv_init,
                           golden::inv_init + golden::inv_n_new * 2);
    std::vector<float> raw(golden::inv_raw,
                           golden::inv_raw + golden::inv_n_train * 2);
    int64_t rng_state[3] = {golden::inv_rng_state[0],
                            golden::inv_rng_state[1],
                            golden::inv_rng_state[2]};
    umap::optimize_layout_inverse(
        emb.data(), golden::inv_n_new, raw.data(), 2, head, tail,
        golden::inv_weight, golden::inv_sigmas, golden::inv_rhos,
        golden::inv_n_epochs, golden::inv_n_train, eps,
        /*a=*/golden::gen_a, /*b=*/golden::gen_b, rng_state, metric,
        /*gamma=*/1.0, /*initial_alpha=*/0.25, /*negative_sample_rate=*/5.0,
        /*move_other=*/false);
    return emb;
}

}  // namespace

TEST_CASE("haversine output metric on float32 rows matches the numba f4 "
          "instantiation") {
    // Spot-check against the f8 golden (the f4 kernel only differs in the
    // float32 element reads; feeding it exactly-representable values must
    // agree with the double path).
    float x[2] = {0.5f, -0.25f};
    float y[2] = {-0.75f, 1.0f};
    double xd[2] = {0.5, -0.25};
    double yd[2] = {-0.75, 1.0};
    double g32[2], g64[2];
    const double d32 = umap::haversine_output_grad(x, y, g32);
    const double d64 = umap::haversine_grad(xd, yd, 2, g64);
    CHECK(d32 == d64);
    CHECK(g32[0] == g64[0]);
    CHECK(g32[1] == g64[1]);
}

TEST_CASE("optimize_layout_generic (haversine, sequential) 50 epochs "
          "bit-exact") {
    std::vector<float> emb = run_generic(golden::gen_n_epochs);
    for (std::size_t i = 0; i < emb.size(); ++i) {
        CAPTURE(i);
        CHECK(emb[i] == golden::gen_final50[i]);  // bit-exact float32
    }
}

TEST_CASE("optimize_layout_generic (haversine, sequential) 200 epochs "
          "bit-exact") {
    std::vector<float> emb = run_generic(200);
    for (std::size_t i = 0; i < emb.size(); ++i) {
        CAPTURE(i);
        CHECK(emb[i] == golden::gen_final200[i]);
    }
}

TEST_CASE("optimize_layout_inverse (haversine, sequential) bit-exact") {
    std::vector<float> emb = run_inverse(umap::make_output_metric("haversine"));
    for (std::size_t i = 0; i < emb.size(); ++i) {
        CAPTURE(i);
        CHECK(emb[i] == golden::inv_final_hav[i]);
    }
}

TEST_CASE("optimize_layout_inverse (euclidean, sequential) bit-exact") {
    std::vector<float> emb = run_inverse(umap::make_output_metric("euclidean"));
    for (std::size_t i = 0; i < emb.size(); ++i) {
        CAPTURE(i);
        CHECK(emb[i] == golden::inv_final_euc[i]);
    }
}
