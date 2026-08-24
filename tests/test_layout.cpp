// Layout SGD (layouts.py port) vs golden values generated from the reference
// numba implementation via scripts/gen_golden_layout.py, plus qualitative
// cluster-preservation checks.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include "doctest.h"
#include "golden_layout.hpp"
#include "umap/fuzzy_simplicial_set.hpp"
#include "umap/layouts.hpp"
#include "umap/sparse_matrix.hpp"

namespace {

template <typename T, std::size_t N>
constexpr std::size_t countof(const T (&)[N]) {
    return N;
}

umap::SparseMatrix golden_graph() {
    umap::SparseMatrix g(golden::graph_n, golden::graph_n);
    g.indptr.assign(golden::graph_indptr,
                    golden::graph_indptr + golden::graph_n + 1);
    g.indices.assign(golden::graph_indices,
                     golden::graph_indices + countof(golden::graph_indices));
    g.data.assign(golden::graph_data,
                  golden::graph_data + countof(golden::graph_data));
    return g;
}

double max_abs_diff(const std::vector<float>& got, const float* want,
                    std::size_t n) {
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - want[i]);
        if (d > m) m = d;
    }
    return m;
}

}  // namespace

TEST_CASE("clip matches reference exactly") {
    for (std::size_t i = 0; i < countof(golden::clip_in); ++i) {
        CHECK(umap::clip(golden::clip_in[i]) == golden::clip_out[i]);
    }
}

TEST_CASE("rdist matches reference float32 arithmetic exactly") {
    CHECK(umap::rdist(golden::rdist_x2, golden::rdist_y2, 2) ==
          golden::rdist_out2);
    CHECK(umap::rdist(golden::rdist_x5, golden::rdist_y5, 5) ==
          golden::rdist_out5);
    CHECK(umap::rdist(golden::rdist_x5, golden::rdist_x5, 5) ==
          golden::rdist_out_same);
}

TEST_CASE("make_epochs_per_sample matches reference exactly") {
    const std::size_t n = countof(golden::meps_weights);
    const struct {
        int n_epochs;
        const double* want;
    } cases[] = {
        {500, golden::meps_out_500},
        {10, golden::meps_out_10},
        {1, golden::meps_out_1},
    };
    for (const auto& c : cases) {
        auto got = umap::make_epochs_per_sample(golden::meps_weights, n,
                                                c.n_epochs);
        REQUIRE(got.size() == n);
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(got[i] == c.want[i]);  // bit-exact float64
        }
    }
}

TEST_CASE("optimize_layout_euclidean matches sequential numba reference") {
    const std::size_t n_edges = countof(golden::layout_eps);
    REQUIRE(countof(golden::layout_head) == n_edges);
    REQUIRE(countof(golden::layout_tail) == n_edges);

    std::vector<int32_t> head(golden::layout_head, golden::layout_head + n_edges);
    std::vector<int32_t> tail(golden::layout_tail, golden::layout_tail + n_edges);
    std::vector<double> eps(golden::layout_eps, golden::layout_eps + n_edges);
    std::vector<float> emb(golden::layout_init,
                           golden::layout_init + countof(golden::layout_init));

    umap::optimize_layout_euclidean(
        emb.data(), golden::graph_n, emb.data(), 2, head, tail,
        golden::layout_n_epochs, golden::graph_n, eps, golden::layout_a,
        golden::layout_b, golden::layout_rng_state, /*gamma=*/1.0,
        /*initial_alpha=*/1.0, /*negative_sample_rate=*/5.0,
        /*parallel=*/false, /*densmap=*/false, /*move_other=*/true);

    const double mad = max_abs_diff(emb, golden::layout_final, emb.size());
    INFO("max abs diff vs Python (50 epochs): " << mad);
    CHECK(mad <= 1e-4);
    // Should in fact be bit-exact: every operation mirrors the numba path.
    WARN_MESSAGE(mad == 0.0, "not bit-exact; max abs diff = ", mad);
}

TEST_CASE("simplicial_set_embedding glue matches reference (500 epochs)") {
    const umap::SparseMatrix graph = golden_graph();
    const std::size_t n_coords = countof(golden::glue_init);

    auto emb = umap::simplicial_set_embedding_from_init(
        graph, golden::glue_init, 2, golden::layout_a, golden::layout_b,
        golden::glue_rng_state, /*n_epochs=*/-1);

    REQUIRE(emb.size() == n_coords);
    const double mad = max_abs_diff(emb, golden::glue_final, n_coords);
    INFO("max abs diff vs Python (500 epochs): " << mad);
    CHECK(mad <= 1e-4);
    WARN_MESSAGE(mad == 0.0, "not bit-exact; max abs diff = ", mad);
}

TEST_CASE("simplicial_set_embedding glue internals match reference") {
    // Rescaled init: exercised indirectly above; check it directly by running
    // zero epochs so only the [0, 10] rescale is applied.
    const umap::SparseMatrix graph = golden_graph();
    auto rescaled = umap::simplicial_set_embedding_from_init(
        graph, golden::glue_init, 2, golden::layout_a, golden::layout_b,
        golden::glue_rng_state, /*n_epochs=*/0);
    // NOTE: n_epochs=0 uses the default-epochs (500) filter threshold like the
    // reference `else` branch, and runs no SGD epochs.
    REQUIRE(rescaled.size() == countof(golden::glue_rescaled_init));
    for (std::size_t i = 0; i < rescaled.size(); ++i) {
        CHECK(rescaled[i] == golden::glue_rescaled_init[i]);
    }

    // Filtered-graph edge weights -> epochs_per_sample (exact), and head order.
    umap::SparseMatrix g = graph;
    if (!g.data.empty()) {
        float mx = g.data[0];
        for (float v : g.data) mx = v > mx ? v : mx;
        const float threshold = mx / static_cast<float>(golden::glue_n_epochs);
        for (float& v : g.data) {
            if (v < threshold) v = 0.0f;
        }
    }
    g.eliminate_zeros();
    auto eps = umap::make_epochs_per_sample(g.data.data(), g.data.size(),
                                            golden::glue_n_epochs);
    REQUIRE(eps.size() == countof(golden::glue_eps));
    for (std::size_t i = 0; i < eps.size(); ++i) {
        CHECK(eps[i] == golden::glue_eps[i]);
    }
    std::size_t e = 0;
    for (int32_t row = 0; row < g.n_rows; ++row) {
        for (int32_t idx = g.indptr[row]; idx < g.indptr[row + 1]; ++idx, ++e) {
            REQUIRE(e < countof(golden::glue_head));
            CHECK(row == golden::glue_head[e]);
        }
    }
    CHECK(e == countof(golden::glue_head));
}

namespace {

// Deterministic pseudo-random doubles in [0, 1) (splitmix64-based).
struct TestRng {
    uint64_t s;
    explicit TestRng(uint64_t seed) : s(seed) {}
    double next() {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<double>(z >> 11) / 9007199254740992.0;
    }
    // Box-Muller normal
    double normal() {
        double u1 = next();
        double u2 = next();
        while (u1 <= 1e-12) u1 = next();
        return std::sqrt(-2.0 * std::log(u1)) *
               std::cos(2.0 * 3.14159265358979323846 * u2);
    }
};

struct ClusterFixture {
    std::vector<float> data;   // n x 2
    std::vector<float> init;   // n x 2 jittered data
    umap::SparseMatrix graph;
    int n = 60;

    ClusterFixture() {
        TestRng rng(20260824);
        const int half = n / 2;
        data.resize(static_cast<std::size_t>(n) * 2);
        for (int i = 0; i < n; ++i) {
            const double cx = i < half ? 0.0 : 20.0;
            const double cy = i < half ? 0.0 : 20.0;
            data[2 * i + 0] = static_cast<float>(cx + rng.normal());
            data[2 * i + 1] = static_cast<float>(cy + rng.normal());
        }
        // jittered init from the data coordinates
        init = data;
        for (auto& v : init) v += static_cast<float>(0.05 * rng.normal());

        // brute-force kNN (k = 8, self included, sorted ascending)
        const int k = 8;
        std::vector<int32_t> knn_idx(static_cast<std::size_t>(n) * k);
        std::vector<float> knn_dist(static_cast<std::size_t>(n) * k);
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<double, int>> d(n);
            for (int j = 0; j < n; ++j) {
                const double dx = static_cast<double>(data[2 * i]) - data[2 * j];
                const double dy =
                    static_cast<double>(data[2 * i + 1]) - data[2 * j + 1];
                d[j] = {std::sqrt(dx * dx + dy * dy), j};
            }
            std::partial_sort(d.begin(), d.begin() + k, d.end());
            for (int m = 0; m < k; ++m) {
                knn_idx[static_cast<std::size_t>(i) * k + m] = d[m].second;
                knn_dist[static_cast<std::size_t>(i) * k + m] =
                    static_cast<float>(d[m].first);
            }
        }
        graph = umap::fuzzy_simplicial_set(n, knn_idx.data(), knn_dist.data(), k)
                    .graph;
    }
};

// centroid distance vs cluster spreads in the embedding
bool clusters_separated(const std::vector<float>& emb, int n) {
    const int half = n / 2;
    double c[2][2] = {{0, 0}, {0, 0}};
    for (int i = 0; i < n; ++i) {
        c[i < half ? 0 : 1][0] += emb[2 * i] / half;
        c[i < half ? 0 : 1][1] += emb[2 * i + 1] / half;
    }
    double r[2] = {0, 0};
    for (int i = 0; i < n; ++i) {
        const int g = i < half ? 0 : 1;
        const double dx = emb[2 * i] - c[g][0];
        const double dy = emb[2 * i + 1] - c[g][1];
        r[g] += (dx * dx + dy * dy) / half;
    }
    const double rms0 = std::sqrt(r[0]);
    const double rms1 = std::sqrt(r[1]);
    const double dx = c[0][0] - c[1][0];
    const double dy = c[0][1] - c[1][1];
    const double centroid_dist = std::sqrt(dx * dx + dy * dy);
    return centroid_dist > 2.0 * (rms0 + rms1);
}

}  // namespace

TEST_CASE("two gaussian clusters remain separated after optimization") {
    ClusterFixture fx;
    const int64_t rng_state[3] = {1234567, -7654321, 55555};
    auto emb = umap::simplicial_set_embedding_from_init(
        fx.graph, fx.init.data(), 2, golden::layout_a, golden::layout_b,
        rng_state);
    for (float v : emb) CHECK(std::isfinite(v));
    CHECK(clusters_separated(emb, fx.n));
}

TEST_CASE("parallel (hogwild) mode runs and preserves cluster structure") {
    ClusterFixture fx;
    const int64_t rng_state[3] = {1234567, -7654321, 55555};
    auto emb = umap::simplicial_set_embedding_from_init(
        fx.graph, fx.init.data(), 2, golden::layout_a, golden::layout_b,
        rng_state, /*n_epochs=*/-1, /*gamma=*/1.0, /*initial_alpha=*/1.0,
        /*negative_sample_rate=*/5.0, /*parallel=*/true);
    for (float v : emb) CHECK(std::isfinite(v));
    CHECK(clusters_separated(emb, fx.n));
}
