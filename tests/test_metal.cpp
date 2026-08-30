// Apple Metal GPU kernels (Phase 3, branch metal-gpu): brute-force kNN vs the
// CPU exact kNN, and GPU SGD layout parity with the CPU path at the
// embedding-quality level (not bitwise — the GPU path has numba-prange hogwild
// race semantics; see include/umap/metal/gpu_layout.hpp).
//
// Only compiled when UMAP_ENABLE_METAL; each case skips gracefully when no
// Metal device / metallib is available at runtime.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "doctest.h"
#include "umap/device.hpp"
#include "umap/distances.hpp"
#include "umap/find_ab_params.hpp"
#include "umap/fuzzy_simplicial_set.hpp"
#include "umap/layouts.hpp"
#include "umap/metal/gpu_knn.hpp"
#include "umap/metal/gpu_layout.hpp"
#include "umap/sparse_matrix.hpp"
// --- phase4 additions --------------------------------------------------------
#include "trustworthiness.hpp"
#include "umap/umap.hpp"
// -----------------------------------------------------------------------------

#define METAL_REQUIRED()                                        \
    do {                                                        \
        if (!umap::metal_available()) {                         \
            MESSAGE("no Metal device/metallib: skipping test"); \
            return;                                             \
        }                                                       \
    } while (0)

namespace {

// Deterministic pseudo-random doubles in [0, 1) (splitmix64-based), matching
// the generator used in tests/test_layout.cpp.
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
    double normal() {
        double u1 = next();
        double u2 = next();
        while (u1 <= 1e-12) u1 = next();
        return std::sqrt(-2.0 * std::log(u1)) *
               std::cos(2.0 * 3.14159265358979323846 * u2);
    }
};

std::vector<float> random_matrix(int n, int dim, uint64_t seed) {
    TestRng rng(seed);
    std::vector<float> m(static_cast<std::size_t>(n) * dim);
    for (auto& v : m) v = static_cast<float>(rng.next());
    return m;
}

double cpu_distance(const float* x, const float* y, int dim, bool cosine) {
    std::vector<double> xd(x, x + dim), yd(y, y + dim);
    return cosine ? umap::cosine(xd.data(), yd.data(), dim)
                  : umap::euclidean(xd.data(), yd.data(), dim);
}

// CPU exact kNN (double precision, ascending, self included).
void cpu_brute_force_knn(const float* data, int n, int dim,
                         const float* queries, int nq, int k, bool cosine,
                         std::vector<int32_t>& idx, std::vector<float>& dist) {
    idx.assign(static_cast<std::size_t>(nq) * k, -1);
    dist.assign(static_cast<std::size_t>(nq) * k, 0.0f);
    std::vector<std::pair<double, int>> d(n);
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < n; ++j) {
            d[j] = {cpu_distance(queries + static_cast<std::size_t>(i) * dim,
                                 data + static_cast<std::size_t>(j) * dim, dim,
                                 cosine),
                    j};
        }
        std::partial_sort(d.begin(), d.begin() + k, d.end());
        for (int m = 0; m < k; ++m) {
            idx[static_cast<std::size_t>(i) * k + m] = d[m].second;
            dist[static_cast<std::size_t>(i) * k + m] =
                static_cast<float>(d[m].first);
        }
    }
}

void check_knn_matches(const float* data, int n, int dim, const float* queries,
                       int nq, int k, bool cosine,
                       const std::vector<int32_t>& gpu_idx,
                       const std::vector<float>& gpu_dist) {
    std::vector<int32_t> cpu_idx;
    std::vector<float> cpu_dist;
    cpu_brute_force_knn(data, n, dim, queries, nq, k, cosine, cpu_idx,
                        cpu_dist);
    int idx_mismatches = 0;
    double max_dist_diff = 0.0;
    for (int i = 0; i < nq; ++i) {
        for (int m = 0; m < k; ++m) {
            const std::size_t e = static_cast<std::size_t>(i) * k + m;
            max_dist_diff = std::max(
                max_dist_diff,
                std::fabs(static_cast<double>(cpu_dist[e]) - gpu_dist[e]));
            if (cpu_idx[e] != gpu_idx[e]) {
                // Allowed only for genuine ties: the CPU distance to the
                // GPU-chosen neighbour must equal the CPU distance at rank m.
                const double d_gpu_choice = cpu_distance(
                    queries + static_cast<std::size_t>(i) * dim,
                    data + static_cast<std::size_t>(gpu_idx[e]) * dim, dim,
                    cosine);
                if (std::fabs(d_gpu_choice - cpu_dist[e]) > 1e-5) {
                    ++idx_mismatches;
                }
            }
        }
    }
    INFO("max |cpu - gpu| distance diff: " << max_dist_diff);
    CHECK(max_dist_diff <= 1e-5);
    CHECK(idx_mismatches == 0);
}

}  // namespace

TEST_CASE("gpu_brute_force_knn matches CPU exact kNN (500x8)") {
    METAL_REQUIRED();
    const int n = 500, dim = 8, k = 10;
    const auto data = random_matrix(n, dim, 20260825);

    for (const bool cosine : {false, true}) {
        CAPTURE(cosine);
        std::vector<int32_t> gpu_idx(static_cast<std::size_t>(n) * k);
        std::vector<float> gpu_dist(static_cast<std::size_t>(n) * k);
        const bool ok = umap::metal::gpu_brute_force_knn(
            data.data(), n, dim, data.data(), n, k,
            cosine ? umap::metal::KnnMetric::cosine
                   : umap::metal::KnnMetric::euclidean,
            gpu_idx.data(), gpu_dist.data());
        REQUIRE(ok);
        // Self-kNN: each row's nearest neighbour is itself at distance 0.
        for (int i = 0; i < n; ++i) {
            CHECK(gpu_idx[static_cast<std::size_t>(i) * k] == i);
        }
        check_knn_matches(data.data(), n, dim, data.data(), n, k, cosine,
                          gpu_idx, gpu_dist);
    }
}

TEST_CASE("gpu_brute_force_knn with distinct query set") {
    METAL_REQUIRED();
    const int n = 300, dim = 8, k = 5, nq = 100;
    const auto data = random_matrix(n, dim, 7);
    const auto queries = random_matrix(nq, dim, 99);

    for (const bool cosine : {false, true}) {
        CAPTURE(cosine);
        std::vector<int32_t> gpu_idx(static_cast<std::size_t>(nq) * k);
        std::vector<float> gpu_dist(static_cast<std::size_t>(nq) * k);
        const bool ok = umap::metal::gpu_brute_force_knn(
            data.data(), n, dim, queries.data(), nq, k,
            cosine ? umap::metal::KnnMetric::cosine
                   : umap::metal::KnnMetric::euclidean,
            gpu_idx.data(), gpu_dist.data());
        REQUIRE(ok);
        check_knn_matches(data.data(), n, dim, queries.data(), nq, k, cosine,
                          gpu_idx, gpu_dist);
    }
}

namespace {

// Gaussian-cluster fixture: `centers` cluster centres in 2D, `per` points
// each; jittered init; fuzzy graph from brute-force kNN.
struct ClusterFixture {
    int n = 0;
    int n_clusters = 0;
    std::vector<float> data;  // n x 2
    std::vector<float> init;  // n x 2
    umap::SparseMatrix graph{0, 0};

    ClusterFixture(std::vector<std::pair<double, double>> centers, int per,
                   uint64_t seed) {
        n_clusters = static_cast<int>(centers.size());
        n = n_clusters * per;
        TestRng rng(seed);
        data.resize(static_cast<std::size_t>(n) * 2);
        for (int i = 0; i < n; ++i) {
            const auto& c = centers[static_cast<std::size_t>(i / per)];
            data[2 * i + 0] = static_cast<float>(c.first + rng.normal());
            data[2 * i + 1] = static_cast<float>(c.second + rng.normal());
        }
        init = data;
        for (auto& v : init) v += static_cast<float>(0.05 * rng.normal());

        const int k = 8;
        std::vector<int32_t> knn_idx;
        std::vector<float> knn_dist;
        cpu_brute_force_knn(data.data(), n, 2, data.data(), n, k, false,
                            knn_idx, knn_dist);
        graph = umap::fuzzy_simplicial_set(n, knn_idx.data(), knn_dist.data(),
                                           k)
                    .graph;
    }

    int cluster_of(int i) const { return i / (n / n_clusters); }
};

// The simplicial_set_embedding glue re-done by hand so the GPU optimizer can
// be driven with exactly the same inputs the CPU glue uses: threshold-filter
// the graph, COO head/tail, epochs_per_sample, [0, 10] rescaled init.
struct GlueParts {
    std::vector<int32_t> head, tail;
    std::vector<double> eps;
    std::vector<float> init;  // rescaled
};

GlueParts make_glue(const umap::SparseMatrix& graph_in, const float* init,
                    int n_components, int n_epochs) {
    GlueParts out;
    umap::SparseMatrix g = graph_in;
    if (!g.data.empty()) {
        float mx = g.data[0];
        for (float v : g.data) mx = v > mx ? v : mx;
        const float threshold = mx / static_cast<float>(n_epochs);
        for (float& v : g.data) {
            if (v < threshold) v = 0.0f;
        }
    }
    g.eliminate_zeros();
    for (int32_t row = 0; row < g.n_rows; ++row) {
        for (int32_t idx = g.indptr[row]; idx < g.indptr[row + 1]; ++idx) {
            out.head.push_back(row);
            out.tail.push_back(g.indices[idx]);
        }
    }
    out.eps = umap::make_epochs_per_sample(g.data.data(), g.data.size(),
                                           n_epochs);
    // [0, 10] per-component float32 rescale (identical to the CPU glue).
    out.init.assign(init, init + static_cast<std::size_t>(g.n_rows) *
                                     n_components);
    for (int d = 0; d < n_components; ++d) {
        float mn = out.init[d], mx = out.init[d];
        for (int i = 1; i < g.n_rows; ++i) {
            const float v =
                out.init[static_cast<std::size_t>(i) * n_components + d];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        const float span = mx - mn;
        for (int i = 0; i < g.n_rows; ++i) {
            float& v = out.init[static_cast<std::size_t>(i) * n_components + d];
            v = (10.0f * (v - mn)) / span;
        }
    }
    return out;
}

// Centroid distance over the summed cluster RMS spreads (the
// tests/test_layout.cpp criterion, exposed as the raw ratio so the racy GPU
// path can assert on order statistics of it — see the flake note below).
double separation_ratio(const std::vector<float>& emb,
                        const ClusterFixture& fx) {
    REQUIRE(fx.n_clusters == 2);
    const int half = fx.n / 2;
    double c[2][2] = {{0, 0}, {0, 0}};
    for (int i = 0; i < fx.n; ++i) {
        c[i < half ? 0 : 1][0] += emb[2 * i] / half;
        c[i < half ? 0 : 1][1] += emb[2 * i + 1] / half;
    }
    double r[2] = {0, 0};
    for (int i = 0; i < fx.n; ++i) {
        const int g = i < half ? 0 : 1;
        const double dx = emb[2 * i] - c[g][0];
        const double dy = emb[2 * i + 1] - c[g][1];
        r[g] += (dx * dx + dy * dy) / half;
    }
    const double dx = c[0][0] - c[1][0];
    const double dy = c[0][1] - c[1][1];
    return std::sqrt(dx * dx + dy * dy) /
           (std::sqrt(r[0]) + std::sqrt(r[1]));
}

bool clusters_separated(const std::vector<float>& emb,
                        const ClusterFixture& fx) {
    return separation_ratio(emb, fx) > 2.0;
}

// Embedding-quality proxy: mean intra-cluster vs mean inter-cluster pairwise
// distance in the 2D embedding.
std::pair<double, double> intra_inter(const std::vector<float>& emb,
                                      const ClusterFixture& fx) {
    double intra = 0.0, inter = 0.0;
    long n_intra = 0, n_inter = 0;
    for (int i = 0; i < fx.n; ++i) {
        for (int j = i + 1; j < fx.n; ++j) {
            const double dx = emb[2 * i] - emb[2 * j];
            const double dy = emb[2 * i + 1] - emb[2 * j + 1];
            const double d = std::sqrt(dx * dx + dy * dy);
            if (fx.cluster_of(i) == fx.cluster_of(j)) {
                intra += d;
                ++n_intra;
            } else {
                inter += d;
                ++n_inter;
            }
        }
    }
    return {intra / static_cast<double>(n_intra),
            inter / static_cast<double>(n_inter)};
}

std::vector<float> run_gpu_layout(const ClusterFixture& fx, int n_epochs,
                                  double a, double b,
                                  const int64_t rng_state[3]) {
    GlueParts glue = make_glue(fx.graph, fx.init.data(), 2, n_epochs);
    std::vector<float> emb = glue.init;
    const bool ok = umap::metal::gpu_optimize_layout_euclidean(
        emb.data(), fx.n, emb.data(), 2, glue.head, glue.tail, n_epochs, fx.n,
        glue.eps, a, b, rng_state, /*gamma=*/1.0, /*initial_alpha=*/1.0,
        /*negative_sample_rate=*/5.0, /*move_other=*/true);
    REQUIRE(ok);
    return emb;
}

}  // namespace

TEST_CASE("gpu sgd: embeddings finite, clusters separate like the CPU path") {
    METAL_REQUIRED();
    ClusterFixture fx({{0.0, 0.0}, {20.0, 20.0}}, 30, 20260824);
    const int64_t rng_state[3] = {1234567, -7654321, 55555};
    const auto ab = umap::find_ab_params(1.0, 0.1);
    const double a = ab.a, b = ab.b;
    const int n_epochs = 500;

    // CPU sequential reference (same graph + same init, via the CPU glue);
    // deterministic, ratio 7.09.
    const auto cpu_emb = umap::simplicial_set_embedding_from_init(
        fx.graph, fx.init.data(), 2, a, b, rng_state, n_epochs);
    REQUIRE(clusters_separated(cpu_emb, fx));

    // The GPU ratio is a RANDOM VARIABLE (hogwild races): measured over 310
    // runs on M-series (2026-08-29, fixed and per-run-varied rng_state):
    // mean 4.6, sd 1.7, observed range [1.01, 9.83], p(ratio < 2.0) ~ 2.6%,
    // p(< 1.5) ~ 1.0%. A single-run `ratio > 2` check therefore flaked about
    // once per ~40 full-suite runs (the rare 1-of-263k-assertions failure).
    // Restructured to order statistics of 5 independent runs (rng_state
    // varied per run): P(median < 1.5) ~ 1e-5 and P(max < 2.0) ~ 1e-8, while
    // a broken kernel still fails hard — flipping the attractive-gradient
    // sign in sgd_epoch drives all 5 ratios to 0.14-0.20 (verified
    // experimentally, then reverted).
    std::vector<double> ratios;
    for (int rep = 0; rep < 5; ++rep) {
        const int64_t rs[3] = {rng_state[0], rng_state[1],
                               rng_state[2] + 7919 * rep};
        const auto gpu_emb = run_gpu_layout(fx, n_epochs, a, b, rs);
        REQUIRE(gpu_emb.size() == cpu_emb.size());
        for (float v : gpu_emb) {
            CHECK(std::isfinite(v));
            // init is in [0, 10]; per-step moves are bounded by |clip| *
            // alpha <= 4, so coordinates stay in a modest range unless
            // updates blew up.
            CHECK(std::fabs(v) < 1e3);
        }
        ratios.push_back(separation_ratio(gpu_emb, fx));
    }
    std::sort(ratios.begin(), ratios.end());
    CAPTURE(ratios[0]);
    CAPTURE(ratios[1]);
    CAPTURE(ratios[2]);
    CAPTURE(ratios[3]);
    CAPTURE(ratios[4]);
    MESSAGE("gpu separation ratios (sorted): " << ratios[0] << " " << ratios[1]
            << " " << ratios[2] << " " << ratios[3] << " " << ratios[4]
            << " (cpu: " << separation_ratio(cpu_emb, fx) << ")");
    CHECK(ratios[2] > 1.5);  // median: typical run separates the clusters
    CHECK(ratios[4] > 2.0);  // max: CPU-grade separation is reached
}

TEST_CASE("gpu sgd quality proxy: 3 clusters, intra << inter for CPU and GPU") {
    METAL_REQUIRED();
    ClusterFixture fx({{0.0, 0.0}, {20.0, 0.0}, {0.0, 20.0}}, 30, 424242);
    const int64_t rng_state[3] = {987654321, -12345, 424242};
    const auto ab = umap::find_ab_params(1.0, 0.1);
    const double a = ab.a, b = ab.b;
    const int n_epochs = 500;

    const auto cpu_emb = umap::simplicial_set_embedding_from_init(
        fx.graph, fx.init.data(), 2, a, b, rng_state, n_epochs);
    const auto cpu_ii = intra_inter(cpu_emb, fx);
    const double cpu_intra = cpu_ii.first, cpu_inter = cpu_ii.second;
    CHECK(cpu_inter > 2.0 * cpu_intra);  // deterministic (10.5x)

    // Same flake-hardening as the 2-cluster case: the GPU inter/intra ratio
    // measured over 310 runs — mean 7.3, sd 1.9, observed min 2.36 — so the
    // single-run `> 2` check sat ~1 sd from the observed floor. Median of 3
    // independent runs: P(median < 2.0) ~ 3e-5, and a broken kernel
    // (attractive-gradient sign flip) fails every run at 0.99-1.00
    // (verified, reverted).
    std::vector<double> ratios;
    for (int rep = 0; rep < 3; ++rep) {
        const int64_t rs[3] = {rng_state[0], rng_state[1],
                               rng_state[2] + 7919 * rep};
        const auto gpu_emb = run_gpu_layout(fx, n_epochs, a, b, rs);
        for (float v : gpu_emb) CHECK(std::isfinite(v));
        const auto gpu_ii = intra_inter(gpu_emb, fx);
        ratios.push_back(gpu_ii.second / gpu_ii.first);
    }
    std::sort(ratios.begin(), ratios.end());
    CAPTURE(ratios[0]);
    CAPTURE(ratios[1]);
    CAPTURE(ratios[2]);
    MESSAGE("quality proxy — CPU inter/intra: " << cpu_inter / cpu_intra
            << ", GPU inter/intra (sorted): " << ratios[0] << " " << ratios[1]
            << " " << ratios[2]);
    CHECK(ratios[1] > 2.0);  // median of 3
}

// Rough CPU-vs-GPU timing on a 5k-point SGD; hidden by default (run with
// `./umap_tests -ns -tc="gpu sgd timing*"`).
TEST_CASE("gpu sgd timing on 5k points" * doctest::skip()) {
    METAL_REQUIRED();
    const int n = 5000, dim = 8, k = 15, n_epochs = 200;
    const auto data = random_matrix(n, dim, 5);
    std::vector<int32_t> knn_idx(static_cast<std::size_t>(n) * k);
    std::vector<float> knn_dist(static_cast<std::size_t>(n) * k);
    REQUIRE(umap::metal::gpu_brute_force_knn(
        data.data(), n, dim, data.data(), n, k,
        umap::metal::KnnMetric::euclidean, knn_idx.data(), knn_dist.data()));
    const auto graph =
        umap::fuzzy_simplicial_set(n, knn_idx.data(), knn_dist.data(), k)
            .graph;

    const auto init = random_matrix(n, 2, 6);
    const auto ab = umap::find_ab_params(1.0, 0.1);
    const double a = ab.a, b = ab.b;
    const int64_t rng_state[3] = {1234567, -7654321, 55555};
    GlueParts glue = make_glue(graph, init.data(), 2, n_epochs);
    MESSAGE("edges: " << glue.head.size());

    using clock = std::chrono::steady_clock;
    std::vector<float> cpu_emb = glue.init;
    auto t0 = clock::now();
    umap::optimize_layout_euclidean(cpu_emb.data(), n, cpu_emb.data(), 2,
                                    glue.head, glue.tail, n_epochs, n,
                                    glue.eps, a, b, rng_state, 1.0, 1.0, 5.0,
                                    /*parallel=*/false, /*densmap=*/false,
                                    /*move_other=*/true);
    auto t1 = clock::now();
    std::vector<float> gpu_emb = glue.init;
    REQUIRE(umap::metal::gpu_optimize_layout_euclidean(
        gpu_emb.data(), n, gpu_emb.data(), 2, glue.head, glue.tail, n_epochs,
        n, glue.eps, a, b, rng_state, 1.0, 1.0, 5.0, /*move_other=*/true));
    auto t2 = clock::now();
    const auto ms = [](auto d) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
    };
    MESSAGE("CPU sequential: " << ms(t1 - t0) << " ms, GPU: " << ms(t2 - t1)
                               << " ms (" << n_epochs << " epochs)");
    for (float v : gpu_emb) CHECK(std::isfinite(v));
}

// --- phase4 additions --------------------------------------------------------
// Runtime device dispatch through the UMAP class (UMAPParams::device).

namespace {

// n_per x len(centers) points, gaussian spread 1.0 around 6-d centers.
std::vector<float> three_cluster_data(int n_per, uint64_t seed,
                                      double separation = 30.0) {
    TestRng rng(seed);
    const int dim = 6;
    std::vector<std::array<double, 6>> centers = {
        {0, 0, 0, 0, 0, 0},
        {separation, 0, 0, separation, 0, 0},
        {0, separation, 0, 0, separation, 0}};
    std::vector<float> X;
    X.reserve(static_cast<std::size_t>(n_per) * centers.size() * dim);
    for (const auto& c : centers) {
        for (int i = 0; i < n_per; ++i) {
            for (int d = 0; d < dim; ++d)
                X.push_back(static_cast<float>(c[d] + rng.normal()));
        }
    }
    return X;
}

}  // namespace

TEST_CASE("phase4 dispatch: device=GPU fit reaches CPU-comparable "
          "trustworthiness on 3 clusters") {
    // The default device must be CPU (every bit-exact golden in the suite
    // fits with UMAPParams{} and proves the CPU path is untouched).
    CHECK(umap::UMAPParams{}.device == umap::Device::CPU);
    METAL_REQUIRED();

    const int n_per = 80, dim = 6, n = 3 * n_per;
    const auto X = three_cluster_data(n_per, 20260828);

    umap::UMAPParams p;
    p.n_neighbors = 10;
    p.init = "random";
    p.random_state = 42;

    umap::UMAP cpu_model(p);
    cpu_model.fit(X.data(), n, dim);

    p.device = umap::Device::GPU;
    umap::UMAP gpu_model(p);
    gpu_model.fit(X.data(), n, dim);

    // Small-data exact kNN on the GPU: neighbour indices identical to the
    // sklearn-replica CPU path (no exact ties in gaussian data), distances to
    // float32-kernel precision.
    REQUIRE(gpu_model.knn_indices().size() == cpu_model.knn_indices().size());
    std::size_t idx_mismatches = 0;
    float max_dist_diff = 0.0f;
    for (std::size_t i = 0; i < cpu_model.knn_indices().size(); ++i) {
        if (gpu_model.knn_indices()[i] != cpu_model.knn_indices()[i])
            ++idx_mismatches;
        max_dist_diff =
            std::max(max_dist_diff, std::fabs(gpu_model.knn_dists()[i] -
                                              cpu_model.knn_dists()[i]));
    }
    CHECK(idx_mismatches == 0);
    CHECK(max_dist_diff <= 1e-3f);

    for (float v : gpu_model.embedding()) CHECK(std::isfinite(v));
    const double trust_cpu = testutil::trustworthiness(
        X.data(), n, dim, cpu_model.embedding().data(), 2, 10);
    const double trust_gpu = testutil::trustworthiness(
        X.data(), n, dim, gpu_model.embedding().data(), 2, 10);
    CAPTURE(trust_cpu);
    CAPTURE(trust_gpu);
    MESSAGE("trustworthiness — CPU: " << trust_cpu << ", GPU: " << trust_gpu);
    CHECK(trust_gpu >= 0.90);
    CHECK(trust_gpu >= trust_cpu - 0.05);

    // transform with device=GPU (GPU kNN + CPU move_other=false SGD): every
    // held-out query should land nearer its own cluster's embedded centroid
    // than any other (allow a couple of stragglers).
    const int nq_per = 10, nq = 3 * nq_per;
    const auto Q = three_cluster_data(nq_per, 977001);
    const auto q_emb = gpu_model.transform(Q.data(), nq);
    REQUIRE(q_emb.size() == static_cast<std::size_t>(nq) * 2);
    std::array<std::array<double, 2>, 3> cent{};
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < n_per; ++i)
            for (int d = 0; d < 2; ++d)
                cent[c][d] +=
                    gpu_model.embedding()[(c * n_per + i) * 2 + d] / n_per;
    }
    int correct = 0;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < nq_per; ++i) {
            const float* e = q_emb.data() + (c * nq_per + i) * 2;
            int best = -1;
            double best_d = 1e300;
            for (int cc = 0; cc < 3; ++cc) {
                const double dx = e[0] - cent[cc][0], dy = e[1] - cent[cc][1];
                const double d2 = dx * dx + dy * dy;
                if (d2 < best_d) {
                    best_d = d2;
                    best = cc;
                }
            }
            if (best == c) ++correct;
        }
    }
    CAPTURE(correct);
    CHECK(correct >= nq - 2);
}

TEST_CASE("phase4 dispatch: unsupported GPU combos fall back to the bit-exact "
          "CPU path") {
    METAL_REQUIRED();
    const int n_per = 20, dim = 6, n = 3 * n_per;
    const auto X = three_cluster_data(n_per, 5150);

    umap::UMAPParams base;
    base.n_neighbors = 8;
    base.init = "random";
    base.random_state = 7;
    base.n_epochs = 60;
    // gpu-knn additions: the engine now covers every named dense metric
    // (manhattan included), so the not-GPU-covered kNN half of this test
    // rides the documented custom-callable fallback instead.
    base.metric = "custom";
    base.custom_metric = [](const float* x, const float* y, std::size_t d) {
        double s = 0.0;
        for (std::size_t i = 0; i < d; ++i)
            s += std::fabs(static_cast<double>(x[i]) -
                           static_cast<double>(y[i]));
        return static_cast<float>(s);
    };

    SUBCASE("densmap") {
        umap::UMAPParams p = base;
        p.densmap = true;
        umap::UMAP cpu_model(p);
        cpu_model.fit(X.data(), n, dim);
        p.device = umap::Device::GPU;
        umap::UMAP gpu_model(p);  // warns, then runs the CPU path
        gpu_model.fit(X.data(), n, dim);
        REQUIRE(gpu_model.embedding().size() == cpu_model.embedding().size());
        for (std::size_t i = 0; i < cpu_model.embedding().size(); ++i)
            CHECK(gpu_model.embedding()[i] == cpu_model.embedding()[i]);
    }
    SUBCASE("non-euclidean output metric") {
        umap::UMAPParams p = base;
        p.output_metric = "haversine";
        umap::UMAP cpu_model(p);
        cpu_model.fit(X.data(), n, dim);
        p.device = umap::Device::GPU;
        umap::UMAP gpu_model(p);
        gpu_model.fit(X.data(), n, dim);
        REQUIRE(gpu_model.embedding().size() == cpu_model.embedding().size());
        for (std::size_t i = 0; i < cpu_model.embedding().size(); ++i)
            CHECK(gpu_model.embedding()[i] == cpu_model.embedding()[i]);
    }
    SUBCASE("n_epochs list") {
        umap::UMAPParams p = base;
        p.n_epochs.reset();
        p.n_epochs_list = std::vector<int>{20, 50};
        umap::UMAP cpu_model(p);
        cpu_model.fit(X.data(), n, dim);
        p.device = umap::Device::GPU;
        umap::UMAP gpu_model(p);
        gpu_model.fit(X.data(), n, dim);
        REQUIRE(gpu_model.embedding_list().size() ==
                cpu_model.embedding_list().size());
        for (std::size_t s = 0; s < cpu_model.embedding_list().size(); ++s)
            for (std::size_t i = 0; i < cpu_model.embedding_list()[s].size();
                 ++i)
                CHECK(gpu_model.embedding_list()[s][i] ==
                      cpu_model.embedding_list()[s][i]);
    }
}
// -----------------------------------------------------------------------------
