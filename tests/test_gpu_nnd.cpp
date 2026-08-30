// GPU NN-descent engine (gpu-nnd pass, 2026-08-29): recall vs the exact GPU
// kNN engine, routing (crossover + $UMAP_GPU_KNN_MODE override), device=GPU
// end-to-end large-n fits, and the hidden three-way timing case backing the
// crossover constant.
//
// Only compiled when UMAP_ENABLE_METAL; each case skips gracefully when no
// Metal device / metallib is available at runtime.
#include <cstdlib>  // setenv/unsetenv (routing override)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "doctest.h"
#include "trustworthiness.hpp"
#include "umap/device.hpp"
#include "umap/metal/gpu_knn.hpp"
#include "umap/metal/gpu_nnd.hpp"
#include "umap/nn_descent.hpp"
#include "umap/umap.hpp"

#define METAL_REQUIRED()                                        \
    do {                                                        \
        if (!umap::metal_available()) {                         \
            MESSAGE("no Metal device/metallib: skipping test"); \
            return;                                             \
        }                                                       \
    } while (0)

namespace {

// Watchdog-hardening consequence (gpu-nnd pass): under adversarial
// concurrent GPU load the OS can abort command buffers, and the engines now
// surface that as a false return (production kNN sites fall back to the CPU
// path with a note) instead of silently corrupt output. Solo suite runs
// never hit it (measured over dozens); under deliberate saturation a
// transient abort usually clears on retry — after three failures the device
// is genuinely unavailable to this process right now, which is the same
// situation METAL_REQUIRED skips on.
template <typename Fn>
bool engine_with_retry(Fn&& fn) {
    for (int attempt = 0; attempt < 3; ++attempt) {
        if (fn()) return true;
    }
    return false;
}

}  // namespace

#define ENGINE_REQUIRE_OR_SKIP(...)                                     \
    do {                                                                \
        if (!engine_with_retry([&] { return (__VA_ARGS__); })) {        \
            MESSAGE("GPU command buffers aborted repeatedly "           \
                    "(concurrent GPU load): skipping");                 \
            return;                                                     \
        }                                                               \
    } while (0)

namespace {

// Deterministic pseudo-random doubles in [0, 1) (splitmix64-based), same
// generator as tests/test_metal.cpp / test_gpu_knn.cpp.
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

// Gaussian-mixture data (10 well-separated clusters, unit noise) —
// representative UMAP input; NN-descent recall on structureless uniform
// noise is not what the routing serves.
std::vector<float> mixture_data(int n, int dim, uint64_t seed,
                                int n_clusters = 10) {
    TestRng rng(seed);
    std::vector<double> centers(static_cast<std::size_t>(n_clusters) * dim);
    for (auto& c : centers) c = 10.0 * (rng.next() - 0.5);
    std::vector<float> X(static_cast<std::size_t>(n) * dim);
    for (int i = 0; i < n; ++i) {
        const int c = i % n_clusters;
        for (int d = 0; d < dim; ++d) {
            X[static_cast<std::size_t>(i) * dim + d] = static_cast<float>(
                centers[static_cast<std::size_t>(c) * dim + d] + rng.normal());
        }
    }
    return X;
}

// Recall of approximate kNN rows against exact rows, ties-tolerant: an
// approximate neighbour counts if its index is in the exact set OR its
// distance is within the exact k-th distance (+ small float32 slack), so
// equal-distance swaps are not penalised.
double knn_recall(const std::vector<int32_t>& exact_idx,
                  const std::vector<float>& exact_dist,
                  const std::vector<int32_t>& idx,
                  const std::vector<float>& dist, int n, int k) {
    std::size_t hit = 0;
    for (int i = 0; i < n; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * k;
        const float kth = exact_dist[base + k - 1];
        const float slack =
            1e-4f * std::max(1.0f, std::fabs(kth));
        for (int j = 0; j < k; ++j) {
            const int32_t cand = idx[base + j];
            if (cand < 0) continue;
            bool ok = dist[base + j] <= kth + slack;
            if (!ok) {
                for (int m = 0; m < k; ++m) {
                    if (exact_idx[base + m] == cand) {
                        ok = true;
                        break;
                    }
                }
            }
            if (ok) ++hit;
        }
    }
    return static_cast<double>(hit) /
           (static_cast<double>(n) * static_cast<double>(k));
}

// Environment override guard (restores the previous value on scope exit).
struct EnvGuard {
    std::string name;
    std::string old;
    bool had;
    EnvGuard(const char* n, const char* v) : name(n) {
        const char* o = std::getenv(n);
        had = o != nullptr;
        if (had) old = o;
        if (v)
            ::setenv(n, v, 1);
        else
            ::unsetenv(n);
    }
    ~EnvGuard() {
        if (had)
            ::setenv(name.c_str(), old.c_str(), 1);
        else
            ::unsetenv(name.c_str());
    }
};

bool throws_with_substr(const std::function<void()>& fn,
                        const std::string& substr) {
    try {
        fn();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(substr) != std::string::npos;
    }
    return false;
}

// Runs the engine + recall vs a shared exact reference for three seeds and
// returns the sorted recalls (the repo's order-statistics policy: recall is
// a random variable — reverse sampling races + per-seed forests). An empty
// result means the device aborted repeatedly under load (caller skips).
std::vector<double> recall_of_three(const std::vector<float>& X, int n,
                                    int dim, int k,
                                    umap::metal::KnnMetric metric) {
    std::vector<int32_t> exact_idx(static_cast<std::size_t>(n) * k);
    std::vector<float> exact_dist(static_cast<std::size_t>(n) * k);
    if (!engine_with_retry([&] {
            return umap::metal::gpu_brute_force_knn(
                X.data(), n, dim, X.data(), n, k, metric, exact_idx.data(),
                exact_dist.data());
        })) {
        return {};
    }
    std::vector<double> recalls;
    for (uint64_t seed : {11ull, 222ull, 3333ull}) {
        std::vector<int32_t> idx(static_cast<std::size_t>(n) * k, -2);
        std::vector<float> dist(static_cast<std::size_t>(n) * k, -1.0f);
        if (!engine_with_retry([&] {
                return umap::metal::gpu_nn_descent_knn(
                    X.data(), n, dim, k, metric, seed, idx.data(),
                    dist.data());
            })) {
            return {};
        }
        // Row conventions: ascending distances, self included at the front
        // (spot rows; ties can reorder equal distances only). Float32 cosine
        // of a row with itself is ~±3e-8 rather than exactly 0 (the exact
        // engine shares this), hence the tolerance.
        for (int i = 0; i < n; i += 997) {
            const std::size_t base = static_cast<std::size_t>(i) * k;
            CHECK(idx[base] == i);
            CHECK(std::fabs(dist[base]) <= 1e-6f);
            for (int j = 1; j < k; ++j)
                CHECK(dist[base + j] >= dist[base + j - 1]);
        }
        recalls.push_back(knn_recall(exact_idx, exact_dist, idx, dist, n, k));
    }
    std::sort(recalls.begin(), recalls.end());
    return recalls;
}

}  // namespace

// ---------------------------------------------------------------------------
// Recall vs the exact engine
// ---------------------------------------------------------------------------

TEST_CASE("gpu nnd: recall vs exact GPU kNN (20k x 16 euclidean)") {
    METAL_REQUIRED();
    const int n = 20000, dim = 16, k = 15;
    const auto X = mixture_data(n, dim, 0xE0C11D);
    const auto r = recall_of_three(X, n, dim, k,
                                   umap::metal::KnnMetric::euclidean);
    if (r.empty()) {
        MESSAGE("GPU command buffers aborted repeatedly (concurrent GPU "
                "load): skipping");
        return;
    }
    CAPTURE(r[0]);
    CAPTURE(r[1]);
    CAPTURE(r[2]);
    MESSAGE("euclidean 20k x 16 recall sorted: " << r[0] << " " << r[1] << " "
                                                 << r[2]);
    // Recall is a random variable (reverse-sampling races; the three seeds
    // also vary the forest). Measured 2026-08-29 (M-series, 5 suite runs x 3
    // seeds = 15 observations): all in [0.99751, 0.99765], spread ~1.4e-4.
    // The 0.90 goal statement is asserted on the median of 3 (order
    // statistics like the other GPU random-variable cases); the per-run 0.98
    // floor is a smoke bound >100 observed-sd below the support (a broken
    // kernel scores far under it).
    CHECK(r[1] >= 0.90);  // median of 3 — the goal statement
    CHECK(r[0] >= 0.98);  // per-run smoke floor
}

TEST_CASE("gpu nnd: recall vs exact GPU kNN (20k x 32 cosine)") {
    METAL_REQUIRED();
    const int n = 20000, dim = 32, k = 15;
    const auto X = mixture_data(n, dim, 0xC0511E);
    const auto r = recall_of_three(X, n, dim, k,
                                   umap::metal::KnnMetric::cosine);
    if (r.empty()) {
        MESSAGE("GPU command buffers aborted repeatedly (concurrent GPU "
                "load): skipping");
        return;
    }
    CAPTURE(r[0]);
    CAPTURE(r[1]);
    CAPTURE(r[2]);
    MESSAGE("cosine 20k x 32 recall sorted: " << r[0] << " " << r[1] << " "
                                              << r[2]);
    // Measured 2026-08-29 (M-series, 5 suite runs x 3 seeds): all in
    // [0.99098, 0.99141], spread ~4e-4 — same policy as the euclidean case
    // (median carries the goal statement, 0.98 is a ~25-sd smoke floor).
    CHECK(r[1] >= 0.90);  // median of 3 — the goal statement
    CHECK(r[0] >= 0.98);  // per-run smoke floor
}

// ---------------------------------------------------------------------------
// Routing: crossover + $UMAP_GPU_KNN_MODE
// ---------------------------------------------------------------------------

TEST_CASE("gpu nnd routing: gpu_knn_mode crossover and env override") {
    using umap::metal::GpuKnnMode;
    using umap::metal::gpu_knn_mode;
    using umap::metal::KnnMetric;
    const int lo = umap::metal::kGpuNndCrossoverN - 1;
    const int hi = umap::metal::kGpuNndCrossoverN + 1;

    {
        EnvGuard unset("UMAP_GPU_KNN_MODE", nullptr);
        // auto: exact below/at the crossover, nnd above.
        CHECK(gpu_knn_mode(lo, 15, KnnMetric::euclidean) == GpuKnnMode::exact);
        CHECK(gpu_knn_mode(hi, 15, KnnMetric::euclidean) == GpuKnnMode::nnd);
        CHECK(gpu_knn_mode(hi, 15, KnnMetric::cosine) == GpuKnnMode::nnd);
        CHECK(gpu_knn_mode(hi, 15, KnnMetric::manhattan) == GpuKnnMode::nnd);
        CHECK(gpu_knn_mode(hi, 15, KnnMetric::sqeuclidean) ==
              GpuKnnMode::nnd);
        // Metrics the descent kernels do not cover stay exact at any n.
        CHECK(gpu_knn_mode(hi, 15, KnnMetric::chebyshev) == GpuKnnMode::exact);
        CHECK(gpu_knn_mode(hi, 15, KnnMetric::correlation) ==
              GpuKnnMode::exact);
        // k above the register-heap cap stays exact.
        CHECK(gpu_knn_mode(hi, umap::metal::kGpuNndMaxK + 1,
                           KnnMetric::euclidean) == GpuKnnMode::exact);
        CHECK(gpu_knn_mode(hi, umap::metal::kGpuNndMaxK,
                           KnnMetric::euclidean) == GpuKnnMode::nnd);
        // Degenerate n stays exact regardless.
        CHECK(gpu_knn_mode(20, 15, KnnMetric::euclidean) == GpuKnnMode::exact);
    }
    {
        EnvGuard force("UMAP_GPU_KNN_MODE", "nnd");
        CHECK(gpu_knn_mode(5000, 15, KnnMetric::euclidean) == GpuKnnMode::nnd);
        // The override cannot force unsupported configurations.
        CHECK(gpu_knn_mode(5000, 15, KnnMetric::chebyshev) ==
              GpuKnnMode::exact);
    }
    {
        EnvGuard force("UMAP_GPU_KNN_MODE", "exact");
        CHECK(gpu_knn_mode(500000, 15, KnnMetric::euclidean) ==
              GpuKnnMode::exact);
    }
    {
        EnvGuard force("UMAP_GPU_KNN_MODE", "auto");
        CHECK(gpu_knn_mode(lo, 15, KnnMetric::euclidean) == GpuKnnMode::exact);
        CHECK(gpu_knn_mode(hi, 15, KnnMetric::euclidean) == GpuKnnMode::nnd);
    }
}

TEST_CASE("gpu nnd routing: device=GPU fits — exact below the crossover, "
          "forced nnd stays approximate-quality") {
    METAL_REQUIRED();
    const int n = 6000, dim = 8, k = 10;
    const auto X = mixture_data(n, dim, 0x60D5EED, 3);

    umap::UMAPParams p;
    p.n_neighbors = k;
    p.init = "random";
    p.random_state = 42;
    p.n_epochs = 20;
    p.device = umap::Device::GPU;

    // Exact reference rows straight from the engine.
    std::vector<int32_t> exact_idx(static_cast<std::size_t>(n) * k);
    std::vector<float> exact_dist(static_cast<std::size_t>(n) * k);
    ENGINE_REQUIRE_OR_SKIP(umap::metal::gpu_brute_force_knn(
        X.data(), n, dim, X.data(), n, k, umap::metal::KnnMetric::euclidean,
        exact_idx.data(), exact_dist.data()));

    {
        // auto mode at 6k < crossover: the fit's graph IS the exact engine
        // output (bitwise — same engine, same input, no disconnection mask
        // at the default +inf threshold).
        EnvGuard unset("UMAP_GPU_KNN_MODE", nullptr);
        umap::UMAP model(p);
        model.fit(X.data(), n, dim);
        REQUIRE_FALSE(model.small_data());
        REQUIRE(model.knn_indices().size() == exact_idx.size());
        std::size_t mismatched = 0;
        for (std::size_t i = 0; i < exact_idx.size(); ++i) {
            if (model.knn_indices()[i] != exact_idx[i] ||
                model.knn_dists()[i] != exact_dist[i])
                ++mismatched;
        }
        CHECK(mismatched == 0);
    }
    {
        // Forced nnd below the crossover: the fit completes on the descent
        // engine; its graph is approximate (recall-level parity, and update()
        // raises exactly like the exact-GPU fit — no search index exists).
        EnvGuard force("UMAP_GPU_KNN_MODE", "nnd");
        umap::UMAP model(p);
        model.fit(X.data(), n, dim);
        REQUIRE_FALSE(model.small_data());
        REQUIRE(model.knn_indices().size() == exact_idx.size());
        std::vector<int32_t> idx(model.knn_indices().begin(),
                                 model.knn_indices().end());
        std::vector<float> dist(model.knn_dists().begin(),
                                model.knn_dists().end());
        const double recall = knn_recall(exact_idx, exact_dist, idx, dist, n,
                                         k);
        CAPTURE(recall);
        MESSAGE("forced-nnd 6k fit graph recall vs exact: " << recall);
        CHECK(recall >= 0.90);
        for (float v : model.embedding()) CHECK(std::isfinite(v));

        const auto upd = mixture_data(6, dim, 99, 3);
        CHECK(throws_with_substr(
            [&] { model.update(upd.data(), 6, dim); },
            "update() requires a search index"));
    }
}

// ---------------------------------------------------------------------------
// End-to-end: large-n device=GPU fit lands on the descent engine
// ---------------------------------------------------------------------------

TEST_CASE("gpu-nnd e2e: 40k device=GPU fit (forced -> nn-descent) reaches "
          "CPU-NN-descent-comparable trustworthiness") {
    METAL_REQUIRED();
    const int n = 40000, dim = 8, k = 10;
    const int nq = 60;  // transform leg below: same centers, tail rows
    // The gpu-knn-gemm P1 kernel rewrite pushed kGpuNndCrossoverN past 40k
    // (the exact engine now out-runs GPU NN-descent through 100k), so auto
    // mode routes 40k to the exact engine. Force the descent engine to keep
    // exercising it end-to-end at this size; the routing decision itself is
    // covered by the "gpu-nnd routing" case above.
    EnvGuard force_nnd("UMAP_GPU_KNN_MODE", "nnd");
    REQUIRE(umap::metal::gpu_knn_mode(n, k, umap::metal::KnnMetric::euclidean) ==
            umap::metal::GpuKnnMode::nnd);
    const auto X = mixture_data(n + nq, dim, 0x40C4FE, 6);

    umap::UMAPParams p;
    p.n_neighbors = k;
    p.init = "random";
    p.random_state = 42;
    p.n_epochs = 100;  // modest for 40k (default would be 200): trust parity
                       // needs the hogwild SGD reasonably converged

    // CPU reference: the standard branch runs the bit-exact NN-descent port.
    umap::UMAP cpu_model(p);
    cpu_model.fit(X.data(), n, dim);
    REQUIRE_FALSE(cpu_model.small_data());

    // Full-matrix trustworthiness is O(n^2) memory (6.4 GB at 40k): evaluate
    // on a fixed row subsample instead — identical treatment for CPU and GPU
    // embeddings, so the parity statement is unaffected.
    const int n_sub = 2000;
    const int stride = n / n_sub;
    std::vector<float> X_sub(static_cast<std::size_t>(n_sub) * dim);
    std::vector<float> emb_sub(static_cast<std::size_t>(n_sub) * 2);
    const auto sub_trust = [&](const std::vector<float>& emb) {
        for (int i = 0; i < n_sub; ++i) {
            std::memcpy(&X_sub[static_cast<std::size_t>(i) * dim],
                        &X[static_cast<std::size_t>(i) * stride * dim],
                        sizeof(float) * dim);
            std::memcpy(&emb_sub[static_cast<std::size_t>(i) * 2],
                        &emb[static_cast<std::size_t>(i) * stride * 2],
                        sizeof(float) * 2);
        }
        return testutil::trustworthiness(X_sub.data(), n_sub, dim,
                                         emb_sub.data(), 2, 10);
    };
    const double trust_cpu = sub_trust(cpu_model.embedding());

    // GPU fit trust is a random variable (descent sampling races + hogwild
    // SGD): median of 3 fits per the hardened-assertion policy, plus a
    // per-fit smoke floor. random_state varies per rep so the forests differ
    // too. Measured 2026-08-29 (M-series, 5 case runs = 15 fits): every fit
    // in [0.8836, 0.8890] vs CPU 0.8939 — worst per-fit gap 0.0103, worst
    // median gap 0.0086, fit-trust sd ~1.3e-3, so the asserted -0.05 median
    // margin sits ~30 sd beyond the observed worst and -0.10 further still.
    p.device = umap::Device::GPU;
    std::vector<double> trusts;
    std::unique_ptr<umap::UMAP> last_model;
    for (uint32_t rep_seed : {42u, 43u, 44u}) {
        p.random_state = rep_seed;
        last_model = std::make_unique<umap::UMAP>(p);
        umap::UMAP& gpu_model = *last_model;
        gpu_model.fit(X.data(), n, dim);
        REQUIRE_FALSE(gpu_model.small_data());
        // Row sanity on the descent graph: self-first, ascending distances.
        const auto& gidx = gpu_model.knn_indices();
        const auto& gdst = gpu_model.knn_dists();
        REQUIRE(gidx.size() == static_cast<std::size_t>(n) * k);
        for (int i = 0; i < n; i += 1999) {
            CHECK(gidx[static_cast<std::size_t>(i) * k] == i);
            for (int m = 1; m < k; ++m)
                CHECK(gdst[static_cast<std::size_t>(i) * k + m] >=
                      gdst[static_cast<std::size_t>(i) * k + m - 1]);
        }
        for (float v : gpu_model.embedding()) CHECK(std::isfinite(v));
        trusts.push_back(sub_trust(gpu_model.embedding()));
    }
    std::sort(trusts.begin(), trusts.end());
    CAPTURE(trust_cpu);
    CAPTURE(trusts[0]);
    CAPTURE(trusts[1]);
    CAPTURE(trusts[2]);
    MESSAGE("40k e2e trustworthiness (2000-row subsample) — CPU(NN-descent): "
            << trust_cpu << ", GPU(nnd) sorted: " << trusts[0] << " "
            << trusts[1] << " " << trusts[2]);
    CHECK(trusts[1] >= trust_cpu - 0.05);  // median parity vs CPU
    CHECK(trusts[0] >= trust_cpu - 0.10);  // per-fit smoke floor

    // Transform against a descent fit: no search index exists, so the kNN
    // runs on the index-free exact engine at any n (O(nq * n), watchdog-
    // safe slices). The queries are the generator's tail rows (identical
    // cluster centers; cluster id of tail row i is (n + i) % 6) — held-out
    // points land finite and nearest an own-cluster training point.
    const float* Q = X.data() + static_cast<std::size_t>(n) * dim;
    const auto q_emb = last_model->transform(Q, nq);
    REQUIRE(q_emb.size() == static_cast<std::size_t>(nq) * 2);
    for (float v : q_emb) CHECK(std::isfinite(v));
    // Majority vote over the 15 embedding-nearest training points (a single
    // stray training point makes nearest-1 flaky at n = 40k).
    const auto& emb = last_model->embedding();
    const int kv = 15;
    int near_own = 0;
    std::vector<std::pair<double, int>> nearest(kv);
    for (int i = 0; i < nq; ++i) {
        nearest.assign(kv, {std::numeric_limits<double>::infinity(), -1});
        for (int j = 0; j < n; ++j) {
            const double dx = static_cast<double>(q_emb[i * 2]) -
                              static_cast<double>(emb[j * 2]);
            const double dy = static_cast<double>(q_emb[i * 2 + 1]) -
                              static_cast<double>(emb[j * 2 + 1]);
            const double d2 = dx * dx + dy * dy;
            if (d2 < nearest.back().first) {
                nearest.back() = {d2, j};
                for (int m = kv - 1;
                     m > 0 && nearest[m].first < nearest[m - 1].first; --m)
                    std::swap(nearest[m], nearest[m - 1]);
            }
        }
        int own = 0;
        for (const auto& pr : nearest)
            if (pr.second >= 0 && pr.second % 6 == (n + i) % 6) ++own;
        if (own > kv / 2) ++near_own;
    }
    CAPTURE(near_own);
    MESSAGE("40k descent-fit transform: " << near_own << "/" << nq
            << " queries with own-cluster majority among 15 nearest");
    // Measured 2026-08-29 (M-series, 6 runs): 60/60 in every run; nq-3
    // leaves room for the racy fit/transform draws.
    CHECK(near_own >= nq - 3);
}

// ---------------------------------------------------------------------------
// Rough timing: GPU NN-descent vs exact engine vs CPU NN-descent (hidden by
// default; run with `./umap_tests -ns -tc="gpu nnd timing*"`). Backs the
// kGpuNndCrossoverN constant and the README coverage table.
// ---------------------------------------------------------------------------

TEST_CASE("gpu nnd timing: descent vs exact engine vs CPU NN-descent"
          * doctest::skip()) {
    METAL_REQUIRED();
    const int dim = 16, k = 15;
    using clock = std::chrono::steady_clock;
    const auto ms = [](clock::duration d) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(d)
            .count();
    };
    for (const int n : {20000, 50000, 100000}) {
        const auto X = mixture_data(n, dim, 4242);
        std::vector<int32_t> idx(static_cast<std::size_t>(n) * k);
        std::vector<float> dst(static_cast<std::size_t>(n) * k);
        std::vector<int32_t> eidx(static_cast<std::size_t>(n) * k);
        std::vector<float> edst(static_cast<std::size_t>(n) * k);

        const auto t0 = clock::now();
        ENGINE_REQUIRE_OR_SKIP(umap::metal::gpu_nn_descent_knn(
            X.data(), n, dim, k, umap::metal::KnnMetric::euclidean, 42,
            idx.data(), dst.data()));
        const auto t1 = clock::now();
        ENGINE_REQUIRE_OR_SKIP(umap::metal::gpu_brute_force_knn(
            X.data(), n, dim, X.data(), n, k,
            umap::metal::KnnMetric::euclidean, eidx.data(), edst.data()));
        const auto t2 = clock::now();
        umap::nnd::nearest_neighbors(X.data(), n, dim, k, "euclidean",
                                     /*angular=*/false, /*random_seed=*/42u,
                                     idx.data(), dst.data());
        const auto t3 = clock::now();
        const double recall = knn_recall(eidx, edst, idx, dst, n, k);
        std::vector<int32_t> gidx(static_cast<std::size_t>(n) * k);
        std::vector<float> gdst(static_cast<std::size_t>(n) * k);
        ENGINE_REQUIRE_OR_SKIP(umap::metal::gpu_nn_descent_knn(
            X.data(), n, dim, k, umap::metal::KnnMetric::euclidean, 42,
            gidx.data(), gdst.data()));
        const double grecall = knn_recall(eidx, edst, gidx, gdst, n, k);
        MESSAGE("n=" << n << ": GPU nnd " << ms(t1 - t0) << " ms (recall "
                     << grecall << "), GPU exact " << ms(t2 - t1)
                     << " ms, CPU NN-descent " << ms(t3 - t2)
                     << " ms (recall " << recall << ")");
    }
}
