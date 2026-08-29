// Auxiliary penalized-kNN GPU kernel (src/metal/knn.metal `knn_penalized`,
// host src/metal/metal_knn.cpp, public API include/umap/metal_knn.h): GPU
// parity vs an in-test CPU reference of the penalized distance
// (cosine-or-euclidean base + wt*|dz_i-dz_j| + wp*|cz_i-cz_j| + wr*|rz_i-rz_j|
// + wk*[tid_i != tid_j]), zero-penalty equivalence with the exact GPU kNN
// engine, -1 / 1e30 padding when k exceeds the n-1 available neighbours, and
// the F > 8 / k > 64 refusal contract.
//
// Compiled in BOTH configs: the GPU cases are gated on UMAP_ENABLE_METAL (and
// skip gracefully when no Metal device / metallib is available at runtime);
// UMAP_ENABLE_METAL=OFF builds instead check the always-false CPU stub
// (src/metal_knn_stub.cpp).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "doctest.h"
#include "umap/metal_knn.h"
#ifdef UMAP_ENABLE_METAL
#include "umap/device.hpp"
#include "umap/metal/gpu_knn.hpp"
#endif

namespace {

// Deterministic pseudo-random doubles in [0, 1) (splitmix64-based), same
// generator as tests/test_metal.cpp / tests/test_gpu_knn.cpp.
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

// Synthetic panel: n*F normal features (unit rows for the cosine base),
// squared row norms, z-scored day/close/return columns, and ticker ids
// i % n_tickers (guaranteed collisions).
struct PenalizedData {
    int n = 0, F = 0;
    std::vector<float> X, sq, dz, cz, rz;
    std::vector<int32_t> tid;
};

PenalizedData make_data(int n, int F, uint64_t seed, int n_tickers,
                        bool unit_rows) {
    TestRng rng(seed);
    PenalizedData D;
    D.n = n;
    D.F = F;
    D.X.resize(static_cast<std::size_t>(n) * F);
    for (auto& v : D.X) v = static_cast<float>(rng.normal());
    if (unit_rows) {
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int f = 0; f < F; ++f) {
                const double v = D.X[static_cast<std::size_t>(i) * F + f];
                s += v * v;
            }
            const float inv = static_cast<float>(1.0 / std::sqrt(s));
            for (int f = 0; f < F; ++f)
                D.X[static_cast<std::size_t>(i) * F + f] *= inv;
        }
    }
    D.sq.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int f = 0; f < F; ++f) {
            const double v = D.X[static_cast<std::size_t>(i) * F + f];
            s += v * v;
        }
        D.sq[static_cast<std::size_t>(i)] = static_cast<float>(s);
    }
    D.dz.resize(static_cast<std::size_t>(n));
    D.cz.resize(static_cast<std::size_t>(n));
    D.rz.resize(static_cast<std::size_t>(n));
    D.tid.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        D.dz[static_cast<std::size_t>(i)] = static_cast<float>(rng.normal());
        D.cz[static_cast<std::size_t>(i)] = static_cast<float>(rng.normal());
        D.rz[static_cast<std::size_t>(i)] = static_cast<float>(rng.normal());
        D.tid[static_cast<std::size_t>(i)] = i % n_tickers;
    }
    return D;
}

#ifdef UMAP_ENABLE_METAL

#define PENALIZED_KNN_REQUIRED()                                \
    do {                                                        \
        if (!gpu_knn_available()) {                             \
            MESSAGE("no Metal device/metallib: skipping test"); \
            return;                                             \
        }                                                       \
    } while (0)

// CPU reference of the kernel's penalized distance, evaluated in float64 on
// the same float32 inputs (incl. the euclidean norm trick
// sqrt(max(sq_i + sq_j - 2<x_i,x_j>, 0)) and the weight>0 penalty gates).
double ref_penalized(const PenalizedData& D, int i, int j, bool cosine,
                     float wt, float wp, float wr, float wk) {
    const int F = D.F;
    double dot = 0.0;
    for (int f = 0; f < F; ++f)
        dot += static_cast<double>(D.X[static_cast<std::size_t>(i) * F + f]) *
               static_cast<double>(D.X[static_cast<std::size_t>(j) * F + f]);
    double base;
    if (cosine) {
        base = 1.0 - dot;
    } else {
        const double d2 = static_cast<double>(D.sq[static_cast<std::size_t>(i)]) +
                          static_cast<double>(D.sq[static_cast<std::size_t>(j)]) -
                          2.0 * dot;
        base = std::sqrt(std::max(d2, 0.0));
    }
    double pen = 0.0;
    if (wt > 0.0f)
        pen += static_cast<double>(wt) *
               std::fabs(static_cast<double>(D.dz[static_cast<std::size_t>(i)]) -
                         static_cast<double>(D.dz[static_cast<std::size_t>(j)]));
    if (wp > 0.0f)
        pen += static_cast<double>(wp) *
               std::fabs(static_cast<double>(D.cz[static_cast<std::size_t>(i)]) -
                         static_cast<double>(D.cz[static_cast<std::size_t>(j)]));
    if (wr > 0.0f)
        pen += static_cast<double>(wr) *
               std::fabs(static_cast<double>(D.rz[static_cast<std::size_t>(i)]) -
                         static_cast<double>(D.rz[static_cast<std::size_t>(j)]));
    if (wk > 0.0f && D.tid[static_cast<std::size_t>(i)] !=
                         D.tid[static_cast<std::size_t>(j)])
        pen += static_cast<double>(wk);
    return base + pen;
}

// CPU reference kNN: self excluded, (distance, index) ascending, and the
// kernel's -1 / 1e30 padding when k > n - 1.
void cpu_penalized_knn(const PenalizedData& D, int k, bool cosine, float wt,
                       float wp, float wr, float wk, std::vector<int32_t>& idx,
                       std::vector<float>& dist) {
    idx.assign(static_cast<std::size_t>(D.n) * k, -1);
    dist.assign(static_cast<std::size_t>(D.n) * k, 1e30f);
    std::vector<std::pair<double, int>> d;
    d.reserve(static_cast<std::size_t>(D.n));
    const int kk = std::min(k, D.n - 1);
    for (int i = 0; i < D.n; ++i) {
        d.clear();
        for (int j = 0; j < D.n; ++j) {
            if (j == i) continue;
            d.emplace_back(ref_penalized(D, i, j, cosine, wt, wp, wr, wk), j);
        }
        std::partial_sort(d.begin(), d.begin() + kk, d.end());
        for (int m = 0; m < kk; ++m) {
            idx[static_cast<std::size_t>(i) * k + m] = d[m].second;
            dist[static_cast<std::size_t>(i) * k + m] =
                static_cast<float>(d[m].first);
        }
    }
}

// Ties-tolerant comparison (same policy as tests/test_gpu_knn.cpp): distances
// within tol, index mismatches allowed only when the CPU distance to the
// GPU-chosen neighbour equals the CPU distance at that rank; padding entries
// must agree exactly (-1 / 1e30).
void check_penalized_match(const PenalizedData& D, int k, bool cosine,
                           float wt, float wp, float wr, float wk,
                           const std::vector<int32_t>& cpu_idx,
                           const std::vector<float>& cpu_dist,
                           const std::vector<int32_t>& gpu_idx,
                           const std::vector<float>& gpu_dist,
                           double tol = 1e-5) {
    int idx_mismatches = 0;
    double max_dist_diff = 0.0;
    for (int i = 0; i < D.n; ++i) {
        for (int m = 0; m < k; ++m) {
            const std::size_t e = static_cast<std::size_t>(i) * k + m;
            if (cpu_idx[e] == -1) {
                CHECK(gpu_idx[e] == -1);
                CHECK(gpu_dist[e] == 1e30f);
                continue;
            }
            max_dist_diff = std::max(
                max_dist_diff, std::fabs(static_cast<double>(cpu_dist[e]) -
                                         static_cast<double>(gpu_dist[e])));
            if (cpu_idx[e] != gpu_idx[e]) {
                if (gpu_idx[e] < 0) {
                    ++idx_mismatches;
                    continue;
                }
                const double d_choice =
                    ref_penalized(D, i, gpu_idx[e], cosine, wt, wp, wr, wk);
                if (std::fabs(d_choice - static_cast<double>(cpu_dist[e])) >
                    tol)
                    ++idx_mismatches;
            }
        }
    }
    INFO("max |cpu - gpu| distance diff: " << max_dist_diff);
    CHECK(max_dist_diff <= tol);
    CHECK(idx_mismatches == 0);
}

#endif  // UMAP_ENABLE_METAL

}  // namespace

#ifdef UMAP_ENABLE_METAL

TEST_CASE("penalized knn: euclidean base + all four penalties vs CPU ref") {
    PENALIZED_KNN_REQUIRED();
    const int n = 400, F = 6, k = 10;
    const auto D = make_data(n, F, 20260829, 25, false);  // 16 rows per ticker
    const float wt = 0.7f, wp = 0.35f, wr = 1.2f, wk = 2.5f;

    std::vector<int32_t> gpu_idx(static_cast<std::size_t>(n) * k);
    std::vector<float> gpu_dist(static_cast<std::size_t>(n) * k);
    REQUIRE(gpu_knn(D.X.data(), D.sq.data(), D.dz.data(), D.cz.data(),
                    D.rz.data(), D.tid.data(), n, F, k, false, wt, wp, wr, wk,
                    gpu_idx.data(), gpu_dist.data()));

    // Rows are sorted ascending and never contain the query point itself.
    for (int i = 0; i < n; ++i) {
        for (int m = 0; m < k; ++m) {
            const std::size_t e = static_cast<std::size_t>(i) * k + m;
            CHECK(gpu_idx[e] != i);
            if (m > 0) CHECK(gpu_dist[e] >= gpu_dist[e - 1]);
        }
    }

    std::vector<int32_t> cpu_idx;
    std::vector<float> cpu_dist;
    cpu_penalized_knn(D, k, false, wt, wp, wr, wk, cpu_idx, cpu_dist);
    check_penalized_match(D, k, false, wt, wp, wr, wk, cpu_idx, cpu_dist,
                          gpu_idx, gpu_dist);
}

TEST_CASE("penalized knn: cosine base (unit rows) + ticker penalty only") {
    PENALIZED_KNN_REQUIRED();
    const int n = 300, F = 8, k = 15;
    const auto D = make_data(n, F, 0xC0511E2026ull, 7, true);
    const float wk = 1.5f;

    std::vector<int32_t> gpu_idx(static_cast<std::size_t>(n) * k);
    std::vector<float> gpu_dist(static_cast<std::size_t>(n) * k);
    REQUIRE(gpu_knn(D.X.data(), D.sq.data(), D.dz.data(), D.cz.data(),
                    D.rz.data(), D.tid.data(), n, F, k, true, 0.0f, 0.0f, 0.0f,
                    wk, gpu_idx.data(), gpu_dist.data()));

    std::vector<int32_t> cpu_idx;
    std::vector<float> cpu_dist;
    cpu_penalized_knn(D, k, true, 0.0f, 0.0f, 0.0f, wk, cpu_idx, cpu_dist);
    check_penalized_match(D, k, true, 0.0f, 0.0f, 0.0f, wk, cpu_idx, cpu_dist,
                          gpu_idx, gpu_dist);

    // The ticker penalty must actually engage: with only ~n/7 same-ticker
    // rows per query and wk >> cosine distances, same-ticker neighbours
    // dominate the head of each row.
    int same_ticker_top1 = 0;
    for (int i = 0; i < n; ++i) {
        const int32_t j = gpu_idx[static_cast<std::size_t>(i) * k];
        if (D.tid[static_cast<std::size_t>(j)] ==
            D.tid[static_cast<std::size_t>(i)])
            ++same_ticker_top1;
    }
    CHECK(same_ticker_top1 == n);
}

TEST_CASE("penalized knn: zero penalties == plain kNN (vs the exact GPU engine)") {
    PENALIZED_KNN_REQUIRED();
    const int n = 350, F = 6, k = 12;
    const auto D = make_data(n, F, 0xABCDEFull, 5, false);

    std::vector<int32_t> gpu_idx(static_cast<std::size_t>(n) * k);
    std::vector<float> gpu_dist(static_cast<std::size_t>(n) * k);
    REQUIRE(gpu_knn(D.X.data(), D.sq.data(), D.dz.data(), D.cz.data(),
                    D.rz.data(), D.tid.data(), n, F, k, false, 0.0f, 0.0f,
                    0.0f, 0.0f, gpu_idx.data(), gpu_dist.data()));

    // Engine self-kNN includes self as each row's first neighbour: ask for
    // k+1 and drop that column.
    std::vector<int32_t> eng_idx(static_cast<std::size_t>(n) * (k + 1));
    std::vector<float> eng_dist(static_cast<std::size_t>(n) * (k + 1));
    REQUIRE(umap::metal::gpu_brute_force_knn(
        D.X.data(), n, F, D.X.data(), n, k + 1,
        umap::metal::KnnMetric::euclidean, eng_idx.data(), eng_dist.data()));

    int idx_mismatches = 0;
    double max_dist_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        REQUIRE(eng_idx[static_cast<std::size_t>(i) * (k + 1)] == i);
        for (int m = 0; m < k; ++m) {
            const std::size_t pe = static_cast<std::size_t>(i) * k + m;
            const std::size_t ee =
                static_cast<std::size_t>(i) * (k + 1) + m + 1;
            max_dist_diff = std::max(
                max_dist_diff, std::fabs(static_cast<double>(gpu_dist[pe]) -
                                         static_cast<double>(eng_dist[ee])));
            if (gpu_idx[pe] != eng_idx[ee]) {
                // Ties-tolerant: the penalized kernel's choice must sit at
                // the same (float64) euclidean distance as the engine's.
                const double d_choice = ref_penalized(
                    D, i, gpu_idx[pe], false, 0.0f, 0.0f, 0.0f, 0.0f);
                if (std::fabs(d_choice -
                              static_cast<double>(eng_dist[ee])) > 1e-5)
                    ++idx_mismatches;
            }
        }
    }
    INFO("max |penalized - engine| distance diff: " << max_dist_diff);
    CHECK(max_dist_diff <= 1e-5);
    CHECK(idx_mismatches == 0);

    // And the in-test CPU reference agrees too (same machinery as the
    // penalty cases, weights all zero).
    std::vector<int32_t> cpu_idx;
    std::vector<float> cpu_dist;
    cpu_penalized_knn(D, k, false, 0.0f, 0.0f, 0.0f, 0.0f, cpu_idx, cpu_dist);
    check_penalized_match(D, k, false, 0.0f, 0.0f, 0.0f, 0.0f, cpu_idx,
                          cpu_dist, gpu_idx, gpu_dist);
}

TEST_CASE("penalized knn: k > available neighbours -> -1 / 1e30 padding") {
    PENALIZED_KNN_REQUIRED();
    const int n = 5, F = 3, k = 8;  // only n-1 = 4 real neighbours per row
    const auto D = make_data(n, F, 0x9ADull, 2, false);
    const float wt = 0.5f, wk = 1.0f;

    std::vector<int32_t> gpu_idx(static_cast<std::size_t>(n) * k);
    std::vector<float> gpu_dist(static_cast<std::size_t>(n) * k);
    REQUIRE(gpu_knn(D.X.data(), D.sq.data(), D.dz.data(), D.cz.data(),
                    D.rz.data(), D.tid.data(), n, F, k, false, wt, 0.0f, 0.0f,
                    wk, gpu_idx.data(), gpu_dist.data()));

    for (int i = 0; i < n; ++i) {
        for (int m = 0; m < k; ++m) {
            const std::size_t e = static_cast<std::size_t>(i) * k + m;
            if (m < n - 1) {
                CHECK(gpu_idx[e] >= 0);
                CHECK(gpu_idx[e] != i);
                CHECK(gpu_dist[e] < 1e30f);
            } else {  // unfilled slots keep the init sentinels
                CHECK(gpu_idx[e] == -1);
                CHECK(gpu_dist[e] == 1e30f);
            }
        }
    }

    std::vector<int32_t> cpu_idx;
    std::vector<float> cpu_dist;
    cpu_penalized_knn(D, k, false, wt, 0.0f, 0.0f, wk, cpu_idx, cpu_dist);
    check_penalized_match(D, k, false, wt, 0.0f, 0.0f, wk, cpu_idx, cpu_dist,
                          gpu_idx, gpu_dist);
}

TEST_CASE("penalized knn: F > 8 or k > 64 -> false, outputs untouched") {
    // Contract holds with or without a Metal device (no skip): the host
    // refuses before touching any buffer, so the caller can always fall back.
    const int n = 20;
    std::vector<int32_t> out_idx(static_cast<std::size_t>(n) * 65, 123);
    std::vector<float> out_dist(static_cast<std::size_t>(n) * 65, -7.0f);

    const auto D9 = make_data(n, 9, 0xF9ull, 4, false);
    CHECK_FALSE(gpu_knn(D9.X.data(), D9.sq.data(), D9.dz.data(), D9.cz.data(),
                        D9.rz.data(), D9.tid.data(), n, 9, 10, false, 1.0f,
                        1.0f, 1.0f, 1.0f, out_idx.data(), out_dist.data()));

    const auto D3 = make_data(n, 3, 0x65ull, 4, false);
    CHECK_FALSE(gpu_knn(D3.X.data(), D3.sq.data(), D3.dz.data(), D3.cz.data(),
                        D3.rz.data(), D3.tid.data(), n, 3, 65, false, 1.0f,
                        1.0f, 1.0f, 1.0f, out_idx.data(), out_dist.data()));

    CHECK(out_idx[0] == 123);
    CHECK(out_dist[0] == -7.0f);
}

#else  // !UMAP_ENABLE_METAL

TEST_CASE("penalized knn: UMAP_ENABLE_METAL=OFF stub returns false") {
    CHECK_FALSE(gpu_knn_available());

    const int n = 8, F = 4, k = 3;
    const auto D = make_data(n, F, 0x0FFull, 3, false);
    std::vector<int32_t> out_idx(static_cast<std::size_t>(n) * k, 123);
    std::vector<float> out_dist(static_cast<std::size_t>(n) * k, -7.0f);
    CHECK_FALSE(gpu_knn(D.X.data(), D.sq.data(), D.dz.data(), D.cz.data(),
                        D.rz.data(), D.tid.data(), n, F, k, false, 1.0f, 1.0f,
                        1.0f, 1.0f, out_idx.data(), out_dist.data()));
    CHECK(out_idx[0] == 123);   // outputs untouched: caller falls back to CPU
    CHECK(out_dist[0] == -7.0f);
}

#endif  // UMAP_ENABLE_METAL
