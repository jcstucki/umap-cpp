// Metal compute kernels for the UMAP C++ port (Phase 3, branch metal-gpu).
//
// Kernel 1: brute_force_knn — exact kNN: one thread per query row, scans all
//   data rows, per-thread insertion-sorted top-k (k <= 64) kept in registers.
//   Metrics: euclidean (0) and cosine (1), formulas matching
//   src/distances.cpp (umap/distances.py) incl. the zero-norm special cases.
//
// Kernel 2: sgd_epoch — one epoch of optimize_layout_euclidean
//   (layouts.py::_optimize_layout_euclidean_single_epoch, densmap off):
//   one thread per 1-simplex (edge), hogwild-style unsynchronized float32
//   updates on the embedding buffer(s). numba's prange(parallel=True) path has
//   exactly the same race semantics (unsequenced racy f4 loads/stores), so this
//   is a faithful *parallel-semantics* port; bit-exactness vs the sequential
//   CPU path is neither possible nor required.
//
// Deviations from the CPU (sequential numba) path — all documented host-side
// in include/umap/metal/gpu_layout.hpp too:
//   - Metal has no double: grad coefficients / alpha / epoch bookkeeping are
//     float32 (CPU uses float64 for those). epoch_of_next_* accumulate in
//     float32 across epochs; drift is O(1e-4) relative over 500 epochs and only
//     perturbs *when* an edge fires, not the gradient math.
//   - Per-edge RNG: a fresh 3-word tau state is derived per (edge, epoch) from
//     the host rng_state via splitmix64, then advanced with the exact
//     tau_rand_int 64-bit recurrence from include/umap/rng.hpp (MSL `long`
//     arithmetic). The CPU path threads per-head-vertex states through the
//     whole run instead; numba's parallel path is likewise not sequence-
//     reproducible, so only distributional equivalence is targeted.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Kernel 1: brute-force kNN
// ---------------------------------------------------------------------------

constant constexpr uint kMaxK = 64;

struct KnnParams {
    uint n;       // number of data rows
    uint nq;      // number of query rows
    uint dim;     // dimensionality
    uint k;       // neighbours to keep (<= kMaxK)
    uint metric;  // 0 = euclidean, 1 = cosine
};

kernel void brute_force_knn(device const float* data [[buffer(0)]],
                            device const float* queries [[buffer(1)]],
                            device int* out_idx [[buffer(2)]],
                            device float* out_dist [[buffer(3)]],
                            constant KnnParams& p [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= p.nq) return;
    const uint K = min(p.k, kMaxK);

    float best_d[kMaxK];
    int best_i[kMaxK];
    for (uint i = 0; i < K; ++i) {
        best_d[i] = INFINITY;
        best_i[i] = -1;
    }

    device const float* q = queries + (ulong)gid * p.dim;

    float q_norm = 0.0f;
    if (p.metric == 1) {
        for (uint d = 0; d < p.dim; ++d) q_norm += q[d] * q[d];
    }

    for (uint j = 0; j < p.n; ++j) {
        device const float* x = data + (ulong)j * p.dim;
        float dist;
        if (p.metric == 0) {
            float s = 0.0f;
            for (uint d = 0; d < p.dim; ++d) {
                const float diff = q[d] - x[d];
                s += diff * diff;
            }
            dist = sqrt(s);
        } else {
            float dot = 0.0f;
            float x_norm = 0.0f;
            for (uint d = 0; d < p.dim; ++d) {
                dot += q[d] * x[d];
                x_norm += x[d] * x[d];
            }
            if (q_norm == 0.0f && x_norm == 0.0f) {
                dist = 0.0f;
            } else if (q_norm == 0.0f || x_norm == 0.0f) {
                dist = 1.0f;
            } else {
                dist = 1.0f - dot / sqrt(q_norm * x_norm);
            }
        }

        // insertion into the ascending top-K list
        if (dist < best_d[K - 1]) {
            uint pos = K - 1;
            while (pos > 0 && best_d[pos - 1] > dist) {
                best_d[pos] = best_d[pos - 1];
                best_i[pos] = best_i[pos - 1];
                --pos;
            }
            best_d[pos] = dist;
            best_i[pos] = (int)j;
        }
    }

    for (uint i = 0; i < K; ++i) {
        out_idx[(ulong)gid * K + i] = best_i[i];
        out_dist[(ulong)gid * K + i] = best_d[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: SGD layout epoch
// ---------------------------------------------------------------------------

// tau_rand_int, bit-for-bit the include/umap/rng.hpp recurrence (numba i8
// semantics: 64-bit wrapping shifts left, arithmetic shifts right).
static inline long shl64(long v, uint s) { return (long)((ulong)v << s); }

static inline int tau_rand_int(thread long state[3]) {
    state[0] = (shl64(state[0] & 4294967294L, 12) & 0xFFFFFFFFL) ^
               (((shl64(state[0], 13) & 0xFFFFFFFFL) ^ state[0]) >> 19);
    state[1] = (shl64(state[1] & 4294967288L, 4) & 0xFFFFFFFFL) ^
               (((shl64(state[1], 2) & 0xFFFFFFFFL) ^ state[1]) >> 25);
    state[2] = (shl64(state[2] & 4294967280L, 17) & 0xFFFFFFFFL) ^
               (((shl64(state[2], 3) & 0xFFFFFFFFL) ^ state[2]) >> 11);
    return (int)(uint)((ulong)(state[0] ^ state[1] ^ state[2]));
}

// splitmix64 — used only to derive well-mixed per-(edge, epoch) tau seeds.
static inline ulong splitmix64(thread ulong& s) {
    s += 0x9E3779B97F4A7C15UL;
    ulong z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

static inline float clipf(float v) { return clamp(v, -4.0f, 4.0f); }

struct SgdParams {
    uint n_edges;
    uint dim;
    uint n_vertices;  // rows of tail_embedding (negative-sample range)
    uint move_other;
    float a;
    float b;
    float gamma;
    float alpha;  // decayed on the host per epoch
    uint epoch;   // n
    uint edge_offset;  // first edge of this sub-dispatch
    long rng0;  // host rng_state[0..2]
    long rng1;
    long rng2;
};

kernel void sgd_epoch(device float* head_embedding [[buffer(0)]],
                      device float* tail_embedding [[buffer(1)]],
                      device const int* head [[buffer(2)]],
                      device const int* tail [[buffer(3)]],
                      device const float* epochs_per_sample [[buffer(4)]],
                      device const float* epochs_per_negative_sample [[buffer(5)]],
                      device float* epoch_of_next_sample [[buffer(6)]],
                      device float* epoch_of_next_negative_sample [[buffer(7)]],
                      constant SgdParams& p [[buffer(8)]],
                      uint gid [[thread_position_in_grid]]) {
    const uint i = gid + p.edge_offset;
    if (i >= p.n_edges) return;
    const float n = (float)p.epoch;
    if (epoch_of_next_sample[i] > n) return;

    const uint dim = p.dim;
    const int j = head[i];
    int k = tail[i];

    device float* current = head_embedding + (ulong)j * dim;
    device float* other = tail_embedding + (ulong)k * dim;

    // Per-(edge, epoch) tau state derived from the host rng_state. The edge
    // and epoch strides are odd 64-bit constants distinct from splitmix64's
    // internal gamma so that no two (edge, epoch, word) combinations ever
    // reuse a splitmix seed (with a stride equal to the gamma, adjacent edges
    // would share two of their three words and negative sampling would be
    // strongly correlated across neighbouring edges).
    long state[3];
    {
        ulong s = (ulong)i * 0xD1342543DE82EF95UL +
                  (ulong)p.epoch * 0xAF251AF3B0F025B5UL;
        state[0] = (long)((ulong)p.rng0 + splitmix64(s));
        state[1] = (long)((ulong)p.rng1 + splitmix64(s));
        state[2] = (long)((ulong)p.rng2 + splitmix64(s));
        tau_rand_int(state);  // warm-up mix
        tau_rand_int(state);
    }

    float dist_squared = 0.0f;  // rdist, float32 like the f4 numba kernel
    for (uint d = 0; d < dim; ++d) {
        const float diff = current[d] - other[d];
        dist_squared += diff * diff;
    }

    float grad_coeff;
    if (dist_squared > 0.0f) {
        grad_coeff = -2.0f * p.a * p.b * pow(dist_squared, p.b - 1.0f);
        grad_coeff /= p.a * pow(dist_squared, p.b) + 1.0f;
    } else {
        grad_coeff = 0.0f;
    }

    for (uint d = 0; d < dim; ++d) {
        const float grad_d = clipf(grad_coeff * (current[d] - other[d]));
        current[d] += grad_d * p.alpha;  // racy on purpose (hogwild)
        if (p.move_other) {
            other[d] += -grad_d * p.alpha;
        }
    }

    epoch_of_next_sample[i] += epochs_per_sample[i];

    const int n_neg_samples =
        (int)((n - epoch_of_next_negative_sample[i]) /
              epochs_per_negative_sample[i]);

    for (int s = 0; s < n_neg_samples; ++s) {
        // Python-modulo semantics: result takes the sign of n_vertices (> 0).
        long r = (long)tau_rand_int(state) % (long)(int)p.n_vertices;
        if (r < 0) r += (long)(int)p.n_vertices;
        k = (int)r;

        other = tail_embedding + (ulong)k * dim;

        dist_squared = 0.0f;
        for (uint d = 0; d < dim; ++d) {
            const float diff = current[d] - other[d];
            dist_squared += diff * diff;
        }

        if (dist_squared > 0.0f) {
            grad_coeff = 2.0f * p.gamma * p.b;
            grad_coeff /= (0.001f + dist_squared) *
                          (p.a * pow(dist_squared, p.b) + 1.0f);
        } else if (j == k) {
            continue;
        } else {
            grad_coeff = 0.0f;
        }

        for (uint d = 0; d < dim; ++d) {
            float grad_d;
            if (grad_coeff > 0.0f) {
                grad_d = clipf(grad_coeff * (current[d] - other[d]));
            } else {
                grad_d = 0.0f;
            }
            current[d] += grad_d * p.alpha;
        }
    }

    epoch_of_next_negative_sample[i] +=
        (float)n_neg_samples * epochs_per_negative_sample[i];
}
