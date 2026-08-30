// Metal compute kernels for the UMAP C++ port (Phase 3, branch metal-gpu;
// exact-kNN engine extended in the gpu-knn pass, 2026-08-29).
//
// Kernels 1a-1d: exact brute-force kNN engine — one thread per query row.
//   Data rows are streamed through a threadgroup-memory tile shared by the
//   whole threadgroup (any n, any nq; rows wider than the tile fall back to
//   direct device reads). Dense metrics are function-constant specializations
//   (one pipeline per metric id, lazily built host-side); sparse CSR kernels
//   cover the norm-trick family (euclidean/sqeuclidean/cosine). Per-thread
//   top-k: ascending insertion list in registers for k <= 64 (*_smallk), a
//   (dist, index)-lexicographic max-heap in thread memory for 64 < k <= 256
//   (*_heap) — both produce the identical ascending (dist, index) order.
//   Distance formulas match src/distances.cpp (umap/distances.py) incl. the
//   zero-norm special cases, evaluated in float32 (the CPU reference uses
//   float64), so distances agree to ~1e-5 and indices up to genuine ties.
//
// Kernel 2: sgd_epoch / sgd_epoch_densmap — one epoch of
//   optimize_layout_euclidean (layouts.py::_optimize_layout_euclidean_single_
//   epoch): one thread per 1-simplex (edge), hogwild-style unsynchronized
//   float32 updates on the embedding buffer(s). numba's prange(parallel=True)
//   path has exactly the same race semantics (unsequenced racy f4 loads/
//   stores), so this is a faithful *parallel-semantics* port; bit-exactness vs
//   the sequential CPU path is neither possible nor required. sgd_epoch_densmap
//   (densmap-gpu additions) adds the per-edge density gradient correction of
//   the densmap_flag branch; the per-epoch scalars (re_std/re_mean/re_cov/
//   mu_tot) and the epoch-init reductions are computed HOST-side between
//   epochs (see src/metal/gpu_layout.cpp) and arrive via DensSgdParams plus
//   the mu/R/phi_sum/re_sum buffers. move_other=false (transform's frozen
//   tail embedding) is a runtime flag: tail rows are then never written, so
//   only same-head-vertex updates can race.
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
// Kernels 1a-1d: exact brute-force kNN engine
// ---------------------------------------------------------------------------

// Metric ids — must match umap::metal::KnnMetric (include/umap/metal/gpu_knn.hpp).
#define UMAP_KNN_EUCLIDEAN 0u
#define UMAP_KNN_COSINE 1u
#define UMAP_KNN_SQEUCLIDEAN 2u
#define UMAP_KNN_MANHATTAN 3u
#define UMAP_KNN_CHEBYSHEV 4u
#define UMAP_KNN_MINKOWSKI 5u
#define UMAP_KNN_CORRELATION 6u
#define UMAP_KNN_HAMMING 7u
#define UMAP_KNN_CANBERRA 8u
#define UMAP_KNN_BRAYCURTIS 9u
#define UMAP_KNN_SEUCLIDEAN 10u
#define UMAP_KNN_WMINKOWSKI 11u
#define UMAP_KNN_MAHALANOBIS 12u

// Dense metric specialization: one pipeline per metric id (dead branches of
// pair_dist are eliminated at pipeline-creation time).
constant uint kMetricId [[function_constant(0)]];

// Threadgroup data tile: 4096 floats = 16 KB (half the 32 KB minimum
// threadgroup memory), leaving headroom for compiler-reserved usage.
constant constexpr uint kTileFloats = 4096;

constant constexpr uint kSmallK = 64;   // register insertion-list cap
constant constexpr uint kHeapK = 256;   // thread-memory heap cap

struct KnnDenseParams {
    uint n;    // number of data rows
    uint nq;   // number of query rows
    uint dim;  // dimensionality
    uint k;    // neighbours to keep
    float p;   // minkowski / wminkowski exponent
};

// Ascending insertion-sorted top-k (k <= kSmallK), the Phase 3 convention:
// candidates equal to the current k-th are NOT inserted (first-seen wins,
// i.e. the lowest data index at an exact k-boundary tie), and equal
// distances inside the list keep ascending data-index order.
struct TopKList {
    float d[kSmallK];
    int i[kSmallK];
    uint k;
    void init(uint kk) {
        k = min(kk, kSmallK);
        for (uint j = 0; j < k; ++j) {
            d[j] = INFINITY;
            i[j] = -1;
        }
    }
    void push(float dist, int idx) {
        if (dist < d[k - 1]) {
            uint pos = k - 1;
            while (pos > 0 && d[pos - 1] > dist) {
                d[pos] = d[pos - 1];
                i[pos] = i[pos - 1];
                --pos;
            }
            d[pos] = dist;
            i[pos] = idx;
        }
    }
    void emit(device int* out_idx, device float* out_dist, ulong row) {
        for (uint j = 0; j < k; ++j) {
            out_idx[row * k + j] = i[j];
            out_dist[row * k + j] = d[j];
        }
    }
};

// (dist, index)-lexicographic max-heap top-k for 64 < k <= kHeapK. The
// lexicographic order makes the kept SET and the final ascending order
// bit-identical to TopKList: at an exact k-boundary tie the lower data index
// wins (the scan visits it first and a later equal candidate compares
// greater-or-equal to the root).
struct TopKHeap {
    float d[kHeapK];
    int i[kHeapK];
    uint k;
    // (d[a], i[a]) < (db, ib)?
    bool entry_less(uint a, float db, int ib) const {
        return d[a] < db || (d[a] == db && i[a] < ib);
    }
    void init(uint kk) {
        k = min(kk, kHeapK);
        for (uint j = 0; j < k; ++j) {
            d[j] = INFINITY;
            i[j] = -1;
        }
    }
    void sift_down(uint root, uint size) {
        for (;;) {
            const uint l = 2 * root + 1;
            if (l >= size) break;
            uint largest = l;
            const uint r = l + 1;
            if (r < size && entry_less(l, d[r], i[r])) largest = r;
            if (!entry_less(root, d[largest], i[largest])) break;
            const float td = d[root];
            const int ti = i[root];
            d[root] = d[largest];
            i[root] = i[largest];
            d[largest] = td;
            i[largest] = ti;
            root = largest;
        }
    }
    void push(float dist, int idx) {
        // Replace the max iff the candidate is strictly (d, i)-smaller.
        if (dist < d[0] || (dist == d[0] && idx < i[0])) {
            d[0] = dist;
            i[0] = idx;
            sift_down(0, k);
        }
    }
    void emit(device int* out_idx, device float* out_dist, ulong row) {
        // In-place heapsort -> ascending (dist, index).
        for (uint size = k; size > 1; --size) {
            const float td = d[0];
            const int ti = i[0];
            d[0] = d[size - 1];
            i[0] = i[size - 1];
            d[size - 1] = td;
            i[size - 1] = ti;
            sift_down(0, size - 1);
        }
        for (uint j = 0; j < k; ++j) {
            out_idx[row * k + j] = i[j];
            out_dist[row * k + j] = d[j];
        }
    }
};

// Per-query precomputed terms (metric-dependent).
struct QueryCtx {
    float q_norm;  // cosine: sum q^2
    float mu_q;    // correlation: mean(q)
};

static inline QueryCtx make_query_ctx(device const float* q, uint dim) {
    QueryCtx qc;
    qc.q_norm = 0.0f;
    qc.mu_q = 0.0f;
    if (kMetricId == UMAP_KNN_COSINE) {
        float s = 0.0f;
        for (uint d = 0; d < dim; ++d) s += q[d] * q[d];
        qc.q_norm = s;
    } else if (kMetricId == UMAP_KNN_CORRELATION) {
        float s = 0.0f;
        for (uint d = 0; d < dim; ++d) s += q[d];
        qc.mu_q = s / (float)dim;
    }
    return qc;
}

// Distance between the query row and a data row (device or threadgroup
// storage); float32 evaluation of the src/distances.cpp formulas.
template <typename XPtr>
static inline float pair_dist(device const float* q, XPtr x, uint dim,
                              float p_exp, device const float* aux,
                              thread const QueryCtx& qc) {
    switch (kMetricId) {
        case UMAP_KNN_EUCLIDEAN:
        case UMAP_KNN_SQEUCLIDEAN: {
            float s = 0.0f;
            for (uint d = 0; d < dim; ++d) {
                const float diff = q[d] - x[d];
                s += diff * diff;
            }
            return kMetricId == UMAP_KNN_EUCLIDEAN ? sqrt(s) : s;
        }
        case UMAP_KNN_COSINE: {
            float dot = 0.0f;
            float x_norm = 0.0f;
            for (uint d = 0; d < dim; ++d) {
                dot += q[d] * x[d];
                x_norm += x[d] * x[d];
            }
            if (qc.q_norm == 0.0f && x_norm == 0.0f) return 0.0f;
            if (qc.q_norm == 0.0f || x_norm == 0.0f) return 1.0f;
            return 1.0f - dot / sqrt(qc.q_norm * x_norm);
        }
        case UMAP_KNN_MANHATTAN: {
            float s = 0.0f;
            for (uint d = 0; d < dim; ++d) s += fabs(q[d] - x[d]);
            return s;
        }
        case UMAP_KNN_CHEBYSHEV: {
            float s = 0.0f;
            for (uint d = 0; d < dim; ++d) s = max(s, fabs(q[d] - x[d]));
            return s;
        }
        case UMAP_KNN_MINKOWSKI: {
            float s = 0.0f;
            for (uint d = 0; d < dim; ++d)
                s += pow(fabs(q[d] - x[d]), p_exp);
            return pow(s, 1.0f / p_exp);
        }
        case UMAP_KNN_CORRELATION: {
            // Two passes over the (threadgroup-resident) data row: mean,
            // then shifted moments — the distances.py formula order.
            float mu_x = 0.0f;
            for (uint d = 0; d < dim; ++d) mu_x += x[d];
            mu_x /= (float)dim;
            float norm_q = 0.0f, norm_x = 0.0f, dot = 0.0f;
            for (uint d = 0; d < dim; ++d) {
                const float sq = q[d] - qc.mu_q;
                const float sx = x[d] - mu_x;
                norm_q += sq * sq;
                norm_x += sx * sx;
                dot += sq * sx;
            }
            if (norm_q == 0.0f && norm_x == 0.0f) return 0.0f;
            if (dot == 0.0f) return 1.0f;
            return 1.0f - dot / sqrt(norm_q * norm_x);
        }
        case UMAP_KNN_HAMMING: {
            float s = 0.0f;
            for (uint d = 0; d < dim; ++d) {
                if (q[d] != x[d]) s += 1.0f;
            }
            return s / (float)dim;
        }
        case UMAP_KNN_CANBERRA: {
            float s = 0.0f;
            for (uint d = 0; d < dim; ++d) {
                const float denom = fabs(q[d]) + fabs(x[d]);
                if (denom > 0.0f) s += fabs(q[d] - x[d]) / denom;
            }
            return s;
        }
        case UMAP_KNN_BRAYCURTIS: {
            float num = 0.0f, denom = 0.0f;
            for (uint d = 0; d < dim; ++d) {
                num += fabs(q[d] - x[d]);
                denom += fabs(q[d] + x[d]);
            }
            return denom > 0.0f ? num / denom : 0.0f;
        }
        case UMAP_KNN_SEUCLIDEAN: {
            // aux = sigma (dim floats)
            float s = 0.0f;
            for (uint d = 0; d < dim; ++d) {
                const float diff = q[d] - x[d];
                s += diff * diff / aux[d];
            }
            return sqrt(s);
        }
        case UMAP_KNN_WMINKOWSKI: {
            // aux = w (dim floats)
            float s = 0.0f;
            for (uint d = 0; d < dim; ++d)
                s += aux[d] * pow(fabs(q[d] - x[d]), p_exp);
            return pow(s, 1.0f / p_exp);
        }
        case UMAP_KNN_MAHALANOBIS: {
            // aux = V^-1 (dim x dim row-major); the O(dim^2) form recomputes
            // diff on the fly (no per-thread scratch array).
            float s = 0.0f;
            for (uint di = 0; di < dim; ++di) {
                float tmp = 0.0f;
                for (uint dj = 0; dj < dim; ++dj)
                    tmp += aux[di * dim + dj] * (q[dj] - x[dj]);
                s += tmp * (q[di] - x[di]);
            }
            return sqrt(s);
        }
    }
    return INFINITY;
}

// Shared dense body: cooperative tile streaming + per-thread top-k. All
// threads of the threadgroup participate in the tile loads and barriers even
// when their query index is out of range (dispatch is by threadgroups).
template <typename TOPK>
static inline void knn_dense_body(device const float* data,
                                  device const float* queries,
                                  device int* out_idx, device float* out_dist,
                                  constant KnnDenseParams& p,
                                  device const float* aux,
                                  threadgroup float* tile, uint lid,
                                  uint tg_id, uint tg_size,
                                  thread TOPK& top) {
    const uint gid = tg_id * tg_size + lid;
    const bool active = gid < p.nq;
    // Inactive threads point at row 0 so their (discarded) reads stay in
    // bounds while they keep servicing the cooperative loads.
    device const float* q = queries + (ulong)(active ? gid : 0u) * p.dim;
    const QueryCtx qc = make_query_ctx(q, p.dim);
    top.init(p.k);

    const uint tile_rows = p.dim <= kTileFloats ? kTileFloats / p.dim : 0u;
    if (tile_rows == 0) {
        // Row wider than the tile: direct device reads (no barriers needed).
        if (active) {
            for (uint j = 0; j < p.n; ++j) {
                const float dist = pair_dist(q, data + (ulong)j * p.dim,
                                             p.dim, p.p, aux, qc);
                top.push(dist, (int)j);
            }
        }
    } else {
        for (uint base = 0; base < p.n; base += tile_rows) {
            const uint rows = min(tile_rows, p.n - base);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = lid; t < rows * p.dim; t += tg_size)
                tile[t] = data[(ulong)base * p.dim + t];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                for (uint r = 0; r < rows; ++r) {
                    const float dist =
                        pair_dist(q, (threadgroup const float*)(tile +
                                                                r * p.dim),
                                  p.dim, p.p, aux, qc);
                    top.push(dist, (int)(base + r));
                }
            }
        }
    }
    if (active) top.emit(out_idx, out_dist, (ulong)gid);
}

kernel void knn_dense_smallk(device const float* data [[buffer(0)]],
                             device const float* queries [[buffer(1)]],
                             device int* out_idx [[buffer(2)]],
                             device float* out_dist [[buffer(3)]],
                             constant KnnDenseParams& p [[buffer(4)]],
                             device const float* aux [[buffer(5)]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint tg_id [[threadgroup_position_in_grid]],
                             uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float tile[kTileFloats];
    TopKList top;
    knn_dense_body(data, queries, out_idx, out_dist, p, aux, tile, lid, tg_id,
                   tg_size, top);
}

kernel void knn_dense_heap(device const float* data [[buffer(0)]],
                           device const float* queries [[buffer(1)]],
                           device int* out_idx [[buffer(2)]],
                           device float* out_dist [[buffer(3)]],
                           constant KnnDenseParams& p [[buffer(4)]],
                           device const float* aux [[buffer(5)]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint tg_id [[threadgroup_position_in_grid]],
                           uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float tile[kTileFloats];
    TopKHeap top;
    knn_dense_body(data, queries, out_idx, out_dist, p, aux, tile, lid, tg_id,
                   tg_size, top);
}

// --------------------------- sparse CSR kNN --------------------------------
// Norm-trick family only: sqeuclidean/euclidean via ||q||^2 + ||x||^2 - 2 q.x
// (clamped at 0 against float32 cancellation) and cosine via the same CSR dot
// with the distances.py zero-norm special cases. Row norms are precomputed
// host-side; CSR column indices must be sorted ascending (the library-wide
// invariant). One thread per query with a serial two-pointer dot per data row.

struct KnnSparseParams {
    uint n;       // data rows
    uint nq;      // query rows
    uint k;       // neighbours to keep
    uint metric;  // 0 = euclidean, 1 = cosine, 2 = sqeuclidean
};

template <typename TOPK>
static inline void knn_sparse_body(
    device const int* d_indptr, device const int* d_indices,
    device const float* d_data, device const int* q_indptr,
    device const int* q_indices, device const float* q_data,
    device const float* d_norms, device const float* q_norms,
    device int* out_idx, device float* out_dist, constant KnnSparseParams& p,
    uint gid, thread TOPK& top) {
    if (gid >= p.nq) return;
    top.init(p.k);
    const uint a0 = (uint)q_indptr[gid];
    const uint a1 = (uint)q_indptr[gid + 1];
    const float qn = q_norms[gid];
    for (uint j = 0; j < p.n; ++j) {
        uint a = a0;
        uint b = (uint)d_indptr[j];
        const uint b1 = (uint)d_indptr[j + 1];
        float dot = 0.0f;
        while (a < a1 && b < b1) {
            const int ia = q_indices[a];
            const int ib = d_indices[b];
            if (ia == ib) {
                dot += q_data[a] * d_data[b];
                ++a;
                ++b;
            } else if (ia < ib) {
                ++a;
            } else {
                ++b;
            }
        }
        const float xn = d_norms[j];
        float dist;
        if (p.metric == 1u) {
            if (qn == 0.0f && xn == 0.0f) {
                dist = 0.0f;
            } else if (qn == 0.0f || xn == 0.0f) {
                dist = 1.0f;
            } else {
                dist = 1.0f - dot / sqrt(qn * xn);
            }
        } else {
            const float d2 = max(qn + xn - 2.0f * dot, 0.0f);
            dist = p.metric == 0u ? sqrt(d2) : d2;
        }
        top.push(dist, (int)j);
    }
    top.emit(out_idx, out_dist, (ulong)gid);
}

kernel void knn_sparse_smallk(device const int* d_indptr [[buffer(0)]],
                              device const int* d_indices [[buffer(1)]],
                              device const float* d_data [[buffer(2)]],
                              device const int* q_indptr [[buffer(3)]],
                              device const int* q_indices [[buffer(4)]],
                              device const float* q_data [[buffer(5)]],
                              device const float* d_norms [[buffer(6)]],
                              device const float* q_norms [[buffer(7)]],
                              device int* out_idx [[buffer(8)]],
                              device float* out_dist [[buffer(9)]],
                              constant KnnSparseParams& p [[buffer(10)]],
                              uint gid [[thread_position_in_grid]]) {
    TopKList top;
    knn_sparse_body(d_indptr, d_indices, d_data, q_indptr, q_indices, q_data,
                    d_norms, q_norms, out_idx, out_dist, p, gid, top);
}

kernel void knn_sparse_heap(device const int* d_indptr [[buffer(0)]],
                            device const int* d_indices [[buffer(1)]],
                            device const float* d_data [[buffer(2)]],
                            device const int* q_indptr [[buffer(3)]],
                            device const int* q_indices [[buffer(4)]],
                            device const float* q_data [[buffer(5)]],
                            device const float* d_norms [[buffer(6)]],
                            device const float* q_norms [[buffer(7)]],
                            device int* out_idx [[buffer(8)]],
                            device float* out_dist [[buffer(9)]],
                            constant KnnSparseParams& p [[buffer(10)]],
                            uint gid [[thread_position_in_grid]]) {
    TopKHeap top;
    knn_sparse_body(d_indptr, d_indices, d_data, q_indptr, q_indices, q_data,
                    d_norms, q_norms, out_idx, out_dist, p, gid, top);
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

// --- densmap-gpu additions --------------------------------------------------
// Per-epoch densMAP scalars, computed on the HOST from the epoch-init
// reductions (src/layouts.cpp::densmap_epoch_stats runs on the shared
// embedding buffer between epochs — the CPU float32/float64 numpy semantics
// are reused verbatim; only the per-edge work below runs on the GPU).
// Mirrors struct DensSgdParams in src/metal/gpu_layout.cpp.
struct DensSgdParams {
    float lambda_;  // densmap_kwds["lambda"] (f8 host-side; f32 on device)
    float mu_tot;   // np.sum(mu_sum) / 2
    float re_std;   // np.sqrt(np.var(re_sum) + var_shift)
    float re_mean;  // np.mean(re_sum)
    float re_cov;   // np.dot(re_sum, R) / (n_vertices - 1)
};
// ----------------------------------------------------------------------------

// Shared edge body of the SGD epoch. DENSMAP=true adds the density gradient
// correction (layouts.py densmap_flag branch) in float32 — the CPU reference
// mixes f8/f4 there; this path is quality-level parity like the rest of the
// kernel. The dens_* pointers are only dereferenced when DENSMAP is true
// (sgd_epoch passes null).
template <bool DENSMAP>
static inline void sgd_epoch_body(device float* head_embedding,
                                  device float* tail_embedding,
                                  device const int* head,
                                  device const int* tail,
                                  device const float* epochs_per_sample,
                                  device const float* epochs_per_negative_sample,
                                  device float* epoch_of_next_sample,
                                  device float* epoch_of_next_negative_sample,
                                  constant SgdParams& p,
                                  device const float* dens_mu,
                                  device const float* dens_R,
                                  device const float* dens_phi_sum,
                                  device const float* dens_re_sum,
                                  thread const DensSgdParams& dp,
                                  uint gid) {
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

    // --- densmap-gpu additions: grad_cor_coeff (float32 evaluation of the
    // layouts.py densmap_flag block; j/k here are still the edge endpoints —
    // negative sampling below reuses k). Compiled out when !DENSMAP.
    float grad_cor_coeff = 0.0f;
    if (DENSMAP) {
        const float dsq = dist_squared;
        const float phi = 1.0f / (1.0f + p.a * pow(dsq, p.b));
        const float dphi_term = p.a * p.b * pow(dsq, p.b - 1.0f) /
                                (1.0f + p.a * pow(dsq, p.b));

        const float q_jk = phi / dens_phi_sum[(uint)k];
        const float q_kj = phi / dens_phi_sum[(uint)j];

        const float drk = q_jk * ((1.0f - p.b * (1.0f - phi)) /
                                      exp(dens_re_sum[(uint)k]) +
                                  dphi_term);
        const float drj = q_kj * ((1.0f - p.b * (1.0f - phi)) /
                                      exp(dens_re_sum[(uint)j]) +
                                  dphi_term);

        const float re_std_sq = dp.re_std * dp.re_std;
        const float weight_k =
            dens_R[(uint)k] -
            dp.re_cov * (dens_re_sum[(uint)k] - dp.re_mean) / re_std_sq;
        const float weight_j =
            dens_R[(uint)j] -
            dp.re_cov * (dens_re_sum[(uint)j] - dp.re_mean) / re_std_sq;

        grad_cor_coeff = dp.lambda_ * dp.mu_tot *
                         (weight_k * drk + weight_j * drj) /
                         (dens_mu[i] * dp.re_std) / (float)p.n_vertices;
    }
    // ------------------------------------------------------------------------

    float grad_coeff;
    if (dist_squared > 0.0f) {
        grad_coeff = -2.0f * p.a * p.b * pow(dist_squared, p.b - 1.0f);
        grad_coeff /= p.a * pow(dist_squared, p.b) + 1.0f;
    } else {
        grad_coeff = 0.0f;
    }

    for (uint d = 0; d < dim; ++d) {
        float grad_d = clipf(grad_coeff * (current[d] - other[d]));
        if (DENSMAP) {
            grad_d += clipf(2.0f * grad_cor_coeff * (current[d] - other[d]));
        }
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
    const DensSgdParams dp{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    sgd_epoch_body<false>(head_embedding, tail_embedding, head, tail,
                          epochs_per_sample, epochs_per_negative_sample,
                          epoch_of_next_sample, epoch_of_next_negative_sample,
                          p, /*dens_mu=*/nullptr, /*dens_R=*/nullptr,
                          /*dens_phi_sum=*/nullptr, /*dens_re_sum=*/nullptr, dp,
                          gid);
}

// --- densmap-gpu additions --------------------------------------------------
kernel void sgd_epoch_densmap(
    device float* head_embedding [[buffer(0)]],
    device float* tail_embedding [[buffer(1)]],
    device const int* head [[buffer(2)]],
    device const int* tail [[buffer(3)]],
    device const float* epochs_per_sample [[buffer(4)]],
    device const float* epochs_per_negative_sample [[buffer(5)]],
    device float* epoch_of_next_sample [[buffer(6)]],
    device float* epoch_of_next_negative_sample [[buffer(7)]],
    constant SgdParams& p [[buffer(8)]],
    device const float* dens_mu [[buffer(9)]],       // per-edge membership
    device const float* dens_R [[buffer(10)]],       // per-vertex log radii (std.)
    device const float* dens_phi_sum [[buffer(11)]], // per-epoch host reduction
    device const float* dens_re_sum [[buffer(12)]],  // per-epoch host reduction
    constant DensSgdParams& dpc [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
    const DensSgdParams dp = dpc;
    sgd_epoch_body<true>(head_embedding, tail_embedding, head, tail,
                         epochs_per_sample, epochs_per_negative_sample,
                         epoch_of_next_sample, epoch_of_next_negative_sample,
                         p, dens_mu, dens_R, dens_phi_sum, dens_re_sum, dp,
                         gid);
}
// ----------------------------------------------------------------------------
