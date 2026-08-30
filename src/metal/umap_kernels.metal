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

// --- gpu-knn-gemm additions -------------------------------------------------
// Register-blocked fast path (PERF_BASELINE.md P1). The scalar body below
// re-reads the query row from device memory for every (data row, dim)
// iteration — 32 lanes x a 4·dim-byte stride touch a fresh cache line per
// load, which is what pinned the old kernel at ~0.3% of fp32 peak. The fast
// path caches the query row in thread registers as float4 chunks (one load
// for dim <= kQChunkDims) and walks the threadgroup tile in blocks of
// kRowBlock rows: per float4 of dims that is 1 register operand + kRowBlock
// tile loads feeding 8 independent accumulator chains (ILP), ~0.9 counted
// flops per instruction against the 1-flop/instr scalar body.
//
// simdgroup_float8x8 GEMM (norms ± Q·Dᵀ) was considered and rejected: on
// pre-M4 Apple GPUs simdgroup matrices lower onto the same FMA ALUs (no
// dedicated matmul units), so MMA only buys operand sharing — which register
// blocking gets too — while the norm-trick expansion ||q||²+||x||²−2q·x
// loses the exact 0.0f distance for self/duplicate pairs (asserted by the
// parity suite) and precision for near-duplicates. The diff form keeps each
// pair's accumulation serial in ascending d, bit-compatible with the scalar
// body.
//
// Covered metrics (per-pipeline function-constant fold): euclidean,
// sqeuclidean, manhattan (single diff accumulator), cosine (dot only — the
// data-row norms arrive precomputed in aux[n], halving the per-pair ALU) and
// correlation (query pre-shifted by mu_q in registers, data row means in
// aux[n], dot + shifted-norm accumulators). Requires dim % 4 == 0 (float4
// tile rows stay 16-byte aligned) and dim <= kTileFloats; everything else —
// chebyshev/minkowski/hamming/canberra/braycurtis/seuclidean/wminkowski and
// the O(dim²) mahalanobis — keeps the scalar tile body, as do fast metrics
// at dim % 4 != 0.
constant constexpr uint kQChunkDims = 64;  // query dims per register chunk
constant constexpr uint kRowBlock = 8;     // tile rows per accumulator block

static inline bool fast_metric() {
    return kMetricId == UMAP_KNN_EUCLIDEAN ||
           kMetricId == UMAP_KNN_SQEUCLIDEAN ||
           kMetricId == UMAP_KNN_MANHATTAN ||
           kMetricId == UMAP_KNN_COSINE ||
           kMetricId == UMAP_KNN_CORRELATION;
}
// ----------------------------------------------------------------------------

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
    float q_norm;  // cosine: sum q^2; correlation: sum (q - mu_q)^2 (fast path)
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
        // gpu-knn-gemm: the query-side centred norm depends only on the
        // query; hoisted here (same serial ascending-d accumulation as the
        // scalar body's per-pair recomputation) for the fast path.
        float nq = 0.0f;
        for (uint d = 0; d < dim; ++d) {
            const float sq = q[d] - qc.mu_q;
            nq += sq * sq;
        }
        qc.q_norm = nq;
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

// --- gpu-knn-gemm additions -------------------------------------------------

// Load one register chunk of the query row. For correlation the query is
// pre-centred here once (qreg = q - mu_q), so the inner loop pays a single
// subtraction per data element instead of two.
static inline void load_qreg(device const float* q, uint c, uint clen,
                             thread float4* qreg, thread const QueryCtx& qc) {
    device const float4* q4 = (device const float4*)(q + c);
    const uint n4 = clen / 4u;
    if (kMetricId == UMAP_KNN_CORRELATION) {
        const float4 shift = float4(qc.mu_q);
        for (uint j = 0; j < n4; ++j) qreg[j] = q4[j] - shift;
    } else {
        for (uint j = 0; j < n4; ++j) qreg[j] = q4[j];
    }
}

// One kRowBlock x clen accumulation step against the threadgroup tile.
// acc is the primary accumulator (diff² / |diff| / dot), acc2 the
// correlation shifted-norm, mux the correlation per-row means. Every
// accumulator advances serially in ascending d (lanes .x...w unrolled in
// order), so per-pair values match the scalar body's accumulation order; the
// ILP comes from the kRowBlock independent chains, the operand reuse from
// qv serving all kRowBlock rows.
static inline void accum_block(thread const float4* qreg,
                               threadgroup const float* x0, uint dim, uint c,
                               uint clen, thread float* acc,
                               thread float* acc2, thread const float* mux) {
    threadgroup const float4* xr[kRowBlock];
    for (uint r = 0; r < kRowBlock; ++r)
        xr[r] = (threadgroup const float4*)(x0 + r * dim + c);
    const uint n4 = clen / 4u;
    for (uint j = 0; j < n4; ++j) {
        const float4 qv = qreg[j];
        for (uint r = 0; r < kRowBlock; ++r) {
            const float4 xv = xr[r][j];
            if (kMetricId == UMAP_KNN_EUCLIDEAN ||
                kMetricId == UMAP_KNN_SQEUCLIDEAN) {
                const float4 t = qv - xv;
                float s = acc[r];
                s += t.x * t.x;
                s += t.y * t.y;
                s += t.z * t.z;
                s += t.w * t.w;
                acc[r] = s;
            } else if (kMetricId == UMAP_KNN_MANHATTAN) {
                const float4 t = fabs(qv - xv);
                float s = acc[r];
                s += t.x;
                s += t.y;
                s += t.z;
                s += t.w;
                acc[r] = s;
            } else if (kMetricId == UMAP_KNN_COSINE) {
                float s = acc[r];
                s += qv.x * xv.x;
                s += qv.y * xv.y;
                s += qv.z * xv.z;
                s += qv.w * xv.w;
                acc[r] = s;
            } else {  // correlation (qv pre-centred by load_qreg)
                const float4 sx = xv - float4(mux[r]);
                float dot = acc[r];
                float nx = acc2[r];
                dot += qv.x * sx.x;
                dot += qv.y * sx.y;
                dot += qv.z * sx.z;
                dot += qv.w * sx.w;
                nx += sx.x * sx.x;
                nx += sx.y * sx.y;
                nx += sx.z * sx.z;
                nx += sx.w * sx.w;
                acc[r] = dot;
                acc2[r] = nx;
            }
        }
    }
}

// Fold the block accumulators into the final distance; formulas (incl. the
// zero-norm special cases) identical to pair_dist. aux_row is the
// precomputed data-row norm for cosine, unused otherwise.
static inline float fast_row_dist(float acc, float acc2, float aux_row,
                                  thread const QueryCtx& qc) {
    switch (kMetricId) {
        case UMAP_KNN_EUCLIDEAN:
            return sqrt(acc);
        case UMAP_KNN_COSINE: {
            const float xn = aux_row;
            if (qc.q_norm == 0.0f && xn == 0.0f) return 0.0f;
            if (qc.q_norm == 0.0f || xn == 0.0f) return 1.0f;
            return 1.0f - acc / sqrt(qc.q_norm * xn);
        }
        case UMAP_KNN_CORRELATION: {
            if (qc.q_norm == 0.0f && acc2 == 0.0f) return 0.0f;
            if (acc == 0.0f) return 1.0f;
            return 1.0f - acc / sqrt(qc.q_norm * acc2);
        }
        default:  // sqeuclidean / manhattan
            return acc;
    }
}

// Fast-path dense body: float4 cooperative tile loads, register-cached query
// chunks, kRowBlock-row accumulator blocks, per-thread full top-k (rows are
// pushed in ascending data order, so the k-boundary tie rule is unchanged).
// Preconditions: fast_metric() && dim % 4 == 0 && dim <= kTileFloats.
template <typename TOPK>
static inline void knn_dense_body_fast(device const float* data,
                                       device const float* q, bool active,
                                       constant KnnDenseParams& p,
                                       device const float* aux,
                                       threadgroup float* tile, uint lid,
                                       uint tg_size,
                                       thread const QueryCtx& qc,
                                       thread TOPK& top) {
    const uint dim = p.dim;
    const uint tile_rows = kTileFloats / dim;
    float4 qreg[kQChunkDims / 4u];
    const bool single_chunk = dim <= kQChunkDims;
    if (single_chunk && active) load_qreg(q, 0u, dim, qreg, qc);
    for (uint base = 0; base < p.n; base += tile_rows) {
        const uint rows = min(tile_rows, p.n - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {  // float4 cooperative load (dim % 4 == 0 keeps rows 16B-aligned)
            device const float4* src4 =
                (device const float4*)(data + (ulong)base * dim);
            threadgroup float4* dst4 = (threadgroup float4*)tile;
            const uint total4 = rows * dim / 4u;
            for (uint t = lid; t < total4; t += tg_size) dst4[t] = src4[t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) continue;
        uint r0 = 0;
        for (; r0 + kRowBlock <= rows; r0 += kRowBlock) {
            float acc[kRowBlock];
            float acc2[kRowBlock];
            float mux[kRowBlock];
            for (uint r = 0; r < kRowBlock; ++r) {
                acc[r] = 0.0f;
                acc2[r] = 0.0f;
                mux[r] = 0.0f;
            }
            if (kMetricId == UMAP_KNN_CORRELATION) {
                for (uint r = 0; r < kRowBlock; ++r)
                    mux[r] = aux[base + r0 + r];
            }
            threadgroup const float* x0 = tile + r0 * dim;
            if (single_chunk) {
                accum_block(qreg, x0, dim, 0u, dim, acc, acc2, mux);
            } else {
                for (uint c = 0; c < dim; c += kQChunkDims) {
                    const uint clen = min(kQChunkDims, dim - c);
                    load_qreg(q, c, clen, qreg, qc);
                    accum_block(qreg, x0, dim, c, clen, acc, acc2, mux);
                }
            }
            for (uint r = 0; r < kRowBlock; ++r) {
                const float aux_row = (kMetricId == UMAP_KNN_COSINE)
                                          ? aux[base + r0 + r]
                                          : 0.0f;
                top.push(fast_row_dist(acc[r], acc2[r], aux_row, qc),
                         (int)(base + r0 + r));
            }
        }
        for (; r0 < rows; ++r0) {  // block tail: scalar per-row path
            const float dist =
                pair_dist(q, (threadgroup const float*)(tile + r0 * dim), dim,
                          p.p, aux, qc);
            top.push(dist, (int)(base + r0));
        }
    }
}

// ----------------------------------------------------------------------------

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
    // gpu-knn-gemm (PERF_BASELINE.md P1): the decomposable metrics take the
    // register-blocked fast body when the tile fits and dim is float4-aligned;
    // all other metrics / geometries fall through to the scalar tile loop.
    // The fast body owns its own tile streaming + top-k (identical push order),
    // so it is a complete replacement of the else-branch for those cases.
    if (tile_rows != 0u && fast_metric() && (p.dim % 4u) == 0u) {
        knn_dense_body_fast(data, q, active, p, aux, tile, lid, tg_size, qc,
                            top);
    } else if (tile_rows == 0) {
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

// splitmix64 — shared rng scrambler: derives the per-(edge, epoch) tau seeds
// of the SGD kernels (kernel 2) and the NN-descent sampling draws (kernels 3).
static inline ulong splitmix64(thread ulong& s) {
    s += 0x9E3779B97F4A7C15UL;
    ulong z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// Kernels 3a-3e: NN-descent kNN graph construction (gpu-nnd additions)
// ---------------------------------------------------------------------------
// GPU port of pynndescent's array-update NN-descent (src/nn_descent.cpp) for
// the device=GPU standard branch ABOVE the exact-engine crossover. Quality
// parity, not bit parity: the CPU port is RandomState-stream bit-exact vs
// Python; this engine only has to deliver a comparable-recall graph fast.
//
// Structure (host driver: src/metal/gpu_nnd.cpp):
//  - init: the RP-forest leaf arrays are built on the HOST (the existing
//    bit-exact rp-tree code, private rng) and inverted into a per-node
//    leaf-membership table; nnd_init then runs one thread per node that
//    scans its own leaves + random top-up draws and serially merges into its
//    own heap row (no races, n-way parallel).
//  - per iteration (serial encoder = inter-kernel barriers):
//      nnd_sample_fwd   one thread per node: own heap entries -> own
//                       new/old candidate lists (no atomics);
//      nnd_sample_rev   one thread per heap entry (j, s): appends j to the
//                       candidate list of its neighbour via an atomic counter
//                       with reservoir replacement past the cap — the ONE
//                       racy step (which reverse candidates survive under
//                       contention is scheduling-dependent; the CPU port
//                       keeps a deterministic random subset instead);
//      nnd_clear_flags  one thread per heap entry: clear "new" iff the entry
//                       was sampled into its owner's new list (the
//                       new_build_candidates semantics);
//      nnd_join         one thread per node i: receiving-side local join.
//                       The CPU joins pairs (a, b) within cand(p) and pushes
//                       to both; since sampling pushes both directions of
//                       every heap edge, cand is (cap-modulo) symmetric, so
//                       node i recovers exactly those pairs by scanning
//                       q in cand(p) for p in cand(i) — new pivots join new
//                       + old targets, old pivots join new targets — and
//                       merges into its OWN register-copy heap serially.
//                       No atomics, no races; each pair is evaluated from
//                       both endpoints (~2x the CPU's distance count, far
//                       cheaper than inbox scatter + a second merge pass).
//                       Accepted pushes are counted into a global counter
//                       the host reads back per iteration (early
//                       termination, delta = 0.001 like pynndescent).
// Metrics: sqeuclidean (serves euclidean/l2 via host-side sqrt), cosine
// (plain float32 form — same ordering as pynndescent's alternative_cosine,
// already the final distance), manhattan. Heap rows use the CPU heap
// semantics (max-heap keyed on distance alone, checked dedup pushes).

#define UMAP_NND_SQEUCLIDEAN 0u
#define UMAP_NND_COSINE 1u
#define UMAP_NND_MANHATTAN 2u

constant constexpr uint kNndMaxK = 64;  // per-node heap cap (register copy)

struct NndParams {
    uint n;        // nodes
    uint dim;      // dimensionality
    uint k;        // heap width (n_neighbors), <= kNndMaxK
    uint n_cand;   // candidate-list cap C (runtime, >= k)
    uint metric;   // UMAP_NND_*
    uint iter;     // current iteration (rng salt)
    uint n_trees;  // node_leaves row length (init only)
    uint leaf_len; // leaf_array row length (init only)
    ulong seed;    // base rng seed
};

static inline float nnd_dist(device const float* x, device const float* y,
                             uint dim, uint metric) {
    switch (metric) {
        case UMAP_NND_COSINE: {
            float dot = 0.0f, nx = 0.0f, ny = 0.0f;
            for (uint d = 0; d < dim; ++d) {
                dot += x[d] * y[d];
                nx += x[d] * x[d];
                ny += y[d] * y[d];
            }
            if (nx == 0.0f && ny == 0.0f) return 0.0f;
            if (nx == 0.0f || ny == 0.0f) return 1.0f;
            return 1.0f - dot / sqrt(nx * ny);
        }
        case UMAP_NND_MANHATTAN: {
            float s = 0.0f;
            for (uint d = 0; d < dim; ++d) s += fabs(x[d] - y[d]);
            return s;
        }
        default: {  // UMAP_NND_SQEUCLIDEAN
            float s = 0.0f;
            for (uint d = 0; d < dim; ++d) {
                const float diff = x[d] - y[d];
                s += diff * diff;
            }
            return s;
        }
    }
}

// Register copy of one heap row; checked_push is the CPU
// checked_flagged_heap_push (max-heap keyed on distance alone, O(k) dedup).
struct NndHeap {
    float d[kNndMaxK];
    int i[kNndMaxK];
    uchar f[kNndMaxK];
    uint k;
    void init(uint kk) {
        k = min(kk, kNndMaxK);
        for (uint j = 0; j < k; ++j) {
            d[j] = INFINITY;
            i[j] = -1;
            f[j] = 0;
        }
    }
    void load(device const float* hd, device const int* hi,
              device const uchar* hf, uint kk, ulong row) {
        k = min(kk, kNndMaxK);
        for (uint j = 0; j < k; ++j) {
            d[j] = hd[row * k + j];
            i[j] = hi[row * k + j];
            f[j] = hf[row * k + j];
        }
    }
    void store(device float* hd, device int* hi, device uchar* hf,
               ulong row) {
        for (uint j = 0; j < k; ++j) {
            hd[row * k + j] = d[j];
            hi[row * k + j] = i[j];
            hf[row * k + j] = f[j];
        }
    }
    int checked_push(float p, int n, uchar flag) {
        if (p >= d[0]) return 0;
        for (uint j = 0; j < k; ++j)
            if (i[j] == n) return 0;
        uint elt = 0;
        for (;;) {
            const uint l = 2 * elt + 1;
            const uint r = l + 1;
            uint sw;
            if (l >= k) {
                break;
            } else if (r >= k) {
                if (d[l] > p) sw = l; else break;
            } else if (d[l] >= d[r]) {
                if (p < d[l]) sw = l; else break;
            } else {
                if (p < d[r]) sw = r; else break;
            }
            d[elt] = d[sw];
            i[elt] = i[sw];
            f[elt] = f[sw];
            elt = sw;
        }
        d[elt] = p;
        i[elt] = n;
        f[elt] = flag;
        return 1;
    }
};

// 3a: heap init from the host-built RP-forest leaves + random top-up.
kernel void nnd_init(device const float* data [[buffer(0)]],
                     device float* heap_dist [[buffer(1)]],
                     device int* heap_idx [[buffer(2)]],
                     device uchar* heap_flag [[buffer(3)]],
                     device const int* leaf_idx [[buffer(4)]],
                     device const int* node_leaves [[buffer(5)]],
                     constant NndParams& p [[buffer(6)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    NndHeap h;
    h.init(p.k);
    device const float* xi = data + (ulong)gid * p.dim;
    for (uint t = 0; t < p.n_trees; ++t) {
        const int row = node_leaves[(ulong)gid * p.n_trees + t];
        if (row < 0) continue;
        device const int* leaf = leaf_idx + (ulong)row * p.leaf_len;
        for (uint m = 0; m < p.leaf_len; ++m) {
            const int q = leaf[m];
            if (q < 0) break;
            if ((uint)q == gid) continue;
            const float dd =
                nnd_dist(xi, data + (ulong)q * p.dim, p.dim, p.metric);
            h.checked_push(dd, q, 1);
        }
    }
    // init_random semantics: top up under-filled rows with random draws.
    if (h.i[0] < 0) {
        uint filled = 0;
        for (uint j = 0; j < p.k; ++j)
            if (h.i[j] >= 0) ++filled;
        ulong s = p.seed ^ ((ulong)(gid + 1) * 0xD1342543DE82EF95UL);
        for (uint r = filled; r < p.k; ++r) {
            const uint q = (uint)(splitmix64(s) % (ulong)p.n);
            const float dd =
                nnd_dist(xi, data + (ulong)q * p.dim, p.dim, p.metric);
            h.checked_push(dd, (int)q, 1);
        }
    }
    h.store(heap_dist, heap_idx, heap_flag, (ulong)gid);
}

// 3b: forward sampling — own heap entries into own candidate lists.
// Candidate arrays are host-reset to -1 and counts to 0 before each
// iteration.
kernel void nnd_sample_fwd(device const int* heap_idx [[buffer(0)]],
                           device const uchar* heap_flag [[buffer(1)]],
                           device int* new_cand [[buffer(2)]],
                           device int* old_cand [[buffer(3)]],
                           device uint* new_count [[buffer(4)]],
                           device uint* old_count [[buffer(5)]],
                           constant NndParams& p [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    uint nc = 0, oc = 0;
    for (uint s = 0; s < p.k; ++s) {
        const int j = heap_idx[(ulong)gid * p.k + s];
        if (j < 0) continue;
        if (heap_flag[(ulong)gid * p.k + s]) {
            if (nc < p.n_cand) new_cand[(ulong)gid * p.n_cand + nc++] = j;
        } else {
            if (oc < p.n_cand) old_cand[(ulong)gid * p.n_cand + oc++] = j;
        }
    }
    new_count[gid] = nc;
    old_count[gid] = oc;
}

// 3c: reverse sampling — heap entry (j, s) appends j to the candidate list
// of heap_idx[j, s]. Atomic append with reservoir replacement past the cap.
kernel void nnd_sample_rev(device const int* heap_idx [[buffer(0)]],
                           device const uchar* heap_flag [[buffer(1)]],
                           device int* new_cand [[buffer(2)]],
                           device int* old_cand [[buffer(3)]],
                           device atomic_uint* new_count [[buffer(4)]],
                           device atomic_uint* old_count [[buffer(5)]],
                           constant NndParams& p [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n * p.k) return;
    const int t = heap_idx[gid];
    if (t < 0) return;
    const uint j = gid / p.k;  // owner of the heap entry
    const bool isnew = heap_flag[gid] != 0;
    device atomic_uint* cnt = (isnew ? new_count : old_count) + (uint)t;
    device int* list =
        (isnew ? new_cand : old_cand) + (ulong)(uint)t * p.n_cand;
    const uint c = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed);
    if (c < p.n_cand) {
        list[c] = (int)j;
    } else {
        ulong s = p.seed ^ ((ulong)(p.iter + 1) * 0xAF251AF3B0F025B5UL) ^
                  ((ulong)(gid + 1) * 0xD1342543DE82EF95UL);
        const uint r = (uint)(splitmix64(s) % (ulong)(c + 1));
        if (r < p.n_cand) list[r] = (int)j;  // racy overwrite by design
    }
}

// 3d: clear the "new" flag of heap entries sampled into their owner's new
// list (new_build_candidates' final loop).
kernel void nnd_clear_flags(device const int* heap_idx [[buffer(0)]],
                            device uchar* heap_flag [[buffer(1)]],
                            device const int* new_cand [[buffer(2)]],
                            device const uint* new_count [[buffer(3)]],
                            constant NndParams& p [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n * p.k) return;
    const int idx = heap_idx[gid];
    if (idx < 0 || heap_flag[gid] == 0) return;
    const uint i = gid / p.k;
    const uint cn = min(new_count[i], p.n_cand);
    for (uint m = 0; m < cn; ++m) {
        if (new_cand[(ulong)i * p.n_cand + m] == idx) {
            heap_flag[gid] = 0;
            return;
        }
    }
}

// Join one candidate list of a pivot against node i's register heap.
static inline uint nnd_join_list(thread NndHeap& h, device const float* xi,
                                 device const float* data,
                                 device const int* list, uint len, uint dim,
                                 uint metric) {
    uint c = 0;
    for (uint m = 0; m < len; ++m) {
        const int q = list[m];
        if (q < 0) continue;
        const float dd =
            nnd_dist(xi, data + (ulong)(uint)q * dim, dim, metric);
        if (dd >= h.d[0]) continue;  // cheap pre-filter before the dedup scan
        c += (uint)h.checked_push(dd, q, 1);
    }
    return c;
}

// 3e: receiving-side local join + per-node serial heap merge.
kernel void nnd_join(device const float* data [[buffer(0)]],
                     device float* heap_dist [[buffer(1)]],
                     device int* heap_idx [[buffer(2)]],
                     device uchar* heap_flag [[buffer(3)]],
                     device const int* new_cand [[buffer(4)]],
                     device const int* old_cand [[buffer(5)]],
                     device const uint* new_count [[buffer(6)]],
                     device const uint* old_count [[buffer(7)]],
                     device atomic_uint* updates [[buffer(8)]],
                     constant NndParams& p [[buffer(9)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    NndHeap h;
    h.load(heap_dist, heap_idx, heap_flag, p.k, (ulong)gid);
    device const float* xi = data + (ulong)gid * p.dim;
    uint c = 0;
    const uint cn = min(new_count[gid], p.n_cand);
    const uint co = min(old_count[gid], p.n_cand);
    for (uint a = 0; a < cn; ++a) {  // new pivots: join new + old targets
        const int pv = new_cand[(ulong)gid * p.n_cand + a];
        if (pv < 0) continue;
        const ulong base = (ulong)(uint)pv * p.n_cand;
        c += nnd_join_list(h, xi, data, new_cand + base,
                           min(new_count[(uint)pv], p.n_cand), p.dim,
                           p.metric);
        c += nnd_join_list(h, xi, data, old_cand + base,
                           min(old_count[(uint)pv], p.n_cand), p.dim,
                           p.metric);
    }
    for (uint a = 0; a < co; ++a) {  // old pivots: join new targets only
        const int pv = old_cand[(ulong)gid * p.n_cand + a];
        if (pv < 0) continue;
        const ulong base = (ulong)(uint)pv * p.n_cand;
        c += nnd_join_list(h, xi, data, new_cand + base,
                           min(new_count[(uint)pv], p.n_cand), p.dim,
                           p.metric);
    }
    h.store(heap_dist, heap_idx, heap_flag, (ulong)gid);
    if (c > 0) atomic_fetch_add_explicit(updates, c, memory_order_relaxed);
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

// splitmix64 (shared, defined above the kernel-3 section) derives the
// well-mixed per-(edge, epoch) tau seeds here.

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
