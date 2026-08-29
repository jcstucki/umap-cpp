#include <metal_stdlib>
using namespace metal;

// Exact penalized k-NN. One thread per query point i; each thread scans all n
// points, computes base distance (cosine or euclidean over F feature dims)
// plus the time/price/return/ticker penalties, and maintains a sorted top-k in
// registers (insertion, with worst-neighbor early reject).
//
// metal-knn additions: auxiliary utility kernel (not used by the UMAP
// pipeline). Compiled to its own .air and linked into the single
// umap_kernels.metallib alongside src/metal/umap_kernels.metal; host in
// src/metal/metal_knn.cpp, public API include/umap/metal_knn.h.
struct Params {
    int   n;
    int   F;
    int   k;
    int   cosine;   // 1 = cosine base, 0 = euclidean
    float wt, wp, wr, wk;
};

kernel void knn_penalized(
    device const float* X    [[buffer(0)]],   // n*F, row-major (unit rows if cosine)
    device const float* sq   [[buffer(1)]],   // n (squared norms, euclidean)
    device const float* dz   [[buffer(2)]],   // n (z-scored day)
    device const float* cz   [[buffer(3)]],   // n (z-scored close)
    device const float* rz   [[buffer(4)]],   // n (z-scored return)
    device const int*   tid  [[buffer(5)]],   // n (ticker id)
    device int*         outI [[buffer(6)]],   // n*k
    device float*       outD [[buffer(7)]],   // n*k
    constant Params&    P    [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int i = int(gid);
    if (i >= P.n) return;
    const int F = P.F, k = P.k;

    float xi[8];
    for (int f = 0; f < F; ++f) xi[f] = X[i * F + f];
    float dzi = dz[i], czi = cz[i], rzi = rz[i];
    int   tii = tid[i];
    float sqi = sq[i];

    float kd[64];
    int   ki[64];
    for (int t = 0; t < k; ++t) { kd[t] = 1e30f; ki[t] = -1; }
    float worst = 1e30f;

    for (int j = 0; j < P.n; ++j) {
        if (j == i) continue;
        float dot = 0.0f;
        for (int f = 0; f < F; ++f) dot += xi[f] * X[j * F + f];
        float base;
        if (P.cosine) {
            base = 1.0f - dot;
        } else {
            float d2 = sqi + sq[j] - 2.0f * dot;
            base = sqrt(max(d2, 0.0f));
        }
        float pen = 0.0f;
        if (P.wt > 0.0f) pen += P.wt * fabs(dzi - dz[j]);
        if (P.wp > 0.0f) pen += P.wp * fabs(czi - cz[j]);
        if (P.wr > 0.0f) pen += P.wr * fabs(rzi - rz[j]);
        if (P.wk > 0.0f && tii != tid[j]) pen += P.wk;
        float d = base + pen;
        if (d < worst) {
            int p = k - 1;
            while (p > 0 && kd[p - 1] > d) { kd[p] = kd[p - 1]; ki[p] = ki[p - 1]; --p; }
            kd[p] = d; ki[p] = j;
            worst = kd[k - 1];
        }
    }
    for (int t = 0; t < k; ++t) {
        outI[i * k + t] = ki[t];
        outD[i * k + t] = kd[t];
    }
}
