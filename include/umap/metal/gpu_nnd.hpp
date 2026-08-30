// GPU (Apple Metal) NN-descent kNN graph construction — gpu-nnd pass,
// 2026-08-29. Completes the device=GPU kNN coverage above the exact
// brute-force engine's O(n²) crossover: large-n dense standard-branch fits
// build their (approximate) kNN graph on the GPU instead of falling back to
// the CPU NN-descent port or paying the quadratic exact cost.
//
// Faithfulness contract: QUALITY parity (like the rest of device=GPU), not
// bit parity. The CPU port (src/nn_descent.cpp) stays the
// RandomState-stream bit-exact reproduction of pynndescent; this engine
// reproduces the same algorithm shape — RP-forest leaf initialisation,
// fwd/reverse candidate sampling capped at max_candidates=60, local-join
// descent iterations with checked heap pushes, delta=0.001 early
// termination, n_iters = max(5, round(log2 n)) — with these documented
// deviations:
//  - the RP forest is built HOST-side by the existing bit-exact rp-tree code
//    but from a PRIVATE NumpyRandomState seeded by the caller (a GPU fit
//    skips the model RandomState's NN-descent draws, exactly like the
//    exact-engine path); leaf-pair distance evaluation runs on the GPU (one
//    thread per node over its own leaf memberships — race-free);
//  - distances are float32 on device (CPU alternative kernels are float32
//    too, but the euclidean correction / cosine form differ in shape:
//    internal sqeuclidean + host sqrt for euclidean/l2, plain float32 cosine
//    instead of the log2 "alternative" form — identical orderings);
//  - candidate sampling keeps ALL forward heap entries (k <= cap) and
//    reservoir-samples the reverse direction under an atomic counter: which
//    reverse candidates survive past the cap is scheduling-dependent, so —
//    unlike the exact engine — repeated GPU NN-descent runs are NOT
//    bit-identical to each other (same family as the hogwild SGD);
//  - the local join is evaluated receiving-side (each node joins the
//    candidate lists of its candidates and merges into its own heap row
//    serially — one thread per node, no atomics, no races) instead of
//    pushing each pivot-side pair to both endpoints.
// Output rows are sorted ascending by (distance, index) — the exact-engine
// convention; each row contains the node itself (distance 0) exactly as the
// CPU NN-descent graph does.
#pragma once

#include <cstdint>

#include "umap/metal/gpu_knn.hpp"  // KnnMetric

namespace umap {
namespace metal {

// Heap rows live in a per-thread register copy during the merge kernels.
constexpr int kGpuNndMaxK = 64;

// Exact-vs-NN-descent crossover for device=GPU standard-branch dense fits
// (auto mode). Measured on M-series (dim 16, k 15, hidden case
// `-ns -tc="gpu nnd timing*"`, 2026-08-29, 3 stable runs): GPU NN-descent is
// faster at every measured size — 20k: 140 ms vs exact 307 ms; 50k: 407 ms
// vs 2251 ms; 100k: 968 ms vs 8963 ms (recall 0.9975 / 0.9952 / 0.9931,
// matching the bit-exact CPU NN-descent port's recall to ~1e-4 at each
// size). Exact nevertheless keeps the sub-30k range: its absolute cost
// there is modest (< ~0.7 s) and it returns recall-1.0 graphs — strictly
// better than what the CPU standard branch would produce — while above 30k
// the O(n²) term dominates (>= ~3x the descent, growing quadratically) for
// a <= ~0.7% recall difference. Interpolated crossing of the "exact costs
// 3x the descent" line sits right around n ≈ 30k.
constexpr int kGpuNndCrossoverN = 30000;

// True for the metrics the NN-descent kernels implement: euclidean/l2,
// sqeuclidean, manhattan (+ aliases), cosine. Other engine metrics stay on
// the exact engine at any n (correct, O(n²)).
bool gpu_nnd_metric_supported(KnnMetric metric);

// Routing decision for a device=GPU standard-branch dense self-kNN.
// exact: the brute-force engine (gpu_brute_force_knn); nnd: this engine.
// Resolution: unsupported metric / k > kGpuNndMaxK / tiny n  -> exact;
// otherwise $UMAP_GPU_KNN_MODE = "exact" | "nnd" forces the mode and
// anything else ("auto", unset) selects nnd iff n > kGpuNndCrossoverN.
enum class GpuKnnMode { exact, nnd };
GpuKnnMode gpu_knn_mode(int n, int k, KnnMetric metric);

// Builds the approximate self-kNN graph of `data` (n x dim row-major
// float32) on the GPU. k in [1, kGpuNndMaxK]; metric must satisfy
// gpu_nnd_metric_supported. seed drives the host RP forest and the device
// sampling draws (fits pass the model random_state; note the run-to-run
// non-determinism above). out_idx/out_dists are n x k, ascending by
// (distance, index). Returns false (outputs untouched) if no Metal
// device/metallib is available or a buffer would exceed the device limit.
// Throws std::invalid_argument for out-of-range k or unsupported metrics.
bool gpu_nn_descent_knn(const float* data, int n, int dim, int k,
                        KnnMetric metric, uint64_t seed, int32_t* out_idx,
                        float* out_dists);

}  // namespace metal
}  // namespace umap
