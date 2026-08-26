// GPU (Apple Metal) port of optimize_layout_euclidean — Phase 3, branch
// metal-gpu. Mirrors umap::optimize_layout_euclidean (include/umap/layouts.hpp)
// with one GPU thread per 1-simplex (edge) per epoch and hogwild-style
// unsynchronized float32 updates on the embedding buffer.
//
// Race semantics: numba's prange(parallel=True) path performs exactly the same
// unsequenced racy float32 loads/stores on the embedding (and the std::thread
// `parallel` mode of the CPU port does too), so the GPU path matches the
// reference's *parallel* semantics: non-deterministic at the bit level,
// equivalent at the embedding-quality level. 32-bit float stores do not tear
// on Metal devices; each embedding scalar is always some previously-written
// value.
//
// Documented deviations from the sequential CPU path (see also the .metal
// source):
//  - Metal has no float64: gradient coefficients, alpha and the
//    epoch_of_next_(negative_)sample bookkeeping are float32 on device (the
//    bookkeeping lives in persistent device buffers across epochs).
//  - RNG: a fresh 3-word tau state is derived per (edge, epoch) from
//    rng_state via splitmix64 and advanced with the exact tau_rand_int
//    recurrence of include/umap/rng.hpp. Exact CPU rng-sequence parity is not
//    provided (numba's parallel path is not sequence-reproducible either).
//  - Alpha decays on the host once per epoch, exactly as in layouts.py.
//  - Gradients are clipped at +/-4, as in layouts.py::clip.
//
// Dispatch structure: epochs are batched several per command buffer; within
// an epoch the edges are split into ~32 sub-dispatches with encoder barriers
// between them (per-epoch barriers match the reference's per-epoch numba
// call). The sub-dispatches bound the number of *concurrently* racing updates
// per vertex at ~degree/32: a single full-grid dispatch would run essentially
// every edge at once, and the resulting lost read-modify-write updates
// measurably weaken cluster attraction on small graphs (numba's prange only
// ever has n_threads concurrent chunks). Override the sub-dispatch size with
// the UMAP_GPU_EDGE_CHUNK environment variable for experiments.
#pragma once

#include <cstdint>
#include <vector>

namespace umap {
namespace metal {

// Signature mirrors umap::optimize_layout_euclidean. head_embedding /
// tail_embedding are n_head_vertices x dim / n_vertices x dim row-major
// float32; when they alias (standard fit), a single device buffer is used.
// head_embedding (and tail_embedding if move_other) are updated in place.
// densmap is not supported on the GPU path (throws, like densmap=true on CPU).
// Returns false (embeddings untouched) if no Metal device is available.
bool gpu_optimize_layout_euclidean(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    int n_epochs, int n_vertices, const std::vector<double>& epochs_per_sample,
    double a, double b, const int64_t rng_state[3], double gamma = 1.0,
    double initial_alpha = 1.0, double negative_sample_rate = 5.0,
    bool move_other = false);

}  // namespace metal
}  // namespace umap
