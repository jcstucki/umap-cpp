// Host side of the SGD layout Metal kernel (src/metal/umap_kernels.metal).
// See include/umap/metal/gpu_layout.hpp for semantics / documented deviations.
#include "umap/metal/gpu_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "metal_context.hpp"
#include "umap/layouts.hpp"  // densmap-gpu additions: DensmapParams + host stats

namespace umap {
namespace metal {

namespace {

// Mirrors struct SgdParams in umap_kernels.metal (10 x 4-byte fields, then
// three 8-byte words at offset 40 — layout identical in C++ and MSL).
struct SgdParams {
    uint32_t n_edges;
    uint32_t dim;
    uint32_t n_vertices;
    uint32_t move_other;
    float a;
    float b;
    float gamma;
    float alpha;
    uint32_t epoch;
    uint32_t edge_offset;
    int64_t rng0;
    int64_t rng1;
    int64_t rng2;
};
static_assert(sizeof(SgdParams) == 64, "SgdParams layout must match MSL");

// Cap on the epochs encoded per command buffer. Each epoch is its own compute
// encoder (encoder boundaries are the per-epoch barriers); batching amortises
// commit/scheduling overhead.
//
// watchdog-hardening additions: the actual batch size adapts at run time
// (below) — a fixed 25-epoch buffer measures a couple hundred ms on 5k-point
// graphs but scales linearly with the edge count, and macOS sporadically
// aborts multi-second GPU submissions (observed on the kNN engines at ~2 s,
// see src/metal/gpu_knn.cpp kChunkWorkBudget), so long fits on large graphs
// could plausibly blow a single batch past the watchdog.
constexpr int kEpochsPerCommandBuffer = 25;

// watchdog-hardening additions: target duration for one command buffer.
// Batches are sized from the measured per-epoch time to stay near this
// budget (the kNN engines' couple-hundred-ms work-budget scale, far below
// observed watchdog aborts); the first buffer probes a single epoch. An
// epoch is the atomic unit — one epoch longer than the budget still runs
// alone (as the densmap-corrected region always does).
constexpr double kBatchBudgetMs = 250.0;

int next_batch_epochs(std::chrono::steady_clock::duration measured,
                      int epochs_run) {
    const double ms =
        std::chrono::duration<double, std::milli>(measured).count();
    if (ms <= 0.0) return kEpochsPerCommandBuffer;
    const double per_epoch = ms / static_cast<double>(std::max(1, epochs_run));
    return std::clamp(static_cast<int>(kBatchBudgetMs / per_epoch), 1,
                      kEpochsPerCommandBuffer);
}

// Each epoch's edges are split into this many sub-dispatches with encoder
// barriers between them. Rationale: hogwild races lose concurrent
// read-modify-write updates on a vertex; numba's prange runs ~n_threads
// concurrent chunks, but a single full-grid GPU dispatch runs (nearly) *all*
// edges concurrently, and on small graphs the lost attractive updates
// measurably degrade embedding quality (see tests). With E/32 edges per
// sub-dispatch the expected number of concurrently-racing edges per vertex is
// ~degree/32 < 1, restoring CPU-hogwild-level quality while keeping tens of
// thousands of threads in flight on large graphs.
constexpr std::size_t kEdgeSubDispatches = 32;
constexpr std::size_t kMinEdgeChunk = 64;

// --- densmap-gpu additions ---------------------------------------------------
// Mirrors struct DensSgdParams in umap_kernels.metal (5 x float32). The
// scalars are the per-epoch densmap_flag values, computed host-side by
// umap::densmap_epoch_stats (see gpu_layout.hpp for the work-split rationale).
struct DensSgdParams {
    float lambda_;
    float mu_tot;
    float re_std;
    float re_mean;
    float re_cov;
};
static_assert(sizeof(DensSgdParams) == 20, "DensSgdParams layout must match MSL");

// The densmap SGD pipeline is a separate kernel entry point sharing the edge
// body with sgd_epoch (not a metal_context named slot — cached here like the
// kNN engine's specialized pipelines).
MTL::ComputePipelineState* densmap_sgd_pipeline() {
    // Magic-static once-init (thread-safe); nullptr failures are cached too.
    static MTL::ComputePipelineState* cached = []() -> MTL::ComputePipelineState* {
        MetalContext& ctx = context();
        if (!ctx.ok()) return nullptr;
        NS::String* fn_name =
            NS::String::string("sgd_epoch_densmap", NS::UTF8StringEncoding);
        MTL::Function* fn = ctx.library->newFunction(fn_name);
        if (!fn) return nullptr;
        NS::Error* error = nullptr;
        MTL::ComputePipelineState* pso =
            ctx.device->newComputePipelineState(fn, &error);
        fn->release();
        return pso;
    }();
    return cached;
}
// -----------------------------------------------------------------------------

}  // namespace

bool gpu_optimize_layout_euclidean(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    int n_epochs, int n_vertices, const std::vector<double>& epochs_per_sample,
    double a, double b, const int64_t rng_state[3], double gamma,
    double initial_alpha, double negative_sample_rate, bool move_other,
    const DensmapParams* densmap_kwds) {
    MTL::ComputePipelineState* pso = pipeline("sgd_epoch");
    if (!pso) return false;
    MetalContext& ctx = context();

    const std::size_t n_edges = epochs_per_sample.size();
    if (n_edges == 0 || n_epochs <= 0) return true;

    // --- densmap-gpu additions ----------------------------------------------
    // First epoch with the density correction active:
    // densmap_flag(n) = lambda > 0 and (n+1)/n_epochs > 1 - frac — monotone
    // in n, so epochs [0, first_dens_epoch) batch exactly like the plain path
    // and [first_dens_epoch, n_epochs) run one host-synchronised command
    // buffer each (the per-epoch scalars depend on the embedding state).
    const bool densmap = densmap_kwds != nullptr;
    int first_dens_epoch = n_epochs;
    if (densmap && densmap_kwds->lambda > 0.0) {
        for (int n = 0; n < n_epochs; ++n) {
            if ((static_cast<double>(n + 1) / static_cast<double>(n_epochs)) >
                (1.0 - densmap_kwds->frac)) {
                first_dens_epoch = n;
                break;
            }
        }
    }
    const bool dens_active = densmap && first_dens_epoch < n_epochs;
    MTL::ComputePipelineState* pso_dens =
        dens_active ? densmap_sgd_pipeline() : nullptr;
    if (dens_active) {
        // The kernel needs per-edge mu and per-vertex R aligned with this
        // graph; the fit configuration also implies aliased head/tail
        // (move_other=true). Anything else is a caller bug — fall back.
        if (!pso_dens || !move_other || head_embedding != tail_embedding ||
            densmap_kwds->mu.size() != n_edges ||
            densmap_kwds->R.size() != static_cast<std::size_t>(n_vertices)) {
            return false;
        }
    }
    // ------------------------------------------------------------------------

    // float32 device copies of the float64 bookkeeping (Metal has no double;
    // documented deviation).
    std::vector<float> eps_f(n_edges), epns_f(n_edges);
    for (std::size_t i = 0; i < n_edges; ++i) {
        eps_f[i] = static_cast<float>(epochs_per_sample[i]);
        epns_f[i] =
            static_cast<float>(epochs_per_sample[i] / negative_sample_rate);
    }

    const bool aliased = (head_embedding == tail_embedding);
    const std::size_t head_bytes =
        static_cast<std::size_t>(n_head_vertices) * dim * sizeof(float);
    const std::size_t tail_bytes =
        static_cast<std::size_t>(n_vertices) * dim * sizeof(float);

    MTL::Buffer* head_buf = ctx.device->newBuffer(
        head_embedding, head_bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* tail_buf =
        aliased ? head_buf
                : ctx.device->newBuffer(tail_embedding, tail_bytes,
                                        MTL::ResourceStorageModeShared);
    MTL::Buffer* head_idx_buf = ctx.device->newBuffer(
        head.data(), n_edges * sizeof(int32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* tail_idx_buf = ctx.device->newBuffer(
        tail.data(), n_edges * sizeof(int32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* eps_buf = ctx.device->newBuffer(
        eps_f.data(), n_edges * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* epns_buf = ctx.device->newBuffer(
        epns_f.data(), n_edges * sizeof(float), MTL::ResourceStorageModeShared);
    // epoch_of_next_sample = epochs_per_sample.copy();
    // epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    MTL::Buffer* eons_buf = ctx.device->newBuffer(
        eps_f.data(), n_edges * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* eonns_buf = ctx.device->newBuffer(
        epns_f.data(), n_edges * sizeof(float), MTL::ResourceStorageModeShared);

    // --- densmap-gpu additions ----------------------------------------------
    // mu / R are constant across epochs; phi_sum / re_sum are refilled from
    // the host stats before each corrected epoch (shared storage, so a plain
    // memcpy into contents() is coherent once the previous buffer completed).
    const std::size_t nv_bytes =
        static_cast<std::size_t>(n_vertices) * sizeof(float);
    MTL::Buffer* dens_mu_buf = nullptr;
    MTL::Buffer* dens_R_buf = nullptr;
    MTL::Buffer* dens_phi_buf = nullptr;
    MTL::Buffer* dens_re_buf = nullptr;
    float dens_lambda = 0.0f;
    float dens_mu_tot = 0.0f;
    std::vector<float> re_sum_host, phi_sum_host;
    if (dens_active) {
        dens_mu_buf =
            ctx.device->newBuffer(densmap_kwds->mu.data(),
                                  n_edges * sizeof(float),
                                  MTL::ResourceStorageModeShared);
        dens_R_buf = ctx.device->newBuffer(densmap_kwds->R.data(), nv_bytes,
                                           MTL::ResourceStorageModeShared);
        dens_phi_buf =
            ctx.device->newBuffer(nv_bytes, MTL::ResourceStorageModeShared);
        dens_re_buf =
            ctx.device->newBuffer(nv_bytes, MTL::ResourceStorageModeShared);
        dens_lambda = static_cast<float>(densmap_kwds->lambda);
        dens_mu_tot = densmap_mu_tot(densmap_kwds->mu_sum);
    }
    // ------------------------------------------------------------------------

    SgdParams params;
    params.n_edges = static_cast<uint32_t>(n_edges);
    params.dim = static_cast<uint32_t>(dim);
    params.n_vertices = static_cast<uint32_t>(n_vertices);
    params.move_other = move_other ? 1u : 0u;
    params.a = static_cast<float>(a);
    params.b = static_cast<float>(b);
    params.gamma = static_cast<float>(gamma);
    params.edge_offset = 0;
    params.rng0 = rng_state[0];
    params.rng1 = rng_state[1];
    params.rng2 = rng_state[2];

    const NS::UInteger tg =
        std::min<NS::UInteger>(pso->maxTotalThreadsPerThreadgroup(), 256);

    // Edge sub-dispatch size (see kEdgeSubDispatches above);
    // UMAP_GPU_EDGE_CHUNK overrides for experiments.
    std::size_t edge_chunk = std::max(
        kMinEdgeChunk,
        (n_edges + kEdgeSubDispatches - 1) / kEdgeSubDispatches);
    if (const char* env = std::getenv("UMAP_GPU_EDGE_CHUNK")) {
        const long v = std::atol(env);
        if (v > 0) edge_chunk = static_cast<std::size_t>(v);
    }

    // Encodes epoch n's sub-dispatches into cmd; dp non-null selects the
    // densmap pipeline + the extra density buffers.
    const auto encode_epoch = [&](MTL::CommandBuffer* cmd, int n,
                                  const DensSgdParams* dp) {
        // Alpha decay on the host, exactly as in layouts.py.
        params.alpha = static_cast<float>(
            initial_alpha *
            (1.0 - (static_cast<double>(n) / static_cast<double>(n_epochs))));
        params.epoch = static_cast<uint32_t>(n);

        for (std::size_t off = 0; off < n_edges; off += edge_chunk) {
            params.edge_offset = static_cast<uint32_t>(off);
            const std::size_t count = std::min(edge_chunk, n_edges - off);
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(dp != nullptr ? pso_dens : pso);
            enc->setBuffer(head_buf, 0, 0);
            enc->setBuffer(tail_buf, 0, 1);
            enc->setBuffer(head_idx_buf, 0, 2);
            enc->setBuffer(tail_idx_buf, 0, 3);
            enc->setBuffer(eps_buf, 0, 4);
            enc->setBuffer(epns_buf, 0, 5);
            enc->setBuffer(eons_buf, 0, 6);
            enc->setBuffer(eonns_buf, 0, 7);
            enc->setBytes(&params, sizeof(params), 8);
            if (dp != nullptr) {
                enc->setBuffer(dens_mu_buf, 0, 9);
                enc->setBuffer(dens_R_buf, 0, 10);
                enc->setBuffer(dens_phi_buf, 0, 11);
                enc->setBuffer(dens_re_buf, 0, 12);
                enc->setBytes(dp, sizeof(DensSgdParams), 13);
            }
            enc->dispatchThreads(MTL::Size::Make(count, 1, 1),
                                 MTL::Size::Make(tg, 1, 1));
            enc->endEncoding();
        }
    };

    // watchdog-hardening additions (mirrors the kNN engines' commit_and_wait,
    // src/metal/gpu_knn.cpp): a command buffer the macOS GPU scheduler
    // aborted or errored leaves the device-side embedding torn, and before
    // this check the epoch loop silently carried on and copied the wreckage
    // back over the caller's arrays. Every buffer's status is now verified;
    // a failure abandons the run WITHOUT the write-back below, so the
    // caller's arrays keep the initial embedding (the device works on
    // copies) and the documented CPU fallback restarts from exactly the
    // state this call received — the cheapest correct recovery: no pristine
    // host copy is ever needed. UMAP_GPU_FAIL_AFTER_BATCH=n (undocumented;
    // fallback-plumbing tests) makes the (n+1)-th command buffer of this
    // call report failure after it really ran.
    long fail_after = -1;
    if (const char* env = std::getenv("UMAP_GPU_FAIL_AFTER_BATCH")) {
        fail_after = std::atol(env);
    }
    long batches_done = 0;
    const auto commit_and_wait = [&](MTL::CommandBuffer* cmd,
                                     int first_epoch) -> bool {
        cmd->commit();
        cmd->waitUntilCompleted();
        bool completed = cmd->status() == MTL::CommandBufferStatusCompleted &&
                         cmd->error() == nullptr;
        if (completed && fail_after >= 0 && batches_done == fail_after) {
            completed = false;  // injected failure (tests)
        }
        ++batches_done;
        if (!completed) {
            std::fprintf(stderr,
                         "UMAP: GPU layout SGD command buffer failed at epoch "
                         "%d/%d (Metal status %ld); leaving the embedding "
                         "untouched for the CPU fallback.\n",
                         first_epoch, n_epochs,
                         static_cast<long>(cmd->status()));
        }
        return completed;
    };

    bool ok = true;
    int batch_epochs = 1;  // probe one epoch, then size batches from its time
    int n = 0;
    while (ok && n < n_epochs) {
        if (n < first_dens_epoch) {
            // Plain epochs: batched per command buffer (adaptive size, see
            // kBatchBudgetMs above).
            const int n0 = n;
            const int n1 =
                std::min({n_epochs, n + batch_epochs, first_dens_epoch});
            MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
            for (; n < n1; ++n) encode_epoch(cmd, n, nullptr);
            const auto t0 = std::chrono::steady_clock::now();
            ok = commit_and_wait(cmd, n0);
            if (ok) {
                batch_epochs = next_batch_epochs(
                    std::chrono::steady_clock::now() - t0, n1 - n0);
            }
        } else {
            // densmap-gpu additions: corrected epoch. The previous command
            // buffer completed, so the shared embedding buffer is coherent;
            // run the CPU epoch-init + scalar reductions on it, publish
            // phi_sum / re_sum to the device, then dispatch the epoch.
            float* emb = static_cast<float*>(head_buf->contents());
            const DensmapEpochScalars s = densmap_epoch_stats(
                emb, emb, head.data(), tail.data(), n_edges, a, b, dim,
                *densmap_kwds, n_vertices, re_sum_host, phi_sum_host);
            std::memcpy(dens_phi_buf->contents(), phi_sum_host.data(),
                        nv_bytes);
            std::memcpy(dens_re_buf->contents(), re_sum_host.data(), nv_bytes);
            const DensSgdParams dp{dens_lambda, dens_mu_tot, s.re_std,
                                   s.re_mean, s.re_cov};
            MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
            encode_epoch(cmd, n, &dp);
            ok = commit_and_wait(cmd, n);
            ++n;
        }
    }

    // watchdog-hardening additions: results are copied back only after every
    // command buffer completed — on failure the host embeddings still hold
    // the initial state (see the fallback contract in gpu_layout.hpp).
    if (ok) {
        std::memcpy(head_embedding, head_buf->contents(), head_bytes);
        if (!aliased) {
            std::memcpy(tail_embedding, tail_buf->contents(), tail_bytes);
        }
    }

    head_buf->release();
    if (!aliased) tail_buf->release();
    head_idx_buf->release();
    tail_idx_buf->release();
    eps_buf->release();
    epns_buf->release();
    eons_buf->release();
    eonns_buf->release();
    // densmap-gpu additions
    if (dens_mu_buf) dens_mu_buf->release();
    if (dens_R_buf) dens_R_buf->release();
    if (dens_phi_buf) dens_phi_buf->release();
    if (dens_re_buf) dens_re_buf->release();
    return ok;
}

}  // namespace metal
}  // namespace umap
