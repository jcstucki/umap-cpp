// Host side of the SGD layout Metal kernel (src/metal/umap_kernels.metal).
// See include/umap/metal/gpu_layout.hpp for semantics / documented deviations.
#include "umap/metal/gpu_layout.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "metal_context.hpp"

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

// Number of epochs encoded per command buffer. Each epoch is its own compute
// encoder (encoder boundaries are the per-epoch barriers); batching amortises
// commit/scheduling overhead.
constexpr int kEpochsPerCommandBuffer = 25;

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

}  // namespace

bool gpu_optimize_layout_euclidean(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    int n_epochs, int n_vertices, const std::vector<double>& epochs_per_sample,
    double a, double b, const int64_t rng_state[3], double gamma,
    double initial_alpha, double negative_sample_rate, bool move_other) {
    MTL::ComputePipelineState* pso = pipeline("sgd_epoch");
    if (!pso) return false;
    MetalContext& ctx = context();

    const std::size_t n_edges = epochs_per_sample.size();
    if (n_edges == 0 || n_epochs <= 0) return true;

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

    for (int n0 = 0; n0 < n_epochs; n0 += kEpochsPerCommandBuffer) {
        const int n1 = std::min(n_epochs, n0 + kEpochsPerCommandBuffer);
        MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
        for (int n = n0; n < n1; ++n) {
            // Alpha decay on the host, exactly as in layouts.py.
            params.alpha = static_cast<float>(
                initial_alpha *
                (1.0 - (static_cast<double>(n) / static_cast<double>(n_epochs))));
            params.epoch = static_cast<uint32_t>(n);

            for (std::size_t off = 0; off < n_edges; off += edge_chunk) {
                params.edge_offset = static_cast<uint32_t>(off);
                const std::size_t count = std::min(edge_chunk, n_edges - off);
                MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(pso);
                enc->setBuffer(head_buf, 0, 0);
                enc->setBuffer(tail_buf, 0, 1);
                enc->setBuffer(head_idx_buf, 0, 2);
                enc->setBuffer(tail_idx_buf, 0, 3);
                enc->setBuffer(eps_buf, 0, 4);
                enc->setBuffer(epns_buf, 0, 5);
                enc->setBuffer(eons_buf, 0, 6);
                enc->setBuffer(eonns_buf, 0, 7);
                enc->setBytes(&params, sizeof(params), 8);
                enc->dispatchThreads(MTL::Size::Make(count, 1, 1),
                                     MTL::Size::Make(tg, 1, 1));
                enc->endEncoding();
            }
        }
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    std::memcpy(head_embedding, head_buf->contents(), head_bytes);
    if (!aliased) {
        std::memcpy(tail_embedding, tail_buf->contents(), tail_bytes);
    }

    head_buf->release();
    if (!aliased) tail_buf->release();
    head_idx_buf->release();
    tail_idx_buf->release();
    eps_buf->release();
    epns_buf->release();
    eons_buf->release();
    eonns_buf->release();
    return true;
}

}  // namespace metal
}  // namespace umap
