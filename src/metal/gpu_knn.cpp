// Host side of the brute-force kNN Metal kernel (src/metal/umap_kernels.metal).
#include "umap/metal/gpu_knn.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "metal_context.hpp"

namespace umap {
namespace metal {

namespace {

// Mirrors struct KnnParams in umap_kernels.metal.
struct KnnParams {
    uint32_t n;
    uint32_t nq;
    uint32_t dim;
    uint32_t k;
    uint32_t metric;
};

}  // namespace

bool gpu_brute_force_knn(const float* data, int n, int dim,
                         const float* queries, int nq, int k, KnnMetric metric,
                         int32_t* out_idx, float* out_dists) {
    if (k < 1 || k > 64) {
        throw std::invalid_argument("gpu_brute_force_knn: k must be in [1, 64]");
    }
    MTL::ComputePipelineState* pso = pipeline("brute_force_knn");
    if (!pso) return false;
    MetalContext& ctx = context();

    const std::size_t data_bytes =
        static_cast<std::size_t>(n) * dim * sizeof(float);
    const std::size_t query_bytes =
        static_cast<std::size_t>(nq) * dim * sizeof(float);
    const std::size_t out_n = static_cast<std::size_t>(nq) * k;

    MTL::Buffer* data_buf = ctx.device->newBuffer(
        data, data_bytes, MTL::ResourceStorageModeShared);
    // queries may alias data (self-kNN): reuse the buffer.
    const bool self_query = (queries == data && nq == n);
    MTL::Buffer* query_buf =
        self_query ? data_buf
                   : ctx.device->newBuffer(queries, query_bytes,
                                           MTL::ResourceStorageModeShared);
    MTL::Buffer* idx_buf = ctx.device->newBuffer(
        out_n * sizeof(int32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* dist_buf = ctx.device->newBuffer(
        out_n * sizeof(float), MTL::ResourceStorageModeShared);

    KnnParams params;
    params.n = static_cast<uint32_t>(n);
    params.nq = static_cast<uint32_t>(nq);
    params.dim = static_cast<uint32_t>(dim);
    params.k = static_cast<uint32_t>(k);
    params.metric = (metric == KnnMetric::euclidean) ? 0u : 1u;

    MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(data_buf, 0, 0);
    enc->setBuffer(query_buf, 0, 1);
    enc->setBuffer(idx_buf, 0, 2);
    enc->setBuffer(dist_buf, 0, 3);
    enc->setBytes(&params, sizeof(params), 4);

    const NS::UInteger tg =
        std::min<NS::UInteger>(pso->maxTotalThreadsPerThreadgroup(), 256);
    enc->dispatchThreads(MTL::Size::Make(nq, 1, 1), MTL::Size::Make(tg, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    std::memcpy(out_idx, idx_buf->contents(), out_n * sizeof(int32_t));
    std::memcpy(out_dists, dist_buf->contents(), out_n * sizeof(float));

    data_buf->release();
    if (!self_query) query_buf->release();
    idx_buf->release();
    dist_buf->release();
    return true;
}

}  // namespace metal
}  // namespace umap
