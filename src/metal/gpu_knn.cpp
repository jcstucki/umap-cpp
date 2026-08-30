// Host side of the exact brute-force kNN Metal engine
// (src/metal/umap_kernels.metal, kernels knn_dense_* / knn_sparse_*).
#include "umap/metal/gpu_knn.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "metal_context.hpp"

namespace umap {
namespace metal {

namespace {

constexpr int kSmallKCap = 64;  // register insertion-list cap (kernel kSmallK)
constexpr int kHeapKCap = 256;  // thread-memory heap cap (kernel kHeapK)

// Mirrors struct KnnDenseParams in umap_kernels.metal.
struct KnnDenseParams {
    uint32_t n;
    uint32_t nq;
    uint32_t dim;
    uint32_t k;
    float p;
};

// Mirrors struct KnnSparseParams in umap_kernels.metal.
struct KnnSparseParams {
    uint32_t n;
    uint32_t nq;
    uint32_t k;
    uint32_t metric;  // 0 = euclidean, 1 = cosine, 2 = sqeuclidean
};

// Lazily-created pipeline cache: dense kernels are function-constant
// specializations (one pipeline per (metric id, k-path)); the two sparse
// kernels dispatch on a runtime metric id and need no specialization.
std::mutex g_knn_mutex;

MTL::ComputePipelineState* dense_pipeline(uint32_t metric_id, bool heap) {
    std::lock_guard<std::mutex> lock(g_knn_mutex);
    static std::unordered_map<uint32_t, MTL::ComputePipelineState*> cache;
    const uint32_t key = metric_id * 2u + (heap ? 1u : 0u);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    MetalContext& ctx = context();
    if (!ctx.ok()) return nullptr;
    MTL::FunctionConstantValues* cv = MTL::FunctionConstantValues::alloc()->init();
    cv->setConstantValue(&metric_id, MTL::DataTypeUInt, NS::UInteger(0));
    NS::String* fn_name = NS::String::string(
        heap ? "knn_dense_heap" : "knn_dense_smallk", NS::UTF8StringEncoding);
    NS::Error* error = nullptr;
    MTL::Function* fn = ctx.library->newFunction(fn_name, cv, &error);
    cv->release();
    if (!fn) return nullptr;
    MTL::ComputePipelineState* pso =
        ctx.device->newComputePipelineState(fn, &error);
    fn->release();
    cache.emplace(key, pso);  // cache nullptr failures too (no retry storms)
    return pso;
}

MTL::ComputePipelineState* sparse_pipeline(bool heap) {
    std::lock_guard<std::mutex> lock(g_knn_mutex);
    static MTL::ComputePipelineState* cache[2] = {nullptr, nullptr};
    static bool tried[2] = {false, false};
    const int slot = heap ? 1 : 0;
    if (tried[slot]) return cache[slot];
    tried[slot] = true;

    MetalContext& ctx = context();
    if (!ctx.ok()) return nullptr;
    NS::String* fn_name = NS::String::string(
        heap ? "knn_sparse_heap" : "knn_sparse_smallk", NS::UTF8StringEncoding);
    MTL::Function* fn = ctx.library->newFunction(fn_name);
    if (!fn) return nullptr;
    NS::Error* error = nullptr;
    cache[slot] = ctx.device->newComputePipelineState(fn, &error);
    fn->release();
    return cache[slot];
}

// newBuffer with a zero length is invalid in Metal; empty CSR arrays (nnz=0)
// still need a live buffer to bind.
MTL::Buffer* copy_buffer(MTL::Device* dev, const void* src,
                         std::size_t bytes) {
    if (bytes == 0) return dev->newBuffer(4, MTL::ResourceStorageModeShared);
    return dev->newBuffer(src, bytes, MTL::ResourceStorageModeShared);
}

// --- gpu-nnd additions -------------------------------------------------------
// Queries are dispatched in slices, one command buffer per slice, so a single
// submission stays well under the macOS GPU watchdog. A monolithic dispatch
// at n = nq = 100k (~2 s) was observed to be sporadically ABORTED by the
// system — and the engine, which never checked the command-buffer status,
// silently returned the zero-filled output buffers (recall collapse in the
// gpu-nnd benchmark uncovered it). The work budget per slice keeps each
// buffer at roughly a couple hundred ms.
constexpr std::size_t kChunkWorkBudget = std::size_t(2) << 33;  // ~1.7e10

std::size_t query_chunk_rows(std::size_t work_per_query, int nq) {
    const std::size_t chunk =
        kChunkWorkBudget / std::max<std::size_t>(1, work_per_query);
    return std::min<std::size_t>(
        static_cast<std::size_t>(nq),
        std::max<std::size_t>(4096, chunk));
}

// waitUntilCompleted + status verification: an errored/aborted buffer means
// the outputs are garbage — callers must fall back (return false).
bool commit_and_wait(MTL::CommandBuffer* cmd) {
    cmd->commit();
    cmd->waitUntilCompleted();
    return cmd->status() == MTL::CommandBufferStatusCompleted &&
           cmd->error() == nullptr;
}
// -----------------------------------------------------------------------------

}  // namespace

bool gpu_brute_force_knn(const float* data, int n, int dim,
                         const float* queries, int nq, int k, KnnMetric metric,
                         int32_t* out_idx, float* out_dists,
                         const KnnMetricParams* metric_params) {
    if (k < 1 || k > kHeapKCap) {
        throw std::invalid_argument(
            "gpu_brute_force_knn: k must be in [1, 256]");
    }
    // Resolve the aux array + exponent for the parameterised metrics
    // (MetricKwds semantics: empty v -> ones, empty vinv -> identity).
    float p_exp = 2.0f;
    std::vector<float> aux(1, 0.0f);
    if (metric == KnnMetric::minkowski || metric == KnnMetric::wminkowski) {
        if (metric_params) p_exp = metric_params->p;
    }
    if (metric == KnnMetric::seuclidean || metric == KnnMetric::wminkowski) {
        if (metric_params && !metric_params->v.empty()) {
            if (metric_params->v.size() != static_cast<std::size_t>(dim)) {
                throw std::invalid_argument(
                    "gpu_brute_force_knn: metric_params->v must have dim "
                    "elements");
            }
            aux = metric_params->v;
        } else {
            aux.assign(static_cast<std::size_t>(dim), 1.0f);
        }
    } else if (metric == KnnMetric::mahalanobis) {
        if (metric_params && !metric_params->vinv.empty()) {
            if (metric_params->vinv.size() !=
                static_cast<std::size_t>(dim) * dim) {
                throw std::invalid_argument(
                    "gpu_brute_force_knn: metric_params->vinv must have "
                    "dim*dim elements");
            }
            aux = metric_params->vinv;
        } else {
            aux.assign(static_cast<std::size_t>(dim) * dim, 0.0f);
            for (int i = 0; i < dim; ++i)
                aux[static_cast<std::size_t>(i) * dim + i] = 1.0f;
        }
    }

    const bool heap = k > kSmallKCap;
    MTL::ComputePipelineState* pso =
        dense_pipeline(static_cast<uint32_t>(metric), heap);
    if (!pso) return false;
    MetalContext& ctx = context();

    const std::size_t data_bytes =
        static_cast<std::size_t>(n) * dim * sizeof(float);
    const std::size_t query_bytes =
        static_cast<std::size_t>(nq) * dim * sizeof(float);
    const std::size_t out_n = static_cast<std::size_t>(nq) * k;
    const std::size_t max_len = ctx.device->maxBufferLength();
    if (data_bytes > max_len || query_bytes > max_len ||
        out_n * sizeof(float) > max_len) {
        return false;  // dataset exceeds a single Metal buffer: CPU fallback
    }

    MTL::Buffer* data_buf = copy_buffer(ctx.device, data, data_bytes);
    // queries may alias data (self-kNN): reuse the buffer.
    const bool self_query = (queries == data && nq == n);
    MTL::Buffer* query_buf =
        self_query ? data_buf : copy_buffer(ctx.device, queries, query_bytes);
    MTL::Buffer* idx_buf = ctx.device->newBuffer(
        out_n * sizeof(int32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* dist_buf = ctx.device->newBuffer(
        out_n * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* aux_buf =
        copy_buffer(ctx.device, aux.data(), aux.size() * sizeof(float));

    KnnDenseParams params;
    params.n = static_cast<uint32_t>(n);
    params.dim = static_cast<uint32_t>(dim);
    params.k = static_cast<uint32_t>(k);
    params.p = p_exp;

    // gpu-nnd additions: watchdog-safe query chunking + status checks (see
    // kChunkWorkBudget above). Query/output rows are sliced via buffer-bind
    // offsets; the kernel is unchanged (data indices stay absolute).
    const std::size_t chunk = query_chunk_rows(
        static_cast<std::size_t>(n) * static_cast<std::size_t>(dim), nq);
    bool ok = true;
    for (std::size_t off = 0; ok && off < static_cast<std::size_t>(nq);
         off += chunk) {
        const std::size_t rows =
            std::min(chunk, static_cast<std::size_t>(nq) - off);
        params.nq = static_cast<uint32_t>(rows);

        MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(data_buf, 0, 0);
        enc->setBuffer(query_buf, off * dim * sizeof(float), 1);
        enc->setBuffer(idx_buf, off * k * sizeof(int32_t), 2);
        enc->setBuffer(dist_buf, off * k * sizeof(float), 3);
        enc->setBytes(&params, sizeof(params), 4);
        enc->setBuffer(aux_buf, 0, 5);

        // Dispatch whole threadgroups: inactive tail threads still take part
        // in the cooperative tile loads/barriers (guarded in the kernel).
        const NS::UInteger tg = std::min<NS::UInteger>(
            pso->maxTotalThreadsPerThreadgroup(), heap ? 64 : 128);
        const NS::UInteger groups =
            (static_cast<NS::UInteger>(rows) + tg - 1) / tg;
        enc->dispatchThreadgroups(MTL::Size::Make(groups, 1, 1),
                                  MTL::Size::Make(tg, 1, 1));
        enc->endEncoding();
        ok = commit_and_wait(cmd);
    }

    if (ok) {
        std::memcpy(out_idx, idx_buf->contents(), out_n * sizeof(int32_t));
        std::memcpy(out_dists, dist_buf->contents(), out_n * sizeof(float));
    }

    data_buf->release();
    if (!self_query) query_buf->release();
    idx_buf->release();
    dist_buf->release();
    aux_buf->release();
    return ok;
}

bool gpu_sparse_brute_force_knn(const int32_t* d_indptr,
                                const int32_t* d_indices, const float* d_data,
                                int n, const int32_t* q_indptr,
                                const int32_t* q_indices, const float* q_data,
                                int nq, int k, KnnMetric metric,
                                int32_t* out_idx, float* out_dists) {
    if (k < 1 || k > kHeapKCap) {
        throw std::invalid_argument(
            "gpu_sparse_brute_force_knn: k must be in [1, 256]");
    }
    uint32_t metric_id;
    switch (metric) {
        case KnnMetric::euclidean:
            metric_id = 0u;
            break;
        case KnnMetric::cosine:
            metric_id = 1u;
            break;
        case KnnMetric::sqeuclidean:
            metric_id = 2u;
            break;
        default:
            throw std::invalid_argument(
                "gpu_sparse_brute_force_knn: only euclidean / sqeuclidean / "
                "cosine are supported for sparse input");
    }

    const bool heap = k > kSmallKCap;
    MTL::ComputePipelineState* pso = sparse_pipeline(heap);
    if (!pso) return false;
    MetalContext& ctx = context();

    const std::size_t d_nnz = static_cast<std::size_t>(d_indptr[n]);
    const std::size_t q_nnz = static_cast<std::size_t>(q_indptr[nq]);
    const std::size_t out_n = static_cast<std::size_t>(nq) * k;
    const std::size_t max_len = ctx.device->maxBufferLength();
    if (d_nnz * sizeof(float) > max_len || q_nnz * sizeof(float) > max_len ||
        out_n * sizeof(float) > max_len) {
        return false;
    }

    // Precompute the squared row norms (float32, like the device dot).
    std::vector<float> d_norms(static_cast<std::size_t>(n), 0.0f);
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int32_t j = d_indptr[i]; j < d_indptr[i + 1]; ++j)
            s += d_data[j] * d_data[j];
        d_norms[static_cast<std::size_t>(i)] = s;
    }
    const bool self_query =
        (q_indptr == d_indptr && q_indices == d_indices && q_data == d_data &&
         nq == n);
    std::vector<float> q_norms;
    if (!self_query) {
        q_norms.assign(static_cast<std::size_t>(nq), 0.0f);
        for (int i = 0; i < nq; ++i) {
            float s = 0.0f;
            for (int32_t j = q_indptr[i]; j < q_indptr[i + 1]; ++j)
                s += q_data[j] * q_data[j];
            q_norms[static_cast<std::size_t>(i)] = s;
        }
    }

    MTL::Device* dev = ctx.device;
    MTL::Buffer* d_indptr_buf = copy_buffer(
        dev, d_indptr, static_cast<std::size_t>(n + 1) * sizeof(int32_t));
    MTL::Buffer* d_indices_buf =
        copy_buffer(dev, d_indices, d_nnz * sizeof(int32_t));
    MTL::Buffer* d_data_buf = copy_buffer(dev, d_data, d_nnz * sizeof(float));
    MTL::Buffer* d_norms_buf =
        copy_buffer(dev, d_norms.data(), d_norms.size() * sizeof(float));
    MTL::Buffer* q_indptr_buf =
        self_query ? d_indptr_buf
                   : copy_buffer(dev, q_indptr,
                                 static_cast<std::size_t>(nq + 1) *
                                     sizeof(int32_t));
    MTL::Buffer* q_indices_buf =
        self_query ? d_indices_buf
                   : copy_buffer(dev, q_indices, q_nnz * sizeof(int32_t));
    MTL::Buffer* q_data_buf =
        self_query ? d_data_buf
                   : copy_buffer(dev, q_data, q_nnz * sizeof(float));
    MTL::Buffer* q_norms_buf =
        self_query ? d_norms_buf
                   : copy_buffer(dev, q_norms.data(),
                                 q_norms.size() * sizeof(float));
    MTL::Buffer* idx_buf =
        dev->newBuffer(out_n * sizeof(int32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* dist_buf =
        dev->newBuffer(out_n * sizeof(float), MTL::ResourceStorageModeShared);

    KnnSparseParams params;
    params.n = static_cast<uint32_t>(n);
    params.k = static_cast<uint32_t>(k);
    params.metric = metric_id;

    // gpu-nnd additions: watchdog-safe query chunking + status checks, like
    // the dense engine. The two-pointer dot walks ~(row nnz + data nnz) per
    // (query, data-row) pair; d_nnz + n bounds the per-query merge work.
    // q_indptr values are absolute offsets into the full q_indices/q_data
    // buffers, so only the indptr/norms/output bindings shift per slice.
    const std::size_t chunk =
        query_chunk_rows(d_nnz + static_cast<std::size_t>(n), nq);
    bool ok = true;
    for (std::size_t off = 0; ok && off < static_cast<std::size_t>(nq);
         off += chunk) {
        const std::size_t rows =
            std::min(chunk, static_cast<std::size_t>(nq) - off);
        params.nq = static_cast<uint32_t>(rows);

        MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(d_indptr_buf, 0, 0);
        enc->setBuffer(d_indices_buf, 0, 1);
        enc->setBuffer(d_data_buf, 0, 2);
        enc->setBuffer(q_indptr_buf, off * sizeof(int32_t), 3);
        enc->setBuffer(q_indices_buf, 0, 4);
        enc->setBuffer(q_data_buf, 0, 5);
        enc->setBuffer(d_norms_buf, 0, 6);
        enc->setBuffer(q_norms_buf, off * sizeof(float), 7);
        enc->setBuffer(idx_buf, off * k * sizeof(int32_t), 8);
        enc->setBuffer(dist_buf, off * k * sizeof(float), 9);
        enc->setBytes(&params, sizeof(params), 10);

        const NS::UInteger tg = std::min<NS::UInteger>(
            pso->maxTotalThreadsPerThreadgroup(), heap ? 64 : 128);
        enc->dispatchThreads(MTL::Size::Make(rows, 1, 1),
                             MTL::Size::Make(tg, 1, 1));
        enc->endEncoding();
        ok = commit_and_wait(cmd);
    }

    if (ok) {
        std::memcpy(out_idx, idx_buf->contents(), out_n * sizeof(int32_t));
        std::memcpy(out_dists, dist_buf->contents(), out_n * sizeof(float));
    }

    d_indptr_buf->release();
    d_indices_buf->release();
    d_data_buf->release();
    d_norms_buf->release();
    if (!self_query) {
        q_indptr_buf->release();
        q_indices_buf->release();
        q_data_buf->release();
        q_norms_buf->release();
    }
    idx_buf->release();
    dist_buf->release();
    return ok;
}

}  // namespace metal
}  // namespace umap
