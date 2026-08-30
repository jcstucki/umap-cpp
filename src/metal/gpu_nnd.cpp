// Host side of the GPU NN-descent engine (src/metal/umap_kernels.metal,
// kernels nnd_*). See include/umap/metal/gpu_nnd.hpp for the contract and
// the documented deviations from the bit-exact CPU port.
#include "umap/metal/gpu_nnd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "metal_context.hpp"
#include "umap/nn_descent.hpp"  // make_forest_leaf_array + NumpyRandomState

namespace umap {
namespace metal {

namespace {

// Mirrors struct NndParams in umap_kernels.metal (8 x 4-byte fields, one
// 8-byte word at offset 32 — layout identical in C++ and MSL).
struct NndParams {
    uint32_t n;
    uint32_t dim;
    uint32_t k;
    uint32_t n_cand;
    uint32_t metric;
    uint32_t iter;
    uint32_t n_trees;
    uint32_t leaf_len;
    uint64_t seed;
};
static_assert(sizeof(NndParams) == 40, "NndParams layout must match MSL");

// Internal metric ids — must match the UMAP_NND_* defines in the kernels.
constexpr uint32_t kNndSqeuclidean = 0u;
constexpr uint32_t kNndCosine = 1u;
constexpr uint32_t kNndManhattan = 2u;

// pynndescent parameters reproduced by the descent loop.
constexpr int64_t kMaxCandidates = 60;  // umap_.py::nearest_neighbors
constexpr double kDelta = 0.001;        // NNDescent default

std::mutex g_nnd_mutex;

// waitUntilCompleted + status verification (the exact engine's
// commit_and_wait twin): an errored/aborted buffer means the device state is
// garbage — abort the descent and let the caller fall back.
bool commit_and_wait(MTL::CommandBuffer* cmd) {
    cmd->commit();
    cmd->waitUntilCompleted();
    return cmd->status() == MTL::CommandBufferStatusCompleted &&
           cmd->error() == nullptr;
}

MTL::ComputePipelineState* nnd_pipeline(const char* name) {
    std::lock_guard<std::mutex> lock(g_nnd_mutex);
    static std::unordered_map<std::string, MTL::ComputePipelineState*> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;

    MetalContext& ctx = context();
    MTL::ComputePipelineState* pso = nullptr;
    if (ctx.ok()) {
        NS::String* fn_name = NS::String::string(name, NS::UTF8StringEncoding);
        MTL::Function* fn = ctx.library->newFunction(fn_name);
        if (fn) {
            NS::Error* error = nullptr;
            pso = ctx.device->newComputePipelineState(fn, &error);
            fn->release();
        }
    }
    cache.emplace(name, pso);  // cache nullptr failures too
    return pso;
}

}  // namespace

bool gpu_nnd_metric_supported(KnnMetric metric) {
    switch (metric) {
        case KnnMetric::euclidean:
        case KnnMetric::sqeuclidean:
        case KnnMetric::manhattan:
        case KnnMetric::cosine:
            return true;
        default:
            return false;
    }
}

GpuKnnMode gpu_knn_mode(int n, int k, KnnMetric metric) {
    if (!gpu_nnd_metric_supported(metric) || k < 1 || k > kGpuNndMaxK ||
        n <= 2 * k) {
        return GpuKnnMode::exact;
    }
    if (const char* env = std::getenv("UMAP_GPU_KNN_MODE")) {
        if (std::strcmp(env, "exact") == 0) return GpuKnnMode::exact;
        if (std::strcmp(env, "nnd") == 0) return GpuKnnMode::nnd;
        // anything else (incl. "auto") falls through to the crossover rule
    }
    return n > kGpuNndCrossoverN ? GpuKnnMode::nnd : GpuKnnMode::exact;
}

bool gpu_nn_descent_knn(const float* data, int n, int dim, int k,
                        KnnMetric metric, uint64_t seed, int32_t* out_idx,
                        float* out_dists) {
    if (k < 1 || k > kGpuNndMaxK) {
        throw std::invalid_argument(
            "gpu_nn_descent_knn: k must be in [1, 64]");
    }
    if (!gpu_nnd_metric_supported(metric)) {
        throw std::invalid_argument(
            "gpu_nn_descent_knn: only euclidean / sqeuclidean / manhattan / "
            "cosine are supported");
    }

    uint32_t metric_id = kNndSqeuclidean;
    bool sqrt_correction = false;
    bool angular = false;
    switch (metric) {
        case KnnMetric::euclidean:
            metric_id = kNndSqeuclidean;
            sqrt_correction = true;  // internal sqeuclidean, host sqrt
            break;
        case KnnMetric::sqeuclidean:
            metric_id = kNndSqeuclidean;
            break;
        case KnnMetric::manhattan:
            metric_id = kNndManhattan;
            break;
        case KnnMetric::cosine:
            metric_id = kNndCosine;
            angular = true;  // pynndescent angular rp-tree list
            break;
        default:
            return false;  // unreachable (guarded above)
    }

    MTL::ComputePipelineState* pso_init = nnd_pipeline("nnd_init");
    MTL::ComputePipelineState* pso_fwd = nnd_pipeline("nnd_sample_fwd");
    MTL::ComputePipelineState* pso_rev = nnd_pipeline("nnd_sample_rev");
    MTL::ComputePipelineState* pso_clear = nnd_pipeline("nnd_clear_flags");
    MTL::ComputePipelineState* pso_join = nnd_pipeline("nnd_join");
    if (!pso_init || !pso_fwd || !pso_rev || !pso_clear || !pso_join)
        return false;
    MetalContext& ctx = context();

    // umap_.py::nearest_neighbors "hacked values".
    const int64_t n_trees = std::min<int64_t>(
        64, 5 + static_cast<int64_t>(
                    std::nearbyint(std::sqrt(static_cast<double>(n)) / 20.0)));
    const int64_t n_iters = std::max<int64_t>(
        5, static_cast<int64_t>(
               std::nearbyint(std::log2(static_cast<double>(n)))));

    // RP forest on the host — the bit-exact rp-tree code with a PRIVATE
    // NumpyRandomState (the model RandomState skips NN-descent's draws under
    // device=GPU, exactly like the exact-engine path).
    nnd::NumpyRandomState forest_rs(
        static_cast<uint32_t>(seed ^ (seed >> 32)));
    const nnd::LeafArray la = nnd::make_forest_leaf_array(
        data, n, dim, k, n_trees, /*leaf_size=*/-1, angular,
        /*max_depth=*/200, forest_rs);

    // Invert into per-node leaf memberships (each node appears in exactly one
    // leaf per tree; -1 pads defensively).
    std::vector<int32_t> node_leaves(
        static_cast<std::size_t>(n) * n_trees, -1);
    {
        std::vector<int32_t> fill(static_cast<std::size_t>(n), 0);
        for (int64_t r = 0; r < la.n_rows; ++r) {
            const int32_t* row = la.row(r);
            for (int64_t m = 0; m < la.row_len; ++m) {
                const int32_t q = row[m];
                if (q < 0) break;
                if (fill[q] < n_trees) {
                    node_leaves[static_cast<std::size_t>(q) * n_trees +
                                fill[q]++] = static_cast<int32_t>(r);
                }
            }
        }
    }

    const int64_t n_cand = std::max<int64_t>(kMaxCandidates, k);

    const std::size_t nk = static_cast<std::size_t>(n) * k;
    const std::size_t nc = static_cast<std::size_t>(n) * n_cand;
    const std::size_t data_bytes =
        static_cast<std::size_t>(n) * dim * sizeof(float);
    const std::size_t leaf_bytes = std::max<std::size_t>(
        la.idx.size() * sizeof(int32_t), sizeof(int32_t));
    const std::size_t max_len = ctx.device->maxBufferLength();
    if (data_bytes > max_len || nk * sizeof(float) > max_len ||
        nc * sizeof(int32_t) > max_len || leaf_bytes > max_len ||
        node_leaves.size() * sizeof(int32_t) > max_len) {
        return false;  // dataset exceeds a single Metal buffer: CPU fallback
    }

    MTL::Device* dev = ctx.device;
    MTL::Buffer* data_buf =
        dev->newBuffer(data, data_bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* heap_dist_buf =
        dev->newBuffer(nk * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* heap_idx_buf =
        dev->newBuffer(nk * sizeof(int32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* heap_flag_buf =
        dev->newBuffer(nk, MTL::ResourceStorageModeShared);
    MTL::Buffer* leaf_buf =
        la.idx.empty()
            ? dev->newBuffer(sizeof(int32_t), MTL::ResourceStorageModeShared)
            : dev->newBuffer(la.idx.data(), leaf_bytes,
                             MTL::ResourceStorageModeShared);
    MTL::Buffer* node_leaves_buf =
        dev->newBuffer(node_leaves.data(),
                       node_leaves.size() * sizeof(int32_t),
                       MTL::ResourceStorageModeShared);
    MTL::Buffer* new_cand_buf =
        dev->newBuffer(nc * sizeof(int32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* old_cand_buf =
        dev->newBuffer(nc * sizeof(int32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* new_count_buf = dev->newBuffer(
        static_cast<std::size_t>(n) * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    MTL::Buffer* old_count_buf = dev->newBuffer(
        static_cast<std::size_t>(n) * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    MTL::Buffer* updates_buf =
        dev->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);

    NndParams params;
    params.n = static_cast<uint32_t>(n);
    params.dim = static_cast<uint32_t>(dim);
    params.k = static_cast<uint32_t>(k);
    params.n_cand = static_cast<uint32_t>(n_cand);
    params.metric = metric_id;
    params.iter = 0;
    params.n_trees =
        static_cast<uint32_t>(la.n_rows > 0 ? n_trees : 0);
    params.leaf_len = static_cast<uint32_t>(la.row_len);
    params.seed = seed;

    const auto dispatch = [&](MTL::CommandBuffer* cmd,
                              MTL::ComputePipelineState* pso,
                              std::size_t threads, NS::UInteger tg_cap,
                              const std::vector<MTL::Buffer*>& bufs) {
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        for (std::size_t b = 0; b < bufs.size(); ++b)
            enc->setBuffer(bufs[b], 0, static_cast<NS::UInteger>(b));
        enc->setBytes(&params, sizeof(params),
                      static_cast<NS::UInteger>(bufs.size()));
        const NS::UInteger tg = std::min<NS::UInteger>(
            pso->maxTotalThreadsPerThreadgroup(), tg_cap);
        enc->dispatchThreads(MTL::Size::Make(threads, 1, 1),
                             MTL::Size::Make(tg, 1, 1));
        enc->endEncoding();
    };

    const auto release_all = [&] {
        data_buf->release();
        heap_dist_buf->release();
        heap_idx_buf->release();
        heap_flag_buf->release();
        leaf_buf->release();
        node_leaves_buf->release();
        new_cand_buf->release();
        old_cand_buf->release();
        new_count_buf->release();
        old_count_buf->release();
        updates_buf->release();
    };

    // Init: forest leaf join + random top-up (one command buffer).
    {
        MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
        dispatch(cmd, pso_init, static_cast<std::size_t>(n), 64,
                 {data_buf, heap_dist_buf, heap_idx_buf, heap_flag_buf,
                  leaf_buf, node_leaves_buf});
        if (!commit_and_wait(cmd)) {
            release_all();
            return false;
        }
    }

    // Descent iterations: sample (fwd, rev), clear flags, join — successive
    // encoders in one command buffer are the inter-phase barriers; the
    // update counter is read back per iteration for early termination.
    for (int64_t iter = 0; iter < n_iters; ++iter) {
        // Host-side resets are coherent here (previous buffer completed).
        std::memset(new_cand_buf->contents(), 0xFF, nc * sizeof(int32_t));
        std::memset(old_cand_buf->contents(), 0xFF, nc * sizeof(int32_t));
        std::memset(new_count_buf->contents(), 0,
                    static_cast<std::size_t>(n) * sizeof(uint32_t));
        std::memset(old_count_buf->contents(), 0,
                    static_cast<std::size_t>(n) * sizeof(uint32_t));
        std::memset(updates_buf->contents(), 0, sizeof(uint32_t));
        params.iter = static_cast<uint32_t>(iter);

        MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
        dispatch(cmd, pso_fwd, static_cast<std::size_t>(n), 256,
                 {heap_idx_buf, heap_flag_buf, new_cand_buf, old_cand_buf,
                  new_count_buf, old_count_buf});
        dispatch(cmd, pso_rev, nk, 256,
                 {heap_idx_buf, heap_flag_buf, new_cand_buf, old_cand_buf,
                  new_count_buf, old_count_buf});
        dispatch(cmd, pso_clear, nk, 256,
                 {heap_idx_buf, heap_flag_buf, new_cand_buf, new_count_buf});
        dispatch(cmd, pso_join, static_cast<std::size_t>(n), 64,
                 {data_buf, heap_dist_buf, heap_idx_buf, heap_flag_buf,
                  new_cand_buf, old_cand_buf, new_count_buf, old_count_buf,
                  updates_buf});
        if (!commit_and_wait(cmd)) {
            release_all();
            return false;
        }

        uint32_t c;
        std::memcpy(&c, updates_buf->contents(), sizeof(uint32_t));
        if (static_cast<double>(c) <=
            kDelta * static_cast<double>(k) * static_cast<double>(n)) {
            break;
        }
    }

    // Emit: ascending (distance, index) per row — the exact-engine
    // convention (deheap_sort's dist order with deterministic tie breaks) —
    // plus the euclidean sqrt correction.
    {
        const float* hd = static_cast<const float*>(heap_dist_buf->contents());
        const int32_t* hi =
            static_cast<const int32_t*>(heap_idx_buf->contents());
        std::vector<std::pair<float, int32_t>> row(static_cast<std::size_t>(k));
        for (int i = 0; i < n; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * k;
            for (int j = 0; j < k; ++j)
                row[j] = {hd[base + j], hi[base + j]};
            std::sort(row.begin(), row.end());
            for (int j = 0; j < k; ++j) {
                out_idx[base + j] = row[j].second;
                out_dists[base + j] =
                    sqrt_correction ? std::sqrt(row[j].first) : row[j].first;
            }
        }
    }

    release_all();
    return true;
}

}  // namespace metal
}  // namespace umap
