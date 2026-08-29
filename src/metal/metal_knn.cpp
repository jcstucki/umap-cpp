// Host side of the auxiliary penalized-kNN kernel (src/metal/knn.metal,
// kernel knn_penalized; public API include/umap/metal_knn.h — global scope,
// called by external user code, not by the UMAP pipeline).
//
// metal-cpp port of the original Objective-C++ host (metal_knn.mm): instead
// of owning its own MTLDevice/queue/pipeline singletons and loading a
// standalone knn.metallib, it shares the process-wide MetalContext — one
// device/queue and the single umap_kernels.metallib, resolved at runtime via
// the existing helper ($UMAP_METALLIB_PATH env -> baked build-tree path ->
// dladdr install-tree lookup; see src/metal/metal_context.cpp). Behaviour is
// otherwise identical: lazy init-once, "knn: "-prefixed stderr diagnostics on
// failure, F > 8 or k > 64 -> false, shared-storage buffers, <= 256
// threads/threadgroup dispatch, command-buffer status check, memcpy out.
#include "umap/metal_knn.h"

#include <cstdio>
#include <cstring>
#include <mutex>

#include "metal_context.hpp"

namespace {

MTL::ComputePipelineState* g_pipe = nullptr;
bool g_ok = false;
std::once_flag g_once;

// Mirrors struct Params in src/metal/knn.metal.
struct Params {
    int n, F, k, cosine;
    float wt, wp, wr, wk;
};

void init_once() {
    std::call_once(g_once, [] {
        umap::metal::MetalContext& ctx = umap::metal::context();
        if (!ctx.device) {
            std::fprintf(stderr, "knn: no Metal device\n");
            return;
        }
        if (!ctx.ok()) {  // device present but the shared metallib didn't load
            std::fprintf(stderr, "knn: metallib load failed\n");
            return;
        }
        NS::String* fn_name =
            NS::String::string("knn_penalized", NS::UTF8StringEncoding);
        MTL::Function* fn = ctx.library->newFunction(fn_name);
        if (!fn) {
            std::fprintf(stderr, "knn: function not found\n");
            return;
        }
        NS::Error* error = nullptr;
        g_pipe = ctx.device->newComputePipelineState(fn, &error);
        fn->release();
        if (!g_pipe) {
            std::fprintf(stderr, "knn: pipeline failed\n");
            return;
        }
        g_ok = true;
    });
}

}  // namespace

bool gpu_knn_available() {
    init_once();
    return g_ok;
}

bool gpu_knn(const float* X, const float* sq, const float* dz, const float* cz,
             const float* rz, const int32_t* tid, int n, int F, int k,
             bool cosine, float wt, float wp, float wr, float wk,
             int32_t* outI, float* outD) {
    init_once();
    if (!g_ok || F > 8 || k > 64) return false;
    umap::metal::MetalContext& ctx = umap::metal::context();
    MTL::Device* dev = ctx.device;
    const auto mkbuf = [dev](const void* p, std::size_t bytes) {
        return dev->newBuffer(p, bytes, MTL::ResourceStorageModeShared);
    };
    const std::size_t nf = static_cast<std::size_t>(n) * F;
    const std::size_t nn = static_cast<std::size_t>(n);
    const std::size_t nk = static_cast<std::size_t>(n) * k;
    MTL::Buffer* bX = mkbuf(X, nf * sizeof(float));
    MTL::Buffer* bsq = mkbuf(sq, nn * sizeof(float));
    MTL::Buffer* bdz = mkbuf(dz, nn * sizeof(float));
    MTL::Buffer* bcz = mkbuf(cz, nn * sizeof(float));
    MTL::Buffer* brz = mkbuf(rz, nn * sizeof(float));
    MTL::Buffer* btid = mkbuf(tid, nn * sizeof(int32_t));
    MTL::Buffer* boI = dev->newBuffer(nk * sizeof(int32_t),
                                      MTL::ResourceStorageModeShared);
    MTL::Buffer* boD = dev->newBuffer(nk * sizeof(float),
                                      MTL::ResourceStorageModeShared);
    Params P{n, F, k, cosine ? 1 : 0, wt, wp, wr, wk};
    MTL::Buffer* bP = mkbuf(&P, sizeof P);

    MTL::CommandBuffer* cb = ctx.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(g_pipe);
    MTL::Buffer* bufs[] = {bX, bsq, bdz, bcz, brz, btid, boI, boD, bP};
    for (int b = 0; b < 9; ++b)
        enc->setBuffer(bufs[b], 0, static_cast<NS::UInteger>(b));
    NS::UInteger tg = g_pipe->maxTotalThreadsPerThreadgroup();
    if (tg > 256) tg = 256;
    enc->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(tg, 1, 1));
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();

    const bool completed = cb->status() == MTL::CommandBufferStatusCompleted;
    if (!completed) {
        std::fprintf(stderr, "knn: command buffer error\n");
    } else {
        std::memcpy(outI, boI->contents(), nk * sizeof(int32_t));
        std::memcpy(outD, boD->contents(), nk * sizeof(float));
    }
    for (MTL::Buffer* b : bufs) b->release();
    return completed;
}
