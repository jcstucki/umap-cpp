// Shared Metal device/queue/library setup + umap::metal_available().
// This is the single TU that instantiates the metal-cpp implementations.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "metal_context.hpp"

#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

#include "umap/device.hpp"

#ifndef UMAP_DEFAULT_METALLIB
#define UMAP_DEFAULT_METALLIB "umap_kernels.metallib"
#endif

namespace umap {
namespace metal {

namespace {

std::mutex g_mutex;

const char* metallib_path() {
    const char* env = std::getenv("UMAP_METALLIB_PATH");
    return (env && *env) ? env : UMAP_DEFAULT_METALLIB;
}

MetalContext make_context() {
    MetalContext ctx;
    ctx.device = MTL::CreateSystemDefaultDevice();
    if (!ctx.device) return ctx;
    ctx.queue = ctx.device->newCommandQueue();

    NS::Error* error = nullptr;
    NS::String* path = NS::String::string(metallib_path(),
                                          NS::UTF8StringEncoding);
    ctx.library = ctx.device->newLibrary(path, &error);
    // ctx.ok() is false if the metallib could not be loaded.
    return ctx;
}

}  // namespace

MetalContext& context() {
    static MetalContext ctx = make_context();
    return ctx;
}

MTL::ComputePipelineState* pipeline(const char* function_name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    MetalContext& ctx = context();
    if (!ctx.ok()) return nullptr;

    MTL::ComputePipelineState** slot = nullptr;
    const std::string name(function_name);
    if (name == "brute_force_knn") {
        slot = &ctx.knn_pipeline;
    } else if (name == "sgd_epoch") {
        slot = &ctx.sgd_pipeline;
    } else {
        return nullptr;
    }
    if (*slot) return *slot;

    NS::String* fn_name = NS::String::string(function_name,
                                             NS::UTF8StringEncoding);
    MTL::Function* fn = ctx.library->newFunction(fn_name);
    if (!fn) return nullptr;
    NS::Error* error = nullptr;
    *slot = ctx.device->newComputePipelineState(fn, &error);
    fn->release();
    return *slot;
}

}  // namespace metal

bool metal_available() { return metal::context().ok(); }

}  // namespace umap
