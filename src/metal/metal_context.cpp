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

// --- phase4 additions --------------------------------------------------------
#include <dlfcn.h>   // dladdr: locate the module containing this code
#include <unistd.h>  // access(): existence checks
// -----------------------------------------------------------------------------

#include "umap/device.hpp"

#ifndef UMAP_DEFAULT_METALLIB
#define UMAP_DEFAULT_METALLIB "umap_kernels.metallib"
#endif

namespace umap {
namespace metal {

namespace {

std::mutex g_mutex;

// --- phase4 additions --------------------------------------------------------
// Metallib search order:
//   1. $UMAP_METALLIB_PATH (explicit override, kept from Phase 3);
//   2. UMAP_DEFAULT_METALLIB -- the absolute build-tree path baked at compile
//      time (works for in-tree builds and tests);
//   3. install-tree resolution via dladdr on this function: for an executable
//      linking the static library (e.g. <prefix>/bin/umap) try
//      <exe dir>/umap_kernels.metallib and
//      <exe dir>/../lib/umapcpp/umap_kernels.metallib -- the location
//      `cmake --install` uses (${CMAKE_INSTALL_LIBDIR}/umapcpp).
// The first existing path wins; if none exists the baked default is returned
// (newLibrary then fails and metal_available() reports false).
bool file_exists(const std::string& p) { return ::access(p.c_str(), R_OK) == 0; }

std::string metallib_path() {
    const char* env = std::getenv("UMAP_METALLIB_PATH");
    if (env && *env) return env;
    const std::string baked = UMAP_DEFAULT_METALLIB;
    if (file_exists(baked)) return baked;
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&metallib_path), &info) != 0 &&
        info.dli_fname != nullptr) {
        std::string module_path(info.dli_fname);
        const std::size_t slash = module_path.find_last_of('/');
        const std::string dir =
            slash == std::string::npos ? "." : module_path.substr(0, slash);
        const std::string beside = dir + "/umap_kernels.metallib";
        if (file_exists(beside)) return beside;
        const std::string installed =
            dir + "/../lib/umapcpp/umap_kernels.metallib";
        if (file_exists(installed)) return installed;
    }
    return baked;
}
// -----------------------------------------------------------------------------

MetalContext make_context() {
    MetalContext ctx;
    ctx.device = MTL::CreateSystemDefaultDevice();
    if (!ctx.device) return ctx;
    ctx.queue = ctx.device->newCommandQueue();

    NS::Error* error = nullptr;
    NS::String* path = NS::String::string(metallib_path().c_str(),
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
