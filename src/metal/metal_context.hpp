// Private shared Metal state for the GPU kernels (Phase 3, branch metal-gpu).
// Not installed; include/umap/device.hpp is the public seam.
#pragma once

#include <Metal/Metal.hpp>

namespace umap {
namespace metal {

// Process-wide lazily-initialised Metal objects. All pointers are owned by the
// context (retained for the process lifetime) and may be null when Metal is
// unavailable — check ok() first.
struct MetalContext {
    MTL::Device* device = nullptr;
    MTL::CommandQueue* queue = nullptr;
    MTL::Library* library = nullptr;  // umap_kernels.metallib
    MTL::ComputePipelineState* knn_pipeline = nullptr;
    MTL::ComputePipelineState* sgd_pipeline = nullptr;

    bool ok() const { return device && queue && library; }
};

// Returns the singleton context; device/queue/library populated on first call.
// The metallib is loaded from $UMAP_METALLIB_PATH if set, else from the
// build-time default path (UMAP_DEFAULT_METALLIB compile definition).
MetalContext& context();

// Lazily builds (and caches) the named pipeline; returns nullptr on failure.
MTL::ComputePipelineState* pipeline(const char* function_name);

}  // namespace metal
}  // namespace umap
