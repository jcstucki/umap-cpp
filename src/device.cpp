// CPU-only fallback for the device-dispatch seam (include/umap/device.hpp).
// When UMAP_ENABLE_METAL is defined, umap::metal_available() is provided by
// src/metal/metal_context.cpp instead.
#include "umap/device.hpp"

#ifndef UMAP_ENABLE_METAL
namespace umap {
bool metal_available() { return false; }
}  // namespace umap
#endif
