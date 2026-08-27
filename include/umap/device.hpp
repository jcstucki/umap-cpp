// Runtime device-dispatch seam (Phase 3, branch metal-gpu).
//
// The future UMAP class takes a Device and routes kNN / layout optimisation to
// the CPU ports or to the Metal kernels (include/umap/metal/*.hpp). GPU is
// only honoured when the library was built with UMAP_ENABLE_METAL *and* a
// Metal device exists at runtime; callers should consult metal_available()
// (or use resolve_device) and fall back to CPU otherwise.
#pragma once

namespace umap {

enum class Device {
    CPU,
    GPU,  // Apple Metal
};

// True iff the library was built with UMAP_ENABLE_METAL and
// MTLCreateSystemDefaultDevice() yields a device at runtime.
bool metal_available();

// GPU if requested and available, else CPU.
inline Device resolve_device(Device requested) {
    if (requested == Device::GPU && metal_available()) return Device::GPU;
    return Device::CPU;
}

}  // namespace umap
