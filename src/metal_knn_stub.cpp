// CPU-only fallback for the auxiliary penalized-kNN GPU API
// (include/umap/metal_knn.h): always false, so callers take their CPU path.
// When UMAP_ENABLE_METAL is defined these symbols are provided by
// src/metal/metal_knn.cpp instead and this TU compiles to nothing (the same
// arrangement as src/device.cpp for umap::metal_available()).
#include "umap/metal_knn.h"

#ifndef UMAP_ENABLE_METAL
bool gpu_knn_available() { return false; }

bool gpu_knn(const float*, const float*, const float*, const float*,
             const float*, const int32_t*, int, int, int, bool, float, float,
             float, float, int32_t*, float*) {
    return false;
}
#endif
