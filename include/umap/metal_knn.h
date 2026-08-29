// GPU exact penalized k-NN (Apple Metal). Returns false if Metal is
// unavailable or the kernel could not be prepared (caller falls back to CPU).
#pragma once
#include <cstdint>

bool gpu_knn_available();

// X: n*F row-major float (rows must be UNIT-normalized when cosine=true).
// sq: n squared norms (used when cosine=false). dz/cz/rz: n z-scored
// day/close/return. tid: n ticker ids. Fills outI (n*k int32) and outD
// (n*k float), each row sorted ascending by distance. F must be <= 8.
bool gpu_knn(const float* X, const float* sq, const float* dz, const float* cz,
             const float* rz, const int32_t* tid, int n, int F, int k,
             bool cosine, float wt, float wp, float wr, float wk,
             int32_t* outI, float* outD);
