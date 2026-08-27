// Faithful C++ port of the sparse distance kernels UMAP uses for sparse
// (CSR) input data: reference/umap/umap/sparse.py (BSD 3 clause) and
// reference/pynndescent/pynndescent/sparse.py (BSD 2 clause). The two files
// contain the same implementations for the metrics ported here; goldens are
// generated against the pynndescent copies (which NN-descent actually calls).
//
// Vectors are slices of a CSR matrix: (ind, data, nnz) with `ind` sorted
// ascending (scipy has_sorted_indices; NNDescent calls sort_indices() before
// use). All kernels replicate numba's float32/float64 promotion behaviour:
//  * sparse_sum / sparse_diff / sparse_mul produce float32 values (zeros
//    dropped) via a sorted index-merge, exactly like the Python versions;
//  * sparse_squared_euclidean / sparse_alternative_cosine (the fast
//    alternatives NN-descent iterates with) accumulate in float32;
//  * the "true" metrics (sparse_euclidean, sparse_manhattan, sparse_cosine)
//    accumulate float32 terms into a float64 result like numba's lazy typing;
//  * utils.py::norm accumulates float32 squares into float64 and returns
//    float32.
//
// Metric registry: resolve_sparse_metric mirrors the subset of
// pynndescent's sparse_named_distances + sparse_fast_distance_alternatives
// that the dense port supports: "euclidean"/"l2" (squared-euclidean
// alternative + sqrt correction), "sqeuclidean", "manhattan"/"l1"/"taxicab",
// and "cosine" (alternative_cosine + correct_alternative_cosine, angular
// trees). Every other name in sparse_named_distances (chebyshev, minkowski,
// hamming, jaccard, correlation, hellinger, ...) throws
// std::invalid_argument("...not implemented...") -- they are valid upstream
// but not ported yet.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace umap {
namespace spdist {

// ------------------------------------------------------------------- aux ops

// utils.py::norm on a data slice: float64 accumulation of float32 products,
// float32 return.
float norm(const float* data, int64_t nnz);

// sparse.py::arr_union / arr_intersect (sorted merge equivalents of
// np.union1d / np.intersect1d; inputs must be sorted + duplicate-free, which
// CSR row index slices are).
std::vector<int32_t> arr_union(const int32_t* ar1, int64_t n1,
                               const int32_t* ar2, int64_t n2);
std::vector<int32_t> arr_intersect(const int32_t* ar1, int64_t n1,
                                   const int32_t* ar2, int64_t n2);

// sparse.py::sparse_sum -- sorted index merge; values added in float32;
// exact zeros are dropped from the result.
void sparse_sum(const int32_t* ind1, const float* data1, int64_t n1,
                const int32_t* ind2, const float* data2, int64_t n2,
                std::vector<int32_t>& result_ind,
                std::vector<float>& result_data);

// sparse.py::sparse_diff == sparse_sum(ind1, data1, ind2, -data2).
void sparse_diff(const int32_t* ind1, const float* data1, int64_t n1,
                 const int32_t* ind2, const float* data2, int64_t n2,
                 std::vector<int32_t>& result_ind,
                 std::vector<float>& result_data);

// sparse.py::sparse_mul -- values multiplied in float32 on the index
// intersection; exact zeros dropped.
void sparse_mul(const int32_t* ind1, const float* data1, int64_t n1,
                const int32_t* ind2, const float* data2, int64_t n2,
                std::vector<int32_t>& result_ind,
                std::vector<float>& result_data);

// ------------------------------------------------------------------- metrics

// True metrics (as tabulated in sparse_named_distances).
float sparse_euclidean(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2);
float sparse_manhattan(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2);
float sparse_cosine(const int32_t* ind1, const float* data1, int64_t n1,
                    const int32_t* ind2, const float* data2, int64_t n2);

// Fast alternatives (sparse_fast_distance_alternatives) used inside
// NN-descent, plus their elementwise corrections.
float sparse_squared_euclidean(const int32_t* ind1, const float* data1,
                               int64_t n1, const int32_t* ind2,
                               const float* data2, int64_t n2);
float sparse_alternative_cosine(const int32_t* ind1, const float* data1,
                                int64_t n1, const int32_t* ind2,
                                const float* data2, int64_t n2);
// np.sqrt correction for "euclidean" (float32 sqrt, numpy semantics).
float sparse_sqrt_correction(float d);
// sparse.py::sparse_correct_alternative_cosine (incl. the isclose(0, |d|,
// atol=1e-7) guard the dense correction does not have).
float sparse_correct_alternative_cosine(float d);

// ------------------------------------------------------------------ registry

using SparseDistFn = float (*)(const int32_t*, const float*, int64_t,
                               const int32_t*, const float*, int64_t);

struct SparseMetricInfo {
    // distance used inside NN-descent (fast alternative when one exists)
    SparseDistFn dist = nullptr;
    // elementwise correction applied to final distances (nullptr = identity)
    float (*correction)(float) = nullptr;
    // the true metric (for brute-force kNN / reported distances)
    SparseDistFn true_dist = nullptr;
    bool angular_trees = false;  // pynndescent's angular metric list
};

// Supported: "euclidean"/"l2", "sqeuclidean", "manhattan"/"taxicab"/"l1",
// "cosine". Throws std::invalid_argument for every other metric name
// (including names upstream supports but this port does not implement yet).
SparseMetricInfo resolve_sparse_metric(const std::string& name);

}  // namespace spdist
}  // namespace umap
