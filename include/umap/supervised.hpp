// Faithful C++ port of the supervised / semi-supervised simplicial-set
// intersection machinery from reference/umap/umap/umap_.py and sparse.py
// (lmcinnes/umap, BSD 3 clause): fast_intersection, fast_metric_intersection,
// reprocess_row / reset_local_metrics, reset_local_connectivity (including the
// sklearn normalize(norm="max") step), discrete_metric_simplicial_set_
// intersection, general_simplicial_set_intersection and its numba core
// sparse.py::general_sset_intersection.
//
// All functions operate on the canonical CSR SparseMatrix; iterating its
// entries row by row reproduces the COO (tocoo()) traversal order of the
// scipy matrices upstream mutates in place.
//
// Faithfulness notes:
//  * fast_intersection / fast_metric_intersection multiply the float32 COO
//    values by float64 np.exp terms; the product is computed in float64 and
//    rounded back to float32 on store, exactly as numba does.
//  * general_sset_intersection computes left_min/right_min from float32
//    .min() reductions promoted by the /2.0 division, keeps left_val/
//    right_val in float64, and *leaves the (ss1+ss2) sum value in place*
//    when neither side exceeds its minimum -- replicating the upstream code
//    path where result_val[idx] is only overwritten inside the guard.
//  * reset_local_connectivity's normalize(simplicial_set, norm="max")
//    replicates sklearn's sparse branch: norms = max(|row min|, row max)
//    with implicit zeros participating (min_max_axis), float32 in-place
//    division, zero-norm rows untouched.
//  * reprocess_row: psum accumulates pow(float32 value, float64 mid) in
//    float64; the final np.power(probabilities, mid) is computed in float64
//    and rounded to float32 on store into the CSR data (numba semantics).
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "umap/sparse_matrix.hpp"

namespace umap {

// umap_.py::fast_intersection -- categorical intersection, mutating the
// graph's values in place. `target` holds one categorical label per sample
// (float64, -1.0 marks "unknown" exactly like upstream).
void fast_intersection(SparseMatrix& graph, const double* target,
                       double unknown_dist = 1.0, double far_dist = 5.0);

// umap_.py::fast_metric_intersection for 1-d discrete targets:
// values[nz] *= exp(-(scale * metric(target[i], target[j]))).
void fast_metric_intersection(SparseMatrix& graph, const double* target,
                              const std::function<double(double, double)>& metric,
                              double scale);

// umap_.py::reprocess_row (numba): rescale a probability row to sum-of-powers
// log2(k) via bisection on the exponent. Returns the reprocessed row.
std::vector<float> reprocess_row(const float* probabilities, int64_t n,
                                 double k = 15.0, int n_iters = 32);

// umap_.py::reset_local_metrics -- reprocess_row applied to every CSR row.
void reset_local_metrics(SparseMatrix& graph);

// umap_.py::reset_local_connectivity: normalize(norm="max"), optional
// reset_local_metrics, then fuzzy union with the transpose
// (A + A^T - A o A^T) and eliminate_zeros.
SparseMatrix reset_local_connectivity(const SparseMatrix& simplicial_set,
                                      bool reset_local_metric = false);

// umap_.py::discrete_metric_simplicial_set_intersection. When `metric` is
// empty the categorical fast_intersection runs with (unknown_dist, far_dist);
// otherwise `metric` is applied via fast_metric_intersection with
// metric_scale. Ends with eliminate_zeros + reset_local_connectivity.
SparseMatrix discrete_metric_simplicial_set_intersection(
    const SparseMatrix& simplicial_set, const double* discrete_space,
    double unknown_dist = 1.0, double far_dist = 5.0,
    const std::function<double(double, double)>& metric = nullptr,
    double metric_scale = 1.0);

// umap_.py::general_simplicial_set_intersection (weight = mix_weight) with
// sparse.py::general_sset_intersection as its core.
SparseMatrix general_simplicial_set_intersection(const SparseMatrix& ss1,
                                                 const SparseMatrix& ss2,
                                                 double weight = 0.5,
                                                 bool right_complement = false);

}  // namespace umap
