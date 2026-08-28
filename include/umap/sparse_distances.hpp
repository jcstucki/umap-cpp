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
// Metric registry: resolve_sparse_metric mirrors pynndescent's
// sparse_named_distances + sparse_fast_distance_alternatives as consumed by
// NNDescent.__init__'s sparse branch: euclidean/l2 (squared-euclidean
// alternative + np.sqrt correction), sqeuclidean, manhattan/l1/taxicab,
// chebyshev/linf/linfty/linfinity, minkowski (p kwd), canberra, braycurtis,
// hamming, jaccard (alternative + correction), dice, matching, kulsinski,
// rogerstanimoto, russellrao, sokalmichener, sokalsneath, cosine
// (alternative + correction), correlation, hellinger (alternative +
// correction), plus umap-only sparse_ll_dirichlet (umap/sparse.py; absent
// from pynndescent's registry, so upstream NN-descent receives it as a
// *callable* -- no fast alternative, no n_features binding, and the tree
// flavour follows UMAP's angular_rp_forest parameter instead of the metric
// name; see SparseMetricInfo::angular_from_param).
//
// Metrics in pynndescent's sparse_need_n_features tuple (hamming, matching,
// kulsinski, rogerstanimoto, russellrao, sokalmichener, correlation) have
// n_features = raw_data.shape[1] appended to their kwds and bound
// positionally (NNDescent._partial_dist_func); resolve_sparse_metric
// replicates that binding via its n_features argument (throws if a metric
// needs it and it was not provided). minkowski's p likewise comes from
// metric_kwds (numba default p=2.0 when absent).
//
// Names that still throw std::invalid_argument, with upstream rationale:
//  * "yule", "dot", "bit_hamming", "bit_jaccard", "sqjaccard", ...:
//    absent from BOTH upstream sparse registries ("dot" only exists as a
//    fast-alternative entry, but is not in sparse_named_distances, so
//    NNDescent itself raises "Metric dot not supported for sparse data");
//  * "kantorovich"/"wasserstein" (needs a ground-metric callable + optimal
//    transport), "wasserstein_1d"/"wasserstein-1d"/"kantorovich-1d",
//    "jensen-shannon"/"jensen_shannon": pynndescent-only sparse metrics that
//    umap's own sparse registry (umap/sparse.py) does not expose -- not
//    ported, mirroring the dense port's treatment of pynndescent-only
//    metrics (tsss, true_angular, ...);
//  * "symmetric-kl"/"symmetric_kl"/"symmetric_kullback_liebler": present in
//    pynndescent's sparse registry but umap_.py takes the
//    dist.named_distances branch first and raises "Metric symmetric_kl is
//    not supported for sparse data" (umap/sparse.py has no entry) -- kept
//    throwing.
// NB: umap_.py's *own* sparse registry also lacks "l2"/"sqeuclidean", so
// UMAP(metric="l2") on sparse raises upstream even though NNDescent accepts
// it; this layer mirrors the NNDescent (pynndescent) registry.
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

// sparse.py::sparse_dot_product -- float32 accumulation over the index
// intersection (declared f4 locals). Used by the sparse search-tree descent
// (rp_trees.py::sparse_select_side). NB: the numba version indexes ind1[0] /
// ind2[0] before checking sizes, so empty inputs are undefined behaviour
// upstream; here they return 0.0f (documented deviation, unreachable for
// non-empty hyperplanes/queries).
float sparse_dot_product(const int32_t* ind1, const float* data1, int64_t n1,
                         const int32_t* ind2, const float* data2, int64_t n2);

// ------------------------------------------------------------------- metrics

// True metrics (as tabulated in sparse_named_distances). All replicate the
// numba typing of the pynndescent copies (see the .cpp for per-kernel notes);
// results are returned as float32 like NNDescent's f4 heaps store them.
float sparse_euclidean(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2);
float sparse_manhattan(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2);
float sparse_chebyshev(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2);
float sparse_cosine(const int32_t* ind1, const float* data1, int64_t n1,
                    const int32_t* ind2, const float* data2, int64_t n2);
float sparse_canberra(const int32_t* ind1, const float* data1, int64_t n1,
                      const int32_t* ind2, const float* data2, int64_t n2);
float sparse_bray_curtis(const int32_t* ind1, const float* data1, int64_t n1,
                         const int32_t* ind2, const float* data2, int64_t n2);
float sparse_jaccard(const int32_t* ind1, const float* data1, int64_t n1,
                     const int32_t* ind2, const float* data2, int64_t n2);
float sparse_dice(const int32_t* ind1, const float* data1, int64_t n1,
                  const int32_t* ind2, const float* data2, int64_t n2);
float sparse_sokal_sneath(const int32_t* ind1, const float* data1, int64_t n1,
                          const int32_t* ind2, const float* data2, int64_t n2);
float sparse_hellinger(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2);
// umap/sparse.py only (not in pynndescent's registry); float32 data typing
// (numba np.sum(f4) norms, f4 log_beta inputs) unlike the f8 dense port.
float sparse_ll_dirichlet(const int32_t* ind1, const float* data1, int64_t n1,
                          const int32_t* ind2, const float* data2, int64_t n2);

// Metrics taking one extra positional argument, bound like pynndescent's
// _partial_dist_func: `extra` is minkowski's p, or n_features (=
// raw_data.shape[1], exact in double) for the sparse_need_n_features metrics.
float sparse_minkowski(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2,
                       double p);
float sparse_hamming(const int32_t* ind1, const float* data1, int64_t n1,
                     const int32_t* ind2, const float* data2, int64_t n2,
                     double n_features);
float sparse_matching(const int32_t* ind1, const float* data1, int64_t n1,
                      const int32_t* ind2, const float* data2, int64_t n2,
                      double n_features);
float sparse_kulsinski(const int32_t* ind1, const float* data1, int64_t n1,
                       const int32_t* ind2, const float* data2, int64_t n2,
                       double n_features);
float sparse_rogers_tanimoto(const int32_t* ind1, const float* data1,
                             int64_t n1, const int32_t* ind2,
                             const float* data2, int64_t n2,
                             double n_features);
float sparse_russellrao(const int32_t* ind1, const float* data1, int64_t n1,
                        const int32_t* ind2, const float* data2, int64_t n2,
                        double n_features);
float sparse_sokal_michener(const int32_t* ind1, const float* data1,
                            int64_t n1, const int32_t* ind2,
                            const float* data2, int64_t n2, double n_features);
float sparse_correlation(const int32_t* ind1, const float* data1, int64_t n1,
                         const int32_t* ind2, const float* data2, int64_t n2,
                         double n_features);

// Fast alternatives (sparse_fast_distance_alternatives) used inside
// NN-descent, plus their elementwise corrections.
float sparse_squared_euclidean(const int32_t* ind1, const float* data1,
                               int64_t n1, const int32_t* ind2,
                               const float* data2, int64_t n2);
float sparse_alternative_cosine(const int32_t* ind1, const float* data1,
                                int64_t n1, const int32_t* ind2,
                                const float* data2, int64_t n2);
float sparse_alternative_jaccard(const int32_t* ind1, const float* data1,
                                 int64_t n1, const int32_t* ind2,
                                 const float* data2, int64_t n2);
float sparse_alternative_hellinger(const int32_t* ind1, const float* data1,
                                   int64_t n1, const int32_t* ind2,
                                   const float* data2, int64_t n2);
// np.sqrt correction for "euclidean" (float32 sqrt, numpy semantics).
float sparse_sqrt_correction(float d);
// sparse.py::sparse_correct_alternative_cosine (incl. the isclose(0, |d|,
// atol=1e-7) guard the dense correction does not have).
float sparse_correct_alternative_cosine(float d);
// sparse.py::correct_alternative_jaccard -- NB no isclose/negative guard.
float sparse_correct_alternative_jaccard(float d);
// sparse.py::sparse_correct_alternative_hellinger (same guard as cosine's).
float sparse_correct_alternative_hellinger(float d);

// ------------------------------------------------------------------ registry

using SparseDistFnPtr = float (*)(const int32_t*, const float*, int64_t,
                                  const int32_t*, const float*, int64_t);
using SparseDistArgFnPtr = float (*)(const int32_t*, const float*, int64_t,
                                     const int32_t*, const float*, int64_t,
                                     double);

// A sparse distance with an optionally bound trailing argument -- the C++
// equivalent of pynndescent's _partial_dist_func closure over
// tuple(metric_kwds.values()). Implicitly constructible from a plain kernel
// pointer so existing call sites/tests keep working.
struct SparseDist {
    SparseDistFnPtr fn = nullptr;
    SparseDistArgFnPtr fn_arg = nullptr;
    double arg = 0.0;

    SparseDist() = default;
    SparseDist(SparseDistFnPtr f) : fn(f) {}  // NOLINT: implicit by design
    SparseDist(SparseDistArgFnPtr f, double a) : fn_arg(f), arg(a) {}

    explicit operator bool() const { return fn != nullptr || fn_arg != nullptr; }
    float operator()(const int32_t* i1, const float* d1, int64_t n1,
                     const int32_t* i2, const float* d2, int64_t n2) const {
        return fn ? fn(i1, d1, n1, i2, d2, n2)
                  : fn_arg(i1, d1, n1, i2, d2, n2, arg);
    }
    friend bool operator==(const SparseDist& a, const SparseDist& b) {
        return a.fn == b.fn && a.fn_arg == b.fn_arg && a.arg == b.arg;
    }
    friend bool operator!=(const SparseDist& a, const SparseDist& b) {
        return !(a == b);
    }
};

using SparseDistFn = SparseDist;

// Optional metric keyword arguments (mirrors upstream metric_kwds; n_features
// is NOT a user kwd -- it is bound from the data shape like pynndescent).
struct SparseMetricKwds {
    double p = 2.0;  // minkowski (numba signature default p=2.0)
};

struct SparseMetricInfo {
    // distance used inside NN-descent (fast alternative when one exists)
    SparseDist dist;
    // elementwise correction applied to final distances (nullptr = identity)
    float (*correction)(float) = nullptr;
    // the true metric (for brute-force kNN / reported distances)
    SparseDist true_dist;
    bool angular_trees = false;  // pynndescent's angular metric list
    // ll_dirichlet only: upstream NN-descent sees a callable, so the tree
    // flavour follows UMAP's angular_rp_forest parameter, not the name.
    bool angular_from_param = false;
};

// Resolves a metric name to its NN-descent kernels, binding n_features (=
// number of columns of the data) for pynndescent's sparse_need_n_features
// metrics and kwds.p for minkowski. Throws std::invalid_argument for names
// this port does not implement (see the header comment for the list and the
// upstream rationale) and for n_features-requiring metrics when
// n_features <= 0.
SparseMetricInfo resolve_sparse_metric(
    const std::string& name, int64_t n_features = 0,
    const SparseMetricKwds& kwds = SparseMetricKwds());

}  // namespace spdist
}  // namespace umap
