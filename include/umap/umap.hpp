// Faithful C++ port of the UMAP class (fit / fit_transform / transform) from
// reference/umap/umap/umap_.py (lmcinnes/umap, BSD 3 clause), for DENSE
// float32 input and the euclidean output metric.
//
// Ported branches of umap_.py (dense path):
//   * fit(): a/b resolution (find_ab_params), parameter validation,
//     n_neighbors truncation for tiny data, the small-data branch
//     (n < 4096 and not force_approximation_algorithm: sklearn-style pairwise
//     distances + fast_knn_indices stable-argsort kNN) and the standard
//     branch (pynndescent NN-descent via nearest_neighbors, with the search
//     index kept for transform), fuzzy_simplicial_set, and
//     _fit_embed_data -> simplicial_set_embedding (n_epochs defaults, weight
//     filtering, random/spectral init + noisy_scale_coords, [0,10] rescale,
//     optimize_layout_euclidean with move_other=True), disconnected-vertex
//     NaN masking.
//   * transform(): input-hash shortcut, kNN of new points (brute force for
//     small data, PreparedIndex::query with epsilon 0.12 / 0.24 for angular
//     metrics otherwise), smooth_knn_dist with
//     local_connectivity' = max(0, local_connectivity - 1),
//     compute_membership_strengths(bipartite=True), init_graph_transform
//     weighted-mean initialisation, the transform n_epochs rule
//     (100 / 30 default, else n_epochs // 3), transform-specific weight
//     filtering, optimize_layout_euclidean with move_other=False and
//     initial_alpha / 4 against the (copied) training embedding.
//
// Feature-bundle additions (Phase 1b, dense paths):
//   * metric="precomputed": X is an n x n distance matrix. Small data reuses
//     the precomputed dmat directly (sklearn pairwise_distances passthrough);
//     the standard branch replicates nearest_neighbors' precomputed path
//     (fast_knn_indices + inf -> -1 pruning, no search index -- transform
//     then raises like upstream). transform() with precomputed after a
//     small-data fit takes the argpartition kNN path with the upstream
//     warning + asserts. The upstream *sparse* precomputed branch is not
//     ported.
//   * precomputed_knn: (indices, dists[, search_index]) tuples bypass kNN
//     computation with upstream's validation/warning semantics, including
//     the elif-chain quirk that skips column pruning when n < 4096 (the
//     fuzzy set then runs on the wider arrays with k = n_neighbors).
//   * unique=true: np.unique(axis=0) row dedup (lexicographic stable sort),
//     fit on the unique rows, inverse-mapped embedding (unique_inverse()).
//   * init="pca" (dense PCA, sklearn svd_flip u_based_decision=False sign
//     convention, noisy_scale_coords; computed in float64 vs sklearn's
//     float32 LAPACK -- documented tolerance) and init as a user ndarray
//     (init_array), including the KDTree duplicate-row jitter branch
//     (exact 2-NN, np.mean pairwise-summation parity).
//   * n_epochs as a list (n_epochs_list): runs to max(list), snapshots at
//     each listed epoch; embedding_list() (inverse-expanded like upstream).
//   * transform_mode="graph": fit skips the embedding exactly like upstream;
//     fit_transform_graph() / transform_graph() return the CSR the reference
//     returns from fit_transform / transform.
//   * disconnection_distance: per-metric DISCONNECTION_DISTANCES defaults,
//     edge removal in all fit branches and in transform's kNN, fully
//     disconnected vertices -> NaN rows + upstream warning texts (stderr).
//
// --- supervised-sparse additions (Phase 1b) ---------------------------------
//   * Supervised / semi-supervised fit(X, y): target_metric="categorical"
//     (fast_intersection with far_dist from target_weight, -1 labels =
//     unknown), the DISCRETE_METRICS branch for "ordinal"/"count"
//     (get_discrete_params + fast_metric_intersection; "string"/"myers"
//     need object-dtype y and "hierarchical_categorical" KeyErrors upstream
//     -- all three raise here), and the general branch (target fuzzy
//     simplicial set with apply_set_operations=False, sklearn float64
//     pairwise for n < 4096 / NN-descent otherwise,
//     general_simplicial_set_intersection + reset_local_connectivity).
//     The disconnected-vertex NaN mask is recomputed from the intersected
//     graph, as upstream does inside the embedding block.
//   * Sparse (CSR) input wiring: fit/fit_transform/transform overloads taking
//     scipy-layout CSR arrays. Small data (n < 4096) replicates sklearn's
//     sparse pairwise_distances float32 semantics for euclidean/l2
//     (_euclidean_distances_upcast: float64 chunks over float32 data) and
//     cosine (inplace_csr_row_normalize_l2 with a float64 accumulator +
//     scipy's fmaf-contracted float32 sparse matmul); other metrics raise
//     (upstream densifies via pairwise_special_metric -- not ported). The
//     standard branch builds a SparsePreparedIndex (continuing the numpy
//     RandomState stream) whose query() serves transform at epsilon 0.12 /
//     0.24 (angular). unique=True uses csr_unique object-row semantics.
//     Not ported for sparse input: metric="precomputed" (upstream's sparse
//     precomputed branch), precomputed_knn, init="pca" (TruncatedSVD).
//     Deviation: transform's small-data kNN breaks exact distance ties in
//     stable order, not numpy argpartition/argsort order (ties are common
//     for sparse cosine at low density -- disjoint rows are all at 1.0).
//   * graph() after a transform_mode="embedding" fit contains explicit zeros
//     at sub-threshold entries: upstream's epoch weight filter writes through
//     scipy's shared tocoo() data array into self.graph_ (replicated via
//     resolve_epochs_and_filter_graph's write_through -- see layouts.hpp),
//     and the NaN disconnection mask is computed from that mutated graph.
// ----------------------------------------------------------------------------
//
// Deferred (upstream branches NOT ported): sparse precomputed distances,
// init="tswspectral", inverse_transform / update / AlignedUMAP.
//
// Seeded parity: when random_state is set, the numpy legacy RandomState
// stream is replicated draw for draw (NN-descent seeding, init draws --
// including the eigsh warm-start normals upstream generates and discards --
// and the SGD rng_state), matching Python's serial path
// (NUMBA_NUM_THREADS=1). On the small-data branch with init="random" and
// explicit a/b, fit and transform are bit-exact against Python end to end
// (see tests/test_umap.cpp). The residual tolerances elsewhere, with causes:
//   * default a/b: our find_ab_params LM reaches scipy curve_fit's minimum
//     only to ~1e-7 relative (different LM path);
//   * NN-descent branch: numba fastmath kernels leave kNN *distances* ~1 ulp
//     off (indices identical), perturbing fuzzy weights and the SGD;
//   * init="spectral": Spectra vs ARPACK eigenvectors agree only to solver
//     precision and up to per-column sign.
// Multi-component spectral init falls back to the tau-prng implementation in
// spectral.hpp (distributional parity only). Unseeded runs draw the seed from
// std::random_device and use the hogwild parallel SGD, mirroring upstream's
// parallel=True when random_state is None.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "umap/fuzzy_simplicial_set.hpp"
#include "umap/nn_descent.hpp"
#include "umap/sparse_matrix.hpp"

namespace umap {

// --- supervised-sparse additions --------------------------------------------
namespace spnnd {
class SparsePreparedIndex;
}
// ----------------------------------------------------------------------------

// --- feature-bundle additions -----------------------------------------------
// The `precomputed_knn=(knn_indices, knn_dists[, knn_search_index])` tuple.
// indices/dists are row-major (n_samples x n_neighbors); search_index may be
// null (upstream warns and transform() becomes unavailable).
struct PrecomputedKnn {
    std::vector<int32_t> indices;
    std::vector<float> dists;
    int64_t n_samples = 0;
    int64_t n_neighbors = 0;
    std::shared_ptr<nnd::PreparedIndex> search_index;
};

// umap_.py::DISCONNECTION_DISTANCES defaults for bounded metrics; every other
// metric maps to +inf.
double default_disconnection_distance(const std::string& metric);
// ----------------------------------------------------------------------------

// Constructor defaults mirror umap.UMAP's signature (the subset ported).
struct UMAPParams {
    int n_neighbors = 15;
    int n_components = 2;
    std::string metric = "euclidean";     // named metrics or "precomputed"
    std::optional<int> n_epochs;          // None -> 500/200 (fit), 100/30 (transform)
    double learning_rate = 1.0;
    std::string init = "spectral";        // "spectral" | "random" | "pca" ("tswspectral" TODO)
    double min_dist = 0.1;
    double spread = 1.0;
    double set_op_mix_ratio = 1.0;
    double local_connectivity = 1.0;
    double repulsion_strength = 1.0;
    int negative_sample_rate = 5;
    double transform_queue_size = 4.0;    // stored like upstream; unused by this transform path
    std::optional<double> a;              // both unset -> find_ab_params(spread, min_dist)
    std::optional<double> b;
    std::optional<uint32_t> random_state; // np.random.RandomState(seed) parity when set
    uint32_t transform_seed = 42;
    bool angular_rp_forest = false;       // accepted for parity; named metrics decide trees
    bool force_approximation_algorithm = false;
    bool verbose = false;
    // --- feature-bundle additions -------------------------------------------
    // n_epochs given as a list (upstream: the same n_epochs parameter). At
    // most one of n_epochs / n_epochs_list may be set; the intermediate
    // embeddings land in UMAP::embedding_list().
    std::optional<std::vector<int>> n_epochs_list;
    // init as a user ndarray (upstream: init=np.ndarray). When set, `init`
    // (the string) is ignored. Row-major n_samples x init_array_cols;
    // init_array_cols = -1 means n_components (the only valid width --
    // upstream raises "init ndarray must match n_components value").
    std::optional<std::vector<float>> init_array;
    int64_t init_array_cols = -1;
    bool unique = false;
    // None -> DISCONNECTION_DISTANCES.get(metric, inf).
    std::optional<double> disconnection_distance;
    std::string transform_mode = "embedding";  // "embedding" | "graph"
    std::optional<PrecomputedKnn> precomputed_knn;
    // ------------------------------------------------------------------------
    // --- densmap additions --------------------------------------------------
    // metric_kwds: extra positional parameters of the input metric, bound like
    // upstream's tuple(metric_kwds.values()) (see distances.hpp::MetricKwds).
    // Threaded into both kNN branches (small-data pairwise + NN-descent) and
    // the transform kNN.
    MetricKwds metric_kwds;
    // output_metric / output_metric_kwds: the embedding-space metric.
    // "euclidean"/"l2" take the specialised euclidean SGD; other named
    // metrics with gradients run optimize_layout_generic; "precomputed" and
    // gradient-less metrics raise like upstream.
    std::string output_metric = "euclidean";
    MetricKwds output_metric_kwds;
    // densMAP (density-augmented objective) + local-radii output.
    bool densmap = false;
    double dens_lambda = 2.0;
    double dens_frac = 0.3;
    double dens_var_shift = 0.1;
    bool output_dens = false;
    // ------------------------------------------------------------------------
    // --- supervised-sparse additions ----------------------------------------
    // Supervised fit target parameters (umap.UMAP defaults). target_metric
    // may be "categorical" (default), a DISCRETE_METRICS name ("ordinal" /
    // "count" supported; get_discrete_params overrides target_metric_kwds
    // exactly like upstream) or any named metric for the general target
    // branch ("l2"/"euclidean" replicate sklearn's float64 pairwise path
    // bit-exactly for 1-d targets).
    std::string target_metric = "categorical";
    MetricKwds target_metric_kwds;
    int target_n_neighbors = -1;  // -1 -> use _n_neighbors
    double target_weight = 0.5;
    // ------------------------------------------------------------------------
};

class UMAP {
 public:
    explicit UMAP(UMAPParams params = UMAPParams());

    // Fit X (row-major n_samples x n_features float32) into an embedded space.
    // y (optional): per-sample target labels for supervised dimension
    // reduction (float; categorical labels with -1.0 marking "unknown" for
    // the default target_metric="categorical", or continuous values for the
    // general target branch), matching upstream's fit(X, y) semantics.
    UMAP& fit(const float* X, int64_t n_samples, int64_t n_features,
              const float* y = nullptr);

    // fit + return a copy of the embedding (n_samples x n_components).
    // transform_mode="graph" raises here (the reference returns the graph from
    // fit_transform; in C++ use fit_transform_graph for that variant).
    std::vector<float> fit_transform(const float* X, int64_t n_samples,
                                     int64_t n_features,
                                     const float* y = nullptr);

    // --- supervised-sparse additions ----------------------------------------
    // Sparse (CSR, scipy layout: sorted duplicate-free column indices per
    // row, float32 data) fit / fit_transform / transform. n_features is the
    // number of columns of the CSR matrix.
    UMAP& fit(const int32_t* indptr, const int32_t* indices, const float* data,
              int64_t n_samples, int64_t n_features, const float* y = nullptr);
    std::vector<float> fit_transform(const int32_t* indptr,
                                     const int32_t* indices, const float* data,
                                     int64_t n_samples, int64_t n_features,
                                     const float* y = nullptr);
    std::vector<float> transform(const int32_t* q_indptr,
                                 const int32_t* q_indices, const float* q_data,
                                 int64_t n_queries) const;
    SparseMatrix transform_graph(const int32_t* q_indptr,
                                 const int32_t* q_indices, const float* q_data,
                                 int64_t n_queries) const;
    // ------------------------------------------------------------------------

    // Transform new points (row-major n_queries x n_features) into the
    // existing embedded space; returns n_queries x n_components.
    std::vector<float> transform(const float* X_new, int64_t n_queries) const;

    // --- feature-bundle additions -------------------------------------------
    // fit_transform for transform_mode="graph": returns self.graph_ (the CSR
    // fuzzy simplicial set) like the reference's fit_transform.
    const SparseMatrix& fit_transform_graph(const float* X, int64_t n_samples,
                                            int64_t n_features);
    // transform for transform_mode="graph": the bipartite membership graph
    // (n_queries x n_train CSR) the reference's transform returns.
    SparseMatrix transform_graph(const float* X_new, int64_t n_queries) const;
    // ------------------------------------------------------------------------

    // -------- fitted attributes (valid after fit) --------
    const std::vector<float>& embedding() const { return embedding_; }
    const SparseMatrix& graph() const { return graph_; }
    const std::vector<int32_t>& knn_indices() const { return knn_indices_; }
    const std::vector<float>& knn_dists() const { return knn_dists_; }
    const std::vector<float>& sigmas() const { return sigmas_; }
    const std::vector<float>& rhos() const { return rhos_; }
    double a() const { return a_; }
    double b() const { return b_; }
    // The initial embedding fed to the SGD (after init + noisy_scale_coords,
    // *before* the [0,10] rescale that simplicial_set_embedding applies);
    // kept for stage-parity tests against the reference.
    const std::vector<float>& fit_init() const { return fit_init_; }
    int effective_n_neighbors() const { return n_neighbors_; }  // _n_neighbors
    bool small_data() const { return small_data_; }
    bool sparse_data() const { return sparse_data_; }  // self._sparse_data
    int64_t n_samples() const { return n_; }
    int64_t n_features() const { return dim_; }
    const UMAPParams& params() const { return params_; }

    // --- feature-bundle additions -------------------------------------------
    // self.embedding_list_ (n_epochs_list fits only): one inverse-expanded
    // snapshot per listed epoch that is reached, plus the final embedding.
    const std::vector<std::vector<float>>& embedding_list() const {
        return embedding_list_;
    }
    // self._unique_inverse_ (unique=true fits only): original row -> row of
    // the deduplicated data (np.unique return_inverse semantics).
    const std::vector<int64_t>& unique_inverse() const {
        return unique_inverse_;
    }
    // self._disconnection_distance after resolution (valid after fit).
    double disconnection_distance() const { return disconnection_distance_; }
    // --- densmap additions --------------------------------------------------
    // self.graph_dists_ (densmap or output_dens fits): max(D, D^T) of the kNN
    // edge distances, as returned by fuzzy_simplicial_set(return_dists=True).
    const SparseMatrix& graph_dists() const { return graph_dists_; }
    // self.rad_orig_ / self.rad_emb_ (output_dens fits only): local radii in
    // the original space and the embedding, inverse-expanded like embedding_.
    const std::vector<float>& rad_orig() const { return rad_orig_; }
    const std::vector<float>& rad_emb() const { return rad_emb_; }
    // ------------------------------------------------------------------------
    // Disconnection bookkeeping from the last fit (raise_disconnected_warning
    // inputs): edges removed by the threshold and vertices of graph_ with an
    // all-zero row.
    int64_t edges_removed() const { return edges_removed_; }
    int64_t vertices_disconnected() const { return vertices_disconnected_; }
    // ------------------------------------------------------------------------

 private:
    void validate_parameters() const;
    std::vector<float> make_fit_init(const float* data, int64_t n_index,
                                     const SparseMatrix& filtered_graph,
                                     nnd::NumpyRandomState& rs) const;
    // The shared fit body (dense when X != nullptr, sparse otherwise; the
    // sparse overload fills the raw CSR members first).
    UMAP& fit_impl(const float* X, int64_t n_samples, int64_t n_features,
                   const float* y);
    // The y-handling block of fit(): intersect graph_ with the target
    // simplicial set (categorical / discrete / general branches).
    void supervise_graph(const float* y, const std::vector<int64_t>& index,
                         int64_t n_samples, nnd::NumpyRandomState& rs);
    // The shared transform prologue: kNN of the new points vs the training
    // data (precomputed / brute-force / query paths), disconnection pruning,
    // smooth_knn_dist and compute_membership_strengths(bipartite=True).
    // Returns the COO triplets in generation order (the transform SGD's edge
    // order upstream).
    MembershipStrengths transform_membership(const float* X_new,
                                             int64_t n_queries) const;
    // --- supervised-sparse additions ----------------------------------------
    // Sparse variant of the transform prologue (brute force vs the training
    // CSR for small data, SparsePreparedIndex::query otherwise).
    MembershipStrengths transform_membership_sparse(const int32_t* q_indptr,
                                                    const int32_t* q_indices,
                                                    const float* q_data,
                                                    int64_t n_queries) const;
    // The shared transform epilogue (init_graph_transform + SGD), used by
    // both the dense and sparse transform overloads.
    std::vector<float> transform_from_membership(const MembershipStrengths& ms,
                                                 int64_t n_queries) const;
    // ------------------------------------------------------------------------

    UMAPParams params_;
    double a_ = 0.0;
    double b_ = 0.0;
    double initial_alpha_ = 1.0;
    int n_neighbors_ = 0;      // self._n_neighbors (truncated for tiny data)
    bool small_data_ = false;
    bool fitted_ = false;
    int64_t n_ = 0;
    int64_t dim_ = 0;
    std::vector<float> raw_data_;                    // training data copy
    std::shared_ptr<nnd::PreparedIndex> knn_index_;  // standard branch only
    // --- supervised-sparse additions ----------------------------------------
    bool sparse_data_ = false;         // self._sparse_data
    std::vector<int32_t> raw_indptr_;  // training CSR copy (sparse fits)
    std::vector<int32_t> raw_indices_;
    std::vector<float> raw_values_;
    std::shared_ptr<spnnd::SparsePreparedIndex> sparse_knn_index_;
    // ------------------------------------------------------------------------
    std::vector<int32_t> knn_indices_;
    std::vector<float> knn_dists_;
    int64_t knn_cols_ = 0;  // knn array width (may exceed n_neighbors_)
    SparseMatrix graph_;
    std::vector<float> sigmas_;
    std::vector<float> rhos_;
    std::vector<float> fit_init_;
    std::vector<float> embedding_;  // n_ x n_components row-major float32
    // --- feature-bundle additions -------------------------------------------
    bool force_approximation_algorithm_ = false;  // effective (pknn may set it)
    double disconnection_distance_ = 0.0;         // _disconnection_distance
    int64_t edges_removed_ = 0;
    int64_t vertices_disconnected_ = 0;
    std::vector<int64_t> unique_inverse_;
    std::vector<std::vector<float>> embedding_list_;
    // ------------------------------------------------------------------------
    // --- densmap additions --------------------------------------------------
    SparseMatrix graph_dists_;   // graph_dists_ (densmap / output_dens fits)
    std::vector<float> rad_orig_;
    std::vector<float> rad_emb_;
    // ------------------------------------------------------------------------
};

// Replication of the small-data branches' pairwise_distances(X, metric=
// self._input_distance_func) call: sklearn routes the umap numba kernel
// through _pairwise_callable (float32 output, zero diagonal when X is Y).
// For "euclidean"/"l2" this is element-exact (float32 ops with a float64
// accumulator, like the numba kernel); other named metrics evaluate the
// double-precision registry in distances.hpp rounded to float32 (may differ
// from numba's float32 kernels in the last ulp). Row-major nx x ny output.
// metric_kwds are bound positionally like upstream (`pairwise_distances(X,
// metric=_m, **self._metric_kwds)`); f32-faithful kernels exist for
// euclidean/l2, manhattan, minkowski and seuclidean, other named metrics
// evaluate the double-precision registry rounded to float32 (may differ from
// numba's float32 kernels in the last ulp).
std::vector<float> pairwise_distances_sklearn(const float* X, int64_t nx,
                                              const float* Y, int64_t ny,
                                              int64_t dim,
                                              const std::string& metric,
                                              bool x_is_y,
                                              const MetricKwds& kwds = {});

// utils.py::fast_knn_indices semantics: per row of the (n x m) distance
// matrix, the n_neighbors smallest entries with stable (mergesort) tie-break
// by ascending column index. Outputs are (n x n_neighbors) row-major.
void fast_knn_from_dmat(const std::vector<float>& dmat, int64_t n, int64_t m,
                        int64_t n_neighbors, int32_t* knn_indices,
                        float* knn_dists);

// umap_.py::init_graph_transform: weighted-mean initialisation of new points
// from a bipartite membership CSR (n_new x n_train) and the training
// embedding, replicating scipy's float32 arithmetic (row sums, 1/sum scaling
// and the CSR @ dense product all in float32). Rows with no entries become
// NaN; rows containing a strength of exactly 1.0 copy that training point.
std::vector<float> init_graph_transform(const SparseMatrix& graph,
                                        const float* embedding, int dim);

// --- feature-bundle additions -----------------------------------------------
// np.unique(X, axis=0, return_index=True, return_inverse=True) semantics for a
// dense row-major float32 matrix: rows are compared lexicographically by value
// (-0.0 == 0.0, like numpy's structured-dtype field comparison) with a stable
// (mergesort) argsort, `index` collects the first original occurrence of each
// unique row in sorted order, `inverse` maps original rows to unique rows.
void unique_rows(const float* X, int64_t n, int64_t dim,
                 std::vector<int64_t>& index, std::vector<int64_t>& inverse);

// sklearn PCA(n_components).fit_transform for dense data: center by column
// means, thin SVD, svd_flip(u, v, u_based_decision=False) sign convention
// (each component's sign makes the largest-|.|-entry of the corresponding row
// of V^T positive), returns U * S truncated to n_components (row-major
// float32). Computed in float64 (sklearn runs float32 LAPACK on float32
// input; documented ~1e-4-level tolerance vs the reference).
std::vector<float> pca_fit_transform(const float* X, int64_t n, int64_t dim,
                                     int n_components);
// ----------------------------------------------------------------------------

// --- supervised-sparse additions -----------------------------------------------
// utils.py::csr_unique semantics: rows of a CSR matrix keyed as the tuple
// (column indices..., values...) and deduplicated with np.unique object-array
// ordering (lexicographic on the concatenated key; a key that is a strict
// prefix of another sorts first). `index` holds the first original occurrence
// of each unique row in key-sorted order, `inverse` maps original rows to
// unique rows.
void csr_unique_rows(const int32_t* indptr, const int32_t* indices,
                     const float* data, int64_t n,
                     std::vector<int64_t>& index,
                     std::vector<int64_t>& inverse);

// Replication of sklearn pairwise_distances on scipy CSR float32 input for
// the metrics the sparse small-data branch supports bit-exactly:
//   * "euclidean"/"l2": _euclidean_distances_upcast -- float64 row norms
//     (sequential over stored order) and dot products (ascending-column
//     merge), (-2*dot + XX_i) + YY_j rounded to float32, maximum(0), zero
//     diagonal when X is Y, float32 sqrt;
//   * "cosine": inplace_csr_row_normalize_l2 (float32 squares accumulated in
//     float64, float64 divide rounded to float32), scipy's float32 sparse
//     matmul with fmaf contraction (ascending-column merge), S = 1 - S,
//     clip(0, 2), zero diagonal when X is Y.
// Everything else throws (upstream falls back to densified
// pairwise_special_metric -- not ported). Row-major nx x ny float32 output.
std::vector<float> sparse_pairwise_distances_sklearn(
    const int32_t* x_indptr, const int32_t* x_indices, const float* x_data,
    int64_t nx, const int32_t* y_indptr, const int32_t* y_indices,
    const float* y_data, int64_t ny, const std::string& metric, bool x_is_y);
// ----------------------------------------------------------------------------

}  // namespace umap
