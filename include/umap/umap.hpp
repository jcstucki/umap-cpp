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
// Deferred (upstream branches NOT ported): sparse input, precomputed
// metric/knn, unique=True, supervised fit (y), densmap / output_dens,
// non-euclidean output metrics, init="pca"/"tswspectral" or ndarray init,
// n_epochs lists, transform_mode="graph", disconnection_distance overrides,
// inverse_transform / update.
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

#include "umap/nn_descent.hpp"
#include "umap/sparse_matrix.hpp"

namespace umap {

// Constructor defaults mirror umap.UMAP's signature (the subset ported).
struct UMAPParams {
    int n_neighbors = 15;
    int n_components = 2;
    std::string metric = "euclidean";
    std::optional<int> n_epochs;          // None -> 500/200 (fit), 100/30 (transform)
    double learning_rate = 1.0;
    std::string init = "spectral";        // "spectral" | "random" ("pca"/"tswspectral" TODO)
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
};

class UMAP {
 public:
    explicit UMAP(UMAPParams params = UMAPParams());

    // Fit X (row-major n_samples x n_features float32) into an embedded space.
    UMAP& fit(const float* X, int64_t n_samples, int64_t n_features);

    // fit + return a copy of the embedding (n_samples x n_components).
    std::vector<float> fit_transform(const float* X, int64_t n_samples,
                                     int64_t n_features);

    // Transform new points (row-major n_queries x n_features) into the
    // existing embedded space; returns n_queries x n_components.
    std::vector<float> transform(const float* X_new, int64_t n_queries) const;

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
    int64_t n_samples() const { return n_; }
    int64_t n_features() const { return dim_; }
    const UMAPParams& params() const { return params_; }

 private:
    void validate_parameters() const;
    std::vector<float> make_fit_init(const SparseMatrix& filtered_graph,
                                     nnd::NumpyRandomState& rs) const;

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
    std::unique_ptr<nnd::PreparedIndex> knn_index_;  // standard branch only
    std::vector<int32_t> knn_indices_;
    std::vector<float> knn_dists_;
    SparseMatrix graph_;
    std::vector<float> sigmas_;
    std::vector<float> rhos_;
    std::vector<float> fit_init_;
    std::vector<float> embedding_;  // n_ x n_components row-major float32
};

// Replication of the small-data branches' pairwise_distances(X, metric=
// self._input_distance_func) call: sklearn routes the umap numba kernel
// through _pairwise_callable (float32 output, zero diagonal when X is Y).
// For "euclidean"/"l2" this is element-exact (float32 ops with a float64
// accumulator, like the numba kernel); other named metrics evaluate the
// double-precision registry in distances.hpp rounded to float32 (may differ
// from numba's float32 kernels in the last ulp). Row-major nx x ny output.
std::vector<float> pairwise_distances_sklearn(const float* X, int64_t nx,
                                              const float* Y, int64_t ny,
                                              int64_t dim,
                                              const std::string& metric,
                                              bool x_is_y);

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

}  // namespace umap
