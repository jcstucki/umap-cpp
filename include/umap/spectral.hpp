// Faithful C++ port of the spectral embedding initialization from
// reference/umap/umap/spectral.py (lmcinnes/umap, BSD 3 clause), plus the
// init-dispatch pieces of umap_.py's simplicial_set_embedding (random init and
// noisy_scale_coords).
//
// Semantics ported:
//   * _spectral_layout / spectral_layout: for a connected graph, the
//     (dim + 1) smallest eigenpairs of the normalized Laplacian
//     L = I - D^{-1/2} A D^{-1/2}, ordered by ascending eigenvalue with the
//     trivial first eigenvector dropped (order = np.argsort(eigenvalues)[1:k]).
//     Eigenvectors are unit 2-norm with arbitrary sign, exactly like ARPACK.
//     On eigensolver failure, upstream warns and falls back to a uniform
//     random layout in [-10, 10]; we do the same (see rng note below).
//   * multi_component_layout: meta-embedding of connected-component centers
//     (deterministic +/- eye construction for n_components <= 2*dim, else
//     component_layout), per-component spectral embedding scaled by
//     data_range / max|coord| into the meta positions, uniform random
//     placement for tiny components (size < 2*dim or size <= dim + 1).
//   * component_layout: centroid per component, euclidean pairwise distances,
//     affinity exp(-d^2), then a spectral embedding of the affinity matrix
//     replicating sklearn.manifold.SpectralEmbedding's selection: normalized
//     Laplacian with zeroed diagonal, smallest (dim + 1) eigenvectors scaled
//     by 1/sqrt(degree), deterministic sign flip (max-abs element positive),
//     drop-first; finally divided by the (signed) max entry. Solved densely
//     (exact) instead of ARPACK -- same selection, tighter accuracy.
//     Only metric="euclidean" is supported; data == nullptr reproduces the
//     upstream "just guess" branch (uniform random in [0, 10)).
//   * connected_components: weak connectivity with scipy.sparse.csgraph
//     label semantics (components labeled in order of first-encountered
//     vertex index).
//
// Deviations (documented, allowed by the port contract):
//   * Random draws use our tau_prng (umap/rng.hpp) instead of numpy
//     RandomState/Generator, so random fallbacks / jitter are not bit-equal
//     to Python -- only distributionally equivalent. Gaussian noise uses
//     Box-Muller on tau_rand.
//   * The eigensolver is Spectra's shift-invert Lanczos (sigma = -1e-5, the
//     value sklearn recommends) rather than ARPACK's regular-mode
//     which="SM"; selection (k = dim + 1 smallest, ncv = max(2k+1, sqrt(n))
//     clamped to n, v0 = ones) and normalization are replicated.
//   * Upstream errors out (uncaught TypeError via scipy) when a connected
//     graph has n <= dim + 1; we throw std::invalid_argument.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "umap/sparse_matrix.hpp"

namespace umap {

// scipy.sparse.csgraph.connected_components(graph) equivalent (weak
// connectivity: edge direction ignored). Fills `labels` (size n_rows) with
// component ids assigned in order of first-encountered vertex; returns the
// number of components.
int32_t connected_components(const SparseMatrix& graph,
                             std::vector<int32_t>& labels);

// The k algebraically-smallest eigenpairs of the normalized Laplacian
// L = I - D^{-1/2} A D^{-1/2} of `graph` (degrees = column sums, matching
// graph.sum(axis=0)). On success returns true with `eigenvalues` ascending
// (size k) and `eigenvectors` row-major (n x k), unit 2-norm columns of
// arbitrary sign. Returns false if the solver fails to converge.
// Requires k < n (throws std::invalid_argument otherwise, mirroring the
// upstream uncaught scipy error for tiny graphs).
// maxiter = 0 keeps the historical default max(1000, 5n); a positive value
// caps the Spectra iterations like upstream's eigsh maxiter (tswspectral
// threads its `maxiter or graph.shape[0] * 5` through here).
bool normalized_laplacian_eigs(const SparseMatrix& graph, int k,
                               std::vector<double>& eigenvalues,
                               std::vector<double>& eigenvectors,
                               int64_t maxiter = 0);

// Port of spectral.component_layout for metric="euclidean" (or data == nullptr).
// data: row-major (n_samples x n_features) or nullptr; component_labels: size
// n_samples with values in [0, n_components). Returns row-major
// (n_components x dim). rng_state is used only for the data == nullptr branch.
std::vector<double> component_layout(const float* data, int64_t n_samples,
                                     int64_t n_features, int32_t n_components,
                                     const std::vector<int32_t>& component_labels,
                                     int dim, int64_t rng_state[3]);

// Port of spectral.multi_component_layout. Returns row-major
// (n_samples x dim) float, one row per vertex of `graph`.
std::vector<float> multi_component_layout(
    const float* data, int64_t n_features, const SparseMatrix& graph,
    int32_t n_components, const std::vector<int32_t>& component_labels, int dim,
    int64_t rng_state[3]);

// Port of spectral.spectral_layout (init="random" eigensolver path): spectral
// embedding of `graph`, dispatching to multi_component_layout when the graph
// is disconnected. `data` (row-major, may be nullptr) is only used when
// multiple components require centroid meta-layout. Returns row-major
// (n x dim) float.
std::vector<float> spectral_layout(const float* data, int64_t n_features,
                                   const SparseMatrix& graph, int dim,
                                   int64_t rng_state[3]);

// Convenience overload without source data (data=None upstream).
inline std::vector<float> spectral_layout(const SparseMatrix& graph, int dim,
                                          int64_t rng_state[3]) {
    return spectral_layout(nullptr, 0, graph, dim, rng_state);
}

// --- tswspectral additions --------------------------------------------------
// Port of spectral.tswspectral_layout, i.e. _spectral_layout(init="tsvd"):
// the same normalized-Laplacian eigenvectors as spectral_layout, but the
// eigensolver is "warmed" with a truncated-SVD guess of the eigenvectors.
// What the reference actually does (current upstream source):
//   * disconnected graphs route to multi_component_layout BEFORE anything
//     tsvd-related, exactly like spectral_layout;
//   * method=None resolves to "eigsh" when n < 2,000,000 else "lobpcg"
//     (any other string raises ValueError "Method should either be None,
//     'eigsh' or 'lobpcg'" -- we throw std::invalid_argument);
//   * the warm start X = TruncatedSVD(n_components=k, random_state).
//     fit_transform(L) (sklearn randomized SVD: k + 10 oversampled Gaussian
//     block, 5 normalizer-stabilized power iterations; k = dim + 1), with
//     X[:, 0] overwritten by the exact first eigenvector
//     sqrt_deg / ||sqrt_deg||;
//   * the "eigsh" branch then DISCARDS X entirely (upstream passes
//     v0=np.ones(n), tol=tol or 1e-4, ncv=max(2k+1, sqrt(n)),
//     maxiter=maxiter or 5n) -- so with the default method the eigen-solve
//     is identical to spectral_layout and only the RNG consumption differs;
//   * the "lobpcg" branch runs scipy.sparse.linalg.lobpcg(L, X,
//     largest=False, tol=tol or 1e-4, maxiter=maxiter or 5n) with the
//     "not reaching the requested tolerance" UserWarning escalated to an
//     error, i.e. NON-CONVERGENCE FALLS BACK TO RANDOM;
//   * order = np.argsort(eigenvalues)[1:k]; return eigenvectors[:, order];
//   * on ArpackError/UserWarning: warn "Spectral initialisation failed!..."
//     and return gen.uniform(-10, 10, (n, dim)).
// Deviations (same contract as spectral_layout): tau_prng replaces the numpy
// Generator/RandomState draws (tsvd Gaussian block, random fallback);
// the eigsh path solves via Spectra shift-invert at tol 1e-10 (tighter than
// upstream's 1e-4 -- ARPACK regular-mode "SM" at loose tol misconverges in
// scipy >= 1.15, which is why upstream skips its tsw test there; we always
// converge to the true eigenspace like the scipy in [1.10, 1.15) window),
// so `tol` only affects the lobpcg method; the randomized-SVD power
// iterations use thin-QR normalization instead of sklearn's LU (same
// subspace quality); metric/metric_kwds are euclidean-only via
// component_layout, as for spectral_layout. `maxiter` caps both solvers.
// `used_random_fallback` (optional) reports whether the random fallback
// fired -- the C++ seam for upstream's pytest.warns(UserWarning) assertion.
//
// UMAP-class wiring for init="tswspectral" (documented here because
// umap.hpp / umap.cpp are owned by another workstream; do the following
// there once this branch merges):
//   1. _validate_parameters: the string-init set becomes
//      ("pca", "spectral", "random", "tswspectral") with the upstream error
//      text 'string init values must be one of: "pca", "tswspectral",
//      "spectral" or "random"'.
//   2. simplicial_set_embedding dispatch (umap_.py:1172): the branch is
//      identical to init == "spectral" but calls
//      tswspectral_layout(data, graph, n_components, random_state,
//      metric=metric, metric_kwds=metric_kwds) -- method/tol/maxiter stay
//      at their defaults -- followed by the same
//      noisy_scale_coords(embedding, random_state, max_coord=10,
//      noise=0.0001). (initialize_embedding below already dispatches
//      EmbeddingInit::TSWSpectral this way.)
//   3. numpy RandomState stream parity (the class replicates draws itself,
//      as its init="spectral" branch does): on a CONNECTED graph,
//      _spectral_layout(init="tsvd") consumes exactly n * (k + 10) rk_gauss
//      draws where k = n_components + 1 -- sklearn
//      TruncatedSVD(algorithm="randomized") -> randomized_range_finder draws
//      random_state.normal(size=(n, k + n_oversamples)) with
//      n_oversamples=10, and nothing else touches the stream (verified
//      against sklearn 1.9). So the class consumes n * (dim + 11) rs.gauss()
//      values (instead of the n * (dim + 1) discarded normals of
//      init="spectral") and then calls normalized_laplacian_eigs exactly as
//      for init="spectral" (method=None resolves to "eigsh" below 2e6 rows,
//      and upstream's eigsh branch discards the warm start). The failure
//      fallback draw is gen.uniform(-10, 10, (n, dim)) from the same stream.
//      Disconnected graphs route to multi_component_layout before any tsvd
//      draw, exactly as for init="spectral".
std::vector<float> tswspectral_layout(const float* data, int64_t n_features,
                                      const SparseMatrix& graph, int dim,
                                      int64_t rng_state[3],
                                      const std::string& method = std::string(),
                                      double tol = 0.0, int64_t maxiter = 0,
                                      bool* used_random_fallback = nullptr);

// Convenience overload without source data (data=None upstream).
inline std::vector<float> tswspectral_layout(
    const SparseMatrix& graph, int dim, int64_t rng_state[3],
    const std::string& method = std::string(), double tol = 0.0,
    int64_t maxiter = 0, bool* used_random_fallback = nullptr) {
    return tswspectral_layout(nullptr, 0, graph, dim, rng_state, method, tol,
                              maxiter, used_random_fallback);
}

// scipy.sparse.linalg.lobpcg(L, X0, largest=False, tol, maxiter) on the
// normalized Laplacian L of `graph`, exposed for tests. X0 is the row-major
// (n x m) initial block; on success (all residual 2-norms <= tol within
// maxiter iterations, scipy's criterion) returns true with `eigenvalues`
// ascending (size m) and `eigenvectors` row-major (n x m, orthonormal
// columns). Returns false when the tolerance was not reached (upstream
// escalates scipy's warning and falls back to random). Mirrors scipy's
// small-problem branch: n < 5 * m solves densely and always "converges".
bool normalized_laplacian_lobpcg(const SparseMatrix& graph, const double* X0,
                                 int m, double tol, int64_t maxiter,
                                 std::vector<double>& eigenvalues,
                                 std::vector<double>& eigenvectors);
// ----------------------------------------------------------------------------

// Port of umap_.noisy_scale_coords: scale so the largest |coordinate| equals
// max_coord (computed in float32 like upstream), then add centered Gaussian
// noise with standard deviation `noise`. Gaussian parity with numpy is NOT
// bit-exact (Box-Muller on tau_rand); documented deviation.
std::vector<float> noisy_scale_coords(std::vector<float> coords,
                                      int64_t rng_state[3],
                                      float max_coord = 10.0f,
                                      float noise = 0.0001f);

// The init-dispatch piece of umap_.simplicial_set_embedding.
enum class EmbeddingInit {
    Random,    // uniform in [-10, 10]
    Spectral,  // spectral_layout + noisy_scale_coords(max_coord=10, noise=1e-4)
    PCA,       // TODO: not implemented (throws)
    // --- tswspectral additions ---
    TSWSpectral,  // tswspectral_layout + noisy_scale_coords(10, 1e-4)
};

// Initial embedding for `graph` (n x dim row-major float32).
// `data` may be nullptr unless required (multi-component Spectral with
// n_components > 2*dim uses it for centroids; PCA would need it).
std::vector<float> initialize_embedding(const float* data, int64_t n_features,
                                        const SparseMatrix& graph, int dim,
                                        EmbeddingInit init,
                                        int64_t rng_state[3]);

}  // namespace umap
