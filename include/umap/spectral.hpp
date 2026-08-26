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
bool normalized_laplacian_eigs(const SparseMatrix& graph, int k,
                               std::vector<double>& eigenvalues,
                               std::vector<double>& eigenvectors);

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
};

// Initial embedding for `graph` (n x dim row-major float32).
// `data` may be nullptr unless required (multi-component Spectral with
// n_components > 2*dim uses it for centroids; PCA would need it).
std::vector<float> initialize_embedding(const float* data, int64_t n_features,
                                        const SparseMatrix& graph, int dim,
                                        EmbeddingInit init,
                                        int64_t rng_state[3]);

}  // namespace umap
