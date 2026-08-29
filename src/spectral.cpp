// Implementation of include/umap/spectral.hpp. See that header for the
// faithfulness notes and documented deviations vs reference/umap/umap/spectral.py.
#include "umap/spectral.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsShiftSolver.h>

#include "umap/rng.hpp"

namespace umap {

namespace {

// random_state.uniform(low, high) equivalent on tau_prng (distributional, not
// bit-equal to numpy; documented deviation).
inline double uniform(int64_t state[3], double low, double high) {
    return low + (high - low) * static_cast<double>(tau_rand(state));
}

// Standard normal via Box-Muller on tau_rand (documented deviation from
// numpy's Gaussian stream).
inline double std_normal(int64_t state[3]) {
    double u1 = static_cast<double>(tau_rand(state));
    double u2 = static_cast<double>(tau_rand(state));
    if (u1 < 1e-12) u1 = 1e-12;
    return std::sqrt(-2.0 * std::log(u1)) *
           std::cos(2.0 * 3.14159265358979323846 * u2);
}

// graph[np.ix_(idx, idx)] for a canonical CSR matrix; idx must be ascending
// (it is: indices of component members in vertex order), so per-row column
// order is preserved.
SparseMatrix submatrix(const SparseMatrix& g, const std::vector<int32_t>& idx) {
    std::vector<int32_t> new_col(static_cast<std::size_t>(g.n_cols), -1);
    for (std::size_t t = 0; t < idx.size(); ++t)
        new_col[static_cast<std::size_t>(idx[t])] = static_cast<int32_t>(t);

    SparseMatrix out(static_cast<int32_t>(idx.size()),
                     static_cast<int32_t>(idx.size()));
    out.indptr.assign(1, 0);
    for (std::size_t t = 0; t < idx.size(); ++t) {
        const int32_t row = idx[t];
        for (int32_t p = g.indptr[row]; p < g.indptr[row + 1]; ++p) {
            const int32_t nc = new_col[static_cast<std::size_t>(g.indices[p])];
            if (nc >= 0) {
                out.indices.push_back(nc);
                out.data.push_back(g.data[p]);
            }
        }
        out.indptr.push_back(static_cast<int32_t>(out.indices.size()));
    }
    return out;
}

// Upstream fallback layout: gen.uniform(low=-10, high=10, size=(n, dim)).
std::vector<double> random_fallback_layout(int64_t n, int dim,
                                           int64_t rng_state[3]) {
    std::vector<double> out(static_cast<std::size_t>(n) * dim);
    for (auto& v : out) v = uniform(rng_state, -10.0, 10.0);
    return out;
}

// Upstream's warn(...) on eigensolver failure (shared by spectral_layout and
// tswspectral_layout; extracted as part of the tswspectral additions).
void warn_spectral_failure() {
    std::fprintf(stderr,
                 "Spectral initialisation failed! The eigenvector solver\n"
                 "failed. This is likely due to too small an eigengap. "
                 "Consider\nadding some noise or jitter to your data.\n\n"
                 "Falling back to random initialisation!\n");
}

// sqrt_deg = np.sqrt(graph.sum(axis=0)) and the normalized Laplacian
// L = I - D^{-1/2} A D^{-1/2} exactly as _spectral_layout builds them.
// (Extracted from normalized_laplacian_eigs so the tswspectral LOBPCG path
// shares the identical construction.)
void build_normalized_laplacian(const SparseMatrix& graph,
                                Eigen::SparseMatrix<double>& L,
                                std::vector<double>& sqrt_deg) {
    const int64_t n = graph.n_rows;

    // sqrt_deg = sqrt(graph.sum(axis=0))  (column sums)
    sqrt_deg.assign(static_cast<std::size_t>(n), 0.0);
    for (int32_t i = 0; i < graph.n_rows; ++i)
        for (int32_t p = graph.indptr[i]; p < graph.indptr[i + 1]; ++p)
            sqrt_deg[static_cast<std::size_t>(graph.indices[p])] +=
                static_cast<double>(graph.data[p]);
    for (auto& d : sqrt_deg) d = std::sqrt(d);

    // L = I - D^{-1/2} A D^{-1/2}
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(graph.nnz() + static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) trips.emplace_back(i, i, 1.0);
    for (int32_t i = 0; i < graph.n_rows; ++i)
        for (int32_t p = graph.indptr[i]; p < graph.indptr[i + 1]; ++p) {
            const int32_t j = graph.indices[p];
            trips.emplace_back(
                i, j,
                -static_cast<double>(graph.data[p]) /
                    (sqrt_deg[static_cast<std::size_t>(i)] *
                     sqrt_deg[static_cast<std::size_t>(j)]));
        }
    L.resize(n, n);
    L.setFromTriplets(trips.begin(), trips.end());
}

// _spectral_layout for the general case, returning float64 like upstream
// (n x dim row-major). Forward declaration for the multi-component recursion.
std::vector<double> spectral_layout_f64(const float* data, int64_t n_features,
                                        const SparseMatrix& graph, int dim,
                                        int64_t rng_state[3]);

}  // namespace

int32_t connected_components(const SparseMatrix& graph,
                             std::vector<int32_t>& labels) {
    // scipy.sparse.csgraph.connected_components defaults: directed=True with
    // connection='weak', i.e. edges are treated as undirected. Components are
    // labeled in order of the first-encountered (lowest-index) vertex.
    const int32_t n = graph.n_rows;
    const SparseMatrix gt = graph.transpose();
    labels.assign(static_cast<std::size_t>(n), -1);
    int32_t n_components = 0;
    std::vector<int32_t> stack;
    for (int32_t start = 0; start < n; ++start) {
        if (labels[start] != -1) continue;
        const int32_t label = n_components++;
        labels[start] = label;
        stack.assign(1, start);
        while (!stack.empty()) {
            const int32_t v = stack.back();
            stack.pop_back();
            for (const SparseMatrix* m : {&graph, &gt}) {
                for (int32_t p = m->indptr[v]; p < m->indptr[v + 1]; ++p) {
                    const int32_t w = m->indices[p];
                    if (labels[w] == -1) {
                        labels[w] = label;
                        stack.push_back(w);
                    }
                }
            }
        }
    }
    return n_components;
}

bool normalized_laplacian_eigs(const SparseMatrix& graph, int k,
                               std::vector<double>& eigenvalues,
                               std::vector<double>& eigenvectors,
                               int64_t maxiter) {
    const int64_t n = graph.n_rows;
    if (k >= n) {
        // Upstream: scipy warns "k >= N ... attempting eigh" then raises an
        // uncaught TypeError, so spectral init on such graphs is an error.
        throw std::invalid_argument(
            "normalized_laplacian_eigs: k must be < n (upstream scipy raises "
            "for k >= N)");
    }

    Eigen::SparseMatrix<double> L;
    std::vector<double> sqrt_deg;
    build_normalized_laplacian(graph, L, sqrt_deg);

    // Upstream eigsh selection: k smallest ("SM", equal to smallest algebraic
    // for the PSD Laplacian), ncv = max(2k + 1, sqrt(n)) (scipy clamps to n),
    // v0 = ones, maxiter = 5n. We solve in shift-invert mode with the small
    // negative sigma sklearn recommends (-1e-5) for robustness; the selected
    // eigenpairs and their (unit 2-norm) normalization match ARPACK's.
    const int64_t ncv64 = std::min<int64_t>(
        n, std::max<int64_t>(2 * static_cast<int64_t>(k) + 1,
                             static_cast<int64_t>(std::sqrt(
                                 static_cast<double>(n)))));
    const int ncv = static_cast<int>(ncv64);
    const int maxit =
        maxiter > 0
            ? static_cast<int>(std::min<int64_t>(maxiter, 1 << 30))
            : std::max<int>(1000,
                            static_cast<int>(std::min<int64_t>(5 * n, 1 << 30)));

    try {
        Spectra::SparseSymShiftSolve<double> op(L);
        Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(
            op, k, ncv, -1e-5);
        Eigen::VectorXd v0 = Eigen::VectorXd::Ones(n);
        eigs.init(v0.data());
        eigs.compute(Spectra::SortRule::LargestMagn, maxit, 1e-10,
                     Spectra::SortRule::SmallestAlge);
        if (eigs.info() != Spectra::CompInfo::Successful) return false;

        const Eigen::VectorXd evals = eigs.eigenvalues();
        const Eigen::MatrixXd evecs = eigs.eigenvectors();
        // order = np.argsort(eigenvalues) (ascending); Spectra already sorts
        // by SmallestAlge but we replicate the explicit argsort regardless.
        std::vector<int> order(static_cast<std::size_t>(k));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return evals[a] < evals[b]; });

        eigenvalues.resize(static_cast<std::size_t>(k));
        eigenvectors.assign(static_cast<std::size_t>(n) * k, 0.0);
        for (int c = 0; c < k; ++c) {
            eigenvalues[static_cast<std::size_t>(c)] = evals[order[c]];
            for (int64_t i = 0; i < n; ++i)
                eigenvectors[static_cast<std::size_t>(i) * k + c] =
                    evecs(i, order[c]);
        }
        return true;
    } catch (const std::exception&) {
        // e.g. numerically-singular factorization; upstream catches
        // ArpackError and falls back to random.
        return false;
    }
}

std::vector<double> component_layout(const float* data, int64_t n_samples,
                                     int64_t n_features, int32_t n_components,
                                     const std::vector<int32_t>& component_labels,
                                     int dim, int64_t rng_state[3]) {
    const std::size_t nc = static_cast<std::size_t>(n_components);
    if (data == nullptr) {
        // "We don't have data to work with; just guess":
        // np.random.random(size=(n_components, dim)) * 10.0
        std::vector<double> out(nc * static_cast<std::size_t>(dim));
        for (auto& v : out) v = uniform(rng_state, 0.0, 1.0) * 10.0;
        return out;
    }
    if (dim + 1 > n_components)
        throw std::invalid_argument(
            "component_layout: need n_components > dim (upstream only calls "
            "this with n_components > 2 * dim)");

    // component_centroids[label] = data[component_labels == label].mean(axis=0)
    Eigen::MatrixXd centroids =
        Eigen::MatrixXd::Zero(n_components, n_features);
    std::vector<int64_t> counts(nc, 0);
    for (int64_t i = 0; i < n_samples; ++i) {
        const int32_t label = component_labels[static_cast<std::size_t>(i)];
        counts[static_cast<std::size_t>(label)] += 1;
        for (int64_t f = 0; f < n_features; ++f)
            centroids(label, f) += static_cast<double>(data[i * n_features + f]);
    }
    for (int32_t c = 0; c < n_components; ++c)
        centroids.row(c) /= static_cast<double>(counts[static_cast<std::size_t>(c)]);

    // affinity_matrix = np.exp(-(pairwise_distances(centroids) ** 2))
    // sklearn's csgraph laplacian zeroes the diagonal of the affinity before
    // computing degrees, so build it with a zero diagonal directly.
    Eigen::MatrixXd affinity(n_components, n_components);
    for (int32_t i = 0; i < n_components; ++i) {
        affinity(i, i) = 0.0;
        for (int32_t j = i + 1; j < n_components; ++j) {
            const double d2 = (centroids.row(i) - centroids.row(j)).squaredNorm();
            const double a = std::exp(-d2);
            affinity(i, j) = a;
            affinity(j, i) = a;
        }
    }

    // sklearn SpectralEmbedding(affinity="precomputed") == spectral_embedding
    // with norm_laplacian=True, drop_first=True:
    //   L = I - D^{-1/2} A D^{-1/2}; k = dim + 1 smallest eigenvectors x;
    //   embedding rows = x / dd, deterministic sign flip, drop first.
    // --- tests-port fix (real bug found by test_umap_ops.py::
    // test_multi_component_layout): scipy csgraph_laplacian(normed=True)
    // guards isolated (zero-degree) vertices -- w[w == 0] = 1 and the
    // Laplacian diagonal is 1 - isolated_node_mask. Without the guard,
    // components whose mutual affinities exp(-d^2) underflow to 0 (very
    // distant blobs) produced a NaN Laplacian and a solver failure, where
    // sklearn/scipy succeed. dd (the returned diag) keeps the guarded values,
    // exactly like scipy's return_diag=True.
    Eigen::VectorXd dd = affinity.rowwise().sum().cwiseSqrt();
    std::vector<bool> isolated(static_cast<std::size_t>(n_components), false);
    for (int32_t i = 0; i < n_components; ++i)
        if (dd(i) == 0.0) {
            isolated[static_cast<std::size_t>(i)] = true;
            dd(i) = 1.0;
        }
    Eigen::MatrixXd L(n_components, n_components);
    for (int32_t i = 0; i < n_components; ++i)
        for (int32_t j = 0; j < n_components; ++j)
            L(i, j) = (i == j)
                          ? (isolated[static_cast<std::size_t>(i)] ? 0.0 : 1.0)
                          : -affinity(i, j) / (dd(i) * dd(j));
    // ------------------------------------------------------------------------
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("component_layout: dense eigensolver failed");

    const int k = dim + 1;
    // rows of `emb` are eigenvectors (ascending eigenvalue), scaled by 1/dd.
    Eigen::MatrixXd emb(k, n_components);
    for (int t = 0; t < k; ++t)
        emb.row(t) =
            (solver.eigenvectors().col(t).array() / dd.array()).transpose();
    // _deterministic_vector_sign_flip: flip each row so its max-|.| entry > 0.
    for (int t = 0; t < k; ++t) {
        int arg = 0;
        for (int c = 1; c < n_components; ++c)
            if (std::abs(emb(t, c)) > std::abs(emb(t, arg))) arg = c;
        const double s =
            (emb(t, arg) > 0.0) ? 1.0 : ((emb(t, arg) < 0.0) ? -1.0 : 0.0);
        emb.row(t) *= s;
    }

    // drop_first, transpose to (n_components x dim), then /= max()
    std::vector<double> out(nc * static_cast<std::size_t>(dim));
    double max_val = emb(1, 0);
    for (int t = 1; t < k; ++t)
        for (int32_t c = 0; c < n_components; ++c)
            max_val = std::max(max_val, emb(t, c));
    for (int32_t c = 0; c < n_components; ++c)
        for (int j = 0; j < dim; ++j)
            out[static_cast<std::size_t>(c) * dim + j] = emb(j + 1, c) / max_val;
    return out;
}

std::vector<float> multi_component_layout(
    const float* data, int64_t n_features, const SparseMatrix& graph,
    int32_t n_components, const std::vector<int32_t>& component_labels, int dim,
    int64_t rng_state[3]) {
    const int64_t n = graph.n_rows;
    std::vector<float> result(static_cast<std::size_t>(n) * dim, 0.0f);

    // Meta-embedding of the component centers.
    std::vector<double> meta;
    if (n_components > 2 * dim) {
        meta = component_layout(data, n, n_features, n_components,
                                component_labels, dim, rng_state);
    } else {
        // k = ceil(nc / 2); base = [eye(k) | 0]; meta = [base; -base][:nc]
        const int32_t k = (n_components + 1) / 2;
        meta.assign(static_cast<std::size_t>(n_components) * dim, 0.0);
        for (int32_t c = 0; c < n_components; ++c) {
            if (c < k)
                meta[static_cast<std::size_t>(c) * dim + c] = 1.0;
            else
                meta[static_cast<std::size_t>(c) * dim + (c - k)] = -1.0;
        }
    }

    for (int32_t label = 0; label < n_components; ++label) {
        std::vector<int32_t> idx;
        for (int64_t i = 0; i < n; ++i)
            if (component_labels[static_cast<std::size_t>(i)] == label)
                idx.push_back(static_cast<int32_t>(i));
        const int64_t m = static_cast<int64_t>(idx.size());

        // data_range = distances[distances > 0].min() / 2 over euclidean
        // distances from this component's meta position to all meta positions.
        double min_pos = -1.0;
        for (int32_t c = 0; c < n_components; ++c) {
            double d2 = 0.0;
            for (int j = 0; j < dim; ++j) {
                const double diff =
                    meta[static_cast<std::size_t>(label) * dim + j] -
                    meta[static_cast<std::size_t>(c) * dim + j];
                d2 += diff * diff;
            }
            const double d = std::sqrt(d2);
            if (d > 0.0 && (min_pos < 0.0 || d < min_pos)) min_pos = d;
        }
        if (min_pos < 0.0)
            throw std::runtime_error(
                "multi_component_layout: degenerate meta embedding (no "
                "positive distances between component centers)");
        const double data_range = min_pos / 2.0;

        if (m < 2 * dim || m <= dim + 1) {
            // Tiny component: random placement around the meta position.
            for (int64_t t = 0; t < m; ++t)
                for (int j = 0; j < dim; ++j)
                    result[static_cast<std::size_t>(idx[t]) * dim + j] =
                        static_cast<float>(
                            uniform(rng_state, -data_range, data_range) +
                            meta[static_cast<std::size_t>(label) * dim + j]);
        } else {
            const SparseMatrix component_graph = submatrix(graph, idx);
            std::vector<double> comp_emb = spectral_layout_f64(
                nullptr, 0, component_graph, dim, rng_state);
            double max_abs = 0.0;
            for (const double v : comp_emb) max_abs = std::max(max_abs, std::abs(v));
            const double expansion = data_range / max_abs;
            for (int64_t t = 0; t < m; ++t)
                for (int j = 0; j < dim; ++j)
                    result[static_cast<std::size_t>(idx[t]) * dim + j] =
                        static_cast<float>(
                            comp_emb[static_cast<std::size_t>(t) * dim + j] *
                                expansion +
                            meta[static_cast<std::size_t>(label) * dim + j]);
        }
    }
    return result;
}

namespace {

std::vector<double> spectral_layout_f64(const float* data, int64_t n_features,
                                        const SparseMatrix& graph, int dim,
                                        int64_t rng_state[3]) {
    std::vector<int32_t> labels;
    const int32_t n_comp = connected_components(graph, labels);
    if (n_comp > 1) {
        const std::vector<float> multi = multi_component_layout(
            data, n_features, graph, n_comp, labels, dim, rng_state);
        return std::vector<double>(multi.begin(), multi.end());
    }

    const int k = dim + 1;
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;  // n x k row-major, ascending eigenvalue
    if (!normalized_laplacian_eigs(graph, k, eigenvalues, eigenvectors)) {
        warn_spectral_failure();
        return random_fallback_layout(graph.n_rows, dim, rng_state);
    }
    // return eigenvectors[:, order] with order = argsort(eigenvalues)[1:k]
    // (the argsort is already applied inside normalized_laplacian_eigs).
    const int64_t n = graph.n_rows;
    std::vector<double> out(static_cast<std::size_t>(n) * dim);
    for (int64_t i = 0; i < n; ++i)
        for (int j = 0; j < dim; ++j)
            out[static_cast<std::size_t>(i) * dim + j] =
                eigenvectors[static_cast<std::size_t>(i) * k + (j + 1)];
    return out;
}

}  // namespace

std::vector<float> spectral_layout(const float* data, int64_t n_features,
                                   const SparseMatrix& graph, int dim,
                                   int64_t rng_state[3]) {
    const std::vector<double> emb =
        spectral_layout_f64(data, n_features, graph, dim, rng_state);
    return std::vector<float>(emb.begin(), emb.end());
}

std::vector<float> noisy_scale_coords(std::vector<float> coords,
                                      int64_t rng_state[3], float max_coord,
                                      float noise) {
    float max_abs = 0.0f;
    for (const float v : coords) max_abs = std::max(max_abs, std::abs(v));
    const double expansion =
        static_cast<double>(max_coord) / static_cast<double>(max_abs);
    for (auto& v : coords) {
        v = static_cast<float>(static_cast<double>(v) * expansion);
        v += static_cast<float>(std_normal(rng_state) *
                                static_cast<double>(noise));
    }
    return coords;
}

std::vector<float> initialize_embedding(const float* data, int64_t n_features,
                                        const SparseMatrix& graph, int dim,
                                        EmbeddingInit init,
                                        int64_t rng_state[3]) {
    switch (init) {
        case EmbeddingInit::Random: {
            // random_state.uniform(low=-10, high=10, size=(n, dim)).astype(f32)
            std::vector<float> out(static_cast<std::size_t>(graph.n_rows) * dim);
            for (auto& v : out)
                v = static_cast<float>(uniform(rng_state, -10.0, 10.0));
            return out;
        }
        case EmbeddingInit::Spectral: {
            std::vector<float> emb =
                spectral_layout(data, n_features, graph, dim, rng_state);
            // "We add a little noise to avoid local minima for optimization"
            return noisy_scale_coords(std::move(emb), rng_state,
                                      /*max_coord=*/10.0f, /*noise=*/0.0001f);
        }
        // --- tswspectral additions ---
        case EmbeddingInit::TSWSpectral: {
            // umap_.py:1172: identical to the "spectral" branch but through
            // tswspectral_layout (method/tol/maxiter at their defaults).
            std::vector<float> emb =
                tswspectral_layout(data, n_features, graph, dim, rng_state);
            return noisy_scale_coords(std::move(emb), rng_state,
                                      /*max_coord=*/10.0f, /*noise=*/0.0001f);
        }
        case EmbeddingInit::PCA:
        default:
            // TODO(port): PCA (dense input) / TruncatedSVD (sparse input) of
            // the source data followed by noisy_scale_coords, as in
            // umap_.simplicial_set_embedding.
            throw std::runtime_error(
                "initialize_embedding: init='pca' is not implemented yet");
    }
}

// --- tswspectral additions --------------------------------------------------
// Port of spectral.tswspectral_layout (_spectral_layout(init="tsvd")) plus a
// scipy-semantics LOBPCG on Eigen sparse. See the header for the port notes.

namespace {

// Thin orthonormal basis of the columns of M (HouseholderQR).
Eigen::MatrixXd thin_q(const Eigen::MatrixXd& M) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    return qr.householderQ() * Eigen::MatrixXd::Identity(M.rows(), M.cols());
}

// Orthonormal basis of span(M) after projecting out the (orthonormal) blocks
// O1 / O2; numerically-dependent directions are dropped via an SVD rank cut
// (the stabilized equivalent of scipy lobpcg's B-orthonormalization with its
// restart-on-Cholesky-breakdown handling).
Eigen::MatrixXd ortho_against(Eigen::MatrixXd M, const Eigen::MatrixXd* O1,
                              const Eigen::MatrixXd* O2 = nullptr) {
    if (M.cols() == 0) return M;
    for (int pass = 0; pass < 2; ++pass) {  // twice-is-enough reorthogonalization
        if (O1 != nullptr && O1->cols() > 0)
            M -= (*O1) * (O1->transpose() * M);
        if (O2 != nullptr && O2->cols() > 0)
            M -= (*O2) * (O2->transpose() * M);
    }
    Eigen::BDCSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU);
    const Eigen::VectorXd& sv = svd.singularValues();
    const double thresh = sv.size() > 0 ? sv(0) * 1e-12 : 0.0;
    int rank = 0;
    while (rank < sv.size() && sv(rank) > thresh && sv(rank) > 0.0) ++rank;
    return svd.matrixU().leftCols(rank);
}

// scipy.sparse.linalg.lobpcg(A, X, largest=False, tol=tol, maxiter=maxiter)
// semantics (no B / M / Y constraints): block Rayleigh-Ritz over
// span[X, R, P] with scipy's monotone soft-locking active mask and its
// convergence criterion (all residual 2-norms <= tol; strictly-greater stays
// active). Returns true iff the tolerance was reached -- scipy otherwise
// warns "...not reaching the requested tolerance", which upstream escalates
// to the random fallback. X / lam hold the final block either way,
// eigenvalues ascending. Mirrors scipy's small-problem branch: n < 5 * m is
// solved densely and always counts as converged (that warning is not
// escalated upstream).
bool lobpcg_smallest(const Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& X,
                     Eigen::VectorXd& lam, double tol, int64_t maxiter) {
    const int64_t n = A.rows();
    const int m = static_cast<int>(X.cols());
    if (n < 5 * static_cast<int64_t>(m)) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es{Eigen::MatrixXd(A)};
        X = es.eigenvectors().leftCols(m);
        lam = es.eigenvalues().head(m);
        return true;
    }

    X = thin_q(X);
    Eigen::MatrixXd AX = A * X;
    {
        // Initial Rayleigh-Ritz on span(X), like scipy's pre-loop step.
        Eigen::MatrixXd G = X.transpose() * AX;
        G = 0.5 * (G + G.transpose().eval());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
        X = X * es.eigenvectors();
        AX = AX * es.eigenvectors();
        lam = es.eigenvalues();
    }

    Eigen::MatrixXd P(n, 0);
    std::vector<bool> still_active(static_cast<std::size_t>(m), true);
    bool converged = false;
    // scipy runs maxiter + 1 residual-check/update passes
    // (iterationNumber = 0..maxiter), then re-checks residuals for the
    // final "not reaching the requested tolerance" verdict.
    for (int64_t it = 0; it <= maxiter; ++it) {
        const Eigen::MatrixXd R = AX - X * lam.asDiagonal();
        std::vector<int> active;
        for (int j = 0; j < m; ++j) {
            if (R.col(j).norm() <= tol)
                still_active[static_cast<std::size_t>(j)] = false;
            if (still_active[static_cast<std::size_t>(j)]) active.push_back(j);
        }
        if (active.empty()) {
            converged = true;
            break;
        }

        const Eigen::Index na = static_cast<Eigen::Index>(active.size());
        Eigen::MatrixXd W(n, na);
        Eigen::MatrixXd Pa(n, P.cols() > 0 ? na : 0);
        for (Eigen::Index t = 0; t < na; ++t) {
            W.col(t) = R.col(active[static_cast<std::size_t>(t)]);
            if (P.cols() > 0)
                Pa.col(t) = P.col(active[static_cast<std::size_t>(t)]);
        }
        W = ortho_against(std::move(W), &X);
        Pa = ortho_against(std::move(Pa), &X, &W);
        if (W.cols() == 0 && Pa.cols() == 0) break;  // breakdown / stagnation

        const Eigen::Index s = m + W.cols() + Pa.cols();
        Eigen::MatrixXd S(n, s);
        S.leftCols(m) = X;
        S.middleCols(m, W.cols()) = W;
        if (Pa.cols() > 0) S.rightCols(Pa.cols()) = Pa;
        const Eigen::MatrixXd AS = A * S;
        Eigen::MatrixXd G = S.transpose() * AS;
        G = 0.5 * (G + G.transpose().eval());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
        const Eigen::MatrixXd C = es.eigenvectors().leftCols(m);  // largest=False
        lam = es.eigenvalues().head(m);
        // New P is the W/P part of the Ritz vectors; new X = X * Cx + P.
        P = S.rightCols(s - m) * C.bottomRows(s - m);
        X = X * C.topRows(m) + P;
        AX = AS * C;
    }
    if (!converged) {
        const Eigen::MatrixXd R = AX - X * lam.asDiagonal();
        double worst = 0.0;
        for (int j = 0; j < m; ++j) worst = std::max(worst, R.col(j).norm());
        converged = worst <= tol;
    }
    return converged;
}

// sklearn TruncatedSVD(n_components=k, random_state=...).fit_transform(L):
// randomized SVD (Halko et al.) with the sklearn defaults n_oversamples=10
// and n_iter=5. Deviations (subspace-equivalent, documented in the header):
// power iterations are thin-QR normalized (sklearn: LU), the Gaussian start
// block comes from tau_prng in numpy's row-major fill order, and no
// svd_flip sign convention is applied (X[:, 0] is overwritten by the caller
// and LOBPCG orthonormalizes the block anyway). Returns U[:, :k] * s[:k]
// exactly like fit_transform.
Eigen::MatrixXd tsvd_warm_start(const Eigen::SparseMatrix<double>& L, int k,
                                int64_t rng_state[3]) {
    const int64_t n = L.rows();
    const int size = static_cast<int>(
        std::min<int64_t>(static_cast<int64_t>(k) + 10, n));
    Eigen::MatrixXd Q(n, size);
    for (int64_t i = 0; i < n; ++i)
        for (int j = 0; j < size; ++j) Q(i, j) = std_normal(rng_state);
    const int n_iter = 5;  // TruncatedSVD default
    for (int i = 0; i < n_iter; ++i) {
        Q = thin_q(L * Q);
        Q = thin_q(L.transpose() * Q);  // L is symmetric; kept for structure
    }
    Q = thin_q(L * Q);
    // B = Q^T L (size x n); L is symmetric so B = (L Q)^T.
    const Eigen::MatrixXd B = (L * Q).transpose();
    Eigen::BDCSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU);
    const Eigen::MatrixXd U = Q * svd.matrixU();  // n x size, sing. values desc
    Eigen::MatrixXd X(n, k);
    for (int j = 0; j < k; ++j) X.col(j) = U.col(j) * svd.singularValues()(j);
    return X;
}

}  // namespace

std::vector<float> tswspectral_layout(const float* data, int64_t n_features,
                                      const SparseMatrix& graph, int dim,
                                      int64_t rng_state[3],
                                      const std::string& method, double tol,
                                      int64_t maxiter,
                                      bool* used_random_fallback) {
    if (used_random_fallback != nullptr) *used_random_fallback = false;

    // The multi-component path runs before anything tsvd-related, exactly as
    // in _spectral_layout (shared with spectral_layout, so given the same
    // rng_state both produce bit-identical multi-component layouts).
    std::vector<int32_t> labels;
    const int32_t n_comp = connected_components(graph, labels);
    if (n_comp > 1)
        return multi_component_layout(data, n_features, graph, n_comp, labels,
                                      dim, rng_state);

    const int64_t n = graph.n_rows;
    const int k = dim + 1;
    if (k >= n) {
        // Same guard as normalized_laplacian_eigs (upstream errors for tiny
        // connected graphs on the eigsh path; see header).
        throw std::invalid_argument(
            "normalized_laplacian_eigs: k must be < n (upstream scipy raises "
            "for k >= N)");
    }

    // if not method: method = "eigsh" if L.shape[0] < 2000000 else "lobpcg"
    std::string resolved = method;
    if (resolved.empty()) resolved = n < 2000000 ? "eigsh" : "lobpcg";

    // init == "tsvd": TruncatedSVD warm start with the exact first
    // eigenvector patched in (X[:, 0] = sqrt_deg / ||sqrt_deg||). Upstream
    // computes this for BOTH methods (the eigsh branch then ignores it), so
    // we do too -- the rng draws are consumed either way.
    Eigen::SparseMatrix<double> L;
    std::vector<double> sqrt_deg;
    build_normalized_laplacian(graph, L, sqrt_deg);
    Eigen::MatrixXd X = tsvd_warm_start(L, k, rng_state);
    {
        double norm = 0.0;
        for (const double d : sqrt_deg) norm += d * d;
        norm = std::sqrt(norm);
        for (int64_t i = 0; i < n; ++i)
            X(i, 0) = sqrt_deg[static_cast<std::size_t>(i)] / norm;
    }

    bool ok = false;
    std::vector<double> evals;
    std::vector<double> evecs;  // n x k row-major, ascending eigenvalue
    if (resolved == "eigsh") {
        // Upstream: eigsh(L, k, which="SM", ncv=max(2k+1, sqrt(n)),
        // tol=tol or 1e-4, v0=np.ones(n), maxiter=maxiter or 5n) -- the warm
        // start is DISCARDED on this branch. Spectra solves tighter than
        // tol=1e-4 (documented deviation); maxiter threads through.
        ok = normalized_laplacian_eigs(graph, k, evals, evecs, maxiter);
    } else if (resolved == "lobpcg") {
        // Upstream: lobpcg(L, X, largest=False, tol=tol or 1e-4,
        // maxiter=maxiter or 5n) with the convergence UserWarning escalated.
        Eigen::VectorXd lam;
        const double eff_tol = tol != 0.0 ? tol : 1e-4;
        const int64_t eff_maxiter = maxiter != 0 ? maxiter : 5 * n;
        if (lobpcg_smallest(L, X, lam, eff_tol, eff_maxiter)) {
            ok = true;
            // order = np.argsort(eigenvalues) ascending (replicated even
            // though the Rayleigh-Ritz values already come out ascending).
            std::vector<int> order(static_cast<std::size_t>(k));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return lam[a] < lam[b]; });
            evals.resize(static_cast<std::size_t>(k));
            evecs.assign(static_cast<std::size_t>(n) * k, 0.0);
            for (int c = 0; c < k; ++c) {
                evals[static_cast<std::size_t>(c)] = lam[order[c]];
                for (int64_t i = 0; i < n; ++i)
                    evecs[static_cast<std::size_t>(i) * k + c] =
                        X(i, order[c]);
            }
        }
    } else {
        // Upstream raises ValueError from inside the try (uncaught by the
        // ArpackError/UserWarning handler).
        throw std::invalid_argument(
            "Method should either be None, 'eigsh' or 'lobpcg'");
    }

    if (!ok) {
        // except (ArpackError, UserWarning): warn + random fallback. The
        // escalated lobpcg "not reaching the requested tolerance" warning
        // lands here as a non-converged solve.
        warn_spectral_failure();
        if (used_random_fallback != nullptr) *used_random_fallback = true;
        const std::vector<double> f64 =
            random_fallback_layout(n, dim, rng_state);
        return std::vector<float>(f64.begin(), f64.end());
    }

    // return eigenvectors[:, order] with order = argsort(eigenvalues)[1:k]
    std::vector<float> out(static_cast<std::size_t>(n) * dim);
    for (int64_t i = 0; i < n; ++i)
        for (int j = 0; j < dim; ++j)
            out[static_cast<std::size_t>(i) * dim + j] = static_cast<float>(
                evecs[static_cast<std::size_t>(i) * k + (j + 1)]);
    return out;
}

bool normalized_laplacian_lobpcg(const SparseMatrix& graph, const double* X0,
                                 int m, double tol, int64_t maxiter,
                                 std::vector<double>& eigenvalues,
                                 std::vector<double>& eigenvectors) {
    const int64_t n = graph.n_rows;
    Eigen::SparseMatrix<double> L;
    std::vector<double> sqrt_deg;
    build_normalized_laplacian(graph, L, sqrt_deg);
    Eigen::MatrixXd X(n, m);
    for (int64_t i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            X(i, j) = X0[static_cast<std::size_t>(i) * m + j];
    Eigen::VectorXd lam;
    const bool converged = lobpcg_smallest(L, X, lam, tol, maxiter);
    // Ascending argsort, matching what tswspectral_layout applies.
    std::vector<int> order(static_cast<std::size_t>(m));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return lam[a] < lam[b]; });
    eigenvalues.resize(static_cast<std::size_t>(m));
    eigenvectors.assign(static_cast<std::size_t>(n) * m, 0.0);
    for (int c = 0; c < m; ++c) {
        eigenvalues[static_cast<std::size_t>(c)] = lam[order[c]];
        for (int64_t i = 0; i < n; ++i)
            eigenvectors[static_cast<std::size_t>(i) * m + c] = X(i, order[c]);
    }
    return converged;
}
// ----------------------------------------------------------------------------

}  // namespace umap
