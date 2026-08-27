#include "umap/fuzzy_simplicial_set.hpp"

#include <cmath>
#include <limits>

namespace umap {

namespace {
constexpr float NPY_FLOATMAX = std::numeric_limits<float>::max();
}

SmoothKnnResult smooth_knn_dist(const float* distances, int n_samples,
                                int n_neighbors, double k, int n_iter,
                                double local_connectivity, double bandwidth) {
    const double target = std::log2(k) * bandwidth;
    SmoothKnnResult res;
    res.rhos.assign(static_cast<std::size_t>(n_samples), 0.0f);
    res.sigmas.assign(static_cast<std::size_t>(n_samples), 0.0f);

    // np.mean over the full (float32) matrix. numba's np.mean on a float32
    // array accumulates and divides in float32, sequentially.
    float total = 0.0f;
    const std::size_t n_total =
        static_cast<std::size_t>(n_samples) * static_cast<std::size_t>(n_neighbors);
    for (std::size_t i = 0; i < n_total; ++i) total += distances[i];
    const float mean_distances = total / static_cast<float>(n_total);

    std::vector<float> non_zero_dists;
    non_zero_dists.reserve(static_cast<std::size_t>(n_neighbors));

    for (int i = 0; i < n_samples; ++i) {
        // The reference declares lo/hi/mid/psum float32 in the numba `locals`
        // dict, but with parallel=True the prange body is lifted into a
        // parfor gufunc where `locals` does NOT apply: the actual compiled
        // code runs the whole bisection in float64 (verified against the
        // reference row by row; only `d` stays float32 since the arrays are
        // float32).
        double lo = 0.0;
        double hi = static_cast<double>(NPY_FLOATMAX);
        double mid = 1.0;

        const float* ith_distances =
            distances + static_cast<std::size_t>(i) * n_neighbors;
        non_zero_dists.clear();
        for (int j = 0; j < n_neighbors; ++j) {
            if (ith_distances[j] > 0.0f) non_zero_dists.push_back(ith_distances[j]);
        }

        if (static_cast<double>(non_zero_dists.size()) >= local_connectivity) {
            int index = static_cast<int>(std::floor(local_connectivity));
            double interpolation = local_connectivity - index;
            if (index > 0) {
                res.rhos[i] = non_zero_dists[static_cast<std::size_t>(index) - 1];
                if (interpolation > SMOOTH_K_TOLERANCE) {
                    // rho[i] += interpolation * (nz[index] - nz[index-1]):
                    // f4 + f8 product evaluated in f8, single-rounded on store.
                    res.rhos[i] = static_cast<float>(
                        static_cast<double>(res.rhos[i]) +
                        interpolation *
                            (non_zero_dists[static_cast<std::size_t>(index)] -
                             non_zero_dists[static_cast<std::size_t>(index) - 1]));
                }
            } else {
                res.rhos[i] =
                    static_cast<float>(interpolation * non_zero_dists[0]);
            }
        } else if (!non_zero_dists.empty()) {
            float mx = non_zero_dists[0];
            for (float v : non_zero_dists) mx = std::max(mx, v);
            res.rhos[i] = mx;
        }

        for (int n = 0; n < n_iter; ++n) {
            double psum = 0.0;
            for (int j = 1; j < n_neighbors; ++j) {
                const float d = ith_distances[j] - res.rhos[i];  // f4 - f4
                if (d > 0.0f) {
                    psum += std::exp(-(static_cast<double>(d) / mid));
                } else {
                    psum += 1.0;
                }
            }

            if (std::fabs(psum - target) < SMOOTH_K_TOLERANCE) {
                break;
            }

            if (psum > target) {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if (hi >= static_cast<double>(NPY_FLOATMAX)) {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
        }

        res.sigmas[i] = static_cast<float>(mid);

        // np.mean(ith_distances): float32 accumulation/division under numba.
        if (res.rhos[i] > 0.0f) {
            float row_total = 0.0f;
            for (int j = 0; j < n_neighbors; ++j) row_total += ith_distances[j];
            const float mean_ith_distances =
                row_total / static_cast<float>(n_neighbors);
            const double clamp =
                MIN_K_DIST_SCALE * static_cast<double>(mean_ith_distances);
            if (static_cast<double>(res.sigmas[i]) < clamp) {
                res.sigmas[i] = static_cast<float>(clamp);
            }
        } else {
            const double clamp =
                MIN_K_DIST_SCALE * static_cast<double>(mean_distances);
            if (static_cast<double>(res.sigmas[i]) < clamp) {
                res.sigmas[i] = static_cast<float>(clamp);
            }
        }
    }

    return res;
}

MembershipStrengths compute_membership_strengths(
    const int32_t* knn_indices, const float* knn_dists, const float* sigmas,
    const float* rhos, int n_samples, int n_neighbors, bool return_dists,
    bool bipartite) {
    const std::size_t size =
        static_cast<std::size_t>(n_samples) * static_cast<std::size_t>(n_neighbors);

    MembershipStrengths out;
    out.rows.assign(size, 0);
    out.cols.assign(size, 0);
    out.vals.assign(size, 0.0f);
    if (return_dists) out.dists.assign(size, 0.0f);

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_neighbors; ++j) {
            const std::size_t pos =
                static_cast<std::size_t>(i) * n_neighbors + static_cast<std::size_t>(j);
            if (knn_indices[pos] == -1) {
                continue;  // We didn't get the full knn for i
            }
            float val;
            // If applied to an adjacency matrix points shouldn't be similar to
            // themselves. If bipartite, row and column index sets differ.
            if (!bipartite && knn_indices[pos] == i) {
                val = 0.0f;
            } else if (knn_dists[pos] - rhos[i] <= 0.0f || sigmas[i] == 0.0f) {
                val = 1.0f;
            } else {
                val = std::exp(-((knn_dists[pos] - rhos[i]) / sigmas[i]));
            }

            out.rows[pos] = i;
            out.cols[pos] = knn_indices[pos];
            out.vals[pos] = val;
            if (return_dists) out.dists[pos] = knn_dists[pos];
        }
    }

    return out;
}

FuzzySimplicialSetResult fuzzy_simplicial_set(
    int n_samples, const int32_t* knn_indices, const float* knn_dists,
    int n_neighbors, double set_op_mix_ratio, double local_connectivity,
    bool apply_set_operations, bool return_dists) {
    FuzzySimplicialSetResult out;

    SmoothKnnResult smooth = smooth_knn_dist(
        knn_dists, n_samples, n_neighbors, static_cast<double>(n_neighbors),
        /*n_iter=*/64, local_connectivity);

    MembershipStrengths ms = compute_membership_strengths(
        knn_indices, knn_dists, smooth.sigmas.data(), smooth.rhos.data(),
        n_samples, n_neighbors, return_dists);

    SparseMatrix result =
        SparseMatrix::from_coo(n_samples, n_samples, ms.rows, ms.cols, ms.vals);
    result.eliminate_zeros();

    if (apply_set_operations) {
        SparseMatrix transpose = result.transpose();
        SparseMatrix prod_matrix = result.multiply(transpose);

        if (set_op_mix_ratio == 1.0) {
            // Pure fuzzy union: A + A^T - A o A^T
            result = result.add(transpose).subtract(prod_matrix);
        } else if (set_op_mix_ratio == 0.0) {
            // Pure fuzzy intersection.
            result = prod_matrix;
        } else {
            const float ratio = static_cast<float>(set_op_mix_ratio);
            const float one_minus_ratio = static_cast<float>(1.0 - set_op_mix_ratio);
            result = result.add(transpose)
                         .subtract(prod_matrix)
                         .scaled(ratio)
                         .add(prod_matrix.scaled(one_minus_ratio));
        }
    }

    result.eliminate_zeros();

    out.graph = std::move(result);
    out.sigmas = std::move(smooth.sigmas);
    out.rhos = std::move(smooth.rhos);

    if (return_dists) {
        SparseMatrix dmat =
            SparseMatrix::from_coo(n_samples, n_samples, ms.rows, ms.cols, ms.dists);
        out.dists = dmat.maximum(dmat.transpose());
        out.dists.eliminate_zeros();
        out.has_dists = true;
    }

    return out;
}

}  // namespace umap
