// Faithful C++ port of the scipy.spatial.Delaunay surface that
// umap_.py::inverse_transform consumes (scipy 1.18.1 _qhull.pyx, BSD 3
// clause), backed by the exact qhull sources scipy vendors (qhull_r 8.0.2 /
// 2020.2 from scipy v1.18.1 subprojects/qhull_r, vendored in
// third_party/qhull_r).
//
// Replicated behaviour, matching Delaunay(points, incremental=True,
// qhull_options="QJ") -- the only construction inverse_transform performs:
//   * option processing: option_set = {"QJ"}; required "Qt" is dropped when
//     "QJ" is present (QJ always produces simplicial output) and the
//     incremental "Q11" is only added when "Qt" survives -- so the command
//     handed to qhull is exactly b"qhull d QJ";
//   * _Qhull.__init__ checks: at least one point, ndim >= 2, no NaNs;
//   * Delaunay._update: qh_triangulate, paraboloid scale/shift (1.0 / 0.0
//     when Qbb is absent, as here), get_simplex_facet_array's facet_list
//     walk (upperdelaunay filter, non-simplicial facet error, the
//     counter-clockwise first/second vertex swap for 2-D triangulations when
//     facet->toporient == qh_ORIENTclock), equations = [normal, offset],
//     min_bound / max_bound;
//   * transform (barycentric): _get_barycentric_transforms with
//     eps = np.finfo(double).eps, LAPACK dgetrf/dgecon/dgetrs on the T
//     matrix, NaN rows for degenerate simplices. The LU factorisation and
//     solves replicate LAPACK's unblocked dgetf2/dgetrs arithmetic
//     (identical to the reference recursive dgetrf2 for the n <= 3 systems
//     n_components <= 4 produces). Documented deviation: the reciprocal
//     condition number is computed exactly (1 / (norm1(T) * norm1(T^-1)))
//     instead of via dgecon's dlacn2 estimator; both sides of the
//     1000 * eps degeneracy threshold agree except for near-singular
//     simplices, which "QJ" joggling avoids;
//   * find_simplex(xi, bruteforce=False, tol=None): eps = 100 * machine
//     epsilon, eps_broad = sqrt(eps), the qh_findbestfacet-style planar
//     walk on the lifted point, the directed barycentric walk (cycle cap
//     1 + nsimplex // 4) with brute-force fallback, and the *shared*
//     `start` state threaded across consecutive query points.
//
// Points are float64 (scipy converts with np.ascontiguousarray(..., double)).
// Errors raise std::runtime_error carrying qhull's message stream (scipy
// raises QhullError).
#pragma once

#include <cstdint>
#include <vector>

namespace umap {

struct Delaunay {
    int ndim = 0;
    int64_t npoints = 0;
    int64_t nsimplex = 0;
    std::vector<double> points;      // npoints x ndim, row-major
    std::vector<int32_t> simplices;  // nsimplex x (ndim+1)
    std::vector<int32_t> neighbors;  // nsimplex x (ndim+1); -1 = no neighbor
    std::vector<double> equations;   // nsimplex x (ndim+2): [normal, offset]
    double paraboloid_scale = 1.0;
    double paraboloid_shift = 0.0;
    std::vector<double> min_bound;   // ndim
    std::vector<double> max_bound;   // ndim
    // Barycentric transforms (the `transform` property): nsimplex x
    // (ndim+1) x ndim; transform[i, :ndim, :ndim] = T^-1 (C layout exactly
    // as scipy stores the LAPACK result), transform[i, ndim, :] = r_n.
    // Computed lazily by find_simplex (like the scipy property); call
    // ensure_transform() to force it.
    mutable std::vector<double> transform;

    void ensure_transform() const;

    // Delaunay.find_simplex(xi, bruteforce=False, tol=None) over n query
    // points (row-major xi, n x ndim, float64). out[i] = simplex index or -1.
    void find_simplex(const double* xi, int64_t n, int32_t* out) const;
};

// scipy.spatial.Delaunay(points, incremental=True, qhull_options="QJ").
// `points` is row-major n x ndim float64.
Delaunay delaunay_qj(const double* pts, int64_t n, int ndim);

}  // namespace umap
