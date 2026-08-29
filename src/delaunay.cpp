// See include/umap/delaunay.hpp. Port of scipy 1.18.1 scipy/spatial/_qhull.pyx
// (BSD 3 clause) restricted to what umap_.py::inverse_transform uses, over the
// qhull_r sources scipy vendors (third_party/qhull_r).
#include "umap/delaunay.hpp"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Reentrant qhull, exactly as scipy compiles it (qhull_misc's
// qh_new_qhull_scipy only differs for halfspace mode, which Delaunay never
// hits -- plain qh_new_qhull is the same code path for mode "d").
extern "C" {
#include "libqhull_r/qhull_ra.h"
}

namespace umap {

namespace {

// ---------------------------------------------------------------- LAPACK-ish
// Unblocked dgetf2 (column-major, lda = n). Matches reference LAPACK
// operation-for-operation: idamax pivot (first maximum), row swap, reciprocal
// scaling when |pivot| >= sfmin, dger rank-1 trailing update.
int dgetf2(int n, double* a, int* ipiv) {
    const double sfmin = DBL_MIN;  // dlamch('S') for IEEE double
    int info = 0;
    for (int j = 0; j < n; ++j) {
        int jp = j;
        double amax = std::fabs(a[j + j * n]);
        for (int i = j + 1; i < n; ++i) {
            const double v = std::fabs(a[i + j * n]);
            if (v > amax) {
                amax = v;
                jp = i;
            }
        }
        ipiv[j] = jp + 1;
        if (a[jp + j * n] != 0.0) {
            if (jp != j) {
                for (int k = 0; k < n; ++k)
                    std::swap(a[j + k * n], a[jp + k * n]);
            }
            if (j < n - 1) {
                if (std::fabs(a[j + j * n]) >= sfmin) {
                    const double alpha = 1.0 / a[j + j * n];
                    for (int i = j + 1; i < n; ++i) a[i + j * n] *= alpha;
                } else {
                    for (int i = j + 1; i < n; ++i)
                        a[i + j * n] /= a[j + j * n];
                }
            }
        } else if (info == 0) {
            info = j + 1;
        }
        if (j < n - 1) {
            // dger: A(j+1:, j+1:) -= A(j+1:, j) * A(j, j+1:)
            for (int k = j + 1; k < n; ++k) {
                const double ajk = a[j + k * n];
                if (ajk != 0.0) {
                    for (int i = j + 1; i < n; ++i)
                        a[i + k * n] -= a[i + j * n] * ajk;
                }
            }
        }
    }
    return info;
}

// dgetrs("N") for nrhs right-hand sides (column-major, lda = ldb = n):
// dlaswp forward, unit-lower solve, upper solve (reference dtrsm loop order).
void dgetrs(int n, int nrhs, const double* a, const int* ipiv, double* b) {
    for (int i = 0; i < n; ++i) {
        const int ip = ipiv[i] - 1;
        if (ip != i) {
            for (int k = 0; k < nrhs; ++k)
                std::swap(b[i + k * n], b[ip + k * n]);
        }
    }
    for (int k = 0; k < nrhs; ++k) {
        double* col = b + k * n;
        // L (unit diagonal), forward.
        for (int j = 0; j < n; ++j) {
            if (col[j] != 0.0) {
                for (int i = j + 1; i < n; ++i)
                    col[i] -= col[j] * a[i + j * n];
            }
        }
        // U, backward.
        for (int j = n - 1; j >= 0; --j) {
            if (col[j] != 0.0) {
                col[j] /= a[j + j * n];
                for (int i = 0; i < j; ++i)
                    col[i] -= col[j] * a[i + j * n];
            }
        }
    }
}

// _qhull.pyx::_matrix_norm1 (column-major).
double matrix_norm1(int n, const double* a) {
    double maxsum = 0.0;
    for (int j = 0; j < n; ++j) {
        double colsum = 0.0;
        for (int i = 0; i < n; ++i) colsum += std::fabs(a[i + j * n]);
        if (maxsum < colsum) maxsum = colsum;
    }
    return maxsum;
}

// ------------------------------------------------------- find_simplex kernels
// Direct translations of the cdef helpers in _qhull.pyx. `d` fields are
// replaced by the Delaunay struct.

int barycentric_inside(int ndim, const double* transform, const double* x,
                       double* c, double eps) {
    c[ndim] = 1.0;
    for (int i = 0; i < ndim; ++i) {
        c[i] = 0.0;
        for (int j = 0; j < ndim; ++j)
            c[i] += transform[ndim * i + j] * (x[j] - transform[ndim * ndim + j]);
        c[ndim] -= c[i];
        if (!(-eps <= c[i] && c[i] <= 1 + eps)) return 0;
    }
    if (!(-eps <= c[ndim] && c[ndim] <= 1 + eps)) return 0;
    return 1;
}

void barycentric_coordinate_single(int ndim, const double* transform,
                                   const double* x, double* c, int i) {
    if (i == ndim) {
        c[ndim] = 1.0;
        for (int j = 0; j < ndim; ++j) c[ndim] -= c[j];
    } else {
        c[i] = 0.0;
        for (int j = 0; j < ndim; ++j)
            c[i] += transform[ndim * i + j] * (x[j] - transform[ndim * ndim + j]);
    }
}

void barycentric_coordinates(int ndim, const double* transform,
                             const double* x, double* c) {
    c[ndim] = 1.0;
    for (int i = 0; i < ndim; ++i) {
        c[i] = 0.0;
        for (int j = 0; j < ndim; ++j)
            c[i] += transform[ndim * i + j] * (x[j] - transform[ndim * ndim + j]);
        c[ndim] -= c[i];
    }
}

void lift_point(const Delaunay& d, const double* x, double* z) {
    z[d.ndim] = 0.0;
    for (int i = 0; i < d.ndim; ++i) {
        z[i] = x[i];
        z[d.ndim] += x[i] * x[i];
    }
    z[d.ndim] *= d.paraboloid_scale;
    z[d.ndim] += d.paraboloid_shift;
}

double distplane(const Delaunay& d, int isimplex, const double* point) {
    double dist = d.equations[static_cast<std::size_t>(isimplex) * (d.ndim + 2) +
                              d.ndim + 1];
    for (int k = 0; k < d.ndim + 1; ++k)
        dist += d.equations[static_cast<std::size_t>(isimplex) * (d.ndim + 2) +
                            k] *
                point[k];
    return dist;
}

int is_point_fully_outside(const Delaunay& d, const double* x, double eps) {
    for (int i = 0; i < d.ndim; ++i) {
        if (x[i] < d.min_bound[static_cast<std::size_t>(i)] - eps ||
            x[i] > d.max_bound[static_cast<std::size_t>(i)] + eps)
            return 1;
    }
    return 0;
}

int find_simplex_bruteforce(const Delaunay& d, double* c, const double* x,
                            double eps, double eps_broad) {
    if (is_point_fully_outside(d, x, eps)) return -1;

    const int ndim = d.ndim;
    for (int isimplex = 0; isimplex < d.nsimplex; ++isimplex) {
        const double* transform =
            d.transform.data() +
            static_cast<std::size_t>(isimplex) * ndim * (ndim + 1);
        if (transform[0] == transform[0]) {
            // transform is valid (non-nan)
            if (barycentric_inside(ndim, transform, x, c, eps))
                return isimplex;
        } else {
            // degenerate simplex: check the neighbors with a larger epsilon
            for (int k = 0; k < ndim + 1; ++k) {
                const int ineighbor =
                    d.neighbors[static_cast<std::size_t>(ndim + 1) * isimplex +
                                k];
                if (ineighbor == -1) continue;
                const double* ntransform =
                    d.transform.data() +
                    static_cast<std::size_t>(ineighbor) * ndim * (ndim + 1);
                if (ntransform[0] != ntransform[0]) continue;  // another bad one
                barycentric_coordinates(ndim, ntransform, x, c);
                int inside = 1;
                for (int m = 0; m < ndim + 1; ++m) {
                    if (d.neighbors[static_cast<std::size_t>(ndim + 1) *
                                        ineighbor +
                                    m] == isimplex) {
                        // extra leeway towards isimplex
                        if (!(-eps_broad <= c[m] && c[m] <= 1 + eps)) {
                            inside = 0;
                            break;
                        }
                    } else {
                        if (!(-eps <= c[m] && c[m] <= 1 + eps)) {
                            inside = 0;
                            break;
                        }
                    }
                }
                if (inside) return ineighbor;
            }
        }
    }
    return -1;
}

int find_simplex_directed(const Delaunay& d, double* c, const double* x,
                          int* start, double eps, double eps_broad) {
    const int ndim = d.ndim;
    int isimplex = *start;
    if (isimplex < 0 || isimplex >= d.nsimplex) isimplex = 0;

    // Cython `for cycle_k in range(...) ... else:` -- the else runs when the
    // loop exhausts without break.
    bool broke = false;
    const int64_t max_iter = 1 + d.nsimplex / 4;
    for (int64_t cycle_k = 0; cycle_k < max_iter; ++cycle_k) {
        if (isimplex == -1) {
            broke = true;
            break;
        }
        const double* transform =
            d.transform.data() +
            static_cast<std::size_t>(isimplex) * ndim * (ndim + 1);

        int inside = 1;
        for (int k = 0; k < ndim + 1; ++k) {
            barycentric_coordinate_single(ndim, transform, x, c, k);
            if (c[k] < -eps) {
                // target point is in the direction of neighbor `k`
                const int m =
                    d.neighbors[static_cast<std::size_t>(ndim + 1) * isimplex +
                                k];
                if (m == -1) {
                    *start = isimplex;
                    return -1;  // outside the triangulation
                }
                isimplex = m;
                inside = -1;
                break;
            } else if (c[k] <= 1 + eps) {
                // inside this simplex (so far)
            } else {
                // outside (or nan -> degenerate)
                inside = 0;
            }
        }

        if (inside == -1) {
            continue;  // hopped to another simplex
        } else if (inside == 1) {
            broke = true;
            break;  // found it
        } else {
            // degenerate simplices in the way: brute force
            isimplex = find_simplex_bruteforce(d, c, x, eps, eps_broad);
            broke = true;
            break;
        }
    }
    if (!broke) {
        // failed to converge -- fall back to brute force
        isimplex = find_simplex_bruteforce(d, c, x, eps, eps_broad);
    }

    *start = isimplex;
    return isimplex;
}

int find_simplex_walk(const Delaunay& d, double* c, const double* x,
                      int* start, double eps, double eps_broad) {
    if (is_point_fully_outside(d, x, eps)) return -1;
    if (d.nsimplex <= 0) return -1;

    const int ndim = d.ndim;
    int isimplex = *start;
    if (isimplex < 0 || isimplex >= d.nsimplex) isimplex = 0;

    // Lift point to the paraboloid and walk to a positive-distance facet.
    std::vector<double> z(static_cast<std::size_t>(ndim) + 1);
    lift_point(d, x, z.data());

    double best_dist = distplane(d, isimplex, z.data());
    int changed = 1;
    while (changed) {
        if (best_dist > 0) break;
        changed = 0;
        for (int k = 0; k < ndim + 1; ++k) {
            const int ineigh =
                d.neighbors[static_cast<std::size_t>(ndim + 1) * isimplex + k];
            if (ineigh == -1) continue;
            const double dist = distplane(d, ineigh, z.data());
            // eps guard against extended-precision non-termination (see pyx)
            if (dist > best_dist + eps * (1 + std::fabs(best_dist))) {
                isimplex = ineigh;
                best_dist = dist;
                changed = 1;
            }
        }
    }

    *start = isimplex;
    return find_simplex_directed(d, c, x, start, eps, eps_broad);
}

// Reads the whole (temp) FILE into a string (qhull's error stream).
std::string read_stream(std::FILE* f) {
    std::string out;
    if (f == nullptr) return out;
    std::fflush(f);
    long size = std::ftell(f);
    if (size <= 0) return out;
    std::rewind(f);
    out.resize(static_cast<std::size_t>(size));
    const std::size_t got = std::fread(&out[0], 1, out.size(), f);
    out.resize(got);
    return out;
}

}  // namespace

// _get_barycentric_transforms(points, simplices, np.finfo(double).eps),
// stored exactly like the scipy `transform` property.
void Delaunay::ensure_transform() const {
    if (!transform.empty()) return;
    const double eps = DBL_EPSILON;          // np.finfo(np.double).eps
    const double rcond_limit = 1000.0 * eps;  // "at least three digits"

    const int n = ndim;
    transform.assign(static_cast<std::size_t>(nsimplex) * (n + 1) * n, 0.0);
    std::vector<double> T(static_cast<std::size_t>(n) * n);
    std::vector<double> Tcopy(static_cast<std::size_t>(n) * n);
    std::vector<double> inv(static_cast<std::size_t>(n) * n);
    std::vector<int> ipiv(static_cast<std::size_t>(n));

    for (int64_t isimplex = 0; isimplex < nsimplex; ++isimplex) {
        double* Tinv = transform.data() +
                       static_cast<std::size_t>(isimplex) * (n + 1) * n;
        // The pyx fills the numpy C-order buffers; LAPACK then reinterprets
        // the same memory as column-major. Replicate the exact memory image:
        //   C  view: T[i][j]      = (r_j - r_n)_i
        //   Fortran view offset i + j*n corresponds to C offset i*n + j,
        // so build the Fortran buffer directly with F(i,j) = C T[j][i]... --
        // equivalently F_buf[i + j*n] = C_buf[i*n + j].
        for (int i = 0; i < n; ++i) {
            Tinv[n * n + i] =
                points[static_cast<std::size_t>(
                           simplices[static_cast<std::size_t>(isimplex) *
                                         (n + 1) +
                                     n]) *
                           n +
                       i];
            for (int j = 0; j < n; ++j) {
                // C-order T[i][j]; the Fortran element (i,j) of the array
                // LAPACK sees is memory slot i + j*n == C slot [j][i].
                T[static_cast<std::size_t>(i) * n + j] =
                    points[static_cast<std::size_t>(
                               simplices[static_cast<std::size_t>(isimplex) *
                                             (n + 1) +
                                         j]) *
                               n +
                           i] -
                    Tinv[n * n + i];
            }
            Tinv[static_cast<std::size_t>(i) * n + i] = 1.0;
        }

        // LAPACK reinterprets the C buffer as column-major: element (i,j) of
        // the factored matrix lives at T[i + j*n] -- i.e. the matrix LAPACK
        // factors is the transpose of the C-order view, exactly as in scipy.
        const double anorm = matrix_norm1(n, T.data());
        int info = dgetf2(n, T.data(), ipiv.data());

        if (info == 0) {
            // dgecon("1"): deviation documented in delaunay.hpp -- exact
            // norm1(T^-1) via n solves instead of the dlacn2 estimate.
            double rcond = 0.0;
            if (anorm != 0.0) {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        inv[static_cast<std::size_t>(j) * n + i] =
                            (i == j) ? 1.0 : 0.0;
                dgetrs(n, n, T.data(), ipiv.data(), inv.data());
                const double ainvnm = matrix_norm1(n, inv.data());
                if (ainvnm != 0.0) rcond = (1.0 / anorm) / ainvnm;
            }
            if (rcond < rcond_limit) info = 1;
        }

        if (info == 0) {
            // Solve into the Tinvs block (the identity seeded above). The
            // block is C-order (ndim+1, ndim) but LAPACK sees the first
            // n*n doubles as a column-major n x n RHS -- identical memory
            // semantics to the pyx call.
            dgetrs(n, n, T.data(), ipiv.data(), Tinv);
        } else {
            for (int i = 0; i < (n + 1) * n; ++i)
                Tinv[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

void Delaunay::find_simplex(const double* xi, int64_t n, int32_t* out) const {
    ensure_transform();
    const double eps = 100.0 * DBL_EPSILON;  // tol=None default
    const double eps_broad = std::sqrt(eps);
    int start = 0;  // shared across queries, exactly like the pyx loop
    std::vector<double> c(static_cast<std::size_t>(ndim) + 2);
    for (int64_t k = 0; k < n; ++k) {
        out[k] = find_simplex_walk(*this, c.data(), xi + k * ndim, &start, eps,
                                   eps_broad);
    }
}

Delaunay delaunay_qj(const double* pts, int64_t n, int ndim) {
    // _Qhull.__init__ input checks (points already float64 C-order here).
    if (n <= 0) throw std::runtime_error("No points given");
    if (ndim < 2) throw std::runtime_error("Need at least 2-D data");
    for (int64_t i = 0; i < n * ndim; ++i) {
        if (std::isnan(pts[i]))
            throw std::runtime_error("Points cannot contain NaN");
    }

    Delaunay out;
    out.ndim = ndim;
    out.npoints = n;
    // incremental=True copies the points (we need our own copy anyway).
    out.points.assign(pts, pts + n * ndim);

    // Option processing for (options="QJ", required="Qt", incremental=True):
    // QJ removes Qt; no Qt -> no Q11; mode "d". Final command: "qhull d QJ".
    char command[] = "qhull d QJ";

    std::FILE* errfile = std::tmpfile();
    qhT* qh = static_cast<qhT*>(std::malloc(sizeof(qhT)));
    if (qh == nullptr) {
        if (errfile) std::fclose(errfile);
        throw std::bad_alloc();
    }
    qh_zero(qh, errfile);

    int exitcode = qh_new_qhull(qh, ndim, static_cast<int>(n),
                                out.points.data(), /*ismalloc=*/False, command,
                                nullptr, errfile);

    auto cleanup = [&]() {
        int curlong = 0, totlong = 0;
        qh_freeqhull(qh, qh_ALL);
        qh_memfreeshort(qh, &curlong, &totlong);
        std::free(qh);
        if (errfile) std::fclose(errfile);
    };

    if (exitcode != 0) {
        std::string msg = read_stream(errfile);
        cleanup();
        throw std::runtime_error("QhullError: " + msg);
    }

    try {
        // Delaunay._update -------------------------------------------------
        qh_triangulate(qh);  // get rid of non-simplical facets

        if (qh->SCALElast) {
            out.paraboloid_scale =
                qh->last_newhigh / (qh->last_high - qh->last_low);
            out.paraboloid_shift = -qh->last_low * out.paraboloid_scale;
        } else {
            out.paraboloid_scale = 1.0;
            out.paraboloid_shift = 0.0;
        }

        // get_simplex_facet_array ------------------------------------------
        const int facet_ndim = ndim + 1;  // _is_delaunay
        std::vector<int32_t> id_map(static_cast<std::size_t>(qh->facet_id),
                                    -1);
        facetT* facet = qh->facet_list;
        int32_t j = 0;
        while (facet && facet->next) {
            if (facet->upperdelaunay == qh->UPPERdelaunay) {
                if (!facet->simplicial &&
                    (qh_setsize(qh, facet->vertices) != facet_ndim ||
                     qh_setsize(qh, facet->neighbors) != facet_ndim)) {
                    throw std::runtime_error(
                        "non-simplical facet encountered: " +
                        std::to_string(qh_setsize(qh, facet->vertices)) +
                        " vertices");
                }
                id_map[static_cast<std::size_t>(facet->id)] = j;
                ++j;
            }
            facet = facet->next;
        }

        out.nsimplex = j;
        out.simplices.assign(static_cast<std::size_t>(j) * facet_ndim, 0);
        out.neighbors.assign(static_cast<std::size_t>(j) * facet_ndim, 0);
        out.equations.assign(static_cast<std::size_t>(j) * (facet_ndim + 1),
                             0.0);

        facet = qh->facet_list;
        j = 0;
        while (facet && facet->next) {
            if (facet->upperdelaunay != qh->UPPERdelaunay) {
                facet = facet->next;
                continue;
            }
            int lower_bound = 0;
            if (facet->toporient == qh_ORIENTclock && facet_ndim == 3) {
                // Swap the first and second indices for counter-clockwise
                // orientation (2-D triangulations only).
                for (int i = 0; i < 2; ++i) {
                    const int swapped_index = 1 ^ i;
                    vertexT* vertex =
                        static_cast<vertexT*>(facet->vertices->e[i].p);
                    out.simplices[static_cast<std::size_t>(j) * facet_ndim +
                                  swapped_index] =
                        qh_pointid(qh, vertex->point);
                    facetT* neighbor =
                        static_cast<facetT*>(facet->neighbors->e[i].p);
                    out.neighbors[static_cast<std::size_t>(j) * facet_ndim +
                                  swapped_index] =
                        id_map[static_cast<std::size_t>(neighbor->id)];
                }
                lower_bound = 2;
            }
            for (int i = lower_bound; i < facet_ndim; ++i) {
                vertexT* vertex =
                    static_cast<vertexT*>(facet->vertices->e[i].p);
                out.simplices[static_cast<std::size_t>(j) * facet_ndim + i] =
                    qh_pointid(qh, vertex->point);
                facetT* neighbor =
                    static_cast<facetT*>(facet->neighbors->e[i].p);
                out.neighbors[static_cast<std::size_t>(j) * facet_ndim + i] =
                    id_map[static_cast<std::size_t>(neighbor->id)];
            }
            for (int i = 0; i < facet_ndim; ++i)
                out.equations[static_cast<std::size_t>(j) * (facet_ndim + 1) +
                              i] = facet->normal[i];
            out.equations[static_cast<std::size_t>(j) * (facet_ndim + 1) +
                          facet_ndim] = facet->offset;
            ++j;
            facet = facet->next;
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();

    // _QhullUser._update: bounds over the (original) points.
    out.min_bound.assign(static_cast<std::size_t>(ndim),
                         std::numeric_limits<double>::infinity());
    out.max_bound.assign(static_cast<std::size_t>(ndim),
                         -std::numeric_limits<double>::infinity());
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < ndim; ++d) {
            const double v = out.points[static_cast<std::size_t>(i) * ndim + d];
            if (v < out.min_bound[static_cast<std::size_t>(d)])
                out.min_bound[static_cast<std::size_t>(d)] = v;
            if (v > out.max_bound[static_cast<std::size_t>(d)])
                out.max_bound[static_cast<std::size_t>(d)] = v;
        }
    }
    return out;
}

}  // namespace umap
