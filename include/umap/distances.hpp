// Faithful C++ port of the distance metrics in reference/umap/umap/distances.py
// (lmcinnes/umap, BSD 3 clause). Formulas and edge-case handling mirror the
// Python/numba implementations exactly.
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace umap {

// Signature shared by all metrics: two dense vectors of length n.
using DistanceFn =
    std::function<double(const double* x, const double* y, std::size_t n)>;

double euclidean(const double* x, const double* y, std::size_t n);
// Squared euclidean ("sqeuclidean"); not in umap/distances.py itself (upstream
// delegates to sklearn/pynndescent) but part of UMAP's supported metric set.
double squared_euclidean(const double* x, const double* y, std::size_t n);
double manhattan(const double* x, const double* y, std::size_t n);
double chebyshev(const double* x, const double* y, std::size_t n);
double minkowski(const double* x, const double* y, std::size_t n, double p);
double cosine(const double* x, const double* y, std::size_t n);
double correlation(const double* x, const double* y, std::size_t n);
double hamming(const double* x, const double* y, std::size_t n);
double jaccard(const double* x, const double* y, std::size_t n);
double dice(const double* x, const double* y, std::size_t n);
double hellinger(const double* x, const double* y, std::size_t n);
double canberra(const double* x, const double* y, std::size_t n);
double bray_curtis(const double* x, const double* y, std::size_t n);

// --- metrics-full additions -------------------------------------------------
// Complete port of the remaining dense metrics in distances.py (numba f8
// instantiations; where upstream hard-codes float32 intermediates -- e.g. the
// mahalanobis `diff` buffer or the dtype=np.float32 gradient arrays -- the
// same rounding is applied here).

// distances.py::sign -- numba helper; note sign(0) == 1 (unlike np.sign).
inline int sign(double a) { return a < 0 ? -1 : 1; }

// distances.py::softmax (always computes/returns float64 in upstream numba:
// the output array is np.empty(n) which defaults to f8).
void softmax(const double* z, std::size_t n, double* out);

// Standardised/weighted distances (extra parameters follow the upstream
// keyword order; see MetricKwds/make_metric below for the kwds mechanism).
double standardised_euclidean(const double* x, const double* y, std::size_t n,
                              const double* sigma);
double weighted_minkowski(const double* x, const double* y, std::size_t n,
                          const double* w, double p);
// vinv is the row-major n x n inverse covariance matrix. Upstream stores the
// difference vector in float32 regardless of input dtype; replicated.
double mahalanobis(const double* x, const double* y, std::size_t n,
                   const double* vinv);

// Binary distances.
double matching(const double* x, const double* y, std::size_t n);
double kulsinski(const double* x, const double* y, std::size_t n);
double rogers_tanimoto(const double* x, const double* y, std::size_t n);
double russellrao(const double* x, const double* y, std::size_t n);
double sokal_michener(const double* x, const double* y, std::size_t n);
double sokal_sneath(const double* x, const double* y, std::size_t n);
double yule(const double* x, const double* y, std::size_t n);

// Other distances. haversine throws std::invalid_argument unless n == 2
// (upstream raises ValueError).
double haversine(const double* x, const double* y, std::size_t n);
double poincare(const double* u, const double* v, std::size_t n);
double ll_dirichlet(const double* data1, const double* data2, std::size_t n);
// Upstream symmetric_kl mutates x and y in place (adds z, normalises). This
// const version computes on internal copies -- identical result for a single
// call; the in-place variant used by the SGD loops is symmetric_kl_grad.
double symmetric_kl(const double* x, const double* y, std::size_t n,
                    double z = 1e-11);
double softmax_hellinger(const double* x, const double* y, std::size_t n);

// ll_dirichlet helpers (exact ports).
double approx_log_Gamma(double x);
double log_beta(double x, double y);
double log_single_beta(double x);

// Special discrete distances (upstream: "where x and y are objects, not
// vectors"). Not part of the vector registry; used by the supervised-fit
// target-metric path (Phase 1b) once that lands.
double categorical_distance(double x, double y);
double ordinal_distance(double x, double y, double support_size = 1.0);
double count_distance(double x, double y, double poisson_lambda = 1.0,
                      double normalisation = 1.0);
// cat_hierarchy: one map per level, category value -> group id. Missing keys
// throw std::out_of_range (upstream numba would raise a KeyError).
double hierarchical_categorical_distance(
    double x, double y,
    const std::vector<std::unordered_map<double, double>>& cat_hierarchy);
double levenshtein(const std::string& x, const std::string& y,
                   double normalisation = 1.0, int max_distance = 20);
double levenshtein_myers_ascii(const std::string& x, const std::string& y,
                               double normalisation = 1.0,
                               int max_distance = 20);

// ---------------------------------------------------------------- gradients
// Port of the `_grad` variants: return the distance and write d(dist)/dx into
// grad (length n, except diagonal_gaussian_energy_grad which requires a
// buffer of at least 6 doubles -- upstream allocates np.empty(6) and only
// initialises the first four entries; we zero the trailing two).
//
// Signature note: x and y are non-const because two upstream gradients mutate
// their inputs in place and the SGD loops rely on it (symmetric_kl_grad
// normalises x and y; gaussian_energy_grad clamps x[2..4]/y[2..4]). All other
// gradients leave x/y untouched.
using GradientFn = std::function<double(double* x, double* y, std::size_t n,
                                        double* grad)>;

double euclidean_grad(const double* x, const double* y, std::size_t n,
                      double* grad);
double standardised_euclidean_grad(const double* x, const double* y,
                                   std::size_t n, const double* sigma,
                                   double* grad);
double manhattan_grad(const double* x, const double* y, std::size_t n,
                      double* grad);
double chebyshev_grad(const double* x, const double* y, std::size_t n,
                      double* grad);
double minkowski_grad(const double* x, const double* y, std::size_t n, double p,
                      double* grad);
double weighted_minkowski_grad(const double* x, const double* y, std::size_t n,
                               const double* w, double p, double* grad);
double mahalanobis_grad(const double* x, const double* y, std::size_t n,
                        const double* vinv, double* grad);
double canberra_grad(const double* x, const double* y, std::size_t n,
                     double* grad);
double bray_curtis_grad(const double* x, const double* y, std::size_t n,
                        double* grad);
double cosine_grad(const double* x, const double* y, std::size_t n,
                   double* grad);
double correlation_grad(const double* x, const double* y, std::size_t n,
                        double* grad);
double hellinger_grad(const double* x, const double* y, std::size_t n,
                      double* grad);
double softmax_hellinger_grad(const double* x, const double* y, std::size_t n,
                              double* grad);
double haversine_grad(const double* x, const double* y, std::size_t n,
                      double* grad);
// Mutates x and y in place (adds z, normalises) exactly like upstream.
double symmetric_kl_grad(double* x, double* y, std::size_t n, double* grad,
                         double z = 1e-11);
// Special embedding gradients (fixed dimensionalities: 3 / 4 / 5 / any).
double spherical_gaussian_energy_grad(const double* x, const double* y,
                                      std::size_t n, double* grad);
double diagonal_gaussian_energy_grad(const double* x, const double* y,
                                     std::size_t n, double* grad);
// Mutates x[2..4] and y[2..4] in place (abs widths/heights, wrapped angle).
double gaussian_energy_grad(double* x, double* y, std::size_t n, double* grad);
double hyperboloid_grad(const double* x, const double* y, std::size_t n,
                        double* grad);

// ------------------------------------------------------------ kwds registry
// Mirror of upstream's metric_kwds mechanism: extra positional parameters of
// parameterised metrics, in the same order upstream binds tuple(kwds.values())
// after (x, y). Metrics ignore the fields they do not use.
struct MetricKwds {
    double p = 2.0;            // minkowski / weighted_minkowski exponent
    std::vector<double> v;     // seuclidean sigma / weighted_minkowski w
                               //   (empty -> ones, upstream's default intent)
    std::vector<double> vinv;  // mahalanobis V^{-1}, row-major n x n
                               //   (empty -> identity)
    double z = 1e-11;          // symmetric_kl smoothing
};

// Factory: bind a named metric with its kwds into a plain DistanceFn, like
// upstream's `_partial_metric` closure over tuple(metric_kwds.values()).
// Throws std::invalid_argument for unknown names or wrong-sized parameters.
DistanceFn make_metric(const std::string& name, const MetricKwds& kwds = {});
GradientFn make_metric_grad(const std::string& name,
                            const MetricKwds& kwds = {});

// Registry mirroring upstream's `named_distances_with_gradients` dict
// (including aliases). Parameterised entries are bound with their upstream
// defaults; use make_metric_grad for explicit kwds.
const std::unordered_map<std::string, GradientFn>&
named_distances_with_gradients();
// ----------------------------------------------------------------------------

// Registry mirroring upstream's `named_distances` dict (including aliases),
// restricted to the metrics ported so far. `minkowski` is registered with its
// upstream default p=2.
const std::unordered_map<std::string, DistanceFn>& named_distances();

// --- densmap additions ------------------------------------------------------
// float32-input metric evaluation, as the numba kernels see it when sklearn's
// _pairwise_callable / pynndescent hand them float32 rows. For a subset of
// metrics the exact numba f4-instantiation arithmetic is replicated
// (verified against the reference environment):
//   * euclidean/l2:  f4 diff, f4 square, f8 accumulate, f8 sqrt;
//   * manhattan/taxicab/l1: f4 diff, f4 abs, f8 accumulate;
//   * minkowski:     f4 diff, f4 abs, f8 pow/accumulate/root;
//   * seuclidean/standardised_euclidean: f4 diff, f4 square, f8 divide by the
//     (float64 kwds) sigma, f8 accumulate, f8 sqrt.
// Every other named metric goes through a documented adapter that widens the
// float32 rows to float64 and evaluates the double-precision registry kernel
// (last-ulp deviations possible vs numba's f4 instantiations).
// `f32_faithful`, when non-null, reports which case was taken. The returned
// value is the float32 rounding of the kernel's (float64) result, matching
// numba results stored into float32 arrays.
using DistanceF32Fn =
    std::function<float(const float* x, const float* y, std::size_t n)>;
DistanceF32Fn make_metric_f32(const std::string& name,
                              const MetricKwds& kwds = {},
                              bool* f32_faithful = nullptr);
// ----------------------------------------------------------------------------

}  // namespace umap
