// Faithful C++ port of the distance metrics in reference/umap/umap/distances.py
// (lmcinnes/umap, BSD 3 clause). Formulas and edge-case handling mirror the
// Python/numba implementations exactly.
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>

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

// Registry mirroring upstream's `named_distances` dict (including aliases),
// restricted to the metrics ported so far. `minkowski` is registered with its
// upstream default p=2.
const std::unordered_map<std::string, DistanceFn>& named_distances();

}  // namespace umap
