// Port of reference/umap/umap/tests/test_umap_grads.py to doctest (upstream
// test names preserved; the pytest dim/p parametrizations run as loops inside
// each case). The sampler draws (np.random.default_rng(0) normal / dirichlet /
// gamma pairs, 100 per dim, plus the uniform w and sigma vectors) are baked in
// tests/upstream_grad_data.hpp by scripts/gen_upstream_fixtures.py.
//
// assert_gradient_matches_finite_diff is ported 1:1: central (or forward)
// finite differences at eps=1e-6 against the analytic gradients, with
// upstream's tolerances (max 1e-5, mean 1e-6) and the skip_close_coords
// handling for hellinger; plus the analytic-vs-plain distance match (< 1e-6).
// test_mahalanobis_gradient diffs a local float64 mahalanobis (upstream's
// dist.mahalanobis_f64) because the production kernel replicates upstream's
// float32 diff buffer, which is too coarse for finite differences.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "doctest.h"
#include "umap/distances.hpp"
#include "upstream_grad_data.hpp"

namespace {

using Dist = std::function<double(const double*, const double*, std::size_t)>;
using Grad =
    std::function<double(double* x, double* y, std::size_t n, double* grad)>;

constexpr int kPairs = 100;
const int kDims[3] = {4, 16, 64};

struct DimData {
    const double* x;
    const double* y;
};

DimData normal_pairs(int dim) {
    switch (dim) {
        case 4: return {ugrad::normal_x4, ugrad::normal_y4};
        case 16: return {ugrad::normal_x16, ugrad::normal_y16};
        default: return {ugrad::normal_x64, ugrad::normal_y64};
    }
}

DimData dirichlet_pairs(int dim) {
    switch (dim) {
        case 4: return {ugrad::dirichlet_x4, ugrad::dirichlet_y4};
        case 16: return {ugrad::dirichlet_x16, ugrad::dirichlet_y16};
        default: return {ugrad::dirichlet_x64, ugrad::dirichlet_y64};
    }
}

DimData abundance_pairs(int dim) {
    switch (dim) {
        case 4: return {ugrad::gamma_x4, ugrad::gamma_y4};
        case 16: return {ugrad::gamma_x16, ugrad::gamma_y16};
        default: return {ugrad::gamma_x64, ugrad::gamma_y64};
    }
}

const double* uniform_w(int dim) {
    switch (dim) {
        case 4: return ugrad::uniform_w4;
        case 16: return ugrad::uniform_w16;
        default: return ugrad::uniform_w64;
    }
}

const double* sigma_vec(int dim) {
    switch (dim) {
        case 4: return ugrad::sigma4;
        case 16: return ugrad::sigma16;
        default: return ugrad::sigma64;
    }
}

struct Opts {
    double max_tol = 1e-5;
    double mean_tol = 1e-6;
    bool forward_only = false;
    bool skip_close_coords = false;
};

// assert_gradient_matches_finite_diff (n_samples=100 pairs of length dim).
void check_gradient(const Dist& dist, const Grad& grad, DimData pairs, int dim,
                    Opts o = Opts()) {
    const double eps = 1e-6;
    double max_err = 0.0, sum_err = 0.0;
    std::vector<double> xbuf(dim), ybuf(dim), gbuf(std::max(dim, 6));
    std::vector<double> num(dim), ana(dim);
    for (int s = 0; s < kPairs; ++s) {
        const double* x = pairs.x + static_cast<std::size_t>(s) * dim;
        const double* y = pairs.y + static_cast<std::size_t>(s) * dim;

        // Numerical gradient wrt x.
        const double fx = o.forward_only ? dist(x, y, dim) : 0.0;
        for (int i = 0; i < dim; ++i) {
            std::vector<double> xf(x, x + dim);
            xf[i] += eps;
            if (o.forward_only) {
                num[i] = (dist(xf.data(), y, dim) - fx) / eps;
            } else {
                std::vector<double> xb(x, x + dim);
                xb[i] -= eps;
                num[i] =
                    (dist(xf.data(), y, dim) - dist(xb.data(), y, dim)) /
                    (2.0 * eps);
            }
        }

        // Analytic gradient (fresh copies: some kernels may mutate inputs).
        xbuf.assign(x, x + dim);
        ybuf.assign(y, y + dim);
        const double analytic_dist =
            grad(xbuf.data(), ybuf.data(), dim, gbuf.data());
        ana.assign(gbuf.begin(), gbuf.begin() + dim);

        if (o.skip_close_coords)
            for (int i = 0; i < dim; ++i)
                if (std::abs(x[i]) < 1e-3 || std::abs(y[i]) < 1e-3)
                    num[i] = ana[i] = 0.0;

        for (int i = 0; i < dim; ++i) {
            const double err = std::abs(num[i] - ana[i]);
            max_err = std::max(max_err, err);
            sum_err += err;
        }

        // Grad dists match non-grad dists.
        const double true_dist = dist(x, y, dim);
        CAPTURE(s);
        CHECK(std::abs(true_dist - analytic_dist) < 1e-6);
    }
    const double mean_err = sum_err / (static_cast<double>(kPairs) * dim);
    CAPTURE(max_err);
    CAPTURE(mean_err);
    CHECK(max_err < o.max_tol);
    CHECK(mean_err < o.mean_tol);
}

Grad wrap(double (*g)(const double*, const double*, std::size_t, double*)) {
    return [g](double* x, double* y, std::size_t n, double* grad) {
        return g(x, y, n, grad);
    };
}

}  // namespace

TEST_SUITE("test_umap_grads.py") {

TEST_CASE("test_euclidean_gradient") {
    for (int dim : kDims) {
        CAPTURE(dim);
        check_gradient(umap::euclidean, wrap(umap::euclidean_grad),
                       normal_pairs(dim), dim);
    }
}

TEST_CASE("test_minkowski_gradient") {
    for (int dim : kDims)
        for (int p = 1; p <= 4; ++p) {
            CAPTURE(dim);
            CAPTURE(p);
            Opts o;
            o.forward_only = (p == 1);
            const double pd = static_cast<double>(p);
            check_gradient(
                [pd](const double* x, const double* y, std::size_t n) {
                    return umap::minkowski(x, y, n, pd);
                },
                [pd](double* x, double* y, std::size_t n, double* grad) {
                    return umap::minkowski_grad(x, y, n, pd, grad);
                },
                normal_pairs(dim), dim, o);
        }
}

TEST_CASE("test_weighted_minkowski_gradient") {
    for (int dim : kDims)
        for (int p = 1; p <= 4; ++p) {
            CAPTURE(dim);
            CAPTURE(p);
            Opts o;
            o.forward_only = (p == 1);
            const double pd = static_cast<double>(p);
            const double* w = uniform_w(dim);
            check_gradient(
                [pd, w](const double* x, const double* y, std::size_t n) {
                    return umap::weighted_minkowski(x, y, n, w, pd);
                },
                [pd, w](double* x, double* y, std::size_t n, double* grad) {
                    return umap::weighted_minkowski_grad(x, y, n, w, pd, grad);
                },
                normal_pairs(dim), dim, o);
        }
}

TEST_CASE("test_cosine_gradient") {
    for (int dim : kDims) {
        CAPTURE(dim);
        check_gradient(umap::cosine, wrap(umap::cosine_grad),
                       normal_pairs(dim), dim);
    }
}

TEST_CASE("test_manhattan_gradient") {
    for (int dim : kDims) {
        CAPTURE(dim);
        Opts o;
        o.forward_only = true;
        check_gradient(umap::manhattan, wrap(umap::manhattan_grad),
                       normal_pairs(dim), dim, o);
    }
}

TEST_CASE("test_chebyshev_gradient") {
    for (int dim : kDims) {
        CAPTURE(dim);
        check_gradient(umap::chebyshev, wrap(umap::chebyshev_grad),
                       normal_pairs(dim), dim);
    }
}

TEST_CASE("test_correlation_gradient") {
    for (int dim : kDims) {
        CAPTURE(dim);
        check_gradient(umap::correlation, wrap(umap::correlation_grad),
                       normal_pairs(dim), dim);
    }
}

TEST_CASE("test_braycurtis_gradient") {
    for (int dim : kDims) {
        CAPTURE(dim);
        check_gradient(umap::bray_curtis, wrap(umap::bray_curtis_grad),
                       abundance_pairs(dim), dim);
    }
}

TEST_CASE("test_hellinger_gradient") {
    for (int dim : kDims) {
        CAPTURE(dim);
        Opts o;
        o.skip_close_coords = true;
        check_gradient(umap::hellinger, wrap(umap::hellinger_grad),
                       dirichlet_pairs(dim), dim, o);
    }
}

TEST_CASE("test_standardised_euclidean_gradient") {
    for (int dim : kDims) {
        CAPTURE(dim);
        const double* sigma = sigma_vec(dim);
        check_gradient(
            [sigma](const double* x, const double* y, std::size_t n) {
                return umap::standardised_euclidean(x, y, n, sigma);
            },
            [sigma](double* x, double* y, std::size_t n, double* grad) {
                return umap::standardised_euclidean_grad(x, y, n, sigma, grad);
            },
            normal_pairs(dim), dim);
    }
}

TEST_CASE("test_mahalanobis_gradient") {
    for (int dim : kDims) {
        CAPTURE(dim);
        // vinv = np.diag(rng.uniform(0.5, 2.0, dim)) -- the sigma draw.
        const double* diag = sigma_vec(dim);
        std::vector<double> vinv(static_cast<std::size_t>(dim) * dim, 0.0);
        for (int i = 0; i < dim; ++i) vinv[i * dim + i] = diag[i];
        const double* v = vinv.data();
        // Full-precision mahalanobis for the finite differences (upstream's
        // dist.mahalanobis_f64).
        const Dist mahal_f64 = [v](const double* x, const double* y,
                                   std::size_t n) {
            std::vector<double> diff(n);
            for (std::size_t i = 0; i < n; ++i) diff[i] = x[i] - y[i];
            double acc = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double tmp = 0.0;
                for (std::size_t j = 0; j < n; ++j)
                    tmp += v[i * n + j] * diff[j];
                acc += tmp * diff[i];
            }
            return std::sqrt(acc);
        };
        check_gradient(
            mahal_f64,
            [v](double* x, double* y, std::size_t n, double* grad) {
                return umap::mahalanobis_grad(x, y, n, v, grad);
            },
            normal_pairs(dim), dim);
    }
}

TEST_CASE("test_softmax_hellinger_gradient") {
    for (int dim : kDims) {
        CAPTURE(dim);
        check_gradient(umap::softmax_hellinger,
                       wrap(umap::softmax_hellinger_grad), normal_pairs(dim),
                       dim);
    }
}

}  // TEST_SUITE
