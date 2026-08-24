#include "umap/find_ab_params.hpp"

#include <cmath>
#include <vector>

namespace umap {

namespace {

// Sum of squared residuals of f(x) = 1/(1 + a x^(2b)) against yv.
double cost(const std::vector<double>& xv, const std::vector<double>& yv,
            double a, double b) {
    double c = 0.0;
    for (std::size_t i = 0; i < xv.size(); ++i) {
        double t = xv[i] > 0.0 ? std::pow(xv[i], 2.0 * b) : 0.0;
        double r = 1.0 / (1.0 + a * t) - yv[i];
        c += r * r;
    }
    return c;
}

}  // namespace

ABParams find_ab_params(double spread, double min_dist) {
    // xv = np.linspace(0, spread * 3, 300)
    constexpr int N = 300;
    std::vector<double> xv(N), yv(N);
    for (int i = 0; i < N; ++i) {
        xv[i] = spread * 3.0 * static_cast<double>(i) / (N - 1);
        yv[i] = xv[i] < min_dist ? 1.0
                                 : std::exp(-(xv[i] - min_dist) / spread);
    }

    // Levenberg-Marquardt on r_i(a, b) = 1/(1 + a x_i^(2b)) - y_i,
    // initial guess (1, 1) like scipy.optimize.curve_fit.
    double a = 1.0;
    double b = 1.0;
    double lambda = 1e-3;
    double prev_cost = cost(xv, yv, a, b);

    for (int iter = 0; iter < 400; ++iter) {
        // Build normal equations from the analytic Jacobian.
        double jtj00 = 0.0, jtj01 = 0.0, jtj11 = 0.0;
        double jtr0 = 0.0, jtr1 = 0.0;
        for (int i = 0; i < N; ++i) {
            double x = xv[i];
            double t = x > 0.0 ? std::pow(x, 2.0 * b) : 0.0;
            double denom = 1.0 + a * t;
            double r = 1.0 / denom - yv[i];
            // d f / d a = -t / denom^2
            double ja = -t / (denom * denom);
            // d f / d b = -2 a t ln(x) / denom^2  (limit 0 as x -> 0)
            double jb = x > 0.0 ? -2.0 * a * t * std::log(x) / (denom * denom) : 0.0;
            jtj00 += ja * ja;
            jtj01 += ja * jb;
            jtj11 += jb * jb;
            jtr0 += ja * r;
            jtr1 += jb * r;
        }

        // Solve (JtJ + lambda * diag(JtJ)) delta = -Jtr
        double m00 = jtj00 * (1.0 + lambda);
        double m11 = jtj11 * (1.0 + lambda);
        double m01 = jtj01;
        double det = m00 * m11 - m01 * m01;
        if (det == 0.0 || !std::isfinite(det)) break;
        double da = (-jtr0 * m11 + jtr1 * m01) / det;
        double db = (-jtr1 * m00 + jtr0 * m01) / det;

        double new_a = a + da;
        double new_b = b + db;
        double new_cost = cost(xv, yv, new_a, new_b);

        if (std::isfinite(new_cost) && new_cost <= prev_cost) {
            a = new_a;
            b = new_b;
            lambda = std::max(lambda * 0.1, 1e-15);
            double step = std::sqrt(da * da + db * db);
            double improvement = prev_cost - new_cost;
            prev_cost = new_cost;
            if (step < 1e-12 || improvement < 1e-15) break;
        } else {
            lambda *= 10.0;
            if (lambda > 1e12) break;
        }
    }

    return {a, b};
}

}  // namespace umap
