#include "umap/distances.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace umap {

double euclidean(const double* x, const double* y, std::size_t n) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return std::sqrt(result);
}

double squared_euclidean(const double* x, const double* y, std::size_t n) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return result;
}

double manhattan(const double* x, const double* y, std::size_t n) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += std::fabs(x[i] - y[i]);
    }
    return result;
}

double chebyshev(const double* x, const double* y, std::size_t n) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result = std::max(result, std::fabs(x[i] - y[i]));
    }
    return result;
}

double minkowski(const double* x, const double* y, std::size_t n, double p) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += std::pow(std::fabs(x[i] - y[i]), p);
    }
    return std::pow(result, 1.0 / p);
}

double cosine(const double* x, const double* y, std::size_t n) {
    double result = 0.0;
    double norm_x = 0.0;
    double norm_y = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }
    if (norm_x == 0.0 && norm_y == 0.0) {
        return 0.0;
    } else if (norm_x == 0.0 || norm_y == 0.0) {
        return 1.0;
    } else {
        return 1.0 - (result / std::sqrt(norm_x * norm_y));
    }
}

double correlation(const double* x, const double* y, std::size_t n) {
    double mu_x = 0.0;
    double mu_y = 0.0;
    double norm_x = 0.0;
    double norm_y = 0.0;
    double dot_product = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        mu_x += x[i];
        mu_y += y[i];
    }
    mu_x /= static_cast<double>(n);
    mu_y /= static_cast<double>(n);

    for (std::size_t i = 0; i < n; ++i) {
        double shifted_x = x[i] - mu_x;
        double shifted_y = y[i] - mu_y;
        norm_x += shifted_x * shifted_x;
        norm_y += shifted_y * shifted_y;
        dot_product += shifted_x * shifted_y;
    }

    if (norm_x == 0.0 && norm_y == 0.0) {
        return 0.0;
    } else if (dot_product == 0.0) {
        return 1.0;
    } else {
        return 1.0 - (dot_product / std::sqrt(norm_x * norm_y));
    }
}

double hamming(const double* x, const double* y, std::size_t n) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] != y[i]) {
            result += 1.0;
        }
    }
    return result / static_cast<double>(n);
}

double jaccard(const double* x, const double* y, std::size_t n) {
    double num_non_zero = 0.0;
    double num_equal = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        bool x_true = x[i] != 0.0;
        bool y_true = y[i] != 0.0;
        num_non_zero += (x_true || y_true) ? 1.0 : 0.0;
        num_equal += (x_true && y_true) ? 1.0 : 0.0;
    }
    if (num_non_zero == 0.0) {
        return 0.0;
    } else {
        return (num_non_zero - num_equal) / num_non_zero;
    }
}

double dice(const double* x, const double* y, std::size_t n) {
    double num_true_true = 0.0;
    double num_not_equal = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        bool x_true = x[i] != 0.0;
        bool y_true = y[i] != 0.0;
        num_true_true += (x_true && y_true) ? 1.0 : 0.0;
        num_not_equal += (x_true != y_true) ? 1.0 : 0.0;
    }
    if (num_not_equal == 0.0) {
        return 0.0;
    } else {
        return num_not_equal / (2.0 * num_true_true + num_not_equal);
    }
}

double hellinger(const double* x, const double* y, std::size_t n) {
    double result = 0.0;
    double l1_norm_x = 0.0;
    double l1_norm_y = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += std::sqrt(x[i] * y[i]);
        l1_norm_x += x[i];
        l1_norm_y += y[i];
    }
    if (l1_norm_x == 0.0 && l1_norm_y == 0.0) {
        return 0.0;
    } else if (l1_norm_x == 0.0 || l1_norm_y == 0.0) {
        return 1.0;
    } else {
        return std::sqrt(1.0 - result / std::sqrt(l1_norm_x * l1_norm_y));
    }
}

double canberra(const double* x, const double* y, std::size_t n) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double denominator = std::fabs(x[i]) + std::fabs(y[i]);
        if (denominator > 0.0) {
            result += std::fabs(x[i] - y[i]) / denominator;
        }
    }
    return result;
}

double bray_curtis(const double* x, const double* y, std::size_t n) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        numerator += std::fabs(x[i] - y[i]);
        denominator += std::fabs(x[i] + y[i]);
    }
    if (denominator > 0.0) {
        return numerator / denominator;
    } else {
        return 0.0;
    }
}

// --- metrics-full additions -------------------------------------------------

void softmax(const double* z, std::size_t n, double* out) {
    double zmax = z[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (z[i] > zmax) zmax = z[i];
    }
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::exp(z[i] - zmax);
        s += out[i];
    }
    if (s == 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = 1.0 / static_cast<double>(n);
        }
    } else {
        const double invs = 1.0 / s;
        for (std::size_t i = 0; i < n; ++i) {
            out[i] *= invs;
        }
    }
}

double standardised_euclidean(const double* x, const double* y, std::size_t n,
                              const double* sigma) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += ((x[i] - y[i]) * (x[i] - y[i])) / sigma[i];
    }
    return std::sqrt(result);
}

double weighted_minkowski(const double* x, const double* y, std::size_t n,
                          const double* w, double p) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += w[i] * std::pow(std::fabs(x[i] - y[i]), p);
    }
    return std::pow(result, 1.0 / p);
}

double mahalanobis(const double* x, const double* y, std::size_t n,
                   const double* vinv) {
    double result = 0.0;
    // diff = np.empty(n, dtype=np.float32) -- float32 regardless of input.
    std::vector<float> diff(n);
    for (std::size_t i = 0; i < n; ++i) {
        diff[i] = static_cast<float>(x[i] - y[i]);
    }
    for (std::size_t i = 0; i < n; ++i) {
        double tmp = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            tmp += vinv[i * n + j] * static_cast<double>(diff[j]);
        }
        result += tmp * static_cast<double>(diff[i]);
    }
    return std::sqrt(result);
}

double matching(const double* x, const double* y, std::size_t n) {
    double num_not_equal = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        bool x_true = x[i] != 0.0;
        bool y_true = y[i] != 0.0;
        num_not_equal += (x_true != y_true) ? 1.0 : 0.0;
    }
    return num_not_equal / static_cast<double>(n);
}

double kulsinski(const double* x, const double* y, std::size_t n) {
    double num_true_true = 0.0;
    double num_not_equal = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        bool x_true = x[i] != 0.0;
        bool y_true = y[i] != 0.0;
        num_true_true += (x_true && y_true) ? 1.0 : 0.0;
        num_not_equal += (x_true != y_true) ? 1.0 : 0.0;
    }
    if (num_not_equal == 0.0) {
        return 0.0;
    } else {
        return (num_not_equal - num_true_true + static_cast<double>(n)) /
               (num_not_equal + static_cast<double>(n));
    }
}

double rogers_tanimoto(const double* x, const double* y, std::size_t n) {
    double num_not_equal = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        bool x_true = x[i] != 0.0;
        bool y_true = y[i] != 0.0;
        num_not_equal += (x_true != y_true) ? 1.0 : 0.0;
    }
    return (2.0 * num_not_equal) / (static_cast<double>(n) + num_not_equal);
}

double russellrao(const double* x, const double* y, std::size_t n) {
    double num_true_true = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        bool x_true = x[i] != 0.0;
        bool y_true = y[i] != 0.0;
        num_true_true += (x_true && y_true) ? 1.0 : 0.0;
    }
    double nnz_x = 0.0;
    double nnz_y = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] != 0.0) nnz_x += 1.0;
        if (y[i] != 0.0) nnz_y += 1.0;
    }
    if (num_true_true == nnz_x && num_true_true == nnz_y) {
        return 0.0;
    } else {
        return (static_cast<double>(n) - num_true_true) /
               static_cast<double>(n);
    }
}

double sokal_michener(const double* x, const double* y, std::size_t n) {
    double num_not_equal = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        bool x_true = x[i] != 0.0;
        bool y_true = y[i] != 0.0;
        num_not_equal += (x_true != y_true) ? 1.0 : 0.0;
    }
    return (2.0 * num_not_equal) / (static_cast<double>(n) + num_not_equal);
}

double sokal_sneath(const double* x, const double* y, std::size_t n) {
    double num_true_true = 0.0;
    double num_not_equal = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        bool x_true = x[i] != 0.0;
        bool y_true = y[i] != 0.0;
        num_true_true += (x_true && y_true) ? 1.0 : 0.0;
        num_not_equal += (x_true != y_true) ? 1.0 : 0.0;
    }
    if (num_not_equal == 0.0) {
        return 0.0;
    } else {
        return num_not_equal / (0.5 * num_true_true + num_not_equal);
    }
}

double yule(const double* x, const double* y, std::size_t n) {
    double num_true_true = 0.0;
    double num_true_false = 0.0;
    double num_false_true = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        bool x_true = x[i] != 0.0;
        bool y_true = y[i] != 0.0;
        num_true_true += (x_true && y_true) ? 1.0 : 0.0;
        num_true_false += (x_true && !y_true) ? 1.0 : 0.0;
        num_false_true += (!x_true && y_true) ? 1.0 : 0.0;
    }
    double num_false_false = static_cast<double>(n) - num_true_true -
                             num_true_false - num_false_true;
    if (num_true_false == 0.0 || num_false_true == 0.0) {
        return 0.0;
    } else {
        return (2.0 * num_true_false * num_false_true) /
               (num_true_true * num_false_false +
                num_true_false * num_false_true);
    }
}

double haversine(const double* x, const double* y, std::size_t n) {
    if (n != 2) {
        throw std::invalid_argument(
            "haversine is only defined for 2 dimensional data");
    }
    double sin_lat = std::sin(0.5 * (x[0] - y[0]));
    double sin_long = std::sin(0.5 * (x[1] - y[1]));
    double result = std::sqrt(sin_lat * sin_lat +
                              std::cos(x[0]) * std::cos(y[0]) * sin_long *
                                  sin_long);
    return 2.0 * std::asin(result);
}

double poincare(const double* u, const double* v, std::size_t n) {
    double sq_u_norm = 0.0;
    double sq_v_norm = 0.0;
    double sq_dist = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sq_u_norm += u[i] * u[i];
        sq_v_norm += v[i] * v[i];
        sq_dist += (u[i] - v[i]) * (u[i] - v[i]);
    }
    return std::acosh(1.0 +
                      2.0 * (sq_dist / ((1.0 - sq_u_norm) * (1.0 - sq_v_norm))));
}

double approx_log_Gamma(double x) {
    if (x == 1.0) return 0.0;
    return x * std::log(x) - x + 0.5 * std::log(2.0 * M_PI / x) +
           1.0 / (x * 12.0);
}

double log_beta(double x, double y) {
    double a = std::min(x, y);
    double b = std::max(x, y);
    if (b < 5.0) {
        double value = -std::log(b);
        // for i in range(1, int(a)) -- int() truncates toward zero.
        const long long ia = static_cast<long long>(a);
        for (long long i = 1; i < ia; ++i) {
            value += std::log(static_cast<double>(i)) -
                     std::log(b + static_cast<double>(i));
        }
        return value;
    } else {
        return approx_log_Gamma(x) + approx_log_Gamma(y) -
               approx_log_Gamma(x + y);
    }
}

double log_single_beta(double x) {
    return std::log(2.0) * (-2.0 * x + 0.5) + 0.5 * std::log(2.0 * M_PI / x) +
           0.125 / x;
}

double ll_dirichlet(const double* data1, const double* data2, std::size_t n) {
    double n1 = 0.0;
    double n2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        n1 += data1[i];
        n2 += data2[i];
    }

    double log_b = 0.0;
    double self_denom1 = 0.0;
    double self_denom2 = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        if (data1[i] * data2[i] > 0.9) {
            log_b += log_beta(data1[i], data2[i]);
            self_denom1 += log_single_beta(data1[i]);
            self_denom2 += log_single_beta(data2[i]);
        } else {
            if (data1[i] > 0.9) {
                self_denom1 += log_single_beta(data1[i]);
            }
            if (data2[i] > 0.9) {
                self_denom2 += log_single_beta(data2[i]);
            }
        }
    }

    return std::sqrt(
        1.0 / n2 *
            (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2))) +
        1.0 / n1 *
            (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1))));
}

namespace {
// Shared core of symmetric_kl / symmetric_kl_grad: normalises x and y in
// place (x[i] = (x[i] + z) / sum(x + z)) and returns the symmetrised KL.
double symmetric_kl_core(double* x, double* y, std::size_t n, double z) {
    double x_sum = 0.0;
    double y_sum = 0.0;
    double kl1 = 0.0;
    double kl2 = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        x[i] += z;
        x_sum += x[i];
        y[i] += z;
        y_sum += y[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        x[i] /= x_sum;
        y[i] /= y_sum;
    }
    for (std::size_t i = 0; i < n; ++i) {
        kl1 += x[i] * std::log(x[i] / y[i]);
        kl2 += y[i] * std::log(y[i] / x[i]);
    }
    return (kl1 + kl2) / 2.0;
}
}  // namespace

double symmetric_kl(const double* x, const double* y, std::size_t n, double z) {
    // Upstream mutates x/y in place; computed on copies here (identical
    // result for a single call).
    std::vector<double> xc(x, x + n);
    std::vector<double> yc(y, y + n);
    return symmetric_kl_core(xc.data(), yc.data(), n, z);
}

double softmax_hellinger(const double* x, const double* y, std::size_t n) {
    std::vector<double> p(n);
    std::vector<double> q(n);
    softmax(x, n, p.data());
    softmax(y, n, q.data());
    return hellinger(p.data(), q.data(), n);
}

// ----------------------------------------------------- special discrete
double categorical_distance(double x, double y) {
    return (x == y) ? 0.0 : 1.0;
}

double ordinal_distance(double x, double y, double support_size) {
    return std::fabs(x - y) / support_size;
}

double count_distance(double x, double y, double poisson_lambda,
                      double normalisation) {
    const long long lo = static_cast<long long>(std::min(x, y));
    const long long hi = static_cast<long long>(std::max(x, y));

    const double log_lambda = std::log(poisson_lambda);

    double log_k_factorial;
    if (lo < 2) {
        log_k_factorial = 0.0;
    } else if (lo < 10) {
        log_k_factorial = 0.0;
        for (long long k = 2; k < lo; ++k) {
            log_k_factorial += std::log(static_cast<double>(k));
        }
    } else {
        log_k_factorial = approx_log_Gamma(static_cast<double>(lo + 1));
    }

    double result = 0.0;
    for (long long k = lo; k < hi; ++k) {
        result += static_cast<double>(k) * log_lambda - poisson_lambda -
                  log_k_factorial;
        log_k_factorial += std::log(static_cast<double>(k));
    }
    return result / normalisation;
}

double hierarchical_categorical_distance(
    double x, double y,
    const std::vector<std::unordered_map<double, double>>& cat_hierarchy) {
    const double n_levels = static_cast<double>(cat_hierarchy.size());
    for (std::size_t level = 0; level < cat_hierarchy.size(); ++level) {
        const auto& cats = cat_hierarchy[level];
        if (cats.at(x) == cats.at(y)) {
            return static_cast<double>(level) / n_levels;
        }
    }
    return 1.0;
}

double levenshtein(const std::string& x, const std::string& y,
                   double normalisation, int max_distance) {
    const std::size_t x_len = x.size();
    const std::size_t y_len = y.size();

    if (std::llabs(static_cast<long long>(x_len) -
                   static_cast<long long>(y_len)) > max_distance) {
        return static_cast<double>(max_distance) / normalisation;
    }
    if (x_len == 0) return static_cast<double>(y_len) / normalisation;
    if (y_len == 0) return static_cast<double>(x_len) / normalisation;

    std::vector<double> v0(y_len + 1);
    std::vector<double> v1(y_len + 1);
    for (std::size_t j = 0; j <= y_len; ++j) {
        v0[j] = static_cast<double>(j);
    }

    for (std::size_t i = 0; i < x_len; ++i) {
        v1[0] = static_cast<double>(i + 1);
        for (std::size_t j = 0; j < y_len; ++j) {
            const double deletion_cost = v0[j + 1] + 1.0;
            const double insertion_cost = v1[j] + 1.0;
            const double substitution_cost =
                v0[j] + ((x[i] != y[j]) ? 1.0 : 0.0);
            v1[j + 1] = std::min({deletion_cost, insertion_cost,
                                  substitution_cost});
        }
        std::swap(v0, v1);
        if (*std::min_element(v0.begin(), v0.end()) >
            static_cast<double>(max_distance)) {
            return static_cast<double>(max_distance) / normalisation;
        }
    }
    return v0[y_len] / normalisation;
}

double levenshtein_myers_ascii(const std::string& x, const std::string& y,
                               double normalisation, int max_distance) {
    const std::size_t x_len = x.size();
    const std::size_t y_len = y.size();

    if (std::llabs(static_cast<long long>(x_len) -
                   static_cast<long long>(y_len)) > max_distance) {
        return static_cast<double>(max_distance) / normalisation;
    }
    // Myers' bit-parallel algorithm is limited to word size.
    if (x_len > 63 || y_len > 63) {
        return levenshtein(x, y, normalisation, max_distance);
    }
    if (x_len == 0) {
        // Upstream would compute top_bit = 1 << -1; guard the trivial case
        // via the DP fallback (same result).
        return levenshtein(x, y, normalisation, max_distance);
    }

    int64_t Peq[128] = {0};
    for (std::size_t i = 0; i < x_len; ++i) {
        const int c = static_cast<unsigned char>(x[i]);
        if (c < 128) {
            Peq[c] |= static_cast<int64_t>(1) << i;
        }
    }

    int64_t Pv = (static_cast<int64_t>(1) << x_len) - 1;
    int64_t Mv = 0;
    long long score = static_cast<long long>(x_len);
    const int64_t top_bit = static_cast<int64_t>(1) << (x_len - 1);

    for (std::size_t j = 0; j < y_len; ++j) {
        const int c = static_cast<unsigned char>(y[j]);
        const int64_t Eq = (c < 128) ? Peq[c] : 0;

        const int64_t Xv = Eq | Mv;
        const int64_t Xh = (((Xv & Pv) + Pv) ^ Pv) | Xv;

        int64_t Ph = Mv | ~(Xh | Pv);
        int64_t Mh = Pv & Xh;

        if (Ph & top_bit) {
            score += 1;
        } else if (Mh & top_bit) {
            score -= 1;
        }

        Ph = (Ph << 1) | 1;
        Mh <<= 1;

        Pv = Mh | ~(Xh | Ph);
        Mv = Ph & Xh;
    }

    if (score > max_distance) {
        return static_cast<double>(max_distance) / normalisation;
    }
    return static_cast<double>(score) / normalisation;
}

// ------------------------------------------------------------- gradients
double euclidean_grad(const double* x, const double* y, std::size_t n,
                      double* grad) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += (x[i] - y[i]) * (x[i] - y[i]);
    }
    const double d = std::sqrt(result);
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = (x[i] - y[i]) / (1e-6 + d);
    }
    return d;
}

double standardised_euclidean_grad(const double* x, const double* y,
                                   std::size_t n, const double* sigma,
                                   double* grad) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += (x[i] - y[i]) * (x[i] - y[i]) / sigma[i];
    }
    const double d = std::sqrt(result);
    // grad = (x - y) / (1e-6 + d * sigma) -- elementwise.
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = (x[i] - y[i]) / (1e-6 + d * sigma[i]);
    }
    return d;
}

double manhattan_grad(const double* x, const double* y, std::size_t n,
                      double* grad) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += std::fabs(x[i] - y[i]);
        // np.sign: 0.0 for a zero difference (unlike the numba sign helper).
        const double diff = x[i] - y[i];
        grad[i] = (diff > 0.0) ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
    }
    return result;
}

double chebyshev_grad(const double* x, const double* y, std::size_t n,
                      double* grad) {
    double result = 0.0;
    std::size_t max_i = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = std::fabs(x[i] - y[i]);
        if (v > result) {
            result = v;
            max_i = i;
        }
    }
    for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
    const double diff = x[max_i] - y[max_i];
    grad[max_i] = (diff > 0.0) ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
    return result;
}

double minkowski_grad(const double* x, const double* y, std::size_t n, double p,
                      double* grad) {
    double S = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        S += std::pow(std::fabs(x[i] - y[i]), p);
    }
    const double dist = std::pow(S, 1.0 / p);
    // grad = np.zeros(n, dtype=np.float32) -- float32 storage.
    for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
    if (S == 0.0) return dist;

    const double inv_denom = std::pow(S, (1.0 - p) / p);
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = static_cast<double>(static_cast<float>(
            std::pow(std::fabs(x[i] - y[i]), p - 1.0) * sign(x[i] - y[i]) *
            inv_denom));
    }
    return dist;
}

double weighted_minkowski_grad(const double* x, const double* y, std::size_t n,
                               const double* w, double p, double* grad) {
    double S = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        S += w[i] * std::pow(std::fabs(x[i] - y[i]), p);
    }
    const double dist = std::pow(S, 1.0 / p);
    for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
    if (S == 0.0) return dist;

    const double inv_denom = std::pow(S, (1.0 - p) / p);
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = static_cast<double>(static_cast<float>(
            w[i] * std::pow(std::fabs(x[i] - y[i]), p - 1.0) *
            sign(x[i] - y[i]) * inv_denom));
    }
    return dist;
}

double mahalanobis_grad(const double* x, const double* y, std::size_t n,
                        const double* vinv, double* grad) {
    double result = 0.0;
    // diff = np.empty(n, dtype=np.float32).
    std::vector<float> diff(n);
    for (std::size_t i = 0; i < n; ++i) {
        diff[i] = static_cast<float>(x[i] - y[i]);
    }
    // grad_tmp = np.zeros(x.shape) -- float64.
    std::vector<double> grad_tmp(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double tmp = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            tmp += vinv[i * n + j] * static_cast<double>(diff[j]);
            grad_tmp[i] += vinv[i * n + j] * static_cast<double>(diff[j]);
        }
        result += tmp * static_cast<double>(diff[i]);
    }
    const double dist = std::sqrt(result);
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = grad_tmp[i] / (1e-6 + dist);
    }
    return dist;
}

double canberra_grad(const double* x, const double* y, std::size_t n,
                     double* grad) {
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double denominator = std::fabs(x[i]) + std::fabs(y[i]);
        if (denominator > 0.0) {
            result += std::fabs(x[i] - y[i]) / denominator;
            const double diff = x[i] - y[i];
            const double sgn_diff =
                (diff > 0.0) ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
            const double sgn_x =
                (x[i] > 0.0) ? 1.0 : (x[i] < 0.0 ? -1.0 : 0.0);
            grad[i] = sgn_diff / denominator -
                      std::fabs(x[i] - y[i]) * sgn_x /
                          (denominator * denominator);
        }
    }
    return result;
}

double bray_curtis_grad(const double* x, const double* y, std::size_t n,
                        double* grad) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        numerator += std::fabs(x[i] - y[i]);
        denominator += std::fabs(x[i] + y[i]);
    }
    if (denominator > 0.0) {
        const double dist = numerator / denominator;
        for (std::size_t i = 0; i < n; ++i) {
            const double diff = x[i] - y[i];
            const double sgn =
                (diff > 0.0) ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
            grad[i] = (sgn - dist) / denominator;
        }
        return dist;
    } else {
        for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
        return 0.0;
    }
}

double cosine_grad(const double* x, const double* y, std::size_t n,
                   double* grad) {
    double result = 0.0;
    double norm_x = 0.0;
    double norm_y = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }
    if (norm_x == 0.0 && norm_y == 0.0) {
        for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
        return 0.0;
    }
    if (norm_x == 0.0 || norm_y == 0.0) {
        for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
        return 1.0;
    }
    const double nx = std::sqrt(norm_x);
    const double ny = std::sqrt(norm_y);
    const double dist = 1.0 - result / (nx * ny);
    const double inv_nx_ny = 1.0 / (nx * ny);
    const double inv_nx3_ny = 1.0 / (norm_x * nx * ny);
    // grad = np.empty(n, dtype=np.float32).
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = static_cast<double>(static_cast<float>(
            x[i] * result * inv_nx3_ny - y[i] * inv_nx_ny));
    }
    return dist;
}

double correlation_grad(const double* x, const double* y, std::size_t n,
                        double* grad) {
    double mu_x = 0.0;
    double mu_y = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        mu_x += x[i];
        mu_y += y[i];
    }
    mu_x /= static_cast<double>(n);
    mu_y /= static_cast<double>(n);

    double dot = 0.0;
    double norm_x = 0.0;
    double norm_y = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double cx = x[i] - mu_x;
        const double cy = y[i] - mu_y;
        dot += cx * cy;
        norm_x += cx * cx;
        norm_y += cy * cy;
    }
    if (norm_x == 0.0 && norm_y == 0.0) {
        for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
        return 0.0;
    }
    if (norm_x == 0.0 || norm_y == 0.0) {
        for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
        return 1.0;
    }
    const double nx = std::sqrt(norm_x);
    const double ny = std::sqrt(norm_y);
    const double dist = 1.0 - dot / (nx * ny);
    const double inv_nx_ny = 1.0 / (nx * ny);
    const double inv_nx3_ny = 1.0 / (norm_x * nx * ny);

    // grad = np.empty(n, dtype=np.float32); mean_grad accumulates the f32
    // entries in f8 and the -= rounds back to f32 on store.
    double mean_grad = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double cx = x[i] - mu_x;
        grad[i] = static_cast<double>(static_cast<float>(
            cx * dot * inv_nx3_ny - (y[i] - mu_y) * inv_nx_ny));
        mean_grad += grad[i];
    }
    mean_grad /= static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = static_cast<double>(
            static_cast<float>(grad[i] - mean_grad));
    }
    return dist;
}

double hellinger_grad(const double* x, const double* y, std::size_t n,
                      double* grad) {
    double result = 0.0;
    double l1_norm_x = 0.0;
    double l1_norm_y = 0.0;
    std::vector<double> grad_term(n);  // np.empty(n) -- float64.
    for (std::size_t i = 0; i < n; ++i) {
        grad_term[i] = std::sqrt(x[i] * y[i]);
        result += grad_term[i];
        l1_norm_x += x[i];
        l1_norm_y += y[i];
    }
    if (l1_norm_x == 0.0 && l1_norm_y == 0.0) {
        for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
        return 0.0;
    }
    if (l1_norm_x == 0.0 || l1_norm_y == 0.0) {
        for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
        return 1.0;
    }
    const double dist_denom = std::sqrt(l1_norm_x * l1_norm_y);
    const double inner = std::max(0.0, 1.0 - result / dist_denom);
    const double dist = std::sqrt(inner);
    if (dist == 0.0) {
        for (std::size_t i = 0; i < n; ++i) grad[i] = 0.0;
        return dist;
    }
    const double grad_denom = 2.0 * dist;
    const double grad_numer_const =
        (l1_norm_y * result) / (2.0 * dist_denom * dist_denom * dist_denom);
    // grad = np.empty(n, dtype=np.float32).
    for (std::size_t i = 0; i < n; ++i) {
        double term;
        if (x[i] > 0.0 && grad_term[i] > 0.0) {
            term = y[i] / (2.0 * grad_term[i] * dist_denom);
        } else {
            term = 0.0;
        }
        grad[i] = static_cast<double>(
            static_cast<float>((grad_numer_const - term) / grad_denom));
    }
    return dist;
}

double softmax_hellinger_grad(const double* x, const double* y, std::size_t n,
                              double* grad) {
    std::vector<double> p(n);
    std::vector<double> q(n);
    softmax(x, n, p.data());
    softmax(y, n, q.data());

    std::vector<double> g_p(n);
    const double dist = hellinger_grad(p.data(), q.data(), n, g_p.data());

    double dot_gp_p = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dot_gp_p += g_p[i] * p[i];
    }
    // grad_x = np.empty(n) -- float64.
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = p[i] * (g_p[i] - dot_gp_p);
    }
    return dist;
}

double haversine_grad(const double* x, const double* y, std::size_t n,
                      double* grad) {
    if (n != 2) {
        throw std::invalid_argument(
            "haversine is only defined for 2 dimensional data");
    }
    const double sin_lat = std::sin(0.5 * (x[0] - y[0]));
    const double cos_lat = std::cos(0.5 * (x[0] - y[0]));
    const double sin_long = std::sin(0.5 * (x[1] - y[1]));
    const double cos_long = std::cos(0.5 * (x[1] - y[1]));

    const double a_0 = std::cos(x[0] + M_PI / 2.0) *
                       std::cos(y[0] + M_PI / 2.0) * sin_long * sin_long;
    const double a_1 = a_0 + sin_lat * sin_lat;

    const double d = 2.0 * std::asin(std::sqrt(
                               std::min(std::max(std::fabs(a_1), 0.0), 1.0)));
    const double denom =
        std::sqrt(std::fabs(a_1 - 1.0)) * std::sqrt(std::fabs(a_1));
    grad[0] = (sin_lat * cos_lat -
               std::sin(x[0] + M_PI / 2.0) * std::cos(y[0] + M_PI / 2.0) *
                   sin_long * sin_long) /
              (denom + 1e-6);
    grad[1] = (std::cos(x[0] + M_PI / 2.0) * std::cos(y[0] + M_PI / 2.0) *
               sin_long * cos_long) /
              (denom + 1e-6);
    return d;
}

double symmetric_kl_grad(double* x, double* y, std::size_t n, double* grad,
                         double z) {
    const double dist = symmetric_kl_core(x, y, n, z);
    // grad = (np.log(y / x) - (x / y) + 1) / 2, on the normalised values.
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = (std::log(y[i] / x[i]) - (x[i] / y[i]) + 1.0) / 2.0;
    }
    return dist;
}

double spherical_gaussian_energy_grad(const double* x, const double* y,
                                      std::size_t n, double* grad) {
    (void)n;  // fixed layout: (mu_1, mu_2, sigma)
    const double mu_1 = x[0] - y[0];
    const double mu_2 = x[1] - y[1];

    const double sigma = std::fabs(x[2]) + std::fabs(y[2]);
    const double sign_sigma =
        (x[2] > 0.0) ? 1.0 : (x[2] < 0.0 ? -1.0 : 0.0);  // np.sign

    const double dist = (mu_1 * mu_1 + mu_2 * mu_2) / (2.0 * sigma) +
                        std::log(sigma) + std::log(2.0 * M_PI);
    // grad = np.empty(3, np.float32).
    grad[0] = static_cast<double>(static_cast<float>(mu_1 / sigma));
    grad[1] = static_cast<double>(static_cast<float>(mu_2 / sigma));
    grad[2] = static_cast<double>(static_cast<float>(
        sign_sigma *
        (1.0 / sigma - (mu_1 * mu_1 + mu_2 * mu_2) / (2.0 * sigma * sigma))));
    return dist;
}

double diagonal_gaussian_energy_grad(const double* x, const double* y,
                                     std::size_t n, double* grad) {
    (void)n;  // fixed layout: (mu_1, mu_2, sigma_11, sigma_22); grad size 6
    const double mu_1 = x[0] - y[0];
    const double mu_2 = x[1] - y[1];

    const double sigma_11 = std::fabs(x[2]) + std::fabs(y[2]);
    const double sigma_12 = 0.0;
    const double sigma_22 = std::fabs(x[3]) + std::fabs(y[3]);

    const double det = sigma_11 * sigma_22;
    const double sign_s1 = (x[2] > 0.0) ? 1.0 : (x[2] < 0.0 ? -1.0 : 0.0);
    const double sign_s2 = (x[3] > 0.0) ? 1.0 : (x[3] < 0.0 ? -1.0 : 0.0);

    if (det == 0.0) {
        // Upstream returns a 4-element array here; entries 4/5 zeroed by us.
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 1.0;
        grad[3] = 1.0;
        grad[4] = 0.0;
        grad[5] = 0.0;
        return mu_1 * mu_1 + mu_2 * mu_2;
    }

    const double cross_term = 2.0 * sigma_12;
    const double m_dist = std::fabs(sigma_22) * (mu_1 * mu_1) -
                          cross_term * mu_1 * mu_2 +
                          std::fabs(sigma_11) * (mu_2 * mu_2);

    const double dist = (m_dist / det + std::log(std::fabs(det))) / 2.0 +
                        std::log(2.0 * M_PI);
    // grad = np.empty(6, dtype=np.float32); upstream leaves grad[4], grad[5]
    // uninitialised -- zeroed here (documented deviation).
    grad[0] = static_cast<double>(static_cast<float>(
        (2.0 * sigma_22 * mu_1 - cross_term * mu_2) / (2.0 * det)));
    grad[1] = static_cast<double>(static_cast<float>(
        (2.0 * sigma_11 * mu_2 - cross_term * mu_1) / (2.0 * det)));
    grad[2] = static_cast<double>(static_cast<float>(
        sign_s1 * (sigma_22 * (det - m_dist) + det * mu_2 * mu_2) /
        (2.0 * det * det)));
    grad[3] = static_cast<double>(static_cast<float>(
        sign_s2 * (sigma_11 * (det - m_dist) + det * mu_1 * mu_1) /
        (2.0 * det * det)));
    grad[4] = 0.0;
    grad[5] = 0.0;
    return dist;
}

double gaussian_energy_grad(double* x, double* y, std::size_t n, double* grad) {
    (void)n;  // fixed layout: (mu_1, mu_2, width, height, angle)
    const double mu_1 = x[0] - y[0];
    const double mu_2 = x[1] - y[1];

    // Ensure widths/heights are positive and angle in [-pi/2, pi/2] --
    // upstream mutates the inputs in place; replicated.
    x[2] = std::fabs(x[2]);
    y[2] = std::fabs(y[2]);
    x[3] = std::fabs(x[3]);
    y[3] = std::fabs(y[3]);
    x[4] = std::asin(std::sin(x[4]));
    y[4] = std::asin(std::sin(y[4]));

    // Covariance entries for y.
    const double a = y[2] * std::cos(y[4]) * std::cos(y[4]) +
                     y[3] * std::sin(y[4]) * std::sin(y[4]);
    const double b = (y[2] - y[3]) * std::sin(y[4]) * std::cos(y[4]);
    const double c = y[3] * std::cos(y[4]) * std::cos(y[4]) +
                     y[2] * std::sin(y[4]) * std::sin(y[4]);

    // Sum of covariance matrices.
    const double sigma_11 = x[2] * std::cos(x[4]) * std::cos(x[4]) +
                            x[3] * std::sin(x[4]) * std::sin(x[4]) + a;
    const double sigma_12 =
        (x[2] - x[3]) * std::sin(x[4]) * std::cos(x[4]) + b;
    const double sigma_22 = x[2] * std::sin(x[4]) * std::sin(x[4]) +
                            x[3] * std::cos(x[4]) * std::cos(x[4]) + c;

    const double det_sigma =
        std::fabs(sigma_11 * sigma_22 - sigma_12 * sigma_12);
    const double x_inv_sigma_y_numerator = sigma_22 * mu_1 * mu_1 -
                                           2.0 * sigma_12 * mu_1 * mu_2 +
                                           sigma_11 * mu_2 * mu_2;

    if (det_sigma < 1e-32) {
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 1.0;
        grad[3] = 1.0;
        grad[4] = 0.0;
        return mu_1 * mu_1 + mu_2 * mu_2;
    }

    const double dist = x_inv_sigma_y_numerator / det_sigma +
                        std::log(det_sigma) + std::log(2.0 * M_PI);

    const double cos_x4 = std::cos(x[4]);
    const double sin_x4 = std::sin(x[4]);

    // grad = np.zeros(5, np.float32) -- every store rounds through f32.
    float g0 = static_cast<float>(
        (2.0 * sigma_22 * mu_1 - 2.0 * sigma_12 * mu_2) / det_sigma);
    float g1 = static_cast<float>(
        (2.0 * sigma_11 * mu_2 - 2.0 * sigma_12 * mu_1) / det_sigma);

    double g2 = mu_2 * (mu_2 * cos_x4 * cos_x4 - mu_1 * cos_x4 * sin_x4);
    g2 = static_cast<float>(g2);
    g2 = static_cast<float>(
        g2 + mu_1 * (mu_1 * sin_x4 * sin_x4 - mu_2 * cos_x4 * sin_x4));
    g2 = static_cast<float>(g2 * det_sigma);
    g2 = static_cast<float>(
        g2 - x_inv_sigma_y_numerator * cos_x4 * cos_x4 * sigma_22);
    g2 = static_cast<float>(
        g2 - x_inv_sigma_y_numerator * sin_x4 * sin_x4 * sigma_11);
    g2 = static_cast<float>(
        g2 + x_inv_sigma_y_numerator * 2.0 * sigma_12 * sin_x4 * cos_x4);
    g2 = static_cast<float>(g2 / (det_sigma * det_sigma + 1e-8));

    double g3 = mu_1 * (mu_1 * cos_x4 * cos_x4 - mu_2 * cos_x4 * sin_x4);
    g3 = static_cast<float>(g3);
    g3 = static_cast<float>(
        g3 + mu_2 * (mu_2 * sin_x4 * sin_x4 - mu_1 * cos_x4 * sin_x4));
    g3 = static_cast<float>(g3 * det_sigma);
    g3 = static_cast<float>(
        g3 - x_inv_sigma_y_numerator * sin_x4 * sin_x4 * sigma_22);
    g3 = static_cast<float>(
        g3 - x_inv_sigma_y_numerator * cos_x4 * cos_x4 * sigma_11);
    g3 = static_cast<float>(
        g3 - x_inv_sigma_y_numerator * 2.0 * sigma_12 * sin_x4 * cos_x4);
    g3 = static_cast<float>(g3 / (det_sigma * det_sigma + 1e-8));

    double g4 = (x[3] - x[2]) * (2.0 * mu_1 * mu_2 * std::cos(2.0 * x[4]) -
                                 (mu_1 * mu_1 - mu_2 * mu_2) *
                                     std::sin(2.0 * x[4]));
    g4 = static_cast<float>(g4);
    g4 = static_cast<float>(g4 * det_sigma);
    g4 = static_cast<float>(g4 - x_inv_sigma_y_numerator * (x[3] - x[2]) *
                                     std::sin(2.0 * x[4]) * sigma_22);
    g4 = static_cast<float>(g4 - x_inv_sigma_y_numerator * (x[2] - x[3]) *
                                     std::sin(2.0 * x[4]) * sigma_11);
    g4 = static_cast<float>(g4 - x_inv_sigma_y_numerator * 2.0 * sigma_12 *
                                     (x[2] - x[3]) * std::cos(2.0 * x[4]));
    g4 = static_cast<float>(g4 / (det_sigma * det_sigma + 1e-8));

    grad[0] = static_cast<double>(g0);
    grad[1] = static_cast<double>(g1);
    grad[2] = static_cast<double>(static_cast<float>(g2));
    grad[3] = static_cast<double>(static_cast<float>(g3));
    grad[4] = static_cast<double>(static_cast<float>(g4));
    return dist;
}

double hyperboloid_grad(const double* x, const double* y, std::size_t n,
                        double* grad) {
    double sx = 0.0;
    double sy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sx += x[i] * x[i];
        sy += y[i] * y[i];
    }
    const double s = std::sqrt(1.0 + sx);
    const double t = std::sqrt(1.0 + sy);

    double B = s * t;
    for (std::size_t i = 0; i < n; ++i) {
        B -= x[i] * y[i];
    }
    if (B <= 1.0) {
        B = 1.0 + 1e-8;
    }
    const double grad_coeff = 1.0 / (std::sqrt(B - 1.0) * std::sqrt(B + 1.0));
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = grad_coeff * (((x[i] * t) / s) - y[i]);
    }
    return std::acosh(B);
}

// ------------------------------------------------------------ kwds factory
namespace {

void check_vec(const std::vector<double>& v, std::size_t n,
               const char* what) {
    if (!v.empty() && v.size() != n) {
        throw std::invalid_argument(std::string("make_metric: ") + what +
                                    " has wrong length for the input vectors");
    }
}

void check_mat(const std::vector<double>& m, std::size_t n) {
    if (!m.empty() && m.size() != n * n) {
        throw std::invalid_argument(
            "make_metric: vinv must be a row-major n x n matrix");
    }
}

}  // namespace

DistanceFn make_metric(const std::string& name, const MetricKwds& kwds) {
    if (name == "minkowski") {
        const double p = kwds.p;
        return [p](const double* x, const double* y, std::size_t n) {
            return minkowski(x, y, n, p);
        };
    }
    if (name == "seuclidean" || name == "standardised_euclidean") {
        const std::vector<double> sigma = kwds.v;
        return [sigma](const double* x, const double* y, std::size_t n) {
            check_vec(sigma, n, "sigma");
            if (sigma.empty()) {
                const std::vector<double> ones(n, 1.0);
                return standardised_euclidean(x, y, n, ones.data());
            }
            return standardised_euclidean(x, y, n, sigma.data());
        };
    }
    if (name == "wminkowski" || name == "weighted_minkowski") {
        const std::vector<double> w = kwds.v;
        const double p = kwds.p;
        return [w, p](const double* x, const double* y, std::size_t n) {
            check_vec(w, n, "w");
            if (w.empty()) {
                const std::vector<double> ones(n, 1.0);
                return weighted_minkowski(x, y, n, ones.data(), p);
            }
            return weighted_minkowski(x, y, n, w.data(), p);
        };
    }
    if (name == "mahalanobis") {
        const std::vector<double> vinv = kwds.vinv;
        return [vinv](const double* x, const double* y, std::size_t n) {
            check_mat(vinv, n);
            if (vinv.empty()) {
                std::vector<double> eye(n * n, 0.0);
                for (std::size_t i = 0; i < n; ++i) eye[i * n + i] = 1.0;
                return mahalanobis(x, y, n, eye.data());
            }
            return mahalanobis(x, y, n, vinv.data());
        };
    }
    if (name == "symmetric_kl") {
        const double z = kwds.z;
        return [z](const double* x, const double* y, std::size_t n) {
            return symmetric_kl(x, y, n, z);
        };
    }
    const auto& reg = named_distances();
    auto it = reg.find(name);
    if (it == reg.end()) {
        throw std::invalid_argument("make_metric: unknown metric '" + name +
                                    "'");
    }
    return it->second;
}

GradientFn make_metric_grad(const std::string& name, const MetricKwds& kwds) {
    if (name == "minkowski") {
        const double p = kwds.p;
        return [p](double* x, double* y, std::size_t n, double* grad) {
            return minkowski_grad(x, y, n, p, grad);
        };
    }
    if (name == "seuclidean" || name == "standardised_euclidean") {
        const std::vector<double> sigma = kwds.v;
        return [sigma](double* x, double* y, std::size_t n, double* grad) {
            check_vec(sigma, n, "sigma");
            if (sigma.empty()) {
                const std::vector<double> ones(n, 1.0);
                return standardised_euclidean_grad(x, y, n, ones.data(), grad);
            }
            return standardised_euclidean_grad(x, y, n, sigma.data(), grad);
        };
    }
    if (name == "wminkowski" || name == "weighted_minkowski") {
        const std::vector<double> w = kwds.v;
        const double p = kwds.p;
        return [w, p](double* x, double* y, std::size_t n, double* grad) {
            check_vec(w, n, "w");
            if (w.empty()) {
                const std::vector<double> ones(n, 1.0);
                return weighted_minkowski_grad(x, y, n, ones.data(), p, grad);
            }
            return weighted_minkowski_grad(x, y, n, w.data(), p, grad);
        };
    }
    if (name == "mahalanobis") {
        const std::vector<double> vinv = kwds.vinv;
        return [vinv](double* x, double* y, std::size_t n, double* grad) {
            check_mat(vinv, n);
            if (vinv.empty()) {
                std::vector<double> eye(n * n, 0.0);
                for (std::size_t i = 0; i < n; ++i) eye[i * n + i] = 1.0;
                return mahalanobis_grad(x, y, n, eye.data(), grad);
            }
            return mahalanobis_grad(x, y, n, vinv.data(), grad);
        };
    }
    if (name == "symmetric_kl") {
        const double z = kwds.z;
        return [z](double* x, double* y, std::size_t n, double* grad) {
            return symmetric_kl_grad(x, y, n, grad, z);
        };
    }
    const auto& reg = named_distances_with_gradients();
    auto it = reg.find(name);
    if (it == reg.end()) {
        throw std::invalid_argument("make_metric_grad: unknown metric '" +
                                    name + "'");
    }
    return it->second;
}

const std::unordered_map<std::string, GradientFn>&
named_distances_with_gradients() {
    static const std::unordered_map<std::string, GradientFn> registry = [] {
        std::unordered_map<std::string, GradientFn> m;
        auto wrap = [](double (*f)(const double*, const double*, std::size_t,
                                   double*)) -> GradientFn {
            return [f](double* x, double* y, std::size_t n, double* grad) {
                return f(x, y, n, grad);
            };
        };
        // general minkowski distances
        m["euclidean"] = wrap(euclidean_grad);
        m["l2"] = wrap(euclidean_grad);
        m["manhattan"] = wrap(manhattan_grad);
        m["taxicab"] = wrap(manhattan_grad);
        m["l1"] = wrap(manhattan_grad);
        m["chebyshev"] = wrap(chebyshev_grad);
        m["linfinity"] = wrap(chebyshev_grad);
        m["linfty"] = wrap(chebyshev_grad);
        m["linf"] = wrap(chebyshev_grad);
        m["minkowski"] = make_metric_grad("minkowski", MetricKwds{});
        // standardised/weighted distances (upstream defaults)
        m["seuclidean"] = make_metric_grad("seuclidean", MetricKwds{});
        m["standardised_euclidean"] = m["seuclidean"];
        m["wminkowski"] = make_metric_grad("wminkowski", MetricKwds{});
        m["weighted_minkowski"] = m["wminkowski"];
        m["mahalanobis"] = make_metric_grad("mahalanobis", MetricKwds{});
        // other distances
        m["canberra"] = wrap(canberra_grad);
        m["cosine"] = wrap(cosine_grad);
        m["correlation"] = wrap(correlation_grad);
        m["hellinger"] = wrap(hellinger_grad);
        m["softmax_hellinger"] = wrap(softmax_hellinger_grad);
        m["haversine"] = wrap(haversine_grad);
        m["braycurtis"] = wrap(bray_curtis_grad);
        m["symmetric_kl"] = [](double* x, double* y, std::size_t n,
                               double* grad) {
            return symmetric_kl_grad(x, y, n, grad);
        };
        // special embeddings
        m["spherical_gaussian_energy"] = wrap(spherical_gaussian_energy_grad);
        m["diagonal_gaussian_energy"] = wrap(diagonal_gaussian_energy_grad);
        m["gaussian_energy"] = gaussian_energy_grad;
        m["hyperboloid"] = wrap(hyperboloid_grad);
        return m;
    }();
    return registry;
}
// ----------------------------------------------------------------------------

const std::unordered_map<std::string, DistanceFn>& named_distances() {
    static const std::unordered_map<std::string, DistanceFn> registry = [] {
        std::unordered_map<std::string, DistanceFn> m;
        auto mink2 = [](const double* x, const double* y, std::size_t n) {
            return minkowski(x, y, n, 2.0);  // upstream default p=2
        };
        // general minkowski distances (aliases as in upstream named_distances)
        m["euclidean"] = euclidean;
        m["l2"] = euclidean;
        m["sqeuclidean"] = squared_euclidean;
        m["manhattan"] = manhattan;
        m["taxicab"] = manhattan;
        m["l1"] = manhattan;
        m["cityblock"] = manhattan;
        m["chebyshev"] = chebyshev;
        m["linfinity"] = chebyshev;
        m["linfty"] = chebyshev;
        m["linf"] = chebyshev;
        m["minkowski"] = mink2;
        // --- metrics-full additions -----------------------------------------
        m["poincare"] = poincare;
        // Standardised/weighted distances registered with their upstream
        // defaults (ones / identity); use make_metric for explicit kwds.
        m["seuclidean"] = make_metric("seuclidean", MetricKwds{});
        m["standardised_euclidean"] = m["seuclidean"];
        m["wminkowski"] = make_metric("wminkowski", MetricKwds{});
        m["weighted_minkowski"] = m["wminkowski"];
        m["mahalanobis"] = make_metric("mahalanobis", MetricKwds{});
        m["haversine"] = haversine;
        m["ll_dirichlet"] = ll_dirichlet;
        m["symmetric_kl"] = [](const double* x, const double* y,
                               std::size_t n) {
            return symmetric_kl(x, y, n);
        };
        m["softmax_hellinger"] = softmax_hellinger;
        m["matching"] = matching;
        m["kulsinski"] = kulsinski;
        m["rogerstanimoto"] = rogers_tanimoto;
        m["russellrao"] = russellrao;
        m["sokalsneath"] = sokal_sneath;
        m["sokalmichener"] = sokal_michener;
        m["yule"] = yule;
        // --------------------------------------------------------------------
        // other distances
        m["canberra"] = canberra;
        m["cosine"] = cosine;
        m["correlation"] = correlation;
        m["hellinger"] = hellinger;
        m["braycurtis"] = bray_curtis;
        // binary distances
        m["hamming"] = hamming;
        m["jaccard"] = jaccard;
        m["dice"] = dice;
        return m;
    }();
    return registry;
}

// --- densmap additions ------------------------------------------------------
// float32-faithful numba-instantiation kernels (see header). Verified against
// the reference environment: with float32 rows, numba computes the elementwise
// difference (and its abs / square) in float32 and only promotes to float64
// where a float64 operand enters the expression (the `result = 0.0`
// accumulator, a Python-float exponent p, or a float64 sigma array).

DistanceF32Fn make_metric_f32(const std::string& name, const MetricKwds& kwds,
                              bool* f32_faithful) {
    if (f32_faithful != nullptr) *f32_faithful = true;
    if (name == "euclidean" || name == "l2") {
        return [](const float* x, const float* y, std::size_t n) {
            double result = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const float d = x[i] - y[i];
                const float sq = d * d;  // (x[i] - y[i]) ** 2 stays f4
                result += static_cast<double>(sq);
            }
            return static_cast<float>(std::sqrt(result));
        };
    }
    if (name == "manhattan" || name == "taxicab" || name == "l1" ||
        name == "cityblock") {
        return [](const float* x, const float* y, std::size_t n) {
            double result = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const float ad = std::fabs(x[i] - y[i]);  // f4 abs
                result += static_cast<double>(ad);
            }
            return static_cast<float>(result);
        };
    }
    if (name == "minkowski") {
        const double p = kwds.p;
        return [p](const float* x, const float* y, std::size_t n) {
            double result = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const float ad = std::fabs(x[i] - y[i]);  // f4 diff + abs
                result += std::pow(static_cast<double>(ad), p);  // f4 ** f8
            }
            return static_cast<float>(std::pow(result, 1.0 / p));
        };
    }
    if (name == "seuclidean" || name == "standardised_euclidean") {
        const std::vector<double> sigma = kwds.v;
        return [sigma](const float* x, const float* y, std::size_t n) {
            if (!sigma.empty() && sigma.size() != n) {
                throw std::invalid_argument(
                    "make_metric_f32: sigma must have the row dimension");
            }
            double result = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const float d = x[i] - y[i];
                const float sq = d * d;  // (x[i] - y[i]) ** 2 stays f4
                const double s = sigma.empty() ? 1.0 : sigma[i];
                result += static_cast<double>(sq) / s;  // f8 divide/accumulate
            }
            return static_cast<float>(std::sqrt(result));
        };
    }
    // --- hardening additions ------------------------------------------------
    // hellinger: numba's f4 instantiation keeps np.sqrt(x[i] * y[i]) in
    // float32 (f4 * f4 product, f4 sqrt) while the accumulators and the
    // final expression are float64. Verified element-exact against
    // pairwise_special_metric on the reference environment (sparse densify
    // fallback + dense small-data route).
    if (name == "hellinger") {
        return [](const float* x, const float* y, std::size_t n) {
            double result = 0.0;
            double l1_norm_x = 0.0;
            double l1_norm_y = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const float prod = x[i] * y[i];  // f4 * f4
                const float sq = std::sqrt(prod);  // np.sqrt(f4) -> f4
                result += static_cast<double>(sq);
                l1_norm_x += static_cast<double>(x[i]);
                l1_norm_y += static_cast<double>(y[i]);
            }
            if (l1_norm_x == 0.0 && l1_norm_y == 0.0) return 0.0f;
            if (l1_norm_x == 0.0 || l1_norm_y == 0.0) return 1.0f;
            return static_cast<float>(
                std::sqrt(1.0 - result / std::sqrt(l1_norm_x * l1_norm_y)));
        };
    }
    // ------------------------------------------------------------------------
    // Adapter: widen the float32 rows to float64 and evaluate the
    // double-precision kernel (documented last-ulp deviation vs numba's f4
    // instantiation, exactly like pairwise_distances_sklearn's registry path).
    if (f32_faithful != nullptr) *f32_faithful = false;
    DistanceFn fn = make_metric(name, kwds);
    return [fn](const float* x, const float* y, std::size_t n) {
        std::vector<double> xb(n), yb(n);
        for (std::size_t i = 0; i < n; ++i) {
            xb[i] = static_cast<double>(x[i]);
            yb[i] = static_cast<double>(y[i]);
        }
        return static_cast<float>(fn(xb.data(), yb.data(), n));
    };
}
// ----------------------------------------------------------------------------

}  // namespace umap
