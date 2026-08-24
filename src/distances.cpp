#include "umap/distances.hpp"

#include <cmath>

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

}  // namespace umap
