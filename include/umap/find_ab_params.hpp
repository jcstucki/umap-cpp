// Faithful C++ port of find_ab_params from reference/umap/umap/umap_.py
// (lmcinnes/umap, BSD 3 clause): least-squares fit of a, b for the curve
// f(x) = 1 / (1 + a * x^(2b)) against an offset exponential decay, via
// Levenberg-Marquardt (mirroring scipy.optimize.curve_fit's default LM path
// with initial guess a = b = 1).
#pragma once

namespace umap {

struct ABParams {
    double a;
    double b;
};

ABParams find_ab_params(double spread, double min_dist);

}  // namespace umap
