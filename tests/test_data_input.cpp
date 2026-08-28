// Port of reference/umap/umap/tests/test_data_input.py to doctest.
//
// Upstream fits with a numba callable metric (nan_dist) that ignores the data
// and returns 0; callable metrics are not portable, so the C++ port runs the
// euclidean metric with init="random" and n_epochs=11 -- the semantics under
// test (the sklearn check_array `ensure_all_finite` gate on fit /
// fit_transform / transform) fire before any distance computation either way.
// update() / inverse_transform() legs are skipped (not ported). See
// tests/PORTING_NOTES.md.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "doctest.h"
#include "umap/umap.hpp"

namespace {

using umap::UMAP;
using umap::UMAPParams;

// all_finite_data fixture: np.arange(100.0).reshape(25, 4).
std::vector<float> all_finite_data() {
    std::vector<float> d(100);
    for (int i = 0; i < 100; ++i) d[i] = static_cast<float>(i);
    return d;
}

std::vector<float> with_inf() {  // inf_data: row 0 = inf
    std::vector<float> d = all_finite_data();
    for (int c = 0; c < 4; ++c) d[c] = std::numeric_limits<float>::infinity();
    return d;
}

std::vector<float> with_nan() {  // nan_data: row 0 = nan
    std::vector<float> d = all_finite_data();
    for (int c = 0; c < 4; ++c) d[c] = std::numeric_limits<float>::quiet_NaN();
    return d;
}

std::vector<float> with_inf_nan() {  // inf_nan_data: row 0 = nan, row 1 = inf
    std::vector<float> d = all_finite_data();
    for (int c = 0; c < 4; ++c) {
        d[c] = std::numeric_limits<float>::quiet_NaN();
        d[4 + c] = std::numeric_limits<float>::infinity();
    }
    return d;
}

// call_umap_functions: fit_transform, fit and transform under one setting.
void call_umap_functions(const std::vector<float>& data,
                         const std::string& ensure_all_finite) {
    UMAPParams p;
    p.init = "random";
    p.n_epochs = 11;
    p.ensure_all_finite = ensure_all_finite;
    {
        UMAP u(p);
        (void)u.fit_transform(data.data(), 25, 4);
    }
    UMAP u(p);
    u.fit(data.data(), 25, 4);
    (void)u.transform(data.data(), 25);
}

}  // namespace

TEST_SUITE("test_data_input.py") {

TEST_CASE("test_check_input_data") {
    // Correct data input is accepted ("true" is upstream's None/True default).
    call_umap_functions(all_finite_data(), "true");

    call_umap_functions(with_nan(), "allow-nan");
    call_umap_functions(all_finite_data(), "allow-nan");

    call_umap_functions(with_inf(), "false");
    call_umap_functions(with_inf_nan(), "false");
    call_umap_functions(with_nan(), "false");
    call_umap_functions(all_finite_data(), "false");

    // Illegal data raises (upstream ValueError -> std::invalid_argument).
    CHECK_THROWS_AS(call_umap_functions(with_nan(), "true"),
                    std::invalid_argument);
    CHECK_THROWS_AS(call_umap_functions(with_inf(), "true"),
                    std::invalid_argument);
    CHECK_THROWS_AS(call_umap_functions(with_inf_nan(), "true"),
                    std::invalid_argument);

    CHECK_THROWS_AS(call_umap_functions(with_inf(), "allow-nan"),
                    std::invalid_argument);
    CHECK_THROWS_AS(call_umap_functions(with_inf_nan(), "allow-nan"),
                    std::invalid_argument);
}

}  // TEST_SUITE
