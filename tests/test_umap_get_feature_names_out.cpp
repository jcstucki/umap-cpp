// Port of reference/umap/umap/tests/test_umap_get_feature_names_out.py to
// doctest. X = make_classification(30, 10, random_state=42) is baked as
// updata::classif30.
//
// Adaptations (see tests/PORTING_NOTES.md):
//   * test_get_feature_names_out passes input_features upstream and asserts
//     they are ignored; the C++ API takes no such argument, so the case
//     reduces to the same assertion as the default variant.
//   * test_get_feature_names_out_featureunion needs sklearn's Pipeline /
//     FeatureUnion machinery -- skipped.
#include <string>
#include <vector>

#include "doctest.h"
#include "umap/umap.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"

namespace {

using umap::UMAP;
using umap::UMAPParams;

UMAP fitted_umap(int n_components) {
    const std::vector<float> X = testutil::to_f32(updata::classif30, 30 * 10);
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.n_epochs = 200;
    p.random_state = 42;
    p.n_components = n_components;
    UMAP u(p);
    u.fit(X.data(), 30, 10);
    return u;
}

}  // namespace

TEST_SUITE("test_umap_get_feature_names_out.py") {

TEST_CASE("test_get_feature_names_out") {
    // input_features must be ignored (no such argument exists in C++).
    const UMAP u = fitted_umap(3);
    const std::vector<std::string> expected = {"umap0", "umap1", "umap2"};
    CHECK(u.get_feature_names_out() == expected);
}

TEST_CASE("test_get_feature_names_out_default") {
    const UMAP u = fitted_umap(3);
    const std::vector<std::string> expected = {"umap0", "umap1", "umap2"};
    CHECK(u.get_feature_names_out() == expected);
}

TEST_CASE("test_get_feature_names_out_multicomponent") {
    const UMAP u = fitted_umap(9);
    const std::vector<std::string> result = u.get_feature_names_out();
    CHECK(result.size() == 9);
    for (int i = 0; i < 9; ++i) {
        CAPTURE(i);
        CHECK(result[i] == "umap" + std::to_string(i));
    }
}

TEST_CASE("test_get_feature_names_out_featureunion" * doctest::skip()) {}

}  // TEST_SUITE
