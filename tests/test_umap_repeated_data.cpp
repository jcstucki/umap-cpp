// Port of reference/umap/umap/tests/test_umap_repeated_data.py to doctest
// (upstream test names preserved). spatial_repeats / binary_repeats are the
// conftest fixtures baked in tests/upstream_data.hpp; repetition_dense is the
// literal conftest array. The sparse variants build conftest's
// sparse_spatial_data_repeats = csr(spatial_repeats * binary_repeats) and
// sparse_binary_data_repeats = csr(binary_repeats) from the baked fixtures.
#include <cstdint>
#include <vector>

#include "doctest.h"
#include "umap/umap.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"

namespace {

using umap::UMAP;
using umap::UMAPParams;

constexpr int64_t kRepN = 14, kRepDim = 20;

// assert np.unique(model.embedding_[0:2], axis=0).shape[0] == 1
bool first_two_rows_identical(const UMAP& model) {
    const std::vector<float>& e = model.embedding();
    const int d = model.params().n_components;
    for (int c = 0; c < d; ++c)
        if (e[c] != e[d + c]) return false;
    return true;
}

// conftest.py::sparse_spatial_data_repeats = csr(spatial_repeats *
// binary_repeats).
testutil::Csr sparse_spatial_repeats_csr() {
    std::vector<float> dense(static_cast<std::size_t>(kRepN) * kRepDim);
    for (std::size_t i = 0; i < dense.size(); ++i)
        dense[i] = static_cast<float>(updata::spatial_repeats[i] *
                                      updata::binary_repeats[i]);
    return testutil::dense_to_csr(dense.data(), kRepN, kRepDim);
}

// conftest.py::sparse_binary_data_repeats = csr(binary_repeats).
testutil::Csr sparse_binary_repeats_csr() {
    const std::vector<float> dense = testutil::to_f32(
        updata::binary_repeats, static_cast<std::size_t>(kRepN) * kRepDim);
    return testutil::dense_to_csr(dense.data(), kRepN, kRepDim);
}

UMAP fit_sparse(const testutil::Csr& csr, bool force_approx, int n_epochs) {
    UMAPParams p;
    p.n_neighbors = 3;
    p.unique = true;
    p.force_approximation_algorithm = force_approx;
    p.n_epochs = n_epochs;
    UMAP model(p);
    model.fit(csr.indptr.data(), csr.indices.data(), csr.data.data(), kRepN,
              kRepDim);
    return model;
}

}  // namespace

TEST_SUITE("test_umap_repeated_data.py") {

TEST_CASE("test_repeated_points_large_sparse_spatial") {
    const UMAP model = fit_sparse(sparse_spatial_repeats_csr(), true, 20);
    CHECK(first_two_rows_identical(model));
}

TEST_CASE("test_repeated_points_small_sparse_spatial") {
    const UMAP model = fit_sparse(sparse_spatial_repeats_csr(), false, 20);
    CHECK(first_two_rows_identical(model));
}

TEST_CASE("test_repeated_points_large_dense_spatial") {
    const std::vector<float> data = testutil::to_f32(
        updata::spatial_repeats, static_cast<std::size_t>(kRepN) * kRepDim);
    UMAPParams p;
    p.n_neighbors = 3;
    p.unique = true;
    p.force_approximation_algorithm = true;
    p.n_epochs = 50;
    UMAP model(p);
    model.fit(data.data(), kRepN, kRepDim);
    CHECK(first_two_rows_identical(model));
}

TEST_CASE("test_repeated_points_small_dense_spatial") {
    const std::vector<float> data = testutil::to_f32(
        updata::spatial_repeats, static_cast<std::size_t>(kRepN) * kRepDim);
    UMAPParams p;
    p.n_neighbors = 3;
    p.unique = true;
    p.n_epochs = 20;
    UMAP model(p);
    model.fit(data.data(), kRepN, kRepDim);
    CHECK(first_two_rows_identical(model));
}

TEST_CASE("test_repeated_points_large_sparse_binary") {
    const UMAP model = fit_sparse(sparse_binary_repeats_csr(), true, 50);
    CHECK(first_two_rows_identical(model));
}

TEST_CASE("test_repeated_points_small_sparse_binary") {
    const UMAP model = fit_sparse(sparse_binary_repeats_csr(), false, 20);
    CHECK(first_two_rows_identical(model));
}

TEST_CASE("test_repeated_points_large_dense_binary") {
    const std::vector<float> data = testutil::to_f32(
        updata::binary_repeats, static_cast<std::size_t>(kRepN) * kRepDim);
    UMAPParams p;
    p.n_neighbors = 3;
    p.unique = true;
    p.force_approximation_algorithm = true;
    p.n_epochs = 20;
    UMAP model(p);
    model.fit(data.data(), kRepN, kRepDim);
    CHECK(first_two_rows_identical(model));
}

TEST_CASE("test_repeated_points_small_dense_binary") {
    const std::vector<float> data = testutil::to_f32(
        updata::binary_repeats, static_cast<std::size_t>(kRepN) * kRepDim);
    // assert np.unique(binary_repeats[0:2], axis=0).shape[0] == 1
    bool rows01_equal = true;
    for (int64_t c = 0; c < kRepDim; ++c)
        if (data[c] != data[kRepDim + c]) rows01_equal = false;
    CHECK(rows01_equal);
    UMAPParams p;
    p.n_neighbors = 3;
    p.unique = true;
    p.n_epochs = 20;
    UMAP model(p);
    model.fit(data.data(), kRepN, kRepDim);
    CHECK(first_two_rows_identical(model));
}

TEST_CASE("test_repeated_points_large_n") {
    // conftest::repetition_dense (9 x 4, 4 distinct rows).
    const float rep[9 * 4] = {
        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8,
        5, 6, 7, 8, 1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 2, 1,
    };
    UMAPParams p;
    p.n_neighbors = 5;
    p.unique = true;
    p.n_epochs = 20;
    UMAP model(p);
    model.fit(rep, 9, 4);
    CHECK(model.effective_n_neighbors() == 3);
}

}  // TEST_SUITE
