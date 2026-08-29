// Port of reference/umap/umap/tests/test_composite_models.py to doctest
// (upstream test names, parameters and thresholds preserved verbatim), plus
// golden sub-checks against scripts/gen_golden_composite.py:
//   * the combined graph_ of every operator (a * b intersection, a + b union,
//     a - b contrast) is compared BIT-EXACT against the Python reference --
//     the operand fits run on the small-data branch where the fuzzy sets are
//     bit-exact, and the set-operation machinery (general_simplicial_set_
//     intersection/union + reset_local_connectivity) is bit-exact too. The
//     goldens include the explicit zeros upstream's epoch weight filter
//     writes back through the shared tocoo() data array;
//   * the composite embedding stage: n_epochs resolution, filtered head/tail
//     edge lists and epochs_per_sample bit-exact, the SGD rng_state via
//     RandomState(42) stream replication, and the post-rescale spectral init
//     up to the per-column eigenvector sign (Spectra vs ARPACK);
//   * embedding quality guards vs the Python composites' trustworthiness.
//
// Skips (mirrored from upstream):
//   * test_composite_trustworthiness_random_init -- @pytest.mark.skip
//     upstream ("Marked as Skipped test").
// Adaptations:
//   * model2's init=model1.embedding_ becomes init_array = model1.embedding()
//     (the C++ ndarray-init seam). The *graphs* being composed are
//     init-independent, so the golden graph comparison holds even though the
//     operand embeddings differ from Python at eigensolver level.
//   * upstream's ValueError for mismatched sample counts maps to
//     std::invalid_argument.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "doctest.h"
#include "golden_composite.hpp"
#include "trustworthiness.hpp"
#include "umap/layouts.hpp"
#include "umap/nn_descent.hpp"
#include "umap/umap.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"

namespace {

using umap::SparseMatrix;
using umap::UMAP;
using umap::UMAPParams;

// The combined graph_ as stored on the composite model (CSR including the
// filter's explicit zeros), value-exact against the Python golden.
void check_graph_bitexact(const SparseMatrix& g, int64_t n, int64_t nnz,
                          const int32_t* indptr, const int32_t* indices,
                          const float* data) {
    REQUIRE(static_cast<int64_t>(g.n_rows) == n);
    REQUIRE(static_cast<int64_t>(g.data.size()) == nnz);
    CHECK(std::memcmp(g.indptr.data(), indptr,
                      static_cast<std::size_t>(n + 1) * sizeof(int32_t)) == 0);
    CHECK(std::memcmp(g.indices.data(), indices,
                      static_cast<std::size_t>(nnz) * sizeof(int32_t)) == 0);
    std::size_t mismatch = 0;
    for (int64_t i = 0; i < nnz; ++i)
        if (g.data[static_cast<std::size_t>(i)] != data[i]) ++mismatch;
    CHECK(mismatch == 0);
}

// The composite embedding stage vs the captured Python
// simplicial_set_embedding inputs: rng_state (RandomState(42) stream
// replication), resolved n_epochs, filtered head/tail, epochs_per_sample
// (all bit-exact) and the post-rescale init up to per-column sign.
void check_embed_stage(const UMAP& model, int requested_epochs,
                       int golden_n_epochs, int init_is_spectral,
                       const int64_t* golden_rng, int64_t golden_n_edges,
                       const int32_t* golden_head, const int32_t* golden_tail,
                       const double* golden_eps, const float* golden_init) {
    const int64_t n = model.n_samples();

    // RandomState(42): spectral init consumes the discarded eigsh warm start
    // normal(size=(n, 3)) + noisy_scale_coords normal(size=(n, 2)); random
    // init consumes uniform(-10, 10, (n, 2)); then randint -> rng_state.
    umap::nnd::NumpyRandomState rs(42);
    if (init_is_spectral) {
        for (int64_t i = 0; i < n * 3; ++i) (void)rs.gauss();
        for (int64_t i = 0; i < n * 2; ++i) (void)rs.normal(0.0, 0.0001);
    } else {
        for (int64_t i = 0; i < n * 2; ++i) (void)rs.uniform(-10.0, 10.0);
    }
    const auto r3 = rs.rng_state3();
    for (int c = 0; c < 3; ++c) CHECK(r3[c] == golden_rng[c]);

    SparseMatrix g = model.graph();
    const int n_epochs =
        umap::resolve_epochs_and_filter_graph(g, requested_epochs);
    CHECK(n_epochs == golden_n_epochs);
    std::vector<int32_t> head, tail;
    for (int32_t row = 0; row < g.n_rows; ++row) {
        for (int32_t k = g.indptr[row]; k < g.indptr[row + 1]; ++k) {
            head.push_back(row);
            tail.push_back(g.indices[k]);
        }
    }
    REQUIRE(static_cast<int64_t>(head.size()) == golden_n_edges);
    CHECK(std::memcmp(head.data(), golden_head,
                      head.size() * sizeof(int32_t)) == 0);
    CHECK(std::memcmp(tail.data(), golden_tail,
                      tail.size() * sizeof(int32_t)) == 0);
    // The operators pass np.int64 n_epochs, so epochs_per_sample uses the
    // int64*f32 -> f64 promotion flavour.
    const std::vector<double> eps = umap::make_epochs_per_sample_i64(
        g.data.data(), g.data.size(), n_epochs);
    double eps_diff = 0.0;
    for (std::size_t i = 0; i < eps.size(); ++i)
        eps_diff = std::max(eps_diff, std::abs(eps[i] - golden_eps[i]));
    CHECK(eps_diff == 0.0);

    // Post-rescale init: exact for random init; up to the per-column
    // eigenvector sign (x -> 10 - x in the rescaled space) for spectral.
    std::vector<float> init = model.fit_init();
    const int dim = 2;
    for (int d = 0; d < dim; ++d) {
        float mn = init[static_cast<std::size_t>(d)];
        float mx = mn;
        for (int64_t i = 1; i < n; ++i) {
            mn = std::min(mn, init[static_cast<std::size_t>(i * dim + d)]);
            mx = std::max(mx, init[static_cast<std::size_t>(i * dim + d)]);
        }
        for (int64_t i = 0; i < n; ++i) {
            float& v = init[static_cast<std::size_t>(i * dim + d)];
            v = (10.0f * (v - mn)) / (mx - mn);
        }
    }
    double init_err = 0.0;
    for (int d = 0; d < dim; ++d) {
        double direct = 0.0, flipped = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double o = init[static_cast<std::size_t>(i * dim + d)];
            const double gv = golden_init[i * dim + d];
            direct = std::max(direct, std::abs(o - gv));
            flipped = std::max(flipped, std::abs((10.0 - o) - gv));
        }
        init_err = std::max(init_err, std::min(direct, flipped));
    }
    CHECK(init_err < 1e-2);
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// iris.data column pair as float32 (150 x 2, row-major).
std::vector<float> iris_cols(int c0, int c1) {
    std::vector<float> out(150 * 2);
    for (int i = 0; i < 150; ++i) {
        out[static_cast<std::size_t>(i) * 2 + 0] =
            static_cast<float>(updata::iris_data[i * 4 + c0]);
        out[static_cast<std::size_t>(i) * 2 + 1] =
            static_cast<float>(updata::iris_data[i * 4 + c1]);
    }
    return out;
}

// The iris pair every iris case uses: UMAP(n_neighbors=10, min_dist=0.01,
// random_state=42, n_epochs=100) on iris.data[:, :2] / iris.data[:, 2:].
UMAPParams iris_pair_params() {
    UMAPParams p;
    p.n_neighbors = 10;
    p.min_dist = 0.01;
    p.random_state = 42;
    p.n_epochs = 100;
    return p;
}

}  // namespace

TEST_SUITE("test_composite_models.py") {

TEST_CASE("test_composite_trustworthiness") {
    namespace gc = golden_composite;
    // data = nn_data[:50]
    const std::vector<float> data = testutil::to_f32(updata::nn_data, 50 * 5);

    UMAPParams p1;
    p1.n_neighbors = 10;
    p1.min_dist = 0.01;
    p1.random_state = 42;
    p1.n_epochs = 50;
    UMAP model1(p1);
    model1.fit(data.data(), 50, 5);

    UMAPParams p2 = p1;
    p2.n_neighbors = 30;
    p2.init_array = model1.embedding();  // init=model1.embedding_
    UMAP model2(p2);
    model2.fit(data.data(), 50, 5);

    const UMAP model3 = model1 * model2;
    double trust = testutil::trustworthiness(data.data(), 50, 5,
                                             model3.embedding().data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.80);
    // Supplementary quality guard vs the Python composite (the upstream
    // threshold above is the ported assertion). The composite SGD amplifies
    // the eigensolver-level init difference, hence the wider margin.
    CHECK(trust >= gc::nn_mul_trust - 0.05);

    const UMAP model4 = model1 + model2;
    trust = testutil::trustworthiness(data.data(), 50, 5,
                                      model4.embedding().data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.80);
    CHECK(trust >= gc::nn_add_trust - 0.05);

    // Golden sub-checks: combined graphs bit-exact vs Python, and the
    // composite embedding stage (rng stream / edges / eps / init).
    check_graph_bitexact(model3.graph(), gc::nn_mul_n, gc::nn_mul_nnz,
                         gc::nn_mul_indptr, gc::nn_mul_indices,
                         gc::nn_mul_data);
    check_embed_stage(model3, /*requested_epochs=*/50, gc::nn_mul_n_epochs,
                      gc::nn_mul_init_is_spectral, gc::nn_mul_rng_state,
                      gc::nn_mul_n_edges, gc::nn_mul_head, gc::nn_mul_tail,
                      gc::nn_mul_eps, gc::nn_mul_init);
    check_graph_bitexact(model4.graph(), gc::nn_add_n, gc::nn_add_nnz,
                         gc::nn_add_indptr, gc::nn_add_indices,
                         gc::nn_add_data);
    check_embed_stage(model4, /*requested_epochs=*/50, gc::nn_add_n_epochs,
                      gc::nn_add_init_is_spectral, gc::nn_add_rng_state,
                      gc::nn_add_n_edges, gc::nn_add_head, gc::nn_add_tail,
                      gc::nn_add_eps, gc::nn_add_init);

    // with pytest.raises(ValueError): model1 <op> iris_model
    // (iris_model fixture: UMAP(n_neighbors=10, min_dist=0.01,
    // random_state=42).fit(iris.data) -- 150 samples vs model1's 50).
    UMAPParams pi;
    pi.n_neighbors = 10;
    pi.min_dist = 0.01;
    pi.random_state = 42;
    UMAP iris_model(pi);
    const std::vector<float> iris_f32 =
        testutil::to_f32(updata::iris_data, 150 * 4);
    iris_model.fit(iris_f32.data(), 150, 4);

    CHECK_THROWS_AS((void)(model1 + iris_model), std::invalid_argument);
    CHECK_THROWS_AS((void)(model1 * iris_model), std::invalid_argument);
    CHECK_THROWS_AS((void)(model1 - iris_model), std::invalid_argument);

    // Unfitted operands raise too (upstream check_is_fitted NotFittedError).
    UMAP unfitted(p1);
    CHECK_THROWS_AS((void)(model1 * unfitted), std::invalid_argument);
}

TEST_CASE("test_composite_trustworthiness_random_init" * doctest::skip()) {}

TEST_CASE("test_composite_trustworthiness_on_iris") {
    namespace gc = golden_composite;
    const std::vector<float> left = iris_cols(0, 1);    // iris.data[:, :2]
    const std::vector<float> right = iris_cols(2, 3);   // iris.data[:, 2:]
    const std::vector<float> full =
        testutil::to_f32(updata::iris_data, 150 * 4);

    UMAP iris_model1(iris_pair_params());
    iris_model1.fit(left.data(), 150, 2);
    UMAP iris_model2(iris_pair_params());
    iris_model2.fit(right.data(), 150, 2);

    const UMAP added = iris_model1 + iris_model2;
    CHECK(all_finite(added.embedding()));
    double trust = testutil::trustworthiness(full.data(), 150, 4,
                                             added.embedding().data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.82);
    CHECK(trust >= gc::iris_add_trust - 0.05);

    const UMAP multiplied = iris_model1 * iris_model2;
    CHECK(all_finite(multiplied.embedding()));
    trust = testutil::trustworthiness(full.data(), 150, 4,
                                      multiplied.embedding().data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.82);
    CHECK(trust >= gc::iris_mul_trust - 0.05);

    // Golden sub-checks.
    check_graph_bitexact(added.graph(), gc::iris_add_n, gc::iris_add_nnz,
                         gc::iris_add_indptr, gc::iris_add_indices,
                         gc::iris_add_data);
    check_embed_stage(added, /*requested_epochs=*/100, gc::iris_add_n_epochs,
                      gc::iris_add_init_is_spectral, gc::iris_add_rng_state,
                      gc::iris_add_n_edges, gc::iris_add_head,
                      gc::iris_add_tail, gc::iris_add_eps, gc::iris_add_init);
    check_graph_bitexact(multiplied.graph(), gc::iris_mul_n, gc::iris_mul_nnz,
                         gc::iris_mul_indptr, gc::iris_mul_indices,
                         gc::iris_mul_data);
    // (Only the rng_state was captured for the * and - iris composites.)
    {
        umap::nnd::NumpyRandomState rs(42);
        REQUIRE(gc::iris_mul_init_is_spectral == 1);
        for (int64_t i = 0; i < 150 * 3; ++i) (void)rs.gauss();
        for (int64_t i = 0; i < 150 * 2; ++i) (void)rs.normal(0.0, 0.0001);
        const auto r3 = rs.rng_state3();
        for (int c = 0; c < 3; ++c) CHECK(r3[c] == gc::iris_mul_rng_state[c]);
    }
}

TEST_CASE("test_contrastive_trustworthiness_on_iris") {
    namespace gc = golden_composite;
    const std::vector<float> left = iris_cols(0, 1);
    const std::vector<float> right = iris_cols(2, 3);
    const std::vector<float> full =
        testutil::to_f32(updata::iris_data, 150 * 4);

    UMAP iris_model1(iris_pair_params());
    iris_model1.fit(left.data(), 150, 2);
    UMAP iris_model2(iris_pair_params());
    iris_model2.fit(right.data(), 150, 2);

    const UMAP contrast = iris_model1 - iris_model2;
    CHECK(all_finite(contrast.embedding()));
    const double trust = testutil::trustworthiness(
        full.data(), 150, 4, contrast.embedding().data(), 2, 10);
    CAPTURE(trust);
    CHECK(trust >= 0.75);
    CHECK(trust >= gc::iris_sub_trust - 0.05);

    check_graph_bitexact(contrast.graph(), gc::iris_sub_n, gc::iris_sub_nnz,
                         gc::iris_sub_indptr, gc::iris_sub_indices,
                         gc::iris_sub_data);
}

}  // TEST_SUITE
