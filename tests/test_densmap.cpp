// Tests for densMAP / output_dens, output_metric wiring and metric_kwds,
// compared end-to-end against goldens generated from the reference Python
// UMAP (tests/golden_densmap.hpp, NUMBA_NUM_THREADS=1, seeded), plus the
// validation-error and transform-behaviour cases the reference raises.
//
// Parity levels (all cases use init="random" + explicit a/b so the
// deterministic paths line up):
//  * dm (densmap + output_dens, small-data branch): bit-exact embedding and
//    rad_orig_/rad_emb_ -- the densMAP epoch init/correction, the f32
//    radii pipeline (NEP 50) and the Accelerate np.dot all reproduce Python.
//  * dd (output_dens only): bit-exact embedding + radii; the embedding is
//    also bitwise identical to a plain fit (output_dens must not perturb the
//    SGD -- the density pass runs afterwards on the same RandomState).
//  * dh (output_metric="haversine"): bit-exact fit AND transform through
//    optimize_layout_generic.
//  * mk/ms (metric_kwds through the small-data pairwise path): bit-exact.
//  * mn (metric_kwds through NN-descent's generic-metric hook): kNN indices
//    identical; distances and embedding checked bit-exact here too (the
//    minkowski kernel's pow() chain came out reproducible, unlike the
//    fastmath euclidean kernels).
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_densmap.hpp"
#include "umap/umap.hpp"

using umap::UMAP;
using umap::UMAPParams;

namespace {

std::size_t count_mismatch(const std::vector<float>& a, const float* b) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++c;
    return c;
}

UMAPParams base_params() {
    UMAPParams p;
    p.init = "random";
    p.random_state = 42;
    p.a = golden_densmap::g_a;
    p.b = golden_densmap::g_b;
    return p;
}

}  // namespace

TEST_CASE("densmap fit: bit-exact embedding and radii (small data)") {
    using namespace golden_densmap;
    UMAPParams p = base_params();
    p.densmap = true;
    p.output_dens = true;
    UMAP model(p);
    model.fit(dm_X, dm_n, dm_dim);

    const std::vector<float>& emb = model.embedding();
    REQUIRE(emb.size() == static_cast<std::size_t>(dm_n) * 2);
    CHECK(count_mismatch(emb, dm_embedding) == 0);

    const std::vector<float>& ro = model.rad_orig();
    const std::vector<float>& re = model.rad_emb();
    REQUIRE(ro.size() == static_cast<std::size_t>(dm_n));
    REQUIRE(re.size() == static_cast<std::size_t>(dm_n));
    CHECK(count_mismatch(ro, dm_rad_orig) == 0);
    CHECK(count_mismatch(re, dm_rad_emb) == 0);

    // graph_dists_ is populated for densmap fits (upstream self.graph_dists_).
    CHECK(model.graph_dists().n_rows == dm_n);
    CHECK(model.graph_dists().nnz() > 0);
}

TEST_CASE("densmap transform: hash shortcut works, new data raises") {
    using namespace golden_densmap;
    UMAPParams p = base_params();
    p.densmap = true;
    UMAP model(p);
    model.fit(dm_X, dm_n, dm_dim);

    // Transforming the training data itself hits the input-hash shortcut and
    // returns embedding_ (works even for densMAP, as upstream).
    const std::vector<float> t = model.transform(dm_X, dm_n);
    CHECK(count_mismatch(t, model.embedding().data()) == 0);

    // New data: NotImplementedError upstream.
    std::vector<float> q(dm_X, dm_X + 5 * dm_dim);
    for (float& v : q) v += 1.0f;
    CHECK_THROWS_WITH_AS(
        model.transform(q.data(), 5),
        "Transforming data into an existing embedding not supported "
        "for densMAP.",
        std::runtime_error);
}

TEST_CASE("output_dens only: radii bit-exact, embedding unperturbed") {
    using namespace golden_densmap;
    UMAPParams p = base_params();
    p.output_dens = true;
    UMAP model(p);
    model.fit(dd_X, dd_n, dd_dim);

    CHECK(count_mismatch(model.embedding(), dd_embedding) == 0);
    CHECK(count_mismatch(model.rad_orig(), dd_rad_orig) == 0);
    CHECK(count_mismatch(model.rad_emb(), dd_rad_emb) == 0);

    // The density pass runs after the SGD: a plain fit must give the
    // bitwise-identical embedding (verified in Python by the generator too).
    UMAPParams p2 = base_params();
    UMAP plain(p2);
    plain.fit(dd_X, dd_n, dd_dim);
    CHECK(count_mismatch(plain.embedding(), dd_embedding) == 0);
    CHECK(plain.rad_orig().empty());
    CHECK(plain.graph_dists().nnz() == 0);
}

TEST_CASE("output_metric=haversine: bit-exact fit and transform") {
    using namespace golden_densmap;
    UMAPParams p = base_params();
    p.output_metric = "haversine";
    p.n_components = 2;
    UMAP model(p);
    model.fit(dh_X, dh_n, dh_dim);
    CHECK(count_mismatch(model.embedding(), dh_embedding) == 0);

    const std::vector<float> t = model.transform(dh_queries, dh_nq);
    REQUIRE(t.size() == static_cast<std::size_t>(dh_nq) * 2);
    CHECK(count_mismatch(t, dh_t_embedding) == 0);
}

TEST_CASE("metric_kwds: minkowski p=3 small-data fit + transform bit-exact") {
    using namespace golden_densmap;
    UMAPParams p = base_params();
    p.metric = "minkowski";
    p.metric_kwds.p = 3.0;
    UMAP model(p);
    model.fit(mk_X, mk_n, mk_dim);
    CHECK(count_mismatch(model.embedding(), mk_embedding) == 0);

    const std::vector<float> t = model.transform(mk_queries, mk_nq);
    CHECK(count_mismatch(t, mk_t_embedding) == 0);
}

TEST_CASE("metric_kwds: seuclidean with sigma array bit-exact") {
    using namespace golden_densmap;
    UMAPParams p = base_params();
    p.metric = "seuclidean";
    p.metric_kwds.v.assign(ms_sigma, ms_sigma + ms_dim);
    UMAP model(p);
    model.fit(ms_X, ms_n, ms_dim);
    CHECK(count_mismatch(model.embedding(), ms_embedding) == 0);
}

TEST_CASE("metric_kwds: minkowski p=3 through NN-descent (generic hook)") {
    using namespace golden_densmap;
    UMAPParams p = base_params();
    p.metric = "minkowski";
    p.metric_kwds.p = 3.0;
    p.force_approximation_algorithm = true;
    UMAP model(p);
    model.fit(mn_X, mn_n, mn_dim);

    // kNN graph: indices must be identical to pynndescent's generic-metric
    // run; distances bit-exact (see file header).
    const std::vector<int32_t>& ki = model.knn_indices();
    const std::vector<float>& kd = model.knn_dists();
    REQUIRE(ki.size() == static_cast<std::size_t>(mn_n) * mn_k);
    std::size_t idx_mismatch = 0;
    for (std::size_t i = 0; i < ki.size(); ++i)
        if (ki[i] != mn_knn_indices[i]) ++idx_mismatch;
    CHECK(idx_mismatch == 0);
    CHECK(count_mismatch(kd, mn_knn_dists) == 0);

    CHECK(count_mismatch(model.embedding(), mn_embedding) == 0);

    const std::vector<float> t = model.transform(mn_queries, mn_nq);
    CHECK(count_mismatch(t, mn_t_embedding) == 0);
}

TEST_CASE("densmap / output_metric parameter validation") {
    using namespace golden_densmap;
    const auto fit_with = [](UMAPParams p) {
        UMAP model(p);
        model.fit(mn_X, mn_n, mn_dim);
    };

    UMAPParams p = base_params();
    p.densmap = true;
    p.dens_lambda = -1.0;
    CHECK_THROWS_WITH_AS(fit_with(p), "dens_lambda cannot be negative",
                         std::invalid_argument);

    p = base_params();
    p.densmap = true;
    p.dens_frac = 1.5;
    CHECK_THROWS_WITH_AS(fit_with(p), "dens_frac must be between 0.0 and 1.0",
                         std::invalid_argument);

    p = base_params();
    p.dens_var_shift = -0.1;
    CHECK_THROWS_WITH_AS(fit_with(p), "dens_var_shift cannot be negative",
                         std::invalid_argument);

    p = base_params();
    p.densmap = true;
    p.output_metric = "haversine";
    p.n_components = 2;
    CHECK_THROWS_WITH_AS(fit_with(p),
                         "Non-Euclidean output metric not supported for "
                         "densMAP.",
                         std::invalid_argument);

    // output_metric elif chain (note upstream's "cannnot" typo).
    p = base_params();
    p.output_metric = "precomputed";
    CHECK_THROWS_WITH_AS(fit_with(p), "output_metric cannnot be 'precomputed'",
                         std::invalid_argument);

    p = base_params();
    p.output_metric = "hamming";  // named distance without a gradient
    CHECK_THROWS_WITH_AS(fit_with(p),
                         "gradient function is not yet implemented for "
                         "hamming.",
                         std::invalid_argument);

    p = base_params();
    p.output_metric = "not_a_metric";
    CHECK_THROWS_WITH_AS(
        fit_with(p), "output_metric is neither callable nor a recognised string",
        std::invalid_argument);
}

TEST_CASE("densmap default epochs: 500 + 200") {
    // The densmap fit above reproduced the Python embedding bit-exactly with
    // upstream's resolved n_epochs; pin the captured defaults down here.
    static_assert(golden_densmap::dm_fit_n_epochs == 700,
                  "densMAP default epochs must be 500 + 200");
    static_assert(golden_densmap::dd_fit_n_epochs == 500,
                  "output_dens alone must not boost the epoch default");
    CHECK(true);
}
