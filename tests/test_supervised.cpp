// Tests for supervised / semi-supervised fit(X, y): the categorical
// (fast_intersection), discrete ("ordinal", get_discrete_params) and general
// (target fuzzy set + general_simplicial_set_intersection +
// reset_local_connectivity) target branches, compared against goldens from
// the reference Python UMAP (tests/golden_supervised.hpp,
// NUMBA_NUM_THREADS=1, seeded, small-data + init="random" + explicit a/b:
// the fully deterministic path, so graph_, init and embedding are expected
// BIT-EXACT), plus a label-separation behavioral test and validation cases.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "doctest.h"
#include "golden_supervised.hpp"
#include "umap/sparse_matrix.hpp"
#include "umap/supervised.hpp"
#include "umap/umap.hpp"

using umap::SparseMatrix;
using umap::UMAP;
using umap::UMAPParams;

namespace {

UMAPParams base_params() {
    UMAPParams p;
    p.init = "random";
    p.random_state = 42;
    p.a = golden_supervised::sup_a;
    p.b = golden_supervised::sup_b;
    return p;
}

std::size_t count_mismatch(const std::vector<float>& a, const float* b) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++c;
    return c;
}

// The [0,10] per-column rescale simplicial_set_embedding applies to the init
// (fit_init() is captured before it; the golden is captured after it).
std::vector<float> rescale_init(std::vector<float> emb, int64_t n, int dim) {
    for (int d = 0; d < dim; ++d) {
        float mn = emb[d], mx = emb[d];
        for (int64_t i = 1; i < n; ++i) {
            const float v = emb[i * dim + d];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        const float span = mx - mn;
        for (int64_t i = 0; i < n; ++i) {
            float& v = emb[i * dim + d];
            v = (10.0f * (v - mn)) / span;
        }
    }
    return emb;
}

void check_graph_exact(const SparseMatrix& g, const int32_t* indptr,
                       const int32_t* indices, const float* data,
                       int64_t nnz_expected) {
    REQUIRE(static_cast<int64_t>(g.indices.size()) == nnz_expected);
    for (std::size_t i = 0; i < g.indptr.size(); ++i)
        CHECK(g.indptr[i] == indptr[i]);
    std::size_t idx_mismatch = 0, val_mismatch = 0;
    for (std::size_t i = 0; i < g.indices.size(); ++i) {
        if (g.indices[i] != indices[i]) ++idx_mismatch;
        if (g.data[i] != data[i]) ++val_mismatch;
    }
    CHECK(idx_mismatch == 0);
    CHECK(val_mismatch == 0);
}

template <std::size_t NP, std::size_t NI, std::size_t NE>
void run_case(const UMAPParams& params, const float* y,
              const int32_t (&g_indptr)[NP], const int32_t (&g_indices)[NI],
              const float (&g_data)[NI], const float (&g_init)[NE],
              const float (&g_emb)[NE]) {
    UMAP model(params);
    const std::vector<float> emb = model.fit_transform(
        golden_supervised::sX, golden_supervised::sn, golden_supervised::sdim,
        y);
    check_graph_exact(model.graph(), g_indptr, g_indices, g_data,
                      static_cast<int64_t>(NI));
    const std::vector<float> init = rescale_init(
        model.fit_init(), golden_supervised::sn, 2);
    std::size_t init_mismatch = 0;
    for (std::size_t i = 0; i < init.size(); ++i)
        if (init[i] != g_init[i]) ++init_mismatch;
    CHECK(init_mismatch == 0);
    CHECK(count_mismatch(emb, g_emb) == 0);
}

// Copy the golden y (f32-representable label/target values) into the float
// array the C++ API takes.
std::vector<float> yf(const float* y, int64_t n) {
    return std::vector<float>(y, y + n);
}

// Mean intra-class / inter-class centroid-distance separation proxy.
double separation_ratio(const std::vector<float>& emb, const int32_t* labels,
                        int64_t n) {
    double cx[3] = {0, 0, 0}, cy[3] = {0, 0, 0};
    int cnt[3] = {0, 0, 0};
    for (int64_t i = 0; i < n; ++i) {
        const int l = labels[i];
        cx[l] += emb[i * 2];
        cy[l] += emb[i * 2 + 1];
        ++cnt[l];
    }
    for (int c = 0; c < 3; ++c) {
        cx[c] /= cnt[c];
        cy[c] /= cnt[c];
    }
    double intra = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const int l = labels[i];
        intra += std::hypot(emb[i * 2] - cx[l], emb[i * 2 + 1] - cy[l]);
    }
    intra /= static_cast<double>(n);
    double inter = 0.0;
    int pairs = 0;
    for (int c = 0; c < 3; ++c)
        for (int d = c + 1; d < 3; ++d) {
            inter += std::hypot(cx[c] - cx[d], cy[c] - cy[d]);
            ++pairs;
        }
    inter /= pairs;
    return inter / intra;
}

}  // namespace

TEST_CASE("supervised fit: categorical with -1 unknowns (s1, bit-exact)") {
    const std::vector<float> y =
        yf(golden_supervised::s1_y, golden_supervised::sn);
    run_case(base_params(), y.data(), golden_supervised::s1_graph_indptr,
             golden_supervised::s1_graph_indices,
             golden_supervised::s1_graph_data, golden_supervised::s1_fit_init,
             golden_supervised::s1_embedding);
}

TEST_CASE("supervised fit: categorical target_weight=0.8 (s2, bit-exact)") {
    UMAPParams p = base_params();
    p.target_weight = 0.8;
    const std::vector<float> y =
        yf(golden_supervised::s2_y, golden_supervised::sn);
    run_case(p, y.data(), golden_supervised::s2_graph_indptr,
             golden_supervised::s2_graph_indices,
             golden_supervised::s2_graph_data, golden_supervised::s2_fit_init,
             golden_supervised::s2_embedding);
}

TEST_CASE("supervised fit: categorical target_weight=1.0 (s3, bit-exact)") {
    UMAPParams p = base_params();
    p.target_weight = 1.0;  // far_dist = 1e12
    const std::vector<float> y =
        yf(golden_supervised::s3_y, golden_supervised::sn);
    run_case(p, y.data(), golden_supervised::s3_graph_indptr,
             golden_supervised::s3_graph_indices,
             golden_supervised::s3_graph_data, golden_supervised::s3_fit_init,
             golden_supervised::s3_embedding);
}

TEST_CASE("supervised fit: general l2 target, target_n_neighbors=8 (s4)") {
    UMAPParams p = base_params();
    p.target_metric = "l2";
    p.target_n_neighbors = 8;
    const std::vector<float> y =
        yf(golden_supervised::s4_y, golden_supervised::sn);
    run_case(p, y.data(), golden_supervised::s4_graph_indptr,
             golden_supervised::s4_graph_indices,
             golden_supervised::s4_graph_data, golden_supervised::s4_fit_init,
             golden_supervised::s4_embedding);
}

TEST_CASE("supervised fit: general l2 target, target_weight=0.3 (s5)") {
    UMAPParams p = base_params();
    p.target_metric = "l2";
    p.target_weight = 0.3;
    const std::vector<float> y =
        yf(golden_supervised::s5_y, golden_supervised::sn);
    run_case(p, y.data(), golden_supervised::s5_graph_indptr,
             golden_supervised::s5_graph_indices,
             golden_supervised::s5_graph_data, golden_supervised::s5_fit_init,
             golden_supervised::s5_embedding);
}

TEST_CASE("supervised fit: discrete ordinal target (s6, bit-exact)") {
    UMAPParams p = base_params();
    p.target_metric = "ordinal";
    const std::vector<float> y =
        yf(golden_supervised::s6_y, golden_supervised::sn);
    run_case(p, y.data(), golden_supervised::s6_graph_indptr,
             golden_supervised::s6_graph_indices,
             golden_supervised::s6_graph_data, golden_supervised::s6_fit_init,
             golden_supervised::s6_embedding);
}

TEST_CASE("supervised fit: discrete count target (s7, bit-exact)") {
    // float32 y end to end: scipy.stats.tmean gives a float32
    // poisson_lambda and count_distance's numba f32 specialization takes
    // np.log(lambda) in float32 (audit 2026-08-29 c07 -- the earlier f64
    // promotion shifted the graph by <= 4 ulp and the embedding entirely).
    UMAPParams p = base_params();
    p.target_metric = "count";
    const std::vector<float> y =
        yf(golden_supervised::s7_y, golden_supervised::sn);
    run_case(p, y.data(), golden_supervised::s7_graph_indptr,
             golden_supervised::s7_graph_indices,
             golden_supervised::s7_graph_data, golden_supervised::s7_fit_init,
             golden_supervised::s7_embedding);
}

TEST_CASE("supervised fit separates labels better than unsupervised") {
    // Unsupervised baseline reproduces the Python baseline bit-exactly, and
    // the supervised embedding must separate the 3 label centroids more.
    UMAP model0(base_params());
    const std::vector<float> emb0 = model0.fit_transform(
        golden_supervised::sX, golden_supervised::sn, golden_supervised::sdim);
    CHECK(count_mismatch(emb0, golden_supervised::s0_embedding) == 0);

    UMAPParams p = base_params();
    p.target_weight = 0.8;
    UMAP model(p);
    const std::vector<float> y =
        yf(golden_supervised::s2_y, golden_supervised::sn);
    const std::vector<float> emb = model.fit_transform(
        golden_supervised::sX, golden_supervised::sn, golden_supervised::sdim,
        y.data());

    const double r0 = separation_ratio(emb0, golden_supervised::s_labels,
                                       golden_supervised::sn);
    const double rs = separation_ratio(emb, golden_supervised::s_labels,
                                       golden_supervised::sn);
    CHECK(rs > r0);
    CHECK(rs > 1.5);
}

TEST_CASE("supervised fit: validation + unsupported targets raise") {
    {
        UMAPParams p = base_params();
        p.target_n_neighbors = 0;
        CHECK_THROWS_AS(UMAP(p).fit(golden_supervised::sX,
                                    golden_supervised::sn,
                                    golden_supervised::sdim),
                        std::invalid_argument);
    }
    for (const char* tm : {"string", "myers", "hierarchical_categorical"}) {
        UMAPParams p = base_params();
        p.target_metric = tm;
        const std::vector<float> y =
            yf(golden_supervised::s2_y, golden_supervised::sn);
        CHECK_THROWS_AS(UMAP(p).fit(golden_supervised::sX,
                                    golden_supervised::sn,
                                    golden_supervised::sdim, y.data()),
                        std::invalid_argument);
    }
}

TEST_CASE("reprocess_row rescales to log2(k) sum of powers") {
    const std::vector<float> row = {1.0f, 0.8f, 0.5f, 0.3f, 0.2f,
                                    0.1f, 0.05f, 0.02f};
    const std::vector<float> re = umap::reprocess_row(
        row.data(), static_cast<int64_t>(row.size()));
    double s = 0.0;
    for (const float v : re) s += static_cast<double>(v);
    CHECK(std::fabs(s - std::log2(15.0)) < 1e-4);
    // Max stays 1 (1^mid == 1).
    CHECK(re[0] == 1.0f);
}
