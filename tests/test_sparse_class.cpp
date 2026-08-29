// Tests for the sparse (CSR) UMAP-class wiring: fit / fit_transform /
// transform on scipy-layout CSR input, compared against goldens from the
// reference Python UMAP (tests/golden_sparse_class.hpp, NUMBA_NUM_THREADS=1,
// seeded, init="random" + explicit a/b).
//
// Parity levels:
//  * small-data branch (p1 euclidean, p2 cosine, p5 unique=True): sklearn's
//    sparse pairwise float32 kernels are replicated exactly, so fit AND
//    transform embeddings are expected BIT-EXACT;
//  * NN-descent branch (p3 euclidean, p4 cosine,
//    force_approximation_algorithm=True): kNN indices / graph structure are
//    identical but numba's fastmath sparse kernels leave dists ~1 ulp off,
//    perturbing the SGD trajectory -- graph indptr/indices compared exactly,
//    weights to 1e-4 relative, embeddings at trustworthiness level against
//    the Python model's own scores.
//
// Documented deviation: transform()'s small-data kNN uses a stable k-smallest
// sort where upstream uses numpy's *unstable* argpartition + argsort. The
// results differ only when distances tie exactly within the top k -- common
// for sparse cosine at low density, where rows sharing no columns are all at
// distance exactly 1.0 (p2 therefore uses density 0.3 data, which has no
// ties). The fit-side kNN (fast_knn_indices) is np.argsort(kind="stable")
// upstream and matches everywhere.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "doctest.h"
#include "golden_sparse_class.hpp"
#include "umap/sparse_matrix.hpp"
#include "umap/umap.hpp"

using umap::SparseMatrix;
using umap::UMAP;
using umap::UMAPParams;

namespace {

UMAPParams base_params() {
    UMAPParams p;
    p.init = "random";
    p.random_state = 42;
    p.a = golden_sparse_class::sp_a;
    p.b = golden_sparse_class::sp_b;
    return p;
}

std::size_t count_mismatch(const std::vector<float>& a, const float* b) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++c;
    return c;
}

bool has_nan(const std::vector<float>& a) {
    for (float v : a)
        if (std::isnan(v)) return true;
    return false;
}

// Port of sklearn.manifold.trustworthiness (euclidean), double precision.
double trustworthiness(const float* X, int64_t n, int64_t dim,
                       const float* emb, int64_t edim, int k) {
    std::vector<double> drow(n);
    std::vector<int32_t> order(n);
    std::vector<int32_t> inverted(static_cast<std::size_t>(n) * n, 0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t d = 0; d < dim; ++d) {
                const double diff = static_cast<double>(X[i * dim + d]) -
                                    static_cast<double>(X[j * dim + d]);
                s += diff * diff;
            }
            drow[j] = (i == j) ? std::numeric_limits<double>::infinity() : s;
        }
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return drow[a] < drow[b];
        });
        for (int64_t r = 0; r < n; ++r)
            inverted[i * n + order[r]] = static_cast<int32_t>(r + 1);
    }
    double t = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t d = 0; d < edim; ++d) {
                const double diff = static_cast<double>(emb[i * edim + d]) -
                                    static_cast<double>(emb[j * edim + d]);
                s += diff * diff;
            }
            drow[j] = (i == j) ? std::numeric_limits<double>::infinity() : s;
        }
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return drow[a] < drow[b];
        });
        for (int r = 0; r < k; ++r) {
            const int rank = inverted[i * n + order[r]] - k;
            if (rank > 0) t += rank;
        }
    }
    return 1.0 - t * (2.0 / (static_cast<double>(n) * k *
                             (2.0 * n - 3.0 * k - 1.0)));
}

}  // namespace

TEST_CASE("sparse class: small-data euclidean fit+transform (p1, bit-exact)") {
    UMAP model(base_params());
    const std::vector<float> emb = model.fit_transform(
        golden_sparse_class::p1_indptr, golden_sparse_class::p1_indices,
        golden_sparse_class::p1_data, golden_sparse_class::p1_n,
        golden_sparse_class::p1_dim);
    CHECK(model.sparse_data());
    CHECK(model.small_data());
    CHECK(count_mismatch(emb, golden_sparse_class::p1_embedding) == 0);

    const std::vector<float> temb = model.transform(
        golden_sparse_class::p1_q_indptr, golden_sparse_class::p1_q_indices,
        golden_sparse_class::p1_q_data, golden_sparse_class::p1_q_n);
    CHECK(count_mismatch(temb, golden_sparse_class::p1_t_embedding) == 0);

    // Input-hash shortcut: transforming the training CSR returns embedding_.
    const std::vector<float> same = model.transform(
        golden_sparse_class::p1_indptr, golden_sparse_class::p1_indices,
        golden_sparse_class::p1_data, golden_sparse_class::p1_n);
    CHECK(count_mismatch(same, emb.data()) == 0);
}

TEST_CASE("sparse class: small-data cosine fit+transform (p2, bit-exact)") {
    UMAPParams p = base_params();
    p.metric = "cosine";
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_sparse_class::p2_indptr, golden_sparse_class::p2_indices,
        golden_sparse_class::p2_data, golden_sparse_class::p2_n,
        golden_sparse_class::p2_dim);
    CHECK(count_mismatch(emb, golden_sparse_class::p2_embedding) == 0);

    const std::vector<float> temb = model.transform(
        golden_sparse_class::p2_q_indptr, golden_sparse_class::p2_q_indices,
        golden_sparse_class::p2_q_data, golden_sparse_class::p2_q_n);
    CHECK(count_mismatch(temb, golden_sparse_class::p2_t_embedding) == 0);
}

// --- hardening additions -----------------------------------------------------
TEST_CASE("sparse class: small-data manhattan fit+transform (p6, bit-exact)") {
    // sklearn's float64 _sparse_manhattan kernel (float32 subtraction on
    // shared columns, float64 fabs-accumulation), float64 fast_knn_indices,
    // astype(np.float32) at the fuzzy_simplicial_set / transform boundary.
    UMAPParams p = base_params();
    p.metric = "manhattan";
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_sparse_class::p6_indptr, golden_sparse_class::p6_indices,
        golden_sparse_class::p6_data, golden_sparse_class::p6_n,
        golden_sparse_class::p6_dim);
    CHECK(model.sparse_data());
    CHECK(model.small_data());
    CHECK(count_mismatch(emb, golden_sparse_class::p6_embedding) == 0);

    const std::vector<float> temb = model.transform(
        golden_sparse_class::p6_q_indptr, golden_sparse_class::p6_q_indices,
        golden_sparse_class::p6_q_data, golden_sparse_class::p6_q_n);
    CHECK(count_mismatch(temb, golden_sparse_class::p6_t_embedding) == 0);
}

TEST_CASE("sparse class: small-data hellinger fit+transform (p7, densify "
          "fallback, bit-exact)") {
    // Upstream: sklearn raises for metric="hellinger" on sparse input, and
    // fit falls back to pairwise_special_metric(X.toarray()) -- the float32
    // _pairwise_callable route with umap's numba hellinger kernel.
    UMAPParams p = base_params();
    p.metric = "hellinger";
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_sparse_class::p7_indptr, golden_sparse_class::p7_indices,
        golden_sparse_class::p7_data, golden_sparse_class::p7_n,
        golden_sparse_class::p7_dim);
    CHECK(model.small_data());
    CHECK(count_mismatch(emb, golden_sparse_class::p7_embedding) == 0);

    const std::vector<float> temb = model.transform(
        golden_sparse_class::p7_q_indptr, golden_sparse_class::p7_q_indices,
        golden_sparse_class::p7_q_data, golden_sparse_class::p7_q_n);
    CHECK(count_mismatch(temb, golden_sparse_class::p7_t_embedding) == 0);
}

TEST_CASE("sparse class: metric validation replicates the upstream elif "
          "chain") {
    const auto sparse_fit_throws = [](const std::string& metric) {
        UMAPParams p = base_params();
        p.metric = metric;
        UMAP model(p);
        CHECK_THROWS_AS(model.fit(golden_sparse_class::p1_indptr,
                                  golden_sparse_class::p1_indices,
                                  golden_sparse_class::p1_data,
                                  golden_sparse_class::p1_n,
                                  golden_sparse_class::p1_dim),
                        std::invalid_argument);
    };
    // "Metric {} is not supported for sparse data": dist.named_distances
    // names missing from umap's sparse registry ...
    sparse_fit_throws("l2");
    sparse_fit_throws("symmetric_kl");
    sparse_fit_throws("seuclidean");
    sparse_fit_throws("yule");
    // ... and pynndescent names missing from its sparse registry.
    sparse_fit_throws("spearmanr");

    // pynndescent-only sparse names pass validation but KeyError upstream on
    // the small-data branch (dist.named_distances[metric] in the fallback).
    {
        UMAPParams p = base_params();
        p.metric = "sqeuclidean";
        UMAP model(p);
        CHECK_THROWS_AS(model.fit(golden_sparse_class::p1_indptr,
                                  golden_sparse_class::p1_indices,
                                  golden_sparse_class::p1_data,
                                  golden_sparse_class::p1_n,
                                  golden_sparse_class::p1_dim),
                        std::out_of_range);
    }

    // The hellinger negative-data guard covers sparse values too.
    {
        std::vector<float> neg(golden_sparse_class::p1_data,
                               golden_sparse_class::p1_data +
                                   golden_sparse_class::p1_indptr
                                       [golden_sparse_class::p1_n]);
        neg[3] = -neg[3];
        UMAPParams p = base_params();
        p.metric = "hellinger";
        UMAP model(p);
        CHECK_THROWS_AS(model.fit(golden_sparse_class::p1_indptr,
                                  golden_sparse_class::p1_indices, neg.data(),
                                  golden_sparse_class::p1_n,
                                  golden_sparse_class::p1_dim),
                        std::invalid_argument);
    }
}
// -----------------------------------------------------------------------------

TEST_CASE("sparse class: unique=True csr_unique fit (p5, bit-exact)") {
    UMAPParams p = base_params();
    p.unique = true;
    p.n_neighbors = 10;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_sparse_class::p5_indptr, golden_sparse_class::p5_indices,
        golden_sparse_class::p5_data, golden_sparse_class::p5_n,
        golden_sparse_class::p5_dim);
    CHECK(count_mismatch(emb, golden_sparse_class::p5_embedding) == 0);
    // Duplicated rows share coordinates via the inverse expansion.
    CHECK(model.unique_inverse().size() ==
          static_cast<std::size_t>(golden_sparse_class::p5_n));
}

namespace {

template <typename Model>
void check_graph_structure(const Model& model, const int32_t* indptr,
                           const int32_t* indices, const float* data,
                           int64_t n) {
    const SparseMatrix& g = model.graph();
    REQUIRE(g.n_rows == n);
    std::size_t ip_mismatch = 0, idx_mismatch = 0;
    for (int64_t i = 0; i <= n; ++i)
        if (g.indptr[i] != indptr[i]) ++ip_mismatch;
    REQUIRE(ip_mismatch == 0);
    double worst_rel = 0.0;
    for (std::size_t k = 0; k < g.indices.size(); ++k) {
        if (g.indices[k] != indices[k]) ++idx_mismatch;
        const double rel = std::fabs(static_cast<double>(g.data[k]) -
                                     static_cast<double>(data[k])) /
                           std::max(1e-12, std::fabs(
                                               static_cast<double>(data[k])));
        worst_rel = std::max(worst_rel, rel);
    }
    CHECK(idx_mismatch == 0);
    CHECK(worst_rel < 1e-4);
}

}  // namespace

TEST_CASE("sparse class: NN-descent euclidean fit+transform (p3)") {
    UMAPParams p = base_params();
    p.force_approximation_algorithm = true;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_sparse_class::p3_indptr, golden_sparse_class::p3_indices,
        golden_sparse_class::p3_data, golden_sparse_class::p3_n,
        golden_sparse_class::p3_dim);
    CHECK(model.sparse_data());
    CHECK(!model.small_data());
    CHECK(!has_nan(emb));
    check_graph_structure(model, golden_sparse_class::p3_graph_indptr,
                          golden_sparse_class::p3_graph_indices,
                          golden_sparse_class::p3_graph_data,
                          golden_sparse_class::p3_n);
    const double tw = trustworthiness(
        golden_sparse_class::p3_dense, golden_sparse_class::p3_n,
        golden_sparse_class::p3_dim, emb.data(), 2, 5);
    CHECK(tw >= golden_sparse_class::p3_fit_trust - 0.05);

    const std::vector<float> temb = model.transform(
        golden_sparse_class::p3_q_indptr, golden_sparse_class::p3_q_indices,
        golden_sparse_class::p3_q_data, golden_sparse_class::p3_q_n);
    CHECK(!has_nan(temb));
    const double twq = trustworthiness(
        golden_sparse_class::p3_q_dense, golden_sparse_class::p3_q_n,
        golden_sparse_class::p3_dim, temb.data(), 2, 5);
    CHECK(twq >= golden_sparse_class::p3_t_trust - 0.05);
}

TEST_CASE("sparse class: NN-descent cosine fit+transform (p4)") {
    UMAPParams p = base_params();
    p.metric = "cosine";
    p.force_approximation_algorithm = true;
    UMAP model(p);
    const std::vector<float> emb = model.fit_transform(
        golden_sparse_class::p4_indptr, golden_sparse_class::p4_indices,
        golden_sparse_class::p4_data, golden_sparse_class::p4_n,
        golden_sparse_class::p4_dim);
    CHECK(!has_nan(emb));
    check_graph_structure(model, golden_sparse_class::p4_graph_indptr,
                          golden_sparse_class::p4_graph_indices,
                          golden_sparse_class::p4_graph_data,
                          golden_sparse_class::p4_n);
    const double tw = trustworthiness(
        golden_sparse_class::p4_dense, golden_sparse_class::p4_n,
        golden_sparse_class::p4_dim, emb.data(), 2, 5);
    CHECK(tw >= golden_sparse_class::p4_fit_trust - 0.05);

    // transform picks epsilon=0.24 for angular-tree metrics.
    const std::vector<float> temb = model.transform(
        golden_sparse_class::p4_q_indptr, golden_sparse_class::p4_q_indices,
        golden_sparse_class::p4_q_data, golden_sparse_class::p4_q_n);
    CHECK(!has_nan(temb));
    const double twq = trustworthiness(
        golden_sparse_class::p4_q_dense, golden_sparse_class::p4_q_n,
        golden_sparse_class::p4_dim, temb.data(), 2, 5);
    CHECK(twq >= golden_sparse_class::p4_t_trust - 0.05);
}

TEST_CASE("sparse class: unsupported branches + overload mixups raise") {
    {
        UMAPParams p = base_params();
        p.metric = "precomputed";
        CHECK_THROWS_AS(
            UMAP(p).fit(golden_sparse_class::p1_indptr,
                        golden_sparse_class::p1_indices,
                        golden_sparse_class::p1_data,
                        golden_sparse_class::p1_n,
                        golden_sparse_class::p1_dim),
            std::invalid_argument);
    }
    {
        UMAPParams p = base_params();
        p.init = "pca";
        CHECK_THROWS_AS(
            UMAP(p).fit(golden_sparse_class::p1_indptr,
                        golden_sparse_class::p1_indices,
                        golden_sparse_class::p1_data,
                        golden_sparse_class::p1_n,
                        golden_sparse_class::p1_dim),
            std::invalid_argument);
    }
    {
        // Dense transform on a sparse-fit model raises (and vice versa).
        UMAP model(base_params());
        model.fit(golden_sparse_class::p1_indptr,
                  golden_sparse_class::p1_indices,
                  golden_sparse_class::p1_data, golden_sparse_class::p1_n,
                  golden_sparse_class::p1_dim);
        const std::vector<float> dense(
            static_cast<std::size_t>(golden_sparse_class::p1_dim), 0.0f);
        CHECK_THROWS_AS(model.transform(dense.data(), 1),
                        std::invalid_argument);
    }
}

TEST_CASE("sparse class: supervised sparse fit runs and separates") {
    // Sparse fit + categorical y (small-data branch); smoke-level: finite
    // output, sparse_data flag, and the graph loses cross-label mass.
    UMAPParams p = base_params();
    p.target_weight = 1.0;
    UMAP model(p);
    std::vector<float> y(static_cast<std::size_t>(golden_sparse_class::p1_n));
    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] = static_cast<float>(i % 2);
    const std::vector<float> emb = model.fit_transform(
        golden_sparse_class::p1_indptr, golden_sparse_class::p1_indices,
        golden_sparse_class::p1_data, golden_sparse_class::p1_n,
        golden_sparse_class::p1_dim, y.data());
    REQUIRE(emb.size() ==
            static_cast<std::size_t>(golden_sparse_class::p1_n) * 2);
    // target_weight=1.0 categorical: every surviving edge connects
    // same-label vertices.
    const SparseMatrix& g = model.graph();
    for (int32_t i = 0; i < g.n_rows; ++i)
        for (int32_t k = g.indptr[i]; k < g.indptr[i + 1]; ++k)
            CHECK((i % 2) == (g.indices[k] % 2));
}
