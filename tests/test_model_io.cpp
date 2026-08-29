// --- phase4 additions --------------------------------------------------------
// Model persistence round trips: UMAP::save / UMAP::load ("UMAPCPP1" format,
// src/model_io.cpp). The bar is bit-exactness of the loaded model's behaviour:
// a loaded model must transform / inverse_transform / update identically to
// the in-memory model it was saved from.
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "doctest.h"
#include "umap/umap.hpp"

namespace {

struct TestRng {
    uint64_t s;
    explicit TestRng(uint64_t seed) : s(seed) {}
    double next() {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<double>(z >> 11) / 9007199254740992.0;
    }
};

std::vector<float> random_matrix(int n, int dim, uint64_t seed) {
    TestRng rng(seed);
    std::vector<float> m(static_cast<std::size_t>(n) * dim);
    for (auto& v : m) v = static_cast<float>(rng.next());
    return m;
}

void check_vec_equal(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const bool same =
            (a[i] == b[i]) || (std::isnan(a[i]) && std::isnan(b[i]));
        if (!same) ++mismatches;
    }
    CHECK(mismatches == 0);
}

void check_graph_equal(const umap::SparseMatrix& a,
                       const umap::SparseMatrix& b) {
    CHECK(a.n_rows == b.n_rows);
    CHECK(a.n_cols == b.n_cols);
    CHECK(a.indptr == b.indptr);
    CHECK(a.indices == b.indices);
    check_vec_equal(a.data, b.data);
}

umap::UMAP roundtrip(const umap::UMAP& model) {
    std::stringstream buf(std::ios::in | std::ios::out | std::ios::binary);
    model.save(buf);
    buf.seekg(0);
    return umap::UMAP::load(buf);
}

}  // namespace

TEST_SUITE("model_io") {

TEST_CASE("small-data fit round trip: state and transform bit-exact") {
    const int n = 150, dim = 4;
    const auto X = random_matrix(n, dim, 11);
    umap::UMAPParams p;
    p.n_neighbors = 8;
    p.init = "random";
    p.random_state = 42;
    p.n_epochs = 80;
    umap::UMAP model(p);
    model.fit(X.data(), n, dim);

    const umap::UMAP loaded = roundtrip(model);

    CHECK(loaded.n_samples() == model.n_samples());
    CHECK(loaded.n_features() == model.n_features());
    CHECK(loaded.small_data() == model.small_data());
    CHECK(loaded.effective_n_neighbors() == model.effective_n_neighbors());
    CHECK(loaded.a() == model.a());
    CHECK(loaded.b() == model.b());
    CHECK(loaded.params().metric == model.params().metric);
    CHECK(loaded.params().random_state == model.params().random_state);
    CHECK(loaded.knn_indices() == model.knn_indices());
    check_vec_equal(loaded.knn_dists(), model.knn_dists());
    check_graph_equal(loaded.graph(), model.graph());
    check_vec_equal(loaded.sigmas(), model.sigmas());
    check_vec_equal(loaded.rhos(), model.rhos());
    check_vec_equal(loaded.embedding(), model.embedding());
    check_vec_equal(loaded.fit_init(), model.fit_init());

    // transform of new points: bit-exact between the two models.
    const int nq = 20;
    const auto Q = random_matrix(nq, dim, 77);
    check_vec_equal(loaded.transform(Q.data(), nq),
                    model.transform(Q.data(), nq));
    // Input-hash shortcut survives the round trip.
    check_vec_equal(loaded.transform(X.data(), n), model.embedding());

    // inverse_transform: identical (Delaunay + inverse SGD are seeded).
    const auto emb_q = model.transform(Q.data(), nq);
    check_vec_equal(loaded.inverse_transform(emb_q.data(), 5),
                    model.inverse_transform(emb_q.data(), 5));
}

TEST_CASE("standard-branch fit round trip: PreparedIndex query and update "
          "bit-exact") {
    const int n = 300, dim = 6;
    const auto X = random_matrix(n, dim, 5);
    umap::UMAPParams p;
    p.n_neighbors = 10;
    p.init = "random";
    p.random_state = 1234;
    p.n_epochs = 50;
    p.force_approximation_algorithm = true;  // NN-descent + PreparedIndex
    umap::UMAP model(p);
    model.fit(X.data(), n, dim);
    REQUIRE_FALSE(model.small_data());

    umap::UMAP loaded = roundtrip(model);
    CHECK(loaded.knn_indices() == model.knn_indices());
    check_vec_equal(loaded.knn_dists(), model.knn_dists());
    check_vec_equal(loaded.embedding(), model.embedding());
    check_graph_equal(loaded.graph(), model.graph());

    // The epsilon graph search (hub tree + search_rng_state + search graph)
    // round trips: identical query results.
    const int nq = 25;
    const auto Q = random_matrix(nq, dim, 999);
    check_vec_equal(loaded.transform(Q.data(), nq),
                    model.transform(Q.data(), nq));

    // update() continues the persisted RandomState + tau rng_state: the
    // in-memory and reloaded models must evolve identically.
    const int n_new = 30;
    const auto Xnew = random_matrix(n_new, dim, 31337);
    model.update(Xnew.data(), n_new, dim);
    loaded.update(Xnew.data(), n_new, dim);
    check_vec_equal(loaded.embedding(), model.embedding());
    CHECK(loaded.knn_indices() == model.knn_indices());
}

TEST_CASE("densmap + output_dens extras round trip") {
    const int n = 120, dim = 4;
    const auto X = random_matrix(n, dim, 21);
    umap::UMAPParams p;
    p.n_neighbors = 8;
    p.init = "random";
    p.random_state = 3;
    p.n_epochs = 60;
    p.densmap = true;
    p.output_dens = true;
    umap::UMAP model(p);
    model.fit(X.data(), n, dim);

    const umap::UMAP loaded = roundtrip(model);
    CHECK(loaded.params().densmap);
    CHECK(loaded.params().output_dens);
    check_vec_equal(loaded.embedding(), model.embedding());
    check_vec_equal(loaded.rad_orig(), model.rad_orig());
    check_vec_equal(loaded.rad_emb(), model.rad_emb());
    check_graph_equal(loaded.graph_dists(), model.graph_dists());
}

TEST_CASE("sparse small-data fit round trips; sparse standard fit refuses") {
    // ~35% dense random CSR, euclidean.
    const int n = 100, dim = 20;
    TestRng rng(2026);
    std::vector<int32_t> indptr(1, 0), indices;
    std::vector<float> data;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (rng.next() < 0.35) {
                indices.push_back(j);
                data.push_back(static_cast<float>(rng.next() + 0.1));
            }
        }
        // Guarantee non-empty rows (stable kNN).
        if (indices.size() == static_cast<std::size_t>(indptr.back())) {
            indices.push_back(i % dim);
            data.push_back(1.0f);
        }
        indptr.push_back(static_cast<int32_t>(indices.size()));
    }

    umap::UMAPParams p;
    p.n_neighbors = 8;
    p.init = "random";
    p.random_state = 9;
    p.n_epochs = 40;
    umap::UMAP model(p);
    model.fit(indptr.data(), indices.data(), data.data(), n, dim);
    CHECK(model.sparse_data());

    const umap::UMAP loaded = roundtrip(model);
    CHECK(loaded.sparse_data());
    check_vec_equal(loaded.embedding(), model.embedding());
    check_vec_equal(
        loaded.transform(indptr.data(), indices.data(), data.data(), 30),
        model.transform(indptr.data(), indices.data(), data.data(), 30));

    // The standard sparse branch keeps a SparsePreparedIndex, which the
    // format does not cover: save must refuse.
    umap::UMAPParams pf = p;
    pf.force_approximation_algorithm = true;
    umap::UMAP forced(pf);
    forced.fit(indptr.data(), indices.data(), data.data(), n, dim);
    std::ostringstream sink;
    CHECK_THROWS_AS(forced.save(sink), std::runtime_error);
}

TEST_CASE("format validation: magic, version, truncation, unfitted") {
    umap::UMAP unfitted;
    std::ostringstream sink;
    CHECK_THROWS_AS(unfitted.save(sink), std::runtime_error);

    const int n = 60, dim = 3;
    const auto X = random_matrix(n, dim, 8);
    umap::UMAPParams p;
    p.n_neighbors = 5;
    p.init = "random";
    p.random_state = 1;
    p.n_epochs = 20;
    umap::UMAP model(p);
    model.fit(X.data(), n, dim);

    std::stringstream buf(std::ios::in | std::ios::out | std::ios::binary);
    model.save(buf);
    const std::string bytes = buf.str();

    {  // bad magic
        std::string bad = bytes;
        bad[0] = 'X';
        std::istringstream is(bad);
        CHECK_THROWS_AS(umap::UMAP::load(is), std::runtime_error);
    }
    {  // bad version
        std::string bad = bytes;
        bad[12] = 99;  // version u32 follows magic(8) + sentinel(4)
        std::istringstream is(bad);
        CHECK_THROWS_AS(umap::UMAP::load(is), std::runtime_error);
    }
    {  // truncated
        std::istringstream is(bytes.substr(0, bytes.size() / 2));
        CHECK_THROWS_AS(umap::UMAP::load(is), std::runtime_error);
    }
}

}  // TEST_SUITE
