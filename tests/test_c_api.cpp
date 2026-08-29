// --- phase4 additions --------------------------------------------------------
// C API coverage. The real consumer lives in tests/c_api_smoke.c (compiled as
// C -- proving umap_c.h is C-compilable); this driver runs it and adds a
// save/load round trip through the C surface.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "doctest.h"
#include "umap/umap_c.h"

extern "C" int umap_c_api_smoke(void);

TEST_SUITE("c_api") {

TEST_CASE("pure-C smoke consumer (fit/transform/supervised/errors)") {
    const int rc = umap_c_api_smoke();
    CAPTURE(rc);
    CHECK(rc == 0);
}

TEST_CASE("save/load round trip through the C API") {
    const int n = 80, dim = 3, ncomp = 2;
    std::vector<float> X(static_cast<std::size_t>(n) * dim);
    uint64_t s = 7;
    for (auto& v : X) {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        v = static_cast<float>(static_cast<double>((z ^ (z >> 31)) >> 11) /
                               9007199254740992.0);
    }

    umap_params p;
    umap_params_init(&p);
    p.n_neighbors = 6;
    p.init = "random";
    p.has_random_state = 1;
    p.random_state = 5;
    p.has_n_epochs = 1;
    p.n_epochs = 30;

    umap_handle* h = umap_create(&p);
    REQUIRE(h != nullptr);
    std::vector<float> emb(static_cast<std::size_t>(n) * ncomp);
    REQUIRE(umap_fit_transform(h, X.data(), n, dim, nullptr, emb.data()) ==
            UMAP_OK);

    const std::string path =
        std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
        "/umapcpp_capi_roundtrip.umap";
    REQUIRE(umap_save(h, path.c_str()) == UMAP_OK);

    umap_handle* h2 = umap_create(nullptr);
    REQUIRE(h2 != nullptr);
    REQUIRE(umap_load(h2, path.c_str()) == UMAP_OK);
    std::vector<float> emb2(static_cast<std::size_t>(n) * ncomp);
    REQUIRE(umap_get_embedding(h2, emb2.data()) == UMAP_OK);
    CHECK(std::memcmp(emb.data(), emb2.data(),
                      emb.size() * sizeof(float)) == 0);

    // Loaded model transforms identically.
    std::vector<float> q1(static_cast<std::size_t>(10) * ncomp);
    std::vector<float> q2(static_cast<std::size_t>(10) * ncomp);
    REQUIRE(umap_transform(h, X.data(), 10, q1.data()) == UMAP_OK);
    REQUIRE(umap_transform(h2, X.data(), 10, q2.data()) == UMAP_OK);
    CHECK(std::memcmp(q1.data(), q2.data(), q1.size() * sizeof(float)) == 0);

    // Load-failure mapping.
    CHECK(umap_load(h2, "/nonexistent/path/model.umap") ==
          UMAP_ERROR_RUNTIME);
    CHECK(std::strlen(umap_last_error(h2)) > 0);

    std::remove(path.c_str());
    umap_destroy(h2);
    umap_destroy(h);
}

}  // TEST_SUITE
