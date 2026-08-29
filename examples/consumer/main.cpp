// --- phase4 additions --------------------------------------------------------
// find_package(umapcpp) consumer smoke test: fit 100 points, check the
// embedding is finite and the two blobs separate. Exit code 0 on success.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "umap/umap.hpp"

int main() {
    const int n = 100, dim = 5;
    std::vector<float> X(static_cast<std::size_t>(n) * dim);
    uint64_t s = 42;
    for (int i = 0; i < n; ++i) {
        const double shift = (i < n / 2) ? 0.0 : 10.0;
        for (int d = 0; d < dim; ++d) {
            s += 0x9E3779B97F4A7C15ULL;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            const double u =
                static_cast<double>((z ^ (z >> 31)) >> 11) / 9007199254740992.0;
            X[static_cast<std::size_t>(i) * dim + d] =
                static_cast<float>(u + shift);
        }
    }

    umap::UMAPParams params;
    params.n_neighbors = 10;
    params.init = "random";
    params.random_state = 42;
    params.n_epochs = 100;
    umap::UMAP model(params);
    const std::vector<float> emb = model.fit_transform(X.data(), n, dim);

    if (emb.size() != static_cast<std::size_t>(n) * 2) {
        std::fprintf(stderr, "consumer_smoke: wrong embedding size\n");
        return 1;
    }
    for (float v : emb) {
        if (!std::isfinite(v)) {
            std::fprintf(stderr, "consumer_smoke: non-finite embedding\n");
            return 1;
        }
    }
    // Blob-separation sanity: centroid distance > mean intra spread.
    double c0[2] = {0, 0}, c1[2] = {0, 0};
    for (int i = 0; i < n / 2; ++i)
        for (int d = 0; d < 2; ++d) {
            c0[d] += emb[static_cast<std::size_t>(i) * 2 + d] / (n / 2);
            c1[d] += emb[static_cast<std::size_t>(i + n / 2) * 2 + d] / (n / 2);
        }
    const double sep = std::hypot(c0[0] - c1[0], c0[1] - c1[1]);
    double spread = 0.0;
    for (int i = 0; i < n; ++i) {
        const double* c = (i < n / 2) ? c0 : c1;
        spread += std::hypot(emb[static_cast<std::size_t>(i) * 2] - c[0],
                             emb[static_cast<std::size_t>(i) * 2 + 1] - c[1]) /
                  n;
    }
    std::printf("consumer_smoke: fit 100 points, separation %.2f, spread %.2f\n",
                sep, spread);
    if (!(sep > spread)) {
        std::fprintf(stderr, "consumer_smoke: blobs failed to separate\n");
        return 1;
    }
    return 0;
}
