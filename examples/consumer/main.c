/* --- phase4 additions -------------------------------------------------------
 * C-API consumer smoke test from an installed tree (find_package(umapcpp)).
 * Fits 60 points through include/umap/umap_c.h; exit 0 on success.
 * -------------------------------------------------------------------------- */
#include <math.h>
#include <stdio.h>

#include "umap/umap_c.h"

int main(void) {
    enum { N = 60, DIM = 4 };
    static float X[N * DIM];
    static float emb[N * 2];
    umap_params p;
    umap_handle* h;
    int i, rc;

    for (i = 0; i < N * DIM; ++i) {
        X[i] = (float)((i * 2654435761u % 1000) / 1000.0 +
                       ((i / DIM) % 2) * 8.0);
    }
    umap_params_init(&p);
    p.n_neighbors = 8;
    p.init = "random";
    p.has_random_state = 1;
    p.random_state = 42;
    p.has_n_epochs = 1;
    p.n_epochs = 50;

    h = umap_create(&p);
    if (h == NULL) return 1;
    rc = umap_fit_transform(h, X, N, DIM, NULL, emb);
    if (rc != UMAP_OK) {
        fprintf(stderr, "consumer_smoke_c: %s\n", umap_last_error(h));
        umap_destroy(h);
        return 1;
    }
    for (i = 0; i < N * 2; ++i) {
        if (isnan(emb[i]) || isinf(emb[i])) {
            umap_destroy(h);
            return 1;
        }
    }
    printf("consumer_smoke_c: fit %d points OK\n", N);
    umap_destroy(h);
    return 0;
}
