/* --- phase4 additions -------------------------------------------------------
 * Pure C consumer of include/umap/umap_c.h, compiled as C (this file proves
 * the header is C-compilable). Returns 0 on success; nonzero values identify
 * the failing step. Driven from the doctest suite by tests/test_c_api.cpp.
 * -------------------------------------------------------------------------- */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "umap/umap_c.h"

/* Deterministic uniform [0,1) (splitmix64), no libc rand() state. */
static uint64_t g_state;
static double next_uniform(void) {
    uint64_t z;
    g_state += 0x9E3779B97F4A7C15ULL;
    z = g_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) / 9007199254740992.0;
}

int umap_c_api_smoke(void) {
    enum { N = 90, DIM = 4, NQ = 12, NCOMP = 2 };
    static float X[N * DIM];
    static float Q[NQ * DIM];
    static float emb[N * NCOMP];
    static float emb2[N * NCOMP];
    static float qemb[NQ * NCOMP];
    umap_params params;
    umap_handle* h;
    int64_t n_out = 0;
    int ncomp_out = 0;
    int rc, i;

    /* Three shifted uniform blobs. */
    g_state = 20260828ULL;
    for (i = 0; i < N * DIM; ++i) {
        const int cluster = (i / DIM) % 3;
        X[i] = (float)(next_uniform() + 8.0 * cluster);
    }
    for (i = 0; i < NQ * DIM; ++i) {
        const int cluster = (i / DIM) % 3;
        Q[i] = (float)(next_uniform() + 8.0 * cluster);
    }

    umap_params_init(&params);
    if (params.n_neighbors != 15 || params.n_components != 2) return 1;
    params.n_neighbors = 8;
    params.init = "random";
    params.has_random_state = 1;
    params.random_state = 42;
    params.has_n_epochs = 1;
    params.n_epochs = 60;

    h = umap_create(&params);
    if (h == NULL) return 2;

    /* Error path before fit: transform must fail with a message. */
    rc = umap_transform(h, Q, NQ, qemb);
    if (rc == UMAP_OK) { umap_destroy(h); return 3; }
    if (strlen(umap_last_error(h)) == 0) { umap_destroy(h); return 4; }

    /* NULL-argument path. */
    rc = umap_fit(h, NULL, N, DIM);
    if (rc != UMAP_ERROR_NULL_ARGUMENT) { umap_destroy(h); return 5; }

    /* Invalid-argument mapping (bad metric name). */
    {
        umap_params bad;
        umap_handle* hb;
        umap_params_init(&bad);
        bad.metric = "no-such-metric";
        hb = umap_create(&bad);
        if (hb == NULL) { umap_destroy(h); return 6; }
        rc = umap_fit(hb, X, N, DIM);
        if (rc != UMAP_ERROR_INVALID_ARGUMENT ||
            strlen(umap_last_error(hb)) == 0) {
            umap_destroy(hb);
            umap_destroy(h);
            return 7;
        }
        umap_destroy(hb);
    }

    /* fit_transform, then introspection. */
    rc = umap_fit_transform(h, X, N, DIM, NULL, emb);
    if (rc != UMAP_OK) {
        fprintf(stderr, "umap_fit_transform: %s\n", umap_last_error(h));
        umap_destroy(h);
        return 8;
    }
    for (i = 0; i < N * NCOMP; ++i) {
        if (!(emb[i] == emb[i]) || fabsf(emb[i]) > 1.0e6f) {
            umap_destroy(h);
            return 9;
        }
    }
    if (umap_n_samples(h, &n_out) != UMAP_OK || n_out != N) {
        umap_destroy(h);
        return 10;
    }
    if (umap_n_components(h, &ncomp_out) != UMAP_OK || ncomp_out != NCOMP) {
        umap_destroy(h);
        return 11;
    }
    if (umap_get_embedding(h, emb2) != UMAP_OK) { umap_destroy(h); return 12; }
    for (i = 0; i < N * NCOMP; ++i) {
        if (emb[i] != emb2[i]) { umap_destroy(h); return 13; }
    }

    /* transform of held-out points. */
    rc = umap_transform(h, Q, NQ, qemb);
    if (rc != UMAP_OK) {
        fprintf(stderr, "umap_transform: %s\n", umap_last_error(h));
        umap_destroy(h);
        return 14;
    }
    for (i = 0; i < NQ * NCOMP; ++i) {
        if (!(qemb[i] == qemb[i])) { umap_destroy(h); return 15; }
    }

    /* Supervised fit (categorical y with an unknown label). */
    {
        static float y[N];
        for (i = 0; i < N; ++i) y[i] = (float)(i % 3);
        y[0] = -1.0f;
        rc = umap_fit_supervised(h, X, N, DIM, y);
        if (rc != UMAP_OK) {
            fprintf(stderr, "umap_fit_supervised: %s\n", umap_last_error(h));
            umap_destroy(h);
            return 16;
        }
    }

    umap_destroy(h);
    umap_destroy(NULL); /* must be a no-op */
    return 0;
}
