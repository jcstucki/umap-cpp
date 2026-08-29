/* --- phase4 additions -------------------------------------------------------
 * C API for the UMAP C++ port (FFI consumers). Wraps umap::UMAP behind an
 * opaque handle; every entry point returns a status code and stores the last
 * error message on the handle (umap_last_error).
 *
 * Status codes:
 *   UMAP_OK (0)                  success
 *   UMAP_ERROR_INVALID_ARGUMENT  parameter/shape/metric errors
 *                                (std::invalid_argument in the C++ core)
 *   UMAP_ERROR_RUNTIME           runtime failures, e.g. transform before fit
 *                                (std::runtime_error / std::exception)
 *   UMAP_ERROR_NULL_ARGUMENT     a required pointer argument was NULL
 *
 * Optional parameters use has_* flags (0 = library default), mirroring the
 * std::optional fields of umap::UMAPParams. Initialise the struct with
 * umap_params_init() and override what you need; passing NULL to
 * umap_create() uses all defaults.
 *
 * All matrices are dense row-major float32. Output buffers are
 * caller-allocated: n_rows x n_components floats for embeddings.
 * -------------------------------------------------------------------------- */
#ifndef UMAP_UMAP_C_H
#define UMAP_UMAP_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    UMAP_OK = 0,
    UMAP_ERROR_INVALID_ARGUMENT = 1,
    UMAP_ERROR_RUNTIME = 2,
    UMAP_ERROR_NULL_ARGUMENT = 3
};

enum {
    UMAP_DEVICE_CPU = 0,
    UMAP_DEVICE_GPU = 1 /* Apple Metal; falls back to CPU with a stderr note
                           when unavailable (see umap.hpp) */
};

typedef struct umap_handle umap_handle;

typedef struct umap_params {
    int n_neighbors;           /* 15 */
    int n_components;          /* 2 */
    const char* metric;        /* "euclidean" (NULL = default) */
    double metric_p;           /* metric_kwds: minkowski/... exponent (2.0) */
    int has_n_epochs;          /* 0: 500/200 fit default */
    int n_epochs;
    double learning_rate;      /* 1.0 */
    const char* init;          /* "spectral" (NULL = default) */
    double min_dist;           /* 0.1 */
    double spread;             /* 1.0 */
    double set_op_mix_ratio;   /* 1.0 */
    double local_connectivity; /* 1.0 */
    double repulsion_strength; /* 1.0 */
    int negative_sample_rate;  /* 5 */
    int has_a;                 /* 0: a/b fit from spread + min_dist */
    double a;
    int has_b;
    double b;
    int has_random_state;      /* 0: unseeded (parallel SGD, like upstream) */
    uint32_t random_state;
    uint32_t transform_seed;   /* 42 */
    const char* output_metric; /* "euclidean" (NULL = default) */
    int densmap;               /* 0 */
    double dens_lambda;        /* 2.0 */
    double dens_frac;          /* 0.3 */
    double dens_var_shift;     /* 0.1 */
    int output_dens;           /* 0 */
    const char* target_metric; /* "categorical" (NULL = default) */
    double target_weight;      /* 0.5 */
    int target_n_neighbors;    /* -1: use n_neighbors */
    int has_disconnection_distance; /* 0: per-metric default */
    double disconnection_distance;
    int unique_rows;           /* 0 (unique=True row dedup when nonzero) */
    int force_approximation;   /* 0 (force the NN-descent branch) */
    int device;                /* UMAP_DEVICE_CPU */
    int verbose;               /* 0 */
} umap_params;

/* Fill `p` with the library defaults listed above. */
void umap_params_init(umap_params* p);

/* Create a model. `p` may be NULL (all defaults). Returns NULL only on
 * allocation failure or invalid parameter strings; check
 * umap_last_error(NULL) is NOT available, so prefer valid inputs here --
 * value errors surface later from umap_fit. */
umap_handle* umap_create(const umap_params* p);
void umap_destroy(umap_handle* h);

/* Message of the last failing call on this handle ("" if none). The pointer
 * stays valid until the next call on the handle (or destroy). */
const char* umap_last_error(const umap_handle* h);

/* Fit `X` (n_samples x n_features row-major float32). */
int umap_fit(umap_handle* h, const float* X, int64_t n_samples,
             int64_t n_features);

/* Supervised fit: `y` holds one float label per sample (-1 = unknown for the
 * default categorical target). */
int umap_fit_supervised(umap_handle* h, const float* X, int64_t n_samples,
                        int64_t n_features, const float* y);

/* Fit and write the embedding (n_samples x n_components) to `out`.
 * `y` may be NULL. */
int umap_fit_transform(umap_handle* h, const float* X, int64_t n_samples,
                       int64_t n_features, const float* y, float* out);

/* Transform new points against a fitted model; writes n_queries x
 * n_components floats to `out`. */
int umap_transform(umap_handle* h, const float* X, int64_t n_queries,
                   float* out);

/* Fitted-model introspection. Return UMAP_ERROR_RUNTIME before fit. */
int umap_n_samples(const umap_handle* h, int64_t* out);
int umap_n_components(const umap_handle* h, int* out);
/* Copy the fitted embedding (n_samples x n_components floats) to `out`. */
int umap_get_embedding(umap_handle* h, float* out);

/* Model persistence ("UMAPCPP1" format; see umap.hpp). umap_load replaces
 * the model held by `h` with the file's contents. */
int umap_save(umap_handle* h, const char* path);
int umap_load(umap_handle* h, const char* path);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* UMAP_UMAP_C_H */
