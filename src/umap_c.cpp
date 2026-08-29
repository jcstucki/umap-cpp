// --- phase4 additions --------------------------------------------------------
// C API implementation (include/umap/umap_c.h): opaque handle around
// umap::UMAP, status-code error handling with per-handle messages.
// -----------------------------------------------------------------------------
#include "umap/umap_c.h"

#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#include "umap/umap.hpp"

struct umap_handle {
    umap::UMAP model;
    std::string last_error;

    explicit umap_handle(umap::UMAPParams p) : model(std::move(p)) {}
};

namespace {

// Run `fn`, mapping C++ exceptions onto status codes + the handle's message.
template <typename Fn>
int guarded(umap_handle* h, Fn&& fn) {
    if (h == nullptr) return UMAP_ERROR_NULL_ARGUMENT;
    try {
        h->last_error.clear();
        fn();
        return UMAP_OK;
    } catch (const std::invalid_argument& e) {
        h->last_error = e.what();
        return UMAP_ERROR_INVALID_ARGUMENT;
    } catch (const std::runtime_error& e) {
        h->last_error = e.what();
        return UMAP_ERROR_RUNTIME;
    } catch (const std::exception& e) {
        h->last_error = e.what();
        return UMAP_ERROR_RUNTIME;
    }
}

umap::UMAPParams to_cpp_params(const umap_params* p) {
    umap::UMAPParams q;  // library defaults
    if (p == nullptr) return q;
    q.n_neighbors = p->n_neighbors;
    q.n_components = p->n_components;
    if (p->metric != nullptr) q.metric = p->metric;
    q.metric_kwds.p = p->metric_p;
    if (p->has_n_epochs) q.n_epochs = p->n_epochs;
    q.learning_rate = p->learning_rate;
    if (p->init != nullptr) q.init = p->init;
    q.min_dist = p->min_dist;
    q.spread = p->spread;
    q.set_op_mix_ratio = p->set_op_mix_ratio;
    q.local_connectivity = p->local_connectivity;
    q.repulsion_strength = p->repulsion_strength;
    q.negative_sample_rate = p->negative_sample_rate;
    if (p->has_a) q.a = p->a;
    if (p->has_b) q.b = p->b;
    if (p->has_random_state) q.random_state = p->random_state;
    q.transform_seed = p->transform_seed;
    if (p->output_metric != nullptr) q.output_metric = p->output_metric;
    q.densmap = p->densmap != 0;
    q.dens_lambda = p->dens_lambda;
    q.dens_frac = p->dens_frac;
    q.dens_var_shift = p->dens_var_shift;
    q.output_dens = p->output_dens != 0;
    if (p->target_metric != nullptr) q.target_metric = p->target_metric;
    q.target_weight = p->target_weight;
    q.target_n_neighbors = p->target_n_neighbors;
    if (p->has_disconnection_distance)
        q.disconnection_distance = p->disconnection_distance;
    q.unique = p->unique_rows != 0;
    q.force_approximation_algorithm = p->force_approximation != 0;
    q.device = p->device == UMAP_DEVICE_GPU ? umap::Device::GPU
                                            : umap::Device::CPU;
    q.verbose = p->verbose != 0;
    return q;
}

}  // namespace

extern "C" {

void umap_params_init(umap_params* p) {
    if (p == nullptr) return;
    std::memset(p, 0, sizeof(*p));
    const umap::UMAPParams d;  // mirror the C++ defaults
    p->n_neighbors = d.n_neighbors;
    p->n_components = d.n_components;
    p->metric = nullptr;  // "euclidean"
    p->metric_p = d.metric_kwds.p;
    p->learning_rate = d.learning_rate;
    p->init = nullptr;  // "spectral"
    p->min_dist = d.min_dist;
    p->spread = d.spread;
    p->set_op_mix_ratio = d.set_op_mix_ratio;
    p->local_connectivity = d.local_connectivity;
    p->repulsion_strength = d.repulsion_strength;
    p->negative_sample_rate = d.negative_sample_rate;
    p->transform_seed = d.transform_seed;
    p->output_metric = nullptr;  // "euclidean"
    p->dens_lambda = d.dens_lambda;
    p->dens_frac = d.dens_frac;
    p->dens_var_shift = d.dens_var_shift;
    p->target_metric = nullptr;  // "categorical"
    p->target_weight = d.target_weight;
    p->target_n_neighbors = d.target_n_neighbors;
    p->device = UMAP_DEVICE_CPU;
}

umap_handle* umap_create(const umap_params* p) {
    try {
        return new umap_handle(to_cpp_params(p));
    } catch (...) {
        return nullptr;
    }
}

void umap_destroy(umap_handle* h) { delete h; }

const char* umap_last_error(const umap_handle* h) {
    if (h == nullptr) return "";
    return h->last_error.c_str();
}

int umap_fit(umap_handle* h, const float* X, int64_t n_samples,
             int64_t n_features) {
    return umap_fit_supervised(h, X, n_samples, n_features, nullptr);
}

int umap_fit_supervised(umap_handle* h, const float* X, int64_t n_samples,
                        int64_t n_features, const float* y) {
    if (h == nullptr) return UMAP_ERROR_NULL_ARGUMENT;
    if (X == nullptr) {
        h->last_error = "X must not be NULL";
        return UMAP_ERROR_NULL_ARGUMENT;
    }
    return guarded(h, [&] { h->model.fit(X, n_samples, n_features, y); });
}

int umap_fit_transform(umap_handle* h, const float* X, int64_t n_samples,
                       int64_t n_features, const float* y, float* out) {
    if (h == nullptr) return UMAP_ERROR_NULL_ARGUMENT;
    if (X == nullptr || out == nullptr) {
        h->last_error = "X and out must not be NULL";
        return UMAP_ERROR_NULL_ARGUMENT;
    }
    return guarded(h, [&] {
        const std::vector<float> emb =
            h->model.fit_transform(X, n_samples, n_features, y);
        std::memcpy(out, emb.data(), emb.size() * sizeof(float));
    });
}

int umap_transform(umap_handle* h, const float* X, int64_t n_queries,
                   float* out) {
    if (h == nullptr) return UMAP_ERROR_NULL_ARGUMENT;
    if (X == nullptr || out == nullptr) {
        h->last_error = "X and out must not be NULL";
        return UMAP_ERROR_NULL_ARGUMENT;
    }
    return guarded(h, [&] {
        const std::vector<float> emb = h->model.transform(X, n_queries);
        std::memcpy(out, emb.data(), emb.size() * sizeof(float));
    });
}

int umap_n_samples(const umap_handle* h, int64_t* out) {
    if (h == nullptr || out == nullptr) return UMAP_ERROR_NULL_ARGUMENT;
    if (h->model.embedding().empty()) return UMAP_ERROR_RUNTIME;
    *out = h->model.n_samples();
    return UMAP_OK;
}

int umap_n_components(const umap_handle* h, int* out) {
    if (h == nullptr || out == nullptr) return UMAP_ERROR_NULL_ARGUMENT;
    *out = h->model.params().n_components;
    return UMAP_OK;
}

int umap_get_embedding(umap_handle* h, float* out) {
    if (h == nullptr) return UMAP_ERROR_NULL_ARGUMENT;
    if (out == nullptr) {
        h->last_error = "out must not be NULL";
        return UMAP_ERROR_NULL_ARGUMENT;
    }
    return guarded(h, [&] {
        const std::vector<float>& emb = h->model.embedding();
        if (emb.empty())
            throw std::runtime_error("model is not fitted");
        std::memcpy(out, emb.data(), emb.size() * sizeof(float));
    });
}

int umap_save(umap_handle* h, const char* path) {
    if (h == nullptr) return UMAP_ERROR_NULL_ARGUMENT;
    if (path == nullptr) {
        h->last_error = "path must not be NULL";
        return UMAP_ERROR_NULL_ARGUMENT;
    }
    return guarded(h, [&] { h->model.save(path); });
}

int umap_load(umap_handle* h, const char* path) {
    if (h == nullptr) return UMAP_ERROR_NULL_ARGUMENT;
    if (path == nullptr) {
        h->last_error = "path must not be NULL";
        return UMAP_ERROR_NULL_ARGUMENT;
    }
    return guarded(h, [&] { h->model = umap::UMAP::load(path); });
}

}  // extern "C"
