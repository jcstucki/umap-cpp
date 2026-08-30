// Faithful C++ port of layouts.py::optimize_layout_euclidean and the
// simplicial_set_embedding glue from umap_.py (lmcinnes/umap, BSD 3 clause).
// See include/umap/layouts.hpp for the semantics notes.
#include "umap/layouts.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "umap/rng.hpp"

// --- phase4 additions --------------------------------------------------------
#ifdef UMAP_ENABLE_METAL
#include "umap/metal/gpu_layout.hpp"
#endif
// -----------------------------------------------------------------------------

namespace umap {

std::vector<double> make_epochs_per_sample(const float* weights,
                                           std::size_t n_weights,
                                           int n_epochs) {
    // result = np.full(n, -1.0, dtype=np.float64)
    // n_samples = n_epochs * (weights / weights.max())   # float32
    // result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[...])
    std::vector<double> result(n_weights, -1.0);
    if (n_weights == 0) return result;
    float w_max = weights[0];
    for (std::size_t i = 1; i < n_weights; ++i) {
        // weights.max() propagates NaN (allow-nan graphs): every n_samples
        // is then NaN, `NaN > 0` is false and all entries stay -1 upstream.
        if (std::isnan(weights[i])) {
            w_max = weights[i];
            break;
        }
        if (weights[i] > w_max) w_max = weights[i];
    }
    const float n_epochs_f = static_cast<float>(n_epochs);
    for (std::size_t i = 0; i < n_weights; ++i) {
        float n_samples = n_epochs_f * (weights[i] / w_max);
        if (n_samples > 0.0f) {
            result[i] =
                static_cast<double>(n_epochs) / static_cast<double>(n_samples);
        }
    }
    return result;
}

// --- inverse-update additions -----------------------------------------------
std::vector<double> make_epochs_per_sample(const double* weights,
                                           std::size_t n_weights,
                                           int n_epochs) {
    // Same routine instantiated on float64 weights (inverse_transform):
    // n_samples = n_epochs * (weights / weights.max()) is all-float64.
    std::vector<double> result(n_weights, -1.0);
    if (n_weights == 0) return result;
    double w_max = weights[0];
    for (std::size_t i = 1; i < n_weights; ++i) {
        if (std::isnan(weights[i])) {  // np.max propagates NaN
            w_max = weights[i];
            break;
        }
        if (weights[i] > w_max) w_max = weights[i];
    }
    for (std::size_t i = 0; i < n_weights; ++i) {
        const double n_samples =
            static_cast<double>(n_epochs) * (weights[i] / w_max);
        if (n_samples > 0.0) {
            result[i] = static_cast<double>(n_epochs) / n_samples;
        }
    }
    return result;
}
// ----------------------------------------------------------------------------

// --- composite additions ------------------------------------------------------
std::vector<double> make_epochs_per_sample_i64(const float* weights,
                                               std::size_t n_weights,
                                               int n_epochs) {
    // The composite operators pass n_epochs = np.max(result.n_epochs), an
    // np.int64; numpy promotes int64 * float32 to FLOAT64, so
    // n_samples = n_epochs * (weights / weights.max()) becomes
    // f8(n_epochs) * f8(f4(w / w_max)) -- the w / w_max quotient is still
    // rounded to float32 first (f4 array / f4 scalar) -- instead of the
    // all-float32 product of the normal fit path (python-int n_epochs).
    std::vector<double> result(n_weights, -1.0);
    if (n_weights == 0) return result;
    float w_max = weights[0];
    for (std::size_t i = 1; i < n_weights; ++i) {
        if (std::isnan(weights[i])) {  // np.max propagates NaN
            w_max = weights[i];
            break;
        }
        if (weights[i] > w_max) w_max = weights[i];
    }
    for (std::size_t i = 0; i < n_weights; ++i) {
        const float q32 = weights[i] / w_max;
        const double n_samples =
            static_cast<double>(n_epochs) * static_cast<double>(q32);
        if (n_samples > 0.0) {
            result[i] = static_cast<double>(n_epochs) / n_samples;
        }
    }
    return result;
}
// ------------------------------------------------------------------------------

// --- densmap additions ------------------------------------------------------
#if defined(__APPLE__)
// numpy's macOS wheels (>= 2.0) link Accelerate; np.dot(f32, f32) is
// cblas_sdot, whose SIMD accumulation order differs from a sequential loop.
// Calling the same library reproduces np.dot bit for bit.
extern "C" float cblas_sdot(int n, const float* x, int incx, const float* y,
                            int incy);
#endif

namespace {

// numpy FLOAT_pairwise_sum (see src/umap.cpp for the float64/float32 twins in
// the umap-class translation unit): 8 accumulators over blocks of 8, reduced
// ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)), recursive split above 128.
float numpy_pairwise_sum_f32(const float* a, int64_t n) {
    if (n < 8) {
        float res = -0.0f;
        for (int64_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        float r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        int64_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (int k = 0; k < 8; ++k) r[k] += a[i + k];
        }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) +
                    ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return numpy_pairwise_sum_f32(a, n2) +
           numpy_pairwise_sum_f32(a + n2, n - n2);
}

// np.sum(float32 array): plain pairwise summation over the whole array
// (verified against np.sum in the reference environment; the
// "first-element-seeds-the-accumulator" form only applies to
// np.add.reduceat, see src/umap.cpp::numpy_add_reduce_segment).
float numpy_sum_f32(const float* a, int64_t n) {
    if (n <= 0) return 0.0f;
    return numpy_pairwise_sum_f32(a, n);
}

// np.mean(float32 array): umr_sum / n, all float32.
float numpy_mean_f32(const float* a, int64_t n) {
    return numpy_sum_f32(a, n) / static_cast<float>(n);
}

// np.var(float32 array) (numpy _methods._var): arrmean = umr_sum/n, then
// x = arr - arrmean, ret = umr_sum(x * x) / n, all float32.
float numpy_var_f32(const float* a, int64_t n) {
    const float mean = numpy_mean_f32(a, n);
    std::vector<float> x(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const float d = a[i] - mean;
        x[static_cast<std::size_t>(i)] = d * d;
    }
    return numpy_sum_f32(x.data(), n) / static_cast<float>(n);
}

// np.dot(float32, float32) -> float32 scalar.
float numpy_dot_f32(const float* a, const float* b, int64_t n) {
#if defined(__APPLE__)
    return cblas_sdot(static_cast<int>(n), a, 1, b, 1);
#else
    // Documented deviation off Apple platforms: sequential f32 accumulation
    // (BLAS vendors block/vectorise differently).
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
#endif
}

// Per-epoch densMAP state handed into the edge loop (mirrors the scalars the
// numba epoch kernel receives; f4/f8 typing preserved).
struct DensmapEpochState {
    bool flag = false;          // densmap_flag for this epoch
    const float* phi_sum = nullptr;  // dens_phi_sum (f4[n_vertices])
    const float* re_sum = nullptr;   // dens_re_sum (f4[n_vertices])
    float re_cov = 0.0f;        // np.dot(re_sum, R) / (n_vertices - 1)  (f4)
    float re_std = 0.0f;        // np.sqrt(np.var(re_sum) + var_shift)   (f4)
    float re_mean = 0.0f;       // np.mean(re_sum)                       (f4)
    double lambda = 0.0;        // densmap_kwds["lambda"]                (f8)
    const float* R = nullptr;   // dens_R (f4[n_vertices])
    const float* mu = nullptr;  // dens_mu (f4[n_edges])
    float mu_tot = 0.0f;        // np.sum(mu_sum) / 2                    (f4)
};

// layouts.py::_optimize_layout_euclidean_densmap_epoch_init (sequential
// instantiation; the prange is a plain range for the seeded path).
// re_sum / phi_sum are float32; phi and the accumulating expressions are
// float64 (literals promote), rounded to float32 on store. The final log runs
// on the float64 `epsilon + (f4 / f4)` sum, rounded to float32 on store.
void densmap_epoch_init(const float* head_embedding,
                        const float* tail_embedding, const int32_t* head,
                        const int32_t* tail, std::size_t n_edges, double a,
                        double b, int dim, std::vector<float>& re_sum,
                        std::vector<float>& phi_sum) {
    std::fill(re_sum.begin(), re_sum.end(), 0.0f);
    std::fill(phi_sum.begin(), phi_sum.end(), 0.0f);

    for (std::size_t i = 0; i < n_edges; ++i) {
        const int32_t j = head[i];
        const int32_t k = tail[i];
        const float* current =
            head_embedding + static_cast<std::size_t>(j) * dim;
        const float* other = tail_embedding + static_cast<std::size_t>(k) * dim;
        const float dist_squared = rdist(current, other, dim);

        const double phi =
            1.0 / (1.0 + a * std::pow(static_cast<double>(dist_squared), b));

        re_sum[static_cast<std::size_t>(j)] = static_cast<float>(
            re_sum[static_cast<std::size_t>(j)] +
            phi * static_cast<double>(dist_squared));
        re_sum[static_cast<std::size_t>(k)] = static_cast<float>(
            re_sum[static_cast<std::size_t>(k)] +
            phi * static_cast<double>(dist_squared));
        phi_sum[static_cast<std::size_t>(j)] =
            static_cast<float>(phi_sum[static_cast<std::size_t>(j)] + phi);
        phi_sum[static_cast<std::size_t>(k)] =
            static_cast<float>(phi_sum[static_cast<std::size_t>(k)] + phi);
    }

    const double epsilon = 1e-8;
    for (std::size_t i = 0; i < re_sum.size(); ++i) {
        const float ratio = re_sum[i] / phi_sum[i];  // f4 / f4 stays f4
        re_sum[i] =
            static_cast<float>(std::log(epsilon + static_cast<double>(ratio)));
    }
}
// ----------------------------------------------------------------------------

// --- perf P2 additions -------------------------------------------------------
// Approximate pow(x, y) for the unseeded hogwild path. Defined and wired in a
// later change (P2-GATED); forward-declared here so pow_sel<true> parses. It is
// never odr-used in this change (dispatch only emits FASTPOW=false), so no
// definition is required for the link.
inline double fast_pow(double x, double y);

// pow selector: the seeded/sequential path (FASTPOW=false) uses libm std::pow
// so the bit-contract with numba holds. FASTPOW=true (fast_pow, unseeded path)
// is enabled in a later change; here only the false instantiation is ever
// generated, so std::pow is the sole code path.
template <bool FASTPOW>
inline double pow_sel(double x, double y) {
    if constexpr (FASTPOW) {
        return fast_pow(x, y);
    } else {
        return std::pow(x, y);
    }
}

// Compile-time-dim reduced euclidean distance. DIM>0 unrolls the loop with the
// SAME left-to-right float32 accumulation order as the runtime rdist (bit
// identical); DIM==0 forwards to the runtime rdist with the passed dim.
template <int DIM>
inline float rdist_t(const float* x, const float* y, int dim) {
    if constexpr (DIM > 0) {
        (void)dim;
        float result = 0.0f;
        for (int i = 0; i < DIM; ++i) {
            const float diff = x[i] - y[i];
            result += diff * diff;
        }
        return result;
    } else {
        return rdist(x, y, dim);
    }
}

// One epoch over edges [i_begin, i_end) — the numba
// _optimize_layout_euclidean_single_epoch body. Templated on:
//   DIM     compile-time n_components (0 = runtime `dim`); the common fit is
//           DIM==2, which unrolls the coordinate loops.
//   DENSMAP whether the densMAP correction is present; when false the entire
//           density branch and its per-edge struct traffic are elided.
//   FASTPOW whether to use the approximate fast_pow (unseeded hogwild path
//           only) instead of libm std::pow.
// The arithmetic is identical to the original scalar kernel for every
// (DIM, DENSMAP=..., FASTPOW=false) instantiation: the specializations only
// reorganize control flow (loop unrolling, dead-branch elision, edge skipping),
// they do not change the sequence of float operations.
template <int DIM, bool DENSMAP, bool FASTPOW>
void optimize_euclidean_epoch_range_t(
    std::size_t i_begin, std::size_t i_end, float* head_embedding,
    float* tail_embedding, const int32_t* head, const int32_t* tail,
    int n_vertices, const double* epochs_per_sample, double a, double b,
    int64_t* rng_state_per_sample, double gamma, int dim, bool move_other,
    double alpha, const double* epochs_per_negative_sample,
    double* epoch_of_next_negative_sample, double* epoch_of_next_sample,
    int n, const DensmapEpochState& dens) {
    const std::size_t stride = static_cast<std::size_t>(DIM > 0 ? DIM : dim);
    for (std::size_t i = i_begin; i < i_end; ++i) {
        // Skip edges whose next-sample epoch hasn't arrived without entering
        // the body (identical to the guarded original; just an early continue).
        if (epoch_of_next_sample[i] > static_cast<double>(n)) continue;

        const int32_t j = head[i];
        int32_t k = tail[i];

        float* current = head_embedding + static_cast<std::size_t>(j) * stride;
        float* other = tail_embedding + static_cast<std::size_t>(k) * stride;

        float dist_squared = rdist_t<DIM>(current, other, dim);

        // --- densmap additions ----------------------------------------------
        // grad_cor_coeff: float64 except where two float32 operands meet
        // (re_std_sq, weight_j/k, mu[i] * re_std), exactly as numba types the
        // reference kernel. np.exp(f4) is float32 expf.
        double grad_cor_coeff = 0.0;
        if constexpr (DENSMAP) {
            if (dens.flag) {
                const double dsq = static_cast<double>(dist_squared);
                const double phi = 1.0 / (1.0 + a * pow_sel<FASTPOW>(dsq, b));
                const double dphi_term = a * b * pow_sel<FASTPOW>(dsq, b - 1.0) /
                                         (1.0 + a * pow_sel<FASTPOW>(dsq, b));

                const double q_jk =
                    phi / static_cast<double>(
                              dens.phi_sum[static_cast<std::size_t>(k)]);
                const double q_kj =
                    phi / static_cast<double>(
                              dens.phi_sum[static_cast<std::size_t>(j)]);

                const double drk =
                    q_jk *
                    ((1.0 - b * (1.0 - phi)) /
                         static_cast<double>(expf(
                             dens.re_sum[static_cast<std::size_t>(k)])) +
                     dphi_term);
                const double drj =
                    q_kj *
                    ((1.0 - b * (1.0 - phi)) /
                         static_cast<double>(expf(
                             dens.re_sum[static_cast<std::size_t>(j)])) +
                     dphi_term);

                const float re_std_sq = dens.re_std * dens.re_std;  // f4
                const float weight_k =                              // all f4
                    dens.R[static_cast<std::size_t>(k)] -
                    dens.re_cov *
                        (dens.re_sum[static_cast<std::size_t>(k)] -
                         dens.re_mean) /
                        re_std_sq;
                const float weight_j =
                    dens.R[static_cast<std::size_t>(j)] -
                    dens.re_cov *
                        (dens.re_sum[static_cast<std::size_t>(j)] -
                         dens.re_mean) /
                        re_std_sq;

                grad_cor_coeff =
                    dens.lambda * static_cast<double>(dens.mu_tot) *
                    (static_cast<double>(weight_k) * drk +
                     static_cast<double>(weight_j) * drj) /
                    static_cast<double>(dens.mu[i] * dens.re_std) /
                    static_cast<double>(n_vertices);
            }
        }
        // --------------------------------------------------------------------

        double grad_coeff;
        if (dist_squared > 0.0f) {
            grad_coeff = -2.0 * a * b *
                         pow_sel<FASTPOW>(static_cast<double>(dist_squared),
                                          b - 1.0);
            grad_coeff /=
                a * pow_sel<FASTPOW>(static_cast<double>(dist_squared), b) + 1.0;
        } else {
            grad_coeff = 0.0;
        }

        const int dloop = DIM > 0 ? DIM : dim;
        for (int d = 0; d < dloop; ++d) {
            double grad_d =
                clip(grad_coeff * static_cast<double>(current[d] - other[d]));

            // --- densmap additions ------------------------------------------
            if constexpr (DENSMAP) {
                if (dens.flag) {
                    grad_d += clip(2.0 * grad_cor_coeff *
                                   static_cast<double>(current[d] - other[d]));
                }
            }
            // ----------------------------------------------------------------

            current[d] = static_cast<float>(current[d] + grad_d * alpha);
            if (move_other) {
                other[d] = static_cast<float>(other[d] + (-grad_d) * alpha);
            }
        }

        epoch_of_next_sample[i] += epochs_per_sample[i];

        // int() in numba truncates toward zero, as does the C++ cast.
        const long long n_neg_samples = static_cast<long long>(
            (static_cast<double>(n) - epoch_of_next_negative_sample[i]) /
            epochs_per_negative_sample[i]);

        int64_t* rng_state =
            rng_state_per_sample + 3 * static_cast<std::size_t>(j);

        for (long long p = 0; p < n_neg_samples; ++p) {
            // numba `%` follows Python semantics: result has the sign of the
            // (positive) divisor.
            long long r = static_cast<long long>(tau_rand_int(rng_state)) %
                          static_cast<long long>(n_vertices);
            if (r < 0) r += n_vertices;
            k = static_cast<int32_t>(r);

            other = tail_embedding + static_cast<std::size_t>(k) * stride;

            dist_squared = rdist_t<DIM>(current, other, dim);

            if (dist_squared > 0.0f) {
                grad_coeff = 2.0 * gamma * b;
                grad_coeff /=
                    (0.001 + static_cast<double>(dist_squared)) *
                    (a * pow_sel<FASTPOW>(static_cast<double>(dist_squared), b) +
                     1.0);
            } else if (j == k) {
                continue;
            } else {
                grad_coeff = 0.0;
            }

            for (int d = 0; d < dloop; ++d) {
                double grad_d;
                if (grad_coeff > 0.0) {
                    grad_d = clip(grad_coeff *
                                  static_cast<double>(current[d] - other[d]));
                } else {
                    grad_d = 0.0;
                }
                current[d] = static_cast<float>(current[d] + grad_d * alpha);
            }
        }

        epoch_of_next_negative_sample[i] +=
            static_cast<double>(n_neg_samples) *
            epochs_per_negative_sample[i];
    }
}

// Runtime dispatch over the compile-time epoch-kernel specializations. Picks
// the DIM==2 unroll for the common 2-D fit and a generic DIM==0 kernel
// otherwise; DENSMAP selects the density variant. Every instantiation here is
// FASTPOW=false and bit-identical to the pre-refactor scalar kernel. (The
// `fastpow` parameter is accepted for forward compatibility with the unseeded
// fast-pow change; it currently only ever selects the libm path.)
inline void dispatch_epoch_range(
    bool densmap_active, bool fastpow, std::size_t i_begin, std::size_t i_end,
    float* head_embedding, float* tail_embedding, const int32_t* head,
    const int32_t* tail, int n_vertices, const double* epochs_per_sample,
    double a, double b, int64_t* rng_state_per_sample, double gamma, int dim,
    bool move_other, double alpha, const double* epochs_per_negative_sample,
    double* epoch_of_next_negative_sample, double* epoch_of_next_sample, int n,
    const DensmapEpochState& dens) {
    (void)fastpow;  // always the libm path in this change
#define UMAP_DISPATCH_CALL(D, DM)                                             \
    optimize_euclidean_epoch_range_t<D, DM, false>(                          \
        i_begin, i_end, head_embedding, tail_embedding, head, tail,          \
        n_vertices, epochs_per_sample, a, b, rng_state_per_sample, gamma,    \
        dim, move_other, alpha, epochs_per_negative_sample,                  \
        epoch_of_next_negative_sample, epoch_of_next_sample, n, dens)
    const bool dim2 = (dim == 2);
    if (densmap_active) {
        if (dim2) UMAP_DISPATCH_CALL(2, true);
        else UMAP_DISPATCH_CALL(0, true);
    } else {
        if (dim2) UMAP_DISPATCH_CALL(2, false);
        else UMAP_DISPATCH_CALL(0, false);
    }
#undef UMAP_DISPATCH_CALL
}
// -----------------------------------------------------------------------------

// --- perf P6 additions -------------------------------------------------------
// Persistent worker pool for the hogwild parallel SGD path. The previous
// implementation spawned/joined n_threads std::threads every epoch (500+ epochs
// x ~10 threads = thousands of spawns per fit); this reuses a fixed set of
// workers across all epochs. The work partition is identical to the old code
// (contiguous static [lo, hi) chunks, chunk t = [t*chunk, (t+1)*chunk)), and
// each worker runs the same dispatch_epoch_range as before -- the hogwild races
// (racy float32 embedding / rng-state writes) are unchanged, so this is the
// SAFE reorganization the non-deterministic contract allows: same work, no
// respawn. The main thread runs chunk 0 itself; workers cover chunks 1..T-1.
class EpochThreadPool {
   public:
    // n_workers background threads handle chunks 1..n_workers (chunk 0 is run
    // inline by the caller in run_epoch).
    explicit EpochThreadPool(unsigned n_workers) : n_workers_(n_workers) {
        workers_.reserve(n_workers_);
        for (unsigned t = 0; t < n_workers_; ++t) {
            workers_.emplace_back([this, t] { worker_loop(t + 1); });
        }
    }

    ~EpochThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            stop_ = true;
            ++generation_;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    EpochThreadPool(const EpochThreadPool&) = delete;
    EpochThreadPool& operator=(const EpochThreadPool&) = delete;

    // Run `task` over chunk indices [0, total_chunks): the caller executes
    // chunk 0 inline; workers pick up 1..total_chunks-1. Blocks until all done.
    void run_epoch(const std::function<void(unsigned)>& task,
                   unsigned total_chunks) {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            task_ = &task;
            total_chunks_ = total_chunks;
            pending_.store((total_chunks > 1) ? (total_chunks - 1) : 0);
            ++generation_;
        }
        cv_.notify_all();
        // Main thread runs chunk 0 (if any).
        if (total_chunks > 0) task(0);
        // Wait for workers to finish their chunks.
        if (pending_.load() != 0) {
            std::unique_lock<std::mutex> lk(done_mtx_);
            done_cv_.wait(lk, [this] { return pending_.load() == 0; });
        }
    }

   private:
    void worker_loop(unsigned chunk_index) {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(unsigned)>* task = nullptr;
            unsigned total = 0;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk,
                         [this, &seen] { return generation_ != seen || stop_; });
                if (stop_) return;
                seen = generation_;
                task = task_;
                total = total_chunks_;
            }
            if (task != nullptr && chunk_index < total) {
                (*task)(chunk_index);
            }
            // Signal completion for this chunk (only chunks 1..total-1 count).
            if (chunk_index < total) {
                if (pending_.fetch_sub(1) == 1) {
                    std::lock_guard<std::mutex> lk(done_mtx_);
                    done_cv_.notify_one();
                }
            }
        }
    }

    const unsigned n_workers_;
    std::vector<std::thread> workers_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::mutex done_mtx_;
    std::condition_variable done_cv_;
    const std::function<void(unsigned)>* task_ = nullptr;
    unsigned total_chunks_ = 0;
    std::atomic<unsigned> pending_{0};
    uint64_t generation_ = 0;
    bool stop_ = false;
};
// -----------------------------------------------------------------------------

}  // namespace

// --- densmap-gpu additions ---------------------------------------------------
// Epoch-level densMAP reductions exported for the Metal host path (see
// layouts.hpp): identical numerics to the in-loop code below — the GPU run
// differs from the CPU one only through the (racy, float32) embedding state
// these are computed FROM.
float densmap_mu_tot(const std::vector<float>& mu_sum) {
    return numpy_sum_f32(mu_sum.data(), static_cast<int64_t>(mu_sum.size())) /
           2.0f;
}

DensmapEpochScalars densmap_epoch_stats(
    const float* head_embedding, const float* tail_embedding,
    const int32_t* head, const int32_t* tail, std::size_t n_edges, double a,
    double b, int dim, const DensmapParams& densmap_kwds, int n_vertices,
    std::vector<float>& re_sum, std::vector<float>& phi_sum) {
    re_sum.resize(static_cast<std::size_t>(n_vertices));
    phi_sum.resize(static_cast<std::size_t>(n_vertices));
    densmap_epoch_init(head_embedding, tail_embedding, head, tail, n_edges, a,
                       b, dim, re_sum, phi_sum);
    // dens_re_std = np.sqrt(np.var(re_sum) + var_shift): float32 (NEP 50: the
    // Python-float var_shift is weak). Likewise mean/cov.
    const int64_t nv = static_cast<int64_t>(re_sum.size());
    const float var = numpy_var_f32(re_sum.data(), nv);
    DensmapEpochScalars s;
    s.re_std = std::sqrt(var + static_cast<float>(densmap_kwds.var_shift));
    s.re_mean = numpy_mean_f32(re_sum.data(), nv);
    s.re_cov = numpy_dot_f32(re_sum.data(), densmap_kwds.R.data(), nv) /
               static_cast<float>(n_vertices - 1);
    return s;
}
// -----------------------------------------------------------------------------

void optimize_layout_euclidean(
    float* head_embedding, int n_head_vertices, float* tail_embedding, int dim,
    const std::vector<int32_t>& head, const std::vector<int32_t>& tail,
    int n_epochs, int n_vertices, const std::vector<double>& epochs_per_sample,
    double a, double b, const int64_t rng_state[3], double gamma,
    double initial_alpha, double negative_sample_rate, bool parallel,
    bool densmap, bool move_other, const std::vector<int>* epochs_list,
    std::vector<std::vector<float>>* embedding_list,
    const DensmapParams* densmap_kwds, int n_threads_cap) {
    // layouts.py: if isinstance(n_epochs, list): epochs_list = n_epochs;
    //             n_epochs = max(epochs_list)
    if (epochs_list != nullptr) {
        if (epochs_list->empty() || embedding_list == nullptr) {
            throw std::invalid_argument(
                "optimize_layout_euclidean: epochs_list requires a non-empty "
                "list and an embedding_list output");
        }
        n_epochs = *std::max_element(epochs_list->begin(), epochs_list->end());
        embedding_list->clear();
    }
    // --- densmap additions --------------------------------------------------
    if (densmap && densmap_kwds == nullptr) {
        throw std::invalid_argument(
            "optimize_layout_euclidean: densmap=true requires densmap_kwds");
    }
    // if densmap: dens_mu_tot = np.sum(densmap_kwds["mu_sum"]) / 2 (float32),
    // dens_phi_sum / dens_re_sum = np.zeros(n_vertices, dtype=np.float32).
    DensmapEpochState dens;
    std::vector<float> dens_phi_sum;
    std::vector<float> dens_re_sum;
    if (densmap) {
        dens.mu_tot = densmap_mu_tot(densmap_kwds->mu_sum);
        dens.lambda = densmap_kwds->lambda;
        dens.R = densmap_kwds->R.data();
        dens.mu = densmap_kwds->mu.data();
        dens_phi_sum.assign(static_cast<std::size_t>(n_vertices), 0.0f);
        dens_re_sum.assign(static_cast<std::size_t>(n_vertices), 0.0f);
        dens.phi_sum = dens_phi_sum.data();
        dens.re_sum = dens_re_sum.data();
    }
    // ------------------------------------------------------------------------

    double alpha = initial_alpha;
    const std::size_t n_edges = epochs_per_sample.size();

    std::vector<double> epochs_per_negative_sample(n_edges);
    for (std::size_t i = 0; i < n_edges; ++i) {
        epochs_per_negative_sample[i] =
            epochs_per_sample[i] / negative_sample_rate;
    }
    std::vector<double> epoch_of_next_negative_sample =
        epochs_per_negative_sample;
    std::vector<double> epoch_of_next_sample = epochs_per_sample;

    // rng_state_per_sample = np.full((n_head, 3), rng_state, dtype=np.int64)
    //     + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)
    // i.e. each head vertex gets its own state, offset by the int64 bit pattern
    // of the double-precision value of its first coordinate at entry (int64
    // addition wraps, as numpy's does).
    std::vector<int64_t> rng_state_per_sample(
        static_cast<std::size_t>(n_head_vertices) * 3);
    for (int j = 0; j < n_head_vertices; ++j) {
        const double v0 = static_cast<double>(
            head_embedding[static_cast<std::size_t>(j) * dim]);
        int64_t offset;
        static_assert(sizeof(offset) == sizeof(v0), "bitcast size");
        std::memcpy(&offset, &v0, sizeof(offset));
        for (int c = 0; c < 3; ++c) {
            rng_state_per_sample[static_cast<std::size_t>(j) * 3 + c] =
                static_cast<int64_t>(static_cast<uint64_t>(rng_state[c]) +
                                     static_cast<uint64_t>(offset));
        }
    }

    unsigned n_threads = 1;
    if (parallel) {
        // --- hardening additions: n_jobs caps the pool (upstream calls
        // numba.set_num_threads(n_jobs) before the prange loop); <= 0 keeps
        // the all-cores default. ------------------------------------------
        if (n_threads_cap > 0) {
            n_threads = static_cast<unsigned>(n_threads_cap);
        } else {
            n_threads = std::thread::hardware_concurrency();
            if (n_threads == 0) n_threads = 4;
        }
        // ------------------------------------------------------------------
        if (static_cast<std::size_t>(n_threads) > n_edges && n_edges > 0) {
            n_threads = static_cast<unsigned>(n_edges);
        }
    }

    // --- perf P6: persistent worker pool for the hogwild path (created once,
    // reused across all epochs). Only spun up when we will actually run the
    // parallel branch (n_threads > 1); n_threads-1 background workers cover
    // chunks 1..n_threads-1 while the main thread runs chunk 0. --------------
    const bool use_pool = parallel && n_threads > 1;
    std::unique_ptr<EpochThreadPool> pool;
    std::size_t chunk_size = 0;
    if (use_pool) {
        pool = std::make_unique<EpochThreadPool>(n_threads - 1);
        chunk_size = (n_edges + n_threads - 1) / n_threads;
    }

    for (int n = 0; n < n_epochs; ++n) {
        // --- densmap additions ----------------------------------------------
        // densmap_flag = densmap and lambda > 0
        //                and (n + 1) / n_epochs > 1 - frac
        dens.flag = densmap && (densmap_kwds->lambda > 0.0) &&
                    ((static_cast<double>(n + 1) /
                      static_cast<double>(n_epochs)) >
                     (1.0 - densmap_kwds->frac));
        if (dens.flag) {
            // densmap-gpu additions: hoisted into densmap_epoch_stats (shared
            // with the Metal host path); numerics unchanged.
            const DensmapEpochScalars s = densmap_epoch_stats(
                head_embedding, tail_embedding, head.data(), tail.data(),
                n_edges, a, b, dim, *densmap_kwds, n_vertices, dens_re_sum,
                dens_phi_sum);
            dens.re_std = s.re_std;
            dens.re_mean = s.re_mean;
            dens.re_cov = s.re_cov;
        } else {
            dens.re_std = 0.0f;
            dens.re_mean = 0.0f;
            dens.re_cov = 0.0f;
        }
        // --------------------------------------------------------------------
        if (!use_pool) {
            // Sequential numba path (parallel=False, or a 1-thread parallel
            // build): edges in order. Routed through the specialized dispatcher
            // (DIM==2 unroll + non-densMAP elision + edge-skip); FASTPOW=false
            // keeps it bit-exact.
            dispatch_epoch_range(
                densmap, /*fastpow=*/false, 0, n_edges, head_embedding,
                tail_embedding, head.data(), tail.data(), n_vertices,
                epochs_per_sample.data(), a, b, rng_state_per_sample.data(),
                gamma, dim, move_other, alpha,
                epochs_per_negative_sample.data(),
                epoch_of_next_negative_sample.data(),
                epoch_of_next_sample.data(), n, dens);
        } else {
            // "Hogwild" mirror of numba prange(parallel=True): contiguous
            // static chunks, racy float32 embedding / rng-state updates (that
            // is numba's semantics too; non-deterministic). The persistent
            // pool reuses workers across epochs; the chunk partition matches
            // the old per-epoch-spawn version exactly.
            const std::size_t chunk = chunk_size;
            std::function<void(unsigned)> epoch_task =
                [&](unsigned t) {
                    const std::size_t lo = static_cast<std::size_t>(t) * chunk;
                    const std::size_t hi = std::min(n_edges, lo + chunk);
                    if (lo >= hi) return;
                    dispatch_epoch_range(
                        densmap, /*fastpow=*/false, lo, hi, head_embedding,
                        tail_embedding, head.data(), tail.data(), n_vertices,
                        epochs_per_sample.data(), a, b,
                        rng_state_per_sample.data(), gamma, dim, move_other,
                        alpha, epochs_per_negative_sample.data(),
                        epoch_of_next_negative_sample.data(),
                        epoch_of_next_sample.data(), n, dens);
                };
            pool->run_epoch(epoch_task, n_threads);
        }

        alpha = initial_alpha *
                (1.0 - (static_cast<double>(n) / static_cast<double>(n_epochs)));

        // if epochs_list is not None and n in epochs_list:
        //     embedding_list.append(head_embedding.copy())
        if (epochs_list != nullptr &&
            std::find(epochs_list->begin(), epochs_list->end(), n) !=
                epochs_list->end()) {
            embedding_list->emplace_back(
                head_embedding,
                head_embedding +
                    static_cast<std::size_t>(n_head_vertices) * dim);
        }
    }

    // Add the last embedding to the list as well.
    if (epochs_list != nullptr) {
        embedding_list->emplace_back(
            head_embedding,
            head_embedding + static_cast<std::size_t>(n_head_vertices) * dim);
    }
}

// --- umap-class additions ---------------------------------------------------
int resolve_epochs_and_filter_graph(SparseMatrix& graph, int n_epochs,
                                    int default_epochs_boost,
                                    SparseMatrix* write_through) {
    // For smaller datasets we can use more epochs (+200 for densMAP, which
    // callers pass through default_epochs_boost).
    const int default_epochs =
        ((graph.n_rows <= 10000) ? 500 : 200) + default_epochs_boost;
    if (n_epochs < 0) n_epochs = default_epochs;  // n_epochs is None
    // For the list-of-epochs variant callers pass max(list) here (upstream's
    // n_epochs_max).
    const int n_epochs_max = n_epochs;

    // graph.data[graph.data < graph.data.max() / n_epochs_max] = 0.0
    // (threshold computed in float32, like np.float32.max() / float(n)).
    // An EMPTY graph raises exactly like numpy's graph.data.max() on a
    // zero-size array in upstream simplicial_set_embedding (e.g. a 2-point
    // fit whose only membership edges are the zeroed self-edges -- audit
    // 2026-08-29 fuzz n2: the previous empty-guard silently succeeded).
    if (graph.data.empty()) {
        throw std::invalid_argument(
            "zero-size array to reduction operation maximum which has no "
            "identity");
    }
    float data_max = graph.data[0];
    for (float v : graph.data) {
        // np.max (maximum.reduce) PROPAGATES NaN: under allow-nan a NaN
        // membership row turns the threshold into NaN and the filter into a
        // no-op (every `v < NaN` is false), exactly like upstream.
        if (std::isnan(v)) {
            data_max = v;
            break;
        }
        if (v > data_max) data_max = v;
    }
    const float threshold =
        data_max / static_cast<float>(n_epochs_max > 10 ? n_epochs_max
                                                        : default_epochs);
    for (float& v : graph.data) {
        if (v < threshold) v = 0.0f;
    }
    // scipy tocoo() data sharing: the filter writes through into the
    // caller's stored graph (structure untouched -- see layouts.hpp).
    if (write_through != nullptr) {
        for (float& v : write_through->data) {
            if (v < threshold) v = 0.0f;
        }
    }
    graph.eliminate_zeros();
    return n_epochs;
}
// ----------------------------------------------------------------------------

std::vector<float> simplicial_set_embedding_from_init(
    const SparseMatrix& graph_in, const float* init_embedding, int n_components,
    double a, double b, const int64_t rng_state[3], int n_epochs, double gamma,
    double initial_alpha, double negative_sample_rate, bool parallel,
    bool graph_prefiltered, const std::vector<int>* epochs_list,
    std::vector<std::vector<float>>* embedding_list,
    const DensmapParams* densmap_kwds, const OutputMetricFn* output_metric,
    bool numpy_int64_epochs, Device device, int n_threads) {
    // --- densmap additions --------------------------------------------------
    if (densmap_kwds != nullptr && output_metric != nullptr) {
        throw std::invalid_argument(
            "Non-Euclidean output metric not supported for densMAP.");
    }
    if (output_metric != nullptr && epochs_list != nullptr) {
        throw std::invalid_argument(
            "simplicial_set_embedding_from_init: n_epochs as a list is not "
            "supported for non-euclidean output metrics (the reference fails "
            "with a numba typing error)");
    }
    // ------------------------------------------------------------------------
    // graph = graph.tocoo(); graph.sum_duplicates() — identity for our
    // canonical CSR (row-major, per-row sorted, no duplicates).
    SparseMatrix graph = graph_in;
    const int n_samples = graph.n_rows;
    const int n_vertices = graph.n_cols;

    if (!graph_prefiltered) {
        // Use more epochs for densMAP (upstream: default_epochs += 200).
        n_epochs = resolve_epochs_and_filter_graph(
            graph, n_epochs, densmap_kwds != nullptr ? 200 : 0);
    } else if (n_epochs < 0) {
        throw std::invalid_argument(
            "simplicial_set_embedding_from_init: graph_prefiltered requires "
            "the resolved n_epochs");
    }
    const int n_epochs_max = n_epochs;

    // head = graph.row; tail = graph.col (COO of the canonical CSR).
    std::vector<int32_t> head;
    std::vector<int32_t> tail;
    head.reserve(graph.nnz());
    tail.reserve(graph.nnz());
    for (int32_t row = 0; row < graph.n_rows; ++row) {
        for (int32_t idx = graph.indptr[row]; idx < graph.indptr[row + 1];
             ++idx) {
            head.push_back(row);
            tail.push_back(graph.indices[idx]);
        }
    }

    std::vector<double> epochs_per_sample =
        numpy_int64_epochs
            ? make_epochs_per_sample_i64(graph.data.data(), graph.data.size(),
                                         n_epochs_max)
            : make_epochs_per_sample(graph.data.data(), graph.data.size(),
                                     n_epochs_max);

    // embedding = (10.0 * (embedding - min0) / (max0 - min0)).astype(f4, "C")
    // per-component, float32 arithmetic, multiplication before division.
    std::vector<float> embedding(
        init_embedding,
        init_embedding + static_cast<std::size_t>(n_samples) * n_components);
    for (int d = 0; d < n_components; ++d) {
        float mn = embedding[d];
        float mx = embedding[d];
        for (int i = 1; i < n_samples; ++i) {
            const float v = embedding[static_cast<std::size_t>(i) * n_components + d];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        const float span = mx - mn;
        for (int i = 0; i < n_samples; ++i) {
            float& v = embedding[static_cast<std::size_t>(i) * n_components + d];
            v = (10.0f * (v - mn)) / span;
        }
    }

    // Spectral/pca/random initialisation (+ noisy_scale_coords) happens
    // before this function; the densMAP aux (mu / mu_sum / R) and the
    // output_dens radii are the caller's concern (UMAP.fit) because they need
    // graph_dists_ / the numpy RandomState.
    if (output_metric != nullptr) {
        // euclidean_output=False: optimize_layout_generic (sequential numba
        // path; there is no parallel instantiation upstream either).
        optimize_layout_generic(embedding.data(), n_samples, embedding.data(),
                                n_components, head, tail, n_epochs, n_vertices,
                                epochs_per_sample, a, b, rng_state,
                                *output_metric, gamma, initial_alpha,
                                negative_sample_rate, /*move_other=*/true);
    } else {
        // --- phase4 additions ------------------------------------------------
        // Device::GPU routes this euclidean epoch loop to the Metal hogwild
        // kernel — including the densMAP correction (densmap-gpu additions;
        // per-epoch reductions stay host-side, see gpu_layout.hpp). Only the
        // snapshot list stays CPU; UMAP::fit gates and warns for that combo
        // before requesting GPU, so the check here is a safety net. Quality-
        // level parity only (see gpu_layout.hpp).
        bool gpu_done = false;
        if (device == Device::GPU && epochs_list == nullptr) {
#ifdef UMAP_ENABLE_METAL
            gpu_done = metal::gpu_optimize_layout_euclidean(
                embedding.data(), n_samples, embedding.data(), n_components,
                head, tail, n_epochs, n_vertices, epochs_per_sample, a, b,
                rng_state, gamma, initial_alpha, negative_sample_rate,
                /*move_other=*/true, densmap_kwds);
#endif
            // watchdog-hardening additions: false also covers a command
            // buffer failing mid-run (macOS GPU watchdog abort); the kernel
            // leaves `embedding` untouched in every false case, so the CPU
            // loop below restarts from the pristine [0, 10]-rescaled init,
            // never from a partially-optimized buffer.
            if (!gpu_done) {
                std::fprintf(stderr,
                             "UMAP: GPU layout SGD requested but the Metal "
                             "kernel is unavailable or failed; running on "
                             "the CPU.\n");
            }
        }
        if (!gpu_done) {
            optimize_layout_euclidean(
                embedding.data(), n_samples, embedding.data(), n_components,
                head, tail, n_epochs, n_vertices, epochs_per_sample, a, b,
                rng_state, gamma, initial_alpha, negative_sample_rate, parallel,
                /*densmap=*/densmap_kwds != nullptr, /*move_other=*/true,
                epochs_list, embedding_list, densmap_kwds, n_threads);
        }
        // ---------------------------------------------------------------------
    }
    return embedding;
}

}  // namespace umap
