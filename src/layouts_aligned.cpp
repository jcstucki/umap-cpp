// Faithful C++ port of layouts.py::optimize_layout_aligned_euclidean and its
// epoch kernel _optimize_layout_aligned_euclidean_single_epoch (lmcinnes/umap,
// BSD 3 clause), plus a bit-exact replication of the numba np.random state the
// kernel drives (seed + shuffle + uniform). See include/umap/aligned_umap.hpp
// for the typing / ordering notes.
//
// The reference kernel is njit(parallel=True) but contains no prange, so it is
// sequential and deterministic; this port is the same.
#include "umap/aligned_umap.hpp"

#include <cmath>
#include <cstdlib>
#include <numeric>

#include "umap/layouts.hpp"  // clip / rdist
#include "umap/rng.hpp"      // tau_rand_int

namespace umap {
namespace aligned {

// ----------------------------------------------------------- NumbaNpRandom
// MT19937 exactly as numba's _random.c (== numpy's legacy scalar seeding ==
// std::mt19937): init_genrand seeding, standard twist, standard tempering.

void NumbaNpRandom::seed(uint32_t s) {
    // numba_rnd_init: Knuth's PRNG seeding as in the MT reference impl.
    for (int pos = 0; pos < 624; ++pos) {
        mt_[pos] = s;
        s = static_cast<uint32_t>(1812433253u * (s ^ (s >> 30)) + pos + 1);
    }
    mti_ = 624;
}

uint32_t NumbaNpRandom::next_u32() {
    if (mti_ >= 624) {
        // numba_rnd_shuffle (the MT twist).
        for (int i = 0; i < 624; ++i) {
            uint32_t y = (mt_[i] & 0x80000000u) | (mt_[(i + 1) % 624] & 0x7fffffffu);
            uint32_t v = mt_[(i + 397) % 624] ^ (y >> 1);
            if (y & 1u) v ^= 0x9908b0dfu;
            mt_[i] = v;
        }
        mti_ = 0;
    }
    uint32_t y = mt_[mti_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

uint32_t NumbaNpRandom::randint_below(uint32_t n) {
    // numba's np.random.randint(0, n): nbits = bit_length(n - 1) masked bits
    // of one tempered draw per attempt, rejecting r >= n (n == 1 draws
    // nothing and returns 0 -- the special case in _randrange_impl).
    if (n <= 1) return 0;
    const uint32_t nm1 = n - 1;
    int nbits = 32 - __builtin_clz(nm1);
    const uint32_t mask =
        (nbits >= 32) ? 0xFFFFFFFFu : ((1u << nbits) - 1u);
    uint32_t r;
    do {
        r = next_u32() & mask;
    } while (r >= n);
    return r;
}

void NumbaNpRandom::shuffle(int32_t* arr, int64_t n) {
    // numba's np.random.shuffle: downward Fisher-Yates, j = randint(i + 1).
    for (int64_t i = n - 1; i > 0; --i) {
        const int64_t j = static_cast<int64_t>(
            randint_below(static_cast<uint32_t>(i + 1)));
        const int32_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

double NumbaNpRandom::uniform(double low, double high) {
    // get_next_double: (a * 2^26 + b) / 9007199254740992.0 from two tempered
    // draws, then low + (high - low) * u. Statements kept split so clang
    // cannot contract the final mul+add into an fma (numba emits separate
    // fmul/fadd; contraction flips low bits -- observed on the golden draws).
    const double a = static_cast<double>(next_u32() >> 5);
    const double b = static_cast<double>(next_u32() >> 6);
    const double scaled_a = a * 67108864.0;  // exact (a < 2^27)
    const double u = (scaled_a + b) / 9007199254740992.0;
    const double span = (high - low) * u;
    return low + span;
}

// ------------------------------------------------------------- SGD kernel

namespace {

// Unchecked-in-spirit tensor reads matching numba's flat pointer arithmetic
// for t[m, r, k]: indices that spill past a row still read within the
// allocation exactly like the jitted code would; reads past the whole
// allocation (true UB upstream) return the "no relation" values.
inline int32_t rel_flat(const Tensor3i& t, int64_t m, int64_t r, int64_t k) {
    const int64_t flat = (m * t.d1 + r) * t.d2 + k;
    if (flat < 0 || flat >= static_cast<int64_t>(t.v.size())) return -1;
    return t.v[static_cast<std::size_t>(flat)];
}

inline float weight_flat(const Tensor3f& t, int64_t m, int64_t r, int64_t k) {
    const int64_t flat = (m * t.d1 + r) * t.d2 + k;
    if (flat < 0 || flat >= static_cast<int64_t>(t.v.size())) return 0.0f;
    return t.v[static_cast<std::size_t>(flat)];
}

}  // namespace

void optimize_layout_aligned_euclidean(
    std::vector<std::vector<float>>& head_embeddings,
    std::vector<std::vector<float>>& tail_embeddings, int dim,
    const std::vector<std::vector<int32_t>>& heads,
    const std::vector<std::vector<int32_t>>& tails, int n_epochs,
    const std::vector<std::vector<double>>& epochs_per_sample,
    const Tensor3f& regularisation_weights, const Tensor3i& relations,
    int64_t rng_state[3], NumbaNpRandom& np_rng, double a, double b,
    double gamma, double lambda_, double initial_alpha,
    double negative_sample_rate, bool move_other) {
    const int64_t n_embeddings = static_cast<int64_t>(heads.size());
    const int64_t window_size = (relations.d1 - 1) / 2;

    // Wrapper-level bookkeeping: float32 typed lists.
    //   epochs_per_negative_sample = eps.astype(f4) / negative_sample_rate
    //     (f4 array / python float -> f4 arithmetic under NEP 50)
    //   epoch_of_next_negative_sample = its copy
    //   epoch_of_next_sample = eps.astype(f4)
    std::vector<std::vector<float>> epochs_per_negative_sample(
        static_cast<std::size_t>(n_embeddings));
    std::vector<std::vector<float>> epoch_of_next_negative_sample(
        static_cast<std::size_t>(n_embeddings));
    std::vector<std::vector<float>> epoch_of_next_sample(
        static_cast<std::size_t>(n_embeddings));
    for (int64_t m = 0; m < n_embeddings; ++m) {
        const auto& eps = epochs_per_sample[static_cast<std::size_t>(m)];
        auto& epns = epochs_per_negative_sample[static_cast<std::size_t>(m)];
        auto& eonns =
            epoch_of_next_negative_sample[static_cast<std::size_t>(m)];
        auto& eons = epoch_of_next_sample[static_cast<std::size_t>(m)];
        epns.resize(eps.size());
        eons.resize(eps.size());
        const float nsr = static_cast<float>(negative_sample_rate);
        for (std::size_t e = 0; e < eps.size(); ++e) {
            const float e_f4 = static_cast<float>(eps[e]);
            epns[e] = e_f4 / nsr;
            eons[e] = e_f4;
        }
        eonns = epns;
    }

    // max_n_edges (recomputed per epoch upstream; constant, hoisted).
    int64_t max_n_edges = 0;
    for (const auto& eps : epochs_per_sample) {
        if (static_cast<int64_t>(eps.size()) >= max_n_edges)
            max_n_edges = static_cast<int64_t>(eps.size());
    }

    double alpha = initial_alpha;
    std::vector<int32_t> embedding_order(
        static_cast<std::size_t>(n_embeddings));

    for (int n = 0; n < n_epochs; ++n) {
        // embedding_order = np.arange(n_embeddings); np.random.seed(
        // abs(rng_state[0])); np.random.shuffle(embedding_order).
        // abs() of the current (mutating) tau word; int64 -> uint32 trunc.
        std::iota(embedding_order.begin(), embedding_order.end(), 0);
        const uint64_t abs_word =
            rng_state[0] < 0
                ? 0ull - static_cast<uint64_t>(rng_state[0])
                : static_cast<uint64_t>(rng_state[0]);
        np_rng.seed(static_cast<uint32_t>(abs_word));
        np_rng.shuffle(embedding_order.data(), n_embeddings);

        for (int64_t i = 0; i < max_n_edges; ++i) {
            for (int64_t mo = 0; mo < n_embeddings; ++mo) {
                const int64_t m = embedding_order[static_cast<std::size_t>(mo)];
                auto& eons = epoch_of_next_sample[static_cast<std::size_t>(m)];
                if (!(i < static_cast<int64_t>(eons.size()) &&
                      static_cast<double>(eons[static_cast<std::size_t>(i)]) <=
                          static_cast<double>(n)))
                    continue;

                const auto& head = heads[static_cast<std::size_t>(m)];
                const auto& tail = tails[static_cast<std::size_t>(m)];
                float* head_emb =
                    head_embeddings[static_cast<std::size_t>(m)].data();
                float* tail_emb =
                    tail_embeddings[static_cast<std::size_t>(m)].data();

                const int32_t j = head[static_cast<std::size_t>(i)];
                int64_t k = tail[static_cast<std::size_t>(i)];

                float* current = head_emb + static_cast<std::size_t>(j) * dim;
                float* other = tail_emb + static_cast<std::size_t>(k) * dim;

                float dist_squared = rdist(current, other, dim);

                double grad_coeff;
                if (dist_squared > 0.0f) {
                    grad_coeff =
                        -2.0 * a * b *
                        std::pow(static_cast<double>(dist_squared), b - 1.0);
                    grad_coeff /=
                        a * std::pow(static_cast<double>(dist_squared), b) +
                        1.0;
                } else {
                    grad_coeff = 0.0;
                }

                for (int d = 0; d < dim; ++d) {
                    double grad_d = clip(
                        grad_coeff *
                        static_cast<double>(current[d] - other[d]));

                    for (int64_t offset = -window_size; offset < window_size;
                         ++offset) {
                        const int64_t neighbor_m = m + offset;
                        // n_embeddings > neighbor_m >= 0 != offset
                        if (neighbor_m < n_embeddings && neighbor_m >= 0 &&
                            offset != 0) {
                            const int32_t identified_index = rel_flat(
                                relations, m, offset + window_size, j);
                            if (identified_index >= 0 &&
                                static_cast<std::size_t>(identified_index) *
                                        dim <
                                    head_embeddings[static_cast<std::size_t>(
                                                        neighbor_m)]
                                        .size()) {
                                const double lam_off =
                                    lambda_ *
                                    std::exp(static_cast<double>(
                                        -(std::llabs(offset) - 1)));
                                const float w = weight_flat(
                                    regularisation_weights, m,
                                    offset + window_size, j);
                                const float diff =
                                    current[d] -
                                    head_embeddings[static_cast<std::size_t>(
                                        neighbor_m)]
                                        [static_cast<std::size_t>(
                                             identified_index) *
                                             dim +
                                         d];
                                grad_d -= clip(lam_off *
                                               static_cast<double>(w) *
                                               static_cast<double>(diff));
                            }
                        }
                    }

                    current[d] = static_cast<float>(
                        current[d] + clip(grad_d) * alpha);
                    if (move_other) {
                        // Recomputed AFTER current[d] moved (upstream order).
                        double other_grad_d = clip(
                            grad_coeff *
                            static_cast<double>(other[d] - current[d]));

                        for (int64_t offset = -window_size;
                             offset < window_size; ++offset) {
                            const int64_t neighbor_m = m + offset;
                            if (neighbor_m < n_embeddings && neighbor_m >= 0 &&
                                offset != 0) {
                                const int32_t identified_index = rel_flat(
                                    relations, m, offset + window_size, k);
                                if (identified_index >= 0 &&
                                    static_cast<std::size_t>(
                                        identified_index) *
                                            dim <
                                        head_embeddings
                                            [static_cast<std::size_t>(
                                                 neighbor_m)]
                                                .size()) {
                                    const double lam_off =
                                        lambda_ *
                                        std::exp(static_cast<double>(
                                            -(std::llabs(offset) - 1)));
                                    const float w = weight_flat(
                                        regularisation_weights, m,
                                        offset + window_size, k);
                                    const float diff =
                                        other[d] -
                                        head_embeddings
                                            [static_cast<std::size_t>(
                                                 neighbor_m)]
                                            [static_cast<std::size_t>(
                                                 identified_index) *
                                                 dim +
                                             d];
                                    other_grad_d -=
                                        clip(lam_off *
                                             static_cast<double>(w) *
                                             static_cast<double>(diff));
                                }
                            }
                        }

                        other[d] = static_cast<float>(
                            other[d] + clip(other_grad_d) * alpha);
                    }
                }

                // epoch_of_next_sample[m][i] += epochs_per_sample[m][i]
                // (f4 + f8 -> f8, rounded back to f4 on store).
                eons[static_cast<std::size_t>(i)] = static_cast<float>(
                    static_cast<double>(eons[static_cast<std::size_t>(i)]) +
                    epochs_per_sample[static_cast<std::size_t>(m)]
                                     [static_cast<std::size_t>(i)]);

                auto& epns =
                    epochs_per_negative_sample[static_cast<std::size_t>(m)];
                auto& eonns = epoch_of_next_negative_sample
                    [static_cast<std::size_t>(m)];
                long long n_neg_samples;
                if (epns[static_cast<std::size_t>(i)] > 0.0f) {
                    n_neg_samples = static_cast<long long>(
                        (static_cast<double>(n) -
                         static_cast<double>(
                             eonns[static_cast<std::size_t>(i)])) /
                        static_cast<double>(epns[static_cast<std::size_t>(i)]));
                } else {
                    n_neg_samples = 0;
                }

                const int64_t n_tail_vertices =
                    static_cast<int64_t>(
                        tail_embeddings[static_cast<std::size_t>(m)].size()) /
                    dim;

                for (long long p = 0; p < n_neg_samples; ++p) {
                    // Python-modulo negative sampling on the SHARED state.
                    long long r =
                        static_cast<long long>(tau_rand_int(rng_state)) %
                        static_cast<long long>(n_tail_vertices);
                    if (r < 0) r += n_tail_vertices;
                    k = r;

                    other = tail_emb + static_cast<std::size_t>(k) * dim;

                    dist_squared = rdist(current, other, dim);

                    if (dist_squared > 0.0f) {
                        grad_coeff = 2.0 * gamma * b;
                        grad_coeff /=
                            (0.001 + static_cast<double>(dist_squared)) *
                            (a * std::pow(static_cast<double>(dist_squared),
                                          b) +
                             1.0);
                    } else if (j == k) {
                        continue;
                    } else {
                        grad_coeff = 0.0;
                    }

                    for (int d = 0; d < dim; ++d) {
                        double grad_d;
                        if (grad_coeff > 0.0) {
                            grad_d = clip(
                                grad_coeff *
                                static_cast<double>(current[d] - other[d]));
                        } else {
                            grad_d = 0.0;
                        }

                        for (int64_t offset = -window_size;
                             offset < window_size; ++offset) {
                            const int64_t neighbor_m = m + offset;
                            if (neighbor_m < n_embeddings && neighbor_m >= 0 &&
                                offset != 0) {
                                const int32_t identified_index = rel_flat(
                                    relations, m, offset + window_size, j);
                                if (identified_index >= 0 &&
                                    static_cast<std::size_t>(
                                        identified_index) *
                                            dim <
                                        head_embeddings
                                            [static_cast<std::size_t>(
                                                 neighbor_m)]
                                                .size()) {
                                    const double lam_off =
                                        lambda_ *
                                        std::exp(static_cast<double>(
                                            -(std::llabs(offset) - 1)));
                                    const float w = weight_flat(
                                        regularisation_weights, m,
                                        offset + window_size, j);
                                    const float diff =
                                        current[d] -
                                        head_embeddings
                                            [static_cast<std::size_t>(
                                                 neighbor_m)]
                                            [static_cast<std::size_t>(
                                                 identified_index) *
                                                 dim +
                                             d];
                                    grad_d -= clip(lam_off *
                                                   static_cast<double>(w) *
                                                   static_cast<double>(diff));
                                }
                            }
                        }

                        current[d] = static_cast<float>(
                            current[d] + clip(grad_d) * alpha);
                    }
                }

                // eonns += n_neg_samples * epns (int64 * f4 -> f8; f4 store).
                eonns[static_cast<std::size_t>(i)] = static_cast<float>(
                    static_cast<double>(eonns[static_cast<std::size_t>(i)]) +
                    static_cast<double>(n_neg_samples) *
                        static_cast<double>(
                            epns[static_cast<std::size_t>(i)]));
            }
        }

        alpha = initial_alpha *
                (1.0 - (static_cast<double>(n) / static_cast<double>(n_epochs)));
    }
}

}  // namespace aligned
}  // namespace umap
