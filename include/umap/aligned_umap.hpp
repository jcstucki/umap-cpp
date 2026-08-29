// Faithful C++ port of AlignedUMAP from reference/umap/umap/aligned_umap.py
// plus optimize_layout_aligned_euclidean from reference/umap/umap/layouts.py
// (lmcinnes/umap, BSD 3 clause).
//
// Pipeline ported (fit):
//   * per-slice UMAP fits with ParamList broadcasting (get_nth_item_or_val
//     semantics for n_neighbors / min_dist / n_epochs / repulsion_strength /
//     learning_rate / spread / negative_sample_rate / local_connectivity /
//     set_op_mix_ratio / unique; everything else -- metric, random_state,
//     a, b, init, target_* ... -- passed to every mapper unchanged);
//   * expand_relations: windowed chaining of the relation dicts (and their
//     inverses) into an int32 (n_models, 2*ws+1, max_n_samples) tensor,
//     including the upstream forward off-by-one (`i + j + 1 >= len(dicts)`
//     blanks chains that would target the last model);
//   * build_neighborhood_similarities: per (model, offset, sample) Jaccard
//     similarity of graph neighborhoods mapped through the relations, with
//     umap.sparse.arr_intersect / arr_union counting semantics (concatenate +
//     sort; duplicates inside ONE side count toward the intersection, and a
//     union with an empty side skips deduplication -- both replicated);
//   * spectral_layout per slice on the *stored* graph_ (which, like upstream,
//     still contains the explicit zeros written through by the epoch-weight
//     filter -- those edges get epochs_per_sample == -1 and are then sampled
//     every epoch by the aligned SGD, an upstream quirk we replicate),
//     scaled by 10 / max|coord| WITHOUT noisy_scale_coords jitter;
//   * procrustes_align of each slice onto the previous one over the
//     offset -1 relation anchors (M = S2^T S1 in float32, R = U @ Vh);
//   * seed_triplet = RandomState(transform_seed).randint(INT32_MIN+1,
//     INT32_MAX-1, 3) *after* the spectral draws (numpy stream continuation);
//   * optimize_layout_aligned_euclidean with upstream's hard-coded defaults:
//     a = 1.576943460405378, b = 0.8950608781227859, gamma = 1.0,
//     initial_alpha = 1.0, negative_sample_rate = 5.0 (fit does NOT forward
//     the per-model a/b/learning_rate/repulsion_strength -- upstream quirk),
//     lambda_ = alignment_regularisation, move_other = true;
//   * NaN masking of disconnected vertices (graph_ row sums == 0).
//
// update(X, relations) ported too: set_aligned_params merging of new
// per-model params, one new mapper fit, old models frozen via
// epochs_per_sample = full(n_samples, n_epochs + 1), init_from_existing
// (relation copy / graph-weighted mean / uniform fallback), fresh
// RandomState(transform_seed) seed triplet, move_other = false (upstream
// omits it), and NO NaN masking afterwards (upstream doesn't).
//
// The aligned SGD kernel replicates the numba typing of
// _optimize_layout_aligned_euclidean_single_epoch exactly:
//   * epoch bookkeeping lists are float32 (unlike the euclidean SGD's f8):
//     epochs_per_negative_sample = f4(epochs_per_sample) / 5.0 in f4
//     arithmetic (NEP 50), and the += updates promote f4 + f8 -> f8 then
//     round back to f4 on store;
//   * n_neg_samples = int((n - f4) / f4) evaluated in f8 (with the aligned
//     kernel's `epochs_per_negative_sample > 0` guard the euclidean kernel
//     lacks), and the bookkeeping increment int64 * f4 -> f8;
//   * gradients in f8 over f4 embedding reads, clip'd twice like upstream;
//     the alignment penalty  clip(lambda_ * exp(-(|offset|-1)) * w * (x_d -
//     x_neighbor_d))  is subtracted inside the per-dimension loop for the
//     window offsets -ws .. ws-1 (the +ws column is never read -- upstream
//     loops `range(-window_size, window_size)`);
//   * move_other recomputes the tail gradient AFTER current[d] was updated
//     (unlike the euclidean kernel's negated-gradient reuse);
//   * ONE shared tau_prng rng_state drives all negative sampling in the
//     shuffled (i, m) traversal order;
//   * per epoch, the embedding traversal order over models is shuffled with
//     numba's np.random: np.random.seed(abs(rng_state[0])) -- the *current*,
//     tau-mutated state word -- then a Fisher-Yates shuffle whose bounded
//     draws take bit_length(i) masked bits per attempt from MT19937
//     (NumbaNpRandom below is a verified bit-exact replication).
//
// Documented deviations:
//   * The eigensolver behind spectral_layout is Spectra (like the rest of the
//     port), so per-slice spectral inits match ARPACK only to ~1e-6 and up to
//     eigenvector sign; downstream SGD is chaotic, so full-fit parity with
//     Python is validated at the upstream tests' quality thresholds, while
//     the aligned SGD itself is validated bit-exactly from Python-captured
//     inputs (tests/test_aligned.cpp).
//   * procrustes_align uses Eigen's JacobiSVD and sequential f32 matmuls
//     rather than numba's LAPACK sgesdd + BLAS sgemm: same math, agreement to
//     ~1e-5 (paired sign flips of the SVD factors cancel in R = U @ Vh).
//   * Relation dicts are canonicalised in ascending key order; upstream
//     Python dicts remember insertion order, which only matters for
//     invert_dict when two keys share a value (last insertion wins upstream,
//     largest key wins here).
//   * init_from_existing's uniform(-10, 10) fallback draws from the same
//     NumbaNpRandom state the epoch shuffles use (numba shares one internal
//     np.random state per thread), so it is bit-exact *given* the C++ SGD ran
//     the preceding fit; upstream's state is process-global rather than
//     per-AlignedUMAP-instance.
//   * n_epochs must stay a scalar: upstream nominally accepts a per-model
//     list but then crashes inside make_epochs_per_sample / numba typing when
//     the aligned pass consumes it, so the C++ port rejects lists up front
//     (and update() rejects changing n_epochs, where upstream would build a
//     tuple and crash the same way).
//   * low_memory is not exposed (the C++ NN-descent implements the
//     low_memory=True code path; upstream AlignedUMAP passes low_memory=False
//     to its mappers, which only changes pynndescent internals on the
//     n >= 4096 branch).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umap/sparse_matrix.hpp"
#include "umap/umap.hpp"

namespace umap {
namespace aligned {

// ------------------------------------------------------------ small tensors

// Row-major 3-D tensors (d0 = n_models, d1 = 2*window_size + 1, d2 =
// max_n_samples). `at` clamps nothing: callers use flat() for the
// numba-style unchecked flat indexing the SGD kernel needs.
struct Tensor3i {
    int64_t d0 = 0, d1 = 0, d2 = 0;
    std::vector<int32_t> v;
    Tensor3i() = default;
    Tensor3i(int64_t a, int64_t b, int64_t c, int32_t fill)
        : d0(a), d1(b), d2(c),
          v(static_cast<std::size_t>(a * b * c), fill) {}
    int32_t at(int64_t a, int64_t b, int64_t c) const {
        return v[static_cast<std::size_t>((a * d1 + b) * d2 + c)];
    }
    int32_t& at(int64_t a, int64_t b, int64_t c) {
        return v[static_cast<std::size_t>((a * d1 + b) * d2 + c)];
    }
};

struct Tensor3f {
    int64_t d0 = 0, d1 = 0, d2 = 0;
    std::vector<float> v;
    Tensor3f() = default;
    Tensor3f(int64_t a, int64_t b, int64_t c, float fill)
        : d0(a), d1(b), d2(c),
          v(static_cast<std::size_t>(a * b * c), fill) {}
    float at(int64_t a, int64_t b, int64_t c) const {
        return v[static_cast<std::size_t>((a * d1 + b) * d2 + c)];
    }
    float& at(int64_t a, int64_t b, int64_t c) {
        return v[static_cast<std::size_t>((a * d1 + b) * d2 + c)];
    }
};

// A relation dict canonicalised to ascending key order (see the insertion
// order deviation note above). Keys are rows of slice t, values rows of
// slice t+1.
using RelationVec = std::vector<std::pair<int32_t, int32_t>>;

RelationVec canonicalize_relations(
    const std::unordered_map<int32_t, int32_t>& d);

// aligned_umap.invert_dict: {value: key}; later keys overwrite on value
// collisions (ascending-key iteration order here).
RelationVec invert_relations(const RelationVec& d);

// ------------------------------------------------- numba np.random parity

// Bit-exact replication of numba's internal np.random state as the aligned
// epoch kernel uses it: np.random.seed(s) (MT19937 init_genrand -- identical
// to std::mt19937 seeding), np.random.shuffle (downward Fisher-Yates whose
// per-step draw takes bit_length(i) masked bits of one tempered 32-bit
// output, rejecting >= i+1) and np.random.uniform (two 32-bit draws per
// double, (a*2^26 + b) / 2^53). Verified draw-for-draw against numba 0.67.
class NumbaNpRandom {
 public:
    // numba states start uninitialised-but-seeded-lazily; every consumer in
    // this port seeds before drawing (the epoch kernel reseeds per epoch).
    void seed(uint32_t s);
    void shuffle(int32_t* arr, int64_t n);
    double uniform(double low, double high);

 private:
    uint32_t next_u32();
    uint32_t randint_below(uint32_t n);  // np.random.randint(0, n), n >= 1
    uint32_t mt_[624];
    int mti_ = 625;
};

// ------------------------------------------------------- pipeline pieces

// aligned_umap.expand_relations: (len(dicts)+1, 2*ws+1, max_n_samples) int32
// tensor of relation chains; -1 marks "no correspondence". Row ws+o holds
// the chain from model i to model i+o. Throws std::invalid_argument for an
// empty dict (upstream dies in max()).
Tensor3i expand_relations(const std::vector<RelationVec>& relation_dicts,
                          int window_size = 3);

// aligned_umap.build_neighborhood_similarities over the models' graph_ CSRs
// (indptr/indices only; explicit zeros participate, as in the reference).
Tensor3f build_neighborhood_similarities(
    const std::vector<const SparseMatrix*>& graphs, const Tensor3i& relations);

// aligned_umap.procrustes_align: rotate `embedding_to_align` (n2 x dim) onto
// `embedding_base` (n1 x dim) over anchor row pairs (anchors_base[i] in the
// base embedding matched with anchors_align[i] in the embedding to align).
// float32 matmuls + SVD of the d x d covariance; returns n2 x dim.
std::vector<float> procrustes_align(const std::vector<float>& embedding_base,
                                    const std::vector<float>& embedding_to_align,
                                    int dim,
                                    const std::vector<int32_t>& anchors_base,
                                    const std::vector<int32_t>& anchors_align);

// aligned_umap.init_from_existing: initial embedding for a new slice from
// the previous slice's embedding. `relation` maps NEW rows -> PREVIOUS rows
// (the caller inverts the update dict). Rows with a direct relation copy the
// previous position; others take the graph-weighted mean of their related
// neighbours' positions; rows with no related neighbour draw
// np.random.uniform(-10, 10) from `np_rng` (numba's shared np state).
std::vector<float> init_from_existing(const std::vector<float>& prev_embedding,
                                      int n_features, const SparseMatrix& graph,
                                      const RelationVec& relation,
                                      NumbaNpRandom& np_rng);

// layouts.optimize_layout_aligned_euclidean. head_embeddings and
// tail_embeddings may reference the same object (they do in both upstream
// call sites); embeddings are modified in place. rng_state is the shared
// 3-word tau_prng seed triplet (mutated); np_rng is numba's np.random state
// (reseeded from abs(rng_state[0]) at each epoch for the model-order
// shuffle). Defaults mirror the upstream signature exactly.
void optimize_layout_aligned_euclidean(
    std::vector<std::vector<float>>& head_embeddings,
    std::vector<std::vector<float>>& tail_embeddings, int dim,
    const std::vector<std::vector<int32_t>>& heads,
    const std::vector<std::vector<int32_t>>& tails, int n_epochs,
    const std::vector<std::vector<double>>& epochs_per_sample,
    const Tensor3f& regularisation_weights, const Tensor3i& relations,
    int64_t rng_state[3], NumbaNpRandom& np_rng,
    double a = 1.576943460405378, double b = 0.8950608781227859,
    double gamma = 1.0, double lambda_ = 5e-3, double initial_alpha = 1.0,
    double negative_sample_rate = 5.0, bool move_other = false);

}  // namespace aligned

// ------------------------------------------------------------- param lists

// A parameter that may be a single value (broadcast to every model) or a
// per-model list, mirroring how AlignedUMAP feeds get_nth_item_or_val.
template <typename T>
class AlignedParam {
 public:
    AlignedParam(T value) : scalar_(value) {}  // NOLINT(runtime/explicit)
    AlignedParam(std::vector<T> values)        // NOLINT(runtime/explicit)
        : list_(std::move(values)) {}

    bool is_list() const { return list_.has_value(); }
    const std::vector<T>& list() const { return *list_; }

    // get_nth_item_or_val: list -> list[n] (IndexError -> std::out_of_range),
    // scalar -> the scalar.
    T get(std::size_t n) const {
        if (!list_) return scalar_;
        if (n >= list_->size())
            throw std::out_of_range(
                "AlignedUMAP: per-model parameter list is shorter than the "
                "number of models");
        return (*list_)[n];
    }

    // set_aligned_params for this parameter: existing list -> append;
    // otherwise a differing new value broadcasts the scalar n_models times
    // and appends (upstream builds a tuple; equal values are a no-op).
    void set_aligned(T new_value, int n_models) {
        if (list_) {
            list_->push_back(new_value);
        } else if (!(new_value == scalar_)) {
            list_.emplace(static_cast<std::size_t>(n_models), scalar_);
            list_->push_back(new_value);
        }
    }

 private:
    T scalar_{};
    std::optional<std::vector<T>> list_;
};

// Per-model parameter overrides handed to update(), i.e. the **fit_params of
// upstream's update(); only the PARAM_NAMES that AlignedUMAP.fit actually
// broadcasts per-model are supported (see header notes for n_epochs).
struct AlignedParamUpdate {
    std::optional<int> n_neighbors;
    std::optional<double> min_dist;
    std::optional<int> n_epochs;  // must equal the fitted value (see notes)
    std::optional<double> repulsion_strength;
    std::optional<double> learning_rate;
    std::optional<double> spread;
    std::optional<int> negative_sample_rate;
    std::optional<double> local_connectivity;
    std::optional<double> set_op_mix_ratio;
    std::optional<bool> unique;
};

// AlignedUMAP.__init__ defaults, verbatim.
struct AlignedUMAPParams {
    AlignedParam<int> n_neighbors = 15;
    int n_components = 2;  // must be a single integer (upstream validates)
    std::string metric = "euclidean";
    MetricKwds metric_kwds;
    std::optional<int> n_epochs;  // None -> 200 (stored back, like upstream)
    AlignedParam<double> learning_rate = 1.0;
    std::string init = "spectral";
    double alignment_regularisation = 1.0e-2;
    int alignment_window_size = 3;
    AlignedParam<double> min_dist = 0.1;
    AlignedParam<double> spread = 1.0;
    AlignedParam<double> set_op_mix_ratio = 1.0;
    AlignedParam<double> local_connectivity = 1.0;
    AlignedParam<double> repulsion_strength = 1.0;
    AlignedParam<int> negative_sample_rate = 5;
    double transform_queue_size = 4.0;
    std::optional<double> a;
    std::optional<double> b;
    std::optional<uint32_t> random_state;  // every mapper gets the same seed
    bool angular_rp_forest = false;
    int target_n_neighbors = -1;
    std::string target_metric = "categorical";
    MetricKwds target_metric_kwds;
    double target_weight = 0.5;
    uint32_t transform_seed = 42;
    bool force_approximation_algorithm = false;
    bool verbose = false;
    AlignedParam<bool> unique = false;
};

class AlignedUMAP {
 public:
    explicit AlignedUMAP(AlignedUMAPParams params = AlignedUMAPParams());

    // Fit a sequence of data slices (each row-major n_i x n_features float32)
    // related pairwise: relations[t] maps row indices of X[t] to row indices
    // of X[t+1] (upstream requires len(X) - 1 == len(relations)). y, when
    // given, holds one per-slice target vector per slice (may contain empty
    // vectors for unsupervised slices, mirroring y=None entries).
    // window_size overrides alignment_window_size for this call only
    // (fit_params["window_size"] upstream).
    AlignedUMAP& fit(const std::vector<std::vector<float>>& X,
                     int64_t n_features,
                     const std::vector<std::unordered_map<int32_t, int32_t>>&
                         relations,
                     const std::vector<std::vector<float>>* y = nullptr,
                     std::optional<int> window_size = std::nullopt);

    const std::vector<std::vector<float>>& fit_transform(
        const std::vector<std::vector<float>>& X, int64_t n_features,
        const std::vector<std::unordered_map<int32_t, int32_t>>& relations,
        const std::vector<std::vector<float>>* y = nullptr,
        std::optional<int> window_size = std::nullopt) {
        fit(X, n_features, relations, y, window_size);
        return embeddings_;
    }

    // AlignedUMAP.update: add one new slice, related to the previous last
    // slice by `relations` (old-slice row -> new-slice row). new_params
    // mirrors upstream's per-model **fit_params merging (set_aligned_params).
    void update(const std::vector<float>& X, int64_t n_features,
                const std::unordered_map<int32_t, int32_t>& relations,
                const std::vector<float>* y = nullptr,
                const AlignedParamUpdate& new_params = AlignedParamUpdate(),
                std::optional<int> window_size = std::nullopt);

    // self.embeddings_ (n_models arrays of n_i x n_components float32).
    const std::vector<std::vector<float>>& embeddings() const {
        return embeddings_;
    }
    int n_models() const { return n_models_; }
    const UMAP& mapper(std::size_t i) const { return *mappers_.at(i); }
    const std::vector<aligned::RelationVec>& dict_relations() const {
        return dict_relations_;
    }
    const AlignedUMAPParams& params() const { return params_; }

 private:
    UMAPParams mapper_params(std::size_t n) const;
    void run_aligned_optimization(const aligned::Tensor3i& relations,
                                  const aligned::Tensor3f& reg_weights,
                                  const std::vector<std::vector<int32_t>>& heads,
                                  const std::vector<std::vector<int32_t>>& tails,
                                  const std::vector<std::vector<double>>& eps,
                                  int n_epochs, int64_t seed_triplet[3],
                                  bool move_other);

    AlignedUMAPParams params_;
    int n_models_ = 0;
    std::vector<std::unique_ptr<UMAP>> mappers_;
    std::vector<aligned::RelationVec> dict_relations_;
    std::vector<std::vector<float>> embeddings_;
    std::vector<std::vector<float>> data_;  // per-slice raw copies
    std::vector<int64_t> slice_sizes_;
    int64_t n_features_ = 0;
    aligned::NumbaNpRandom np_rng_;  // numba's shared np.random state
};

}  // namespace umap
