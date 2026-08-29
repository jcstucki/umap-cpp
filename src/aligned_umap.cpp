// Implementation of include/umap/aligned_umap.hpp (the AlignedUMAP class and
// the deterministic pipeline pieces of reference/umap/umap/aligned_umap.py;
// the SGD kernel lives in src/layouts_aligned.cpp). lmcinnes/umap, BSD 3
// clause. See the header for semantics and deviation notes.
#include "umap/aligned_umap.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <Eigen/Dense>

#include "umap/layouts.hpp"   // make_epochs_per_sample
#include "umap/spectral.hpp"  // connected_components / eigs / multi-component

namespace umap {
namespace aligned {

// ------------------------------------------------------------- dict helpers

RelationVec canonicalize_relations(
    const std::unordered_map<int32_t, int32_t>& d) {
    RelationVec out(d.begin(), d.end());
    std::sort(out.begin(), out.end());
    return out;
}

RelationVec invert_relations(const RelationVec& d) {
    // {value: key for key, value in d.items()} -- on value collisions the
    // later key wins (ascending-key iteration order here; see header note).
    std::map<int32_t, int32_t> inv;
    for (const auto& kv : d) inv[kv.second] = kv.first;
    return RelationVec(inv.begin(), inv.end());
}

namespace {

// aligned_umap.relation_lut: lut[key] = value, else -1.
std::vector<int32_t> relation_lut(const RelationVec& d, int64_t n_samples) {
    std::vector<int32_t> lut(static_cast<std::size_t>(n_samples), -1);
    for (const auto& kv : d)
        lut[static_cast<std::size_t>(kv.first)] = kv.second;
    return lut;
}

// aligned_umap.chain_relation_lut: mapping[i] -> lut[mapping[i]], -1 sticky.
void chain_relation_lut(const std::vector<int32_t>& lut,
                        std::vector<int32_t>& mapping) {
    for (auto& v : mapping) v = (v >= 0) ? lut[static_cast<std::size_t>(v)] : -1;
}

}  // namespace

// ---------------------------------------------------------- expand_relations

Tensor3i expand_relations(const std::vector<RelationVec>& relation_dicts,
                          int window_size) {
    // max_n_samples = max over all keys and values + 1 (upstream max() dies
    // on an empty dict).
    int32_t mx = std::numeric_limits<int32_t>::min();
    for (const auto& d : relation_dicts) {
        if (d.empty())
            throw std::invalid_argument(
                "AlignedUMAP: relation dictionaries may not be empty");
        for (const auto& kv : d) {
            mx = std::max(mx, kv.first);
            mx = std::max(mx, kv.second);
        }
    }
    if (relation_dicts.empty())
        throw std::invalid_argument(
            "AlignedUMAP: at least one relation dictionary is required");
    const int64_t max_n_samples = static_cast<int64_t>(mx) + 1;
    const int64_t n_dicts = static_cast<int64_t>(relation_dicts.size());

    Tensor3i result(n_dicts + 1, 2 * static_cast<int64_t>(window_size) + 1,
                    max_n_samples, -1);

    std::vector<std::vector<int32_t>> luts, reverse_luts;
    luts.reserve(relation_dicts.size());
    reverse_luts.reserve(relation_dicts.size());
    for (const auto& d : relation_dicts) {
        luts.push_back(relation_lut(d, max_n_samples));
        reverse_luts.push_back(relation_lut(invert_relations(d), max_n_samples));
    }

    std::vector<int32_t> mapping;
    for (int64_t i = 0; i < result.d0; ++i) {
        // Forward offsets +1 .. +window_size. NOTE upstream's `>=`: a chain
        // that would land on the LAST model is blanked too (off-by-one kept).
        for (int64_t j = 0; j < window_size; ++j) {
            const int64_t result_index = window_size + (j + 1);
            if (i + j + 1 >= n_dicts) continue;  // stays -1
            mapping.resize(static_cast<std::size_t>(max_n_samples));
            std::iota(mapping.begin(), mapping.end(), 0);
            for (int64_t k = 0; k <= j; ++k)
                chain_relation_lut(luts[static_cast<std::size_t>(i + k)],
                                   mapping);
            for (int64_t s = 0; s < max_n_samples; ++s)
                result.at(i, result_index, s) =
                    mapping[static_cast<std::size_t>(s)];
        }

        // Backward offsets -1 .. -window_size (j = 0, -1, ..., -(ws-1)).
        for (int64_t j = 0; j > -static_cast<int64_t>(window_size); --j) {
            const int64_t result_index = window_size + (j - 1);
            if (i + j - 1 < 0) continue;  // stays -1
            mapping.resize(static_cast<std::size_t>(max_n_samples));
            std::iota(mapping.begin(), mapping.end(), 0);
            for (int64_t k = 0; k >= j; --k)  // range(0, j - 1, -1)
                chain_relation_lut(
                    reverse_luts[static_cast<std::size_t>(i + k - 1)], mapping);
            for (int64_t s = 0; s < max_n_samples; ++s)
                result.at(i, result_index, s) =
                    mapping[static_cast<std::size_t>(s)];
        }
    }

    return result;
}

// ------------------------------------------- build_neighborhood_similarities

namespace {

// umap.sparse.arr_intersect(...).shape[0]: concatenate + sort, count entries
// equal to their successor. Duplicates within ONE input count too (upstream
// quirk kept -- callers may pass unsorted arrays with repeats).
int64_t upstream_intersect_size(const std::vector<int32_t>& a,
                                const std::vector<int32_t>& b,
                                std::vector<int32_t>& scratch) {
    scratch.clear();
    scratch.insert(scratch.end(), a.begin(), a.end());
    scratch.insert(scratch.end(), b.begin(), b.end());
    std::sort(scratch.begin(), scratch.end());
    int64_t count = 0;
    for (std::size_t t = 1; t < scratch.size(); ++t)
        if (scratch[t] == scratch[t - 1]) ++count;
    return count;
}

// umap.sparse.arr_union(...).shape[0]: empty side -> the other side AS IS
// (no dedup!), else unique(concatenate).size.
int64_t upstream_union_size(const std::vector<int32_t>& a,
                            const std::vector<int32_t>& b,
                            std::vector<int32_t>& scratch) {
    if (a.empty()) return static_cast<int64_t>(b.size());
    if (b.empty()) return static_cast<int64_t>(a.size());
    scratch.clear();
    scratch.insert(scratch.end(), a.begin(), a.end());
    scratch.insert(scratch.end(), b.begin(), b.end());
    std::sort(scratch.begin(), scratch.end());
    scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
    return static_cast<int64_t>(scratch.size());
}

}  // namespace

Tensor3f build_neighborhood_similarities(
    const std::vector<const SparseMatrix*>& graphs, const Tensor3i& relations) {
    Tensor3f result(relations.d0, relations.d1, relations.d2, 0.0f);
    const int64_t center_index = (relations.d1 - 1) / 2;
    const int64_t n_samples = relations.d2;
    const int64_t n_graphs = static_cast<int64_t>(graphs.size());

    std::vector<uint8_t> present(static_cast<std::size_t>(n_samples));
    std::vector<int32_t> base_indices, comparison_indices, scratch;

    for (int64_t i = 0; i < relations.d0; ++i) {
        const SparseMatrix& base_graph = *graphs[static_cast<std::size_t>(i)];
        for (int64_t j = 0; j < relations.d1; ++j) {
            const int64_t cmp_model = i + j - center_index;
            if (cmp_model < 0 || cmp_model >= n_graphs) continue;
            const SparseMatrix& comparison_graph =
                *graphs[static_cast<std::size_t>(cmp_model)];

            // present[v] for values of relations[i, j] (v < n_samples always
            // holds by construction of max_n_samples).
            std::fill(present.begin(), present.end(), 0);
            for (int64_t s = 0; s < n_samples; ++s) {
                const int32_t v = relations.at(i, j, s);
                if (v >= 0) present[static_cast<std::size_t>(v)] = 1;
            }

            for (int64_t k = 0; k < n_samples; ++k) {
                const int32_t comparison_index = relations.at(i, j, k);
                if (comparison_index < 0) continue;

                // base_indices = relations[i, j][base row k's neighbours
                // (indices < n_samples)] with -1 entries dropped.
                base_indices.clear();
                if (k + 1 < static_cast<int64_t>(base_graph.indptr.size())) {
                    for (int32_t t = base_graph.indptr[static_cast<std::size_t>(k)];
                         t < base_graph.indptr[static_cast<std::size_t>(k + 1)];
                         ++t) {
                        const int32_t nb =
                            base_graph.indices[static_cast<std::size_t>(t)];
                        if (nb < n_samples) {
                            const int32_t mapped = relations.at(i, j, nb);
                            if (mapped >= 0) base_indices.push_back(mapped);
                        }
                    }
                }

                // comparison_indices = comparison row's neighbours restricted
                // to values present in relations[i, j].
                comparison_indices.clear();
                if (comparison_index + 1 <
                    static_cast<int64_t>(comparison_graph.indptr.size())) {
                    for (int32_t t = comparison_graph.indptr
                             [static_cast<std::size_t>(comparison_index)];
                         t < comparison_graph.indptr[static_cast<std::size_t>(
                             comparison_index + 1)];
                         ++t) {
                        const int32_t c =
                            comparison_graph.indices[static_cast<std::size_t>(t)];
                        if (c < n_samples && present[static_cast<std::size_t>(c)])
                            comparison_indices.push_back(c);
                    }
                }

                const int64_t intersection_size = upstream_intersect_size(
                    base_indices, comparison_indices, scratch);
                const int64_t union_size = upstream_union_size(
                    base_indices, comparison_indices, scratch);

                result.at(i, j, k) =
                    union_size > 0
                        ? static_cast<float>(
                              static_cast<double>(intersection_size) /
                              static_cast<double>(union_size))
                        : 1.0f;
            }
        }
    }

    return result;
}

// ------------------------------------------------------------ procrustes

std::vector<float> procrustes_align(const std::vector<float>& embedding_base,
                                    const std::vector<float>& embedding_to_align,
                                    int dim,
                                    const std::vector<int32_t>& anchors_base,
                                    const std::vector<int32_t>& anchors_align) {
    // subset1 = base[anchors[0]]; subset2 = to_align[anchors[1]];
    // M = subset2.T @ subset1 (f4); U, S, V = svd(M); R = U @ V;
    // return to_align @ R (f4). Deviation: Eigen JacobiSVD + sequential f4
    // accumulation instead of LAPACK sgesdd + BLAS sgemm (see header).
    const std::size_t n_anchors = anchors_base.size();
    Eigen::MatrixXf M = Eigen::MatrixXf::Zero(dim, dim);
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            float acc = 0.0f;
            for (std::size_t l = 0; l < n_anchors; ++l) {
                const float s2 =
                    embedding_to_align[static_cast<std::size_t>(
                                           anchors_align[l]) *
                                           dim +
                                       r];
                const float s1 =
                    embedding_base[static_cast<std::size_t>(anchors_base[l]) *
                                       dim +
                                   c];
                const float prod = s2 * s1;
                acc += prod;
            }
            M(r, c) = acc;
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(
        M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // numpy's svd returns vh with M = U diag(S) vh; Eigen's V satisfies
    // M = U S V^T, so R = U @ vh = U * V^T.
    Eigen::MatrixXf R = svd.matrixU() * svd.matrixV().transpose();

    const std::size_t n_rows = embedding_to_align.size() /
                               static_cast<std::size_t>(dim);
    std::vector<float> out(embedding_to_align.size());
    for (std::size_t l = 0; l < n_rows; ++l) {
        for (int c = 0; c < dim; ++c) {
            float acc = 0.0f;
            for (int r = 0; r < dim; ++r) {
                const float prod =
                    embedding_to_align[l * dim + r] * R(r, c);
                acc += prod;
            }
            out[l * dim + c] = acc;
        }
    }
    return out;
}

// ------------------------------------------------------- init_from_existing

std::vector<float> init_from_existing(const std::vector<float>& prev_embedding,
                                      int n_features, const SparseMatrix& graph,
                                      const RelationVec& relation,
                                      NumbaNpRandom& np_rng) {
    const int64_t n_samples =
        static_cast<int64_t>(graph.indptr.size()) - 1;
    std::vector<float> result(
        static_cast<std::size_t>(n_samples * n_features), 0.0f);
    std::map<int32_t, int32_t> rel(relation.begin(), relation.end());

    for (int64_t i = 0; i < n_samples; ++i) {
        float* row = result.data() + static_cast<std::size_t>(i) * n_features;
        const auto hit = rel.find(static_cast<int32_t>(i));
        if (hit != rel.end()) {
            const float* prev =
                prev_embedding.data() +
                static_cast<std::size_t>(hit->second) * n_features;
            for (int d = 0; d < n_features; ++d) row[d] = prev[d];
        } else {
            double normalisation = 0.0;  // f8 accumulator (python float)
            for (int32_t idx = graph.indptr[static_cast<std::size_t>(i)];
                 idx < graph.indptr[static_cast<std::size_t>(i) + 1]; ++idx) {
                const int32_t jn = graph.indices[static_cast<std::size_t>(idx)];
                const auto rj = rel.find(jn);
                if (rj != rel.end()) {
                    const float w = graph.data[static_cast<std::size_t>(idx)];
                    normalisation += static_cast<double>(w);
                    const float* prev =
                        prev_embedding.data() +
                        static_cast<std::size_t>(rj->second) * n_features;
                    // result[i] += w * previous_embedding[...] -- f4 product,
                    // f4 add (kept as separate statements: no fma).
                    for (int d = 0; d < n_features; ++d) {
                        const float prod = w * prev[d];
                        row[d] += prod;
                    }
                }
            }
            if (normalisation == 0.0) {
                // np.random.uniform(-10.0, 10.0, n_features) from numba's
                // shared np state (f8 draws stored to f4).
                for (int d = 0; d < n_features; ++d)
                    row[d] = static_cast<float>(np_rng.uniform(-10.0, 10.0));
            } else {
                // result[i] /= normalisation (f4 /= f8: f8 divide, f4 store).
                for (int d = 0; d < n_features; ++d)
                    row[d] = static_cast<float>(
                        static_cast<double>(row[d]) / normalisation);
            }
        }
    }

    return result;
}

}  // namespace aligned

// =================================================================== class

namespace {

// spectral.spectral_layout with numpy RandomState stream parity, as
// AlignedUMAP.fit calls it (NO noisy_scale_coords). Mirrors the spectral
// branch of UMAP::make_fit_init in src/umap.cpp: consume the
// gen.normal(size=(n, dim + 1)) eigsh warm-start draws, Spectra eigensolver,
// random [-10, 10] fallback on failure, tau-prng multi-component path
// (documented deviation there and here).
std::vector<double> spectral_layout_numpy(const float* data, int64_t n_features,
                                          const SparseMatrix& graph, int dim,
                                          nnd::NumpyRandomState& rs) {
    const int64_t n = graph.n_rows;
    std::vector<double> emb64(static_cast<std::size_t>(n) * dim);

    std::vector<int32_t> labels;
    const int32_t n_comp = connected_components(graph, labels);
    if (n_comp > 1) {
        auto s = rs.rng_state3();
        int64_t tau_state[3] = {s[0], s[1], s[2]};
        const std::vector<float> multi = multi_component_layout(
            data, n_features, graph, n_comp, labels, dim, tau_state);
        for (std::size_t i = 0; i < emb64.size(); ++i)
            emb64[i] = static_cast<double>(multi[i]);
        return emb64;
    }

    const int k = dim + 1;
    for (int64_t i = 0; i < n * k; ++i) (void)rs.gauss();
    std::vector<double> eigenvalues, eigenvectors;
    if (normalized_laplacian_eigs(graph, k, eigenvalues, eigenvectors)) {
        // order = np.argsort(eigenvalues)[1:k] (ascending already).
        for (int64_t i = 0; i < n; ++i)
            for (int j = 0; j < dim; ++j)
                emb64[static_cast<std::size_t>(i) * dim + j] =
                    eigenvectors[static_cast<std::size_t>(i) * k + j + 1];
    } else {
        std::fprintf(stderr,
                     "WARNING: spectral initialisation failed! The eigenvector "
                     "solver\nfailed. This is likely due to too small an eigengap."
                     " Consider\nadding some noise or jitter to your data.\n\n"
                     "Falling back to random initialisation!\n");
        for (auto& v : emb64) v = rs.uniform(-10.0, 10.0);
    }
    return emb64;
}

// (init * (10.0 / np.abs(init).max())).astype(np.float32)
std::vector<float> expand_to_f32(const std::vector<double>& init) {
    double max_abs = 0.0;
    for (double v : init) max_abs = std::max(max_abs, std::abs(v));
    const double expansion = 10.0 / max_abs;
    std::vector<float> out(init.size());
    for (std::size_t i = 0; i < init.size(); ++i)
        out[i] = static_cast<float>(init[i] * expansion);
    return out;
}

// COO view of a stored CSR graph_ (explicit zeros INCLUDED, like tocoo()).
void graph_to_coo(const SparseMatrix& g, std::vector<int32_t>& rows,
                  std::vector<int32_t>& cols, std::vector<float>& vals) {
    rows.clear();
    cols.clear();
    vals.clear();
    rows.reserve(g.data.size());
    cols.reserve(g.data.size());
    vals.reserve(g.data.size());
    for (int32_t r = 0; r < g.n_rows; ++r) {
        for (int32_t t = g.indptr[static_cast<std::size_t>(r)];
             t < g.indptr[static_cast<std::size_t>(r) + 1]; ++t) {
            rows.push_back(r);
            cols.push_back(g.indices[static_cast<std::size_t>(t)]);
            vals.push_back(g.data[static_cast<std::size_t>(t)]);
        }
    }
}

}  // namespace

AlignedUMAP::AlignedUMAP(AlignedUMAPParams params)
    : params_(std::move(params)) {}

UMAPParams AlignedUMAP::mapper_params(std::size_t n) const {
    UMAPParams p;
    p.n_neighbors = params_.n_neighbors.get(n);
    p.min_dist = params_.min_dist.get(n);
    p.n_epochs = params_.n_epochs;  // scalar (resolved to 200 by fit/update)
    p.repulsion_strength = params_.repulsion_strength.get(n);
    p.learning_rate = params_.learning_rate.get(n);
    p.init = params_.init;
    p.spread = params_.spread.get(n);
    p.negative_sample_rate = params_.negative_sample_rate.get(n);
    p.local_connectivity = params_.local_connectivity.get(n);
    p.set_op_mix_ratio = params_.set_op_mix_ratio.get(n);
    p.unique = params_.unique.get(n);
    p.n_components = params_.n_components;
    p.metric = params_.metric;
    p.metric_kwds = params_.metric_kwds;
    p.random_state = params_.random_state;
    p.angular_rp_forest = params_.angular_rp_forest;
    p.transform_queue_size = params_.transform_queue_size;
    p.target_n_neighbors = params_.target_n_neighbors;
    p.target_metric = params_.target_metric;
    p.target_metric_kwds = params_.target_metric_kwds;
    p.target_weight = params_.target_weight;
    p.transform_seed = params_.transform_seed;
    p.force_approximation_algorithm = params_.force_approximation_algorithm;
    p.verbose = params_.verbose;
    p.a = params_.a;
    p.b = params_.b;
    return p;
}

void AlignedUMAP::run_aligned_optimization(
    const aligned::Tensor3i& relations, const aligned::Tensor3f& reg_weights,
    const std::vector<std::vector<int32_t>>& heads,
    const std::vector<std::vector<int32_t>>& tails,
    const std::vector<std::vector<double>>& eps, int n_epochs,
    int64_t seed_triplet[3], bool move_other) {
    aligned::optimize_layout_aligned_euclidean(
        embeddings_, embeddings_, params_.n_components, heads, tails, n_epochs,
        eps, reg_weights, relations, seed_triplet, np_rng_,
        /*a=*/1.576943460405378, /*b=*/0.8950608781227859, /*gamma=*/1.0,
        /*lambda_=*/params_.alignment_regularisation, /*initial_alpha=*/1.0,
        /*negative_sample_rate=*/5.0, move_other);
}

AlignedUMAP& AlignedUMAP::fit(
    const std::vector<std::vector<float>>& X, int64_t n_features,
    const std::vector<std::unordered_map<int32_t, int32_t>>& relations,
    const std::vector<std::vector<float>>* y,
    std::optional<int> window_size) {
    if (X.empty())
        throw std::invalid_argument(
            "AlignedUMAP requires at least one data slice");
    if (relations.size() != X.size() - 1)
        throw std::invalid_argument(
            "Aligned UMAP requires relations between data to be specified: "
            "len(X) - 1 == len(relations) must hold");
    if (y != nullptr && y->size() != X.size())
        throw std::invalid_argument(
            "AlignedUMAP: (len(y) - 1) == len(relations) must hold");

    dict_relations_.clear();
    for (const auto& d : relations)
        dict_relations_.push_back(aligned::canonicalize_relations(d));

    n_models_ = static_cast<int>(X.size());
    n_features_ = n_features;
    data_ = X;
    slice_sizes_.clear();
    for (const auto& slice : X)
        slice_sizes_.push_back(static_cast<int64_t>(slice.size()) / n_features);

    if (!params_.n_epochs.has_value()) params_.n_epochs = 200;
    const int n_epochs = *params_.n_epochs;

    mappers_.clear();
    for (int n = 0; n < n_models_; ++n) {
        auto m = std::make_unique<UMAP>(mapper_params(static_cast<std::size_t>(n)));
        const float* yn = nullptr;
        if (y != nullptr && !(*y)[static_cast<std::size_t>(n)].empty())
            yn = (*y)[static_cast<std::size_t>(n)].data();
        m->fit(X[static_cast<std::size_t>(n)].data(),
               slice_sizes_[static_cast<std::size_t>(n)], n_features, yn);
        mappers_.push_back(std::move(m));
    }

    const int ws = window_size.value_or(params_.alignment_window_size);
    const aligned::Tensor3i rel = aligned::expand_relations(dict_relations_, ws);

    std::vector<std::vector<int32_t>> heads, tails;
    std::vector<std::vector<double>> epochs_per_samples;
    std::vector<const SparseMatrix*> graphs;
    for (const auto& mapper : mappers_) {
        const SparseMatrix& g = mapper->graph();
        graphs.push_back(&g);
        std::vector<int32_t> rows, cols;
        std::vector<float> vals;
        graph_to_coo(g, rows, cols, vals);
        heads.push_back(std::move(rows));
        tails.push_back(std::move(cols));
        epochs_per_samples.push_back(
            make_epochs_per_sample(vals.data(), vals.size(), n_epochs));
    }

    nnd::NumpyRandomState rng_state_transform(params_.transform_seed);
    const aligned::Tensor3f regularisation_weights =
        aligned::build_neighborhood_similarities(graphs, rel);

    // First slice: spectral_layout scaled to +-10, float32, no jitter.
    const int dim = params_.n_components;
    embeddings_.clear();
    {
        const std::vector<double> first_init = spectral_layout_numpy(
            data_[0].data(), n_features_, mappers_[0]->graph(), dim,
            rng_state_transform);
        embeddings_.push_back(expand_to_f32(first_init));
    }
    for (int i = 1; i < n_models_; ++i) {
        const std::vector<double> next_init = spectral_layout_numpy(
            data_[static_cast<std::size_t>(i)].data(), n_features_,
            mappers_[static_cast<std::size_t>(i)]->graph(), dim,
            rng_state_transform);
        const std::vector<float> next_embedding = expand_to_f32(next_init);
        // anchor_data = relations[i][window_size - 1] (the offset -1 row).
        std::vector<int32_t> left_anchors, right_anchors;
        for (int64_t s = 0; s < rel.d2; ++s) {
            const int32_t v = rel.at(i, ws - 1, s);
            if (v >= 0) {
                left_anchors.push_back(v);
                right_anchors.push_back(static_cast<int32_t>(s));
            }
        }
        embeddings_.push_back(aligned::procrustes_align(
            embeddings_.back(), next_embedding, dim, left_anchors,
            right_anchors));
    }

    auto st = rng_state_transform.rng_state3();
    int64_t seed_triplet[3] = {st[0], st[1], st[2]};
    run_aligned_optimization(rel, regularisation_weights, heads, tails,
                             epochs_per_samples, n_epochs, seed_triplet,
                             /*move_other=*/true);

    // embedding[graph_.sum(axis=1) == 0] = NaN (f4 row sums of stored data).
    for (int i = 0; i < n_models_; ++i) {
        const SparseMatrix& g = mappers_[static_cast<std::size_t>(i)]->graph();
        auto& emb = embeddings_[static_cast<std::size_t>(i)];
        for (int32_t r = 0; r < g.n_rows; ++r) {
            float row_sum = 0.0f;
            for (int32_t t = g.indptr[static_cast<std::size_t>(r)];
                 t < g.indptr[static_cast<std::size_t>(r) + 1]; ++t)
                row_sum += g.data[static_cast<std::size_t>(t)];
            if (row_sum == 0.0f) {
                for (int d = 0; d < dim; ++d)
                    emb[static_cast<std::size_t>(r) * dim + d] =
                        std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    return *this;
}

void AlignedUMAP::update(const std::vector<float>& X, int64_t n_features,
                         const std::unordered_map<int32_t, int32_t>& relations,
                         const std::vector<float>* y,
                         const AlignedParamUpdate& new_params,
                         std::optional<int> window_size) {
    if (mappers_.empty())
        throw std::logic_error("AlignedUMAP::update requires a prior fit()");
    if (n_features != n_features_)
        throw std::invalid_argument(
            "AlignedUMAP::update: n_features must match the fitted slices");

    // set_aligned_params(fit_params, self.__dict__, self.n_models_).
    if (new_params.n_neighbors)
        params_.n_neighbors.set_aligned(*new_params.n_neighbors, n_models_);
    if (new_params.min_dist)
        params_.min_dist.set_aligned(*new_params.min_dist, n_models_);
    if (new_params.n_epochs && params_.n_epochs &&
        *new_params.n_epochs != *params_.n_epochs)
        throw std::invalid_argument(
            "AlignedUMAP::update: n_epochs cannot change between fits (the "
            "reference would build a per-model tuple and crash in the aligned "
            "optimization)");
    if (new_params.repulsion_strength)
        params_.repulsion_strength.set_aligned(*new_params.repulsion_strength,
                                               n_models_);
    if (new_params.learning_rate)
        params_.learning_rate.set_aligned(*new_params.learning_rate, n_models_);
    if (new_params.spread)
        params_.spread.set_aligned(*new_params.spread, n_models_);
    if (new_params.negative_sample_rate)
        params_.negative_sample_rate.set_aligned(
            *new_params.negative_sample_rate, n_models_);
    if (new_params.local_connectivity)
        params_.local_connectivity.set_aligned(*new_params.local_connectivity,
                                               n_models_);
    if (new_params.set_op_mix_ratio)
        params_.set_op_mix_ratio.set_aligned(*new_params.set_op_mix_ratio,
                                             n_models_);
    if (new_params.unique)
        params_.unique.set_aligned(*new_params.unique, n_models_);

    if (!params_.n_epochs.has_value()) params_.n_epochs = 200;
    const int n_epochs = *params_.n_epochs;

    // New mapper with the parameters at index n_models_ (pre-increment).
    auto new_mapper = std::make_unique<UMAP>(
        mapper_params(static_cast<std::size_t>(n_models_)));
    const int64_t n_new =
        static_cast<int64_t>(X.size()) / n_features;
    new_mapper->fit(X.data(), n_new, n_features,
                    (y != nullptr && !y->empty()) ? y->data() : nullptr);

    n_models_ += 1;
    mappers_.push_back(std::move(new_mapper));
    data_.push_back(X);
    slice_sizes_.push_back(n_new);
    dict_relations_.push_back(aligned::canonicalize_relations(relations));

    const int ws = window_size.value_or(params_.alignment_window_size);
    const aligned::Tensor3i new_relations =
        aligned::expand_relations(dict_relations_, ws);

    std::vector<std::vector<int32_t>> heads, tails;
    std::vector<std::vector<double>> epochs_per_samples;
    std::vector<const SparseMatrix*> graphs;
    for (int i = 0; i < n_models_; ++i) {
        const SparseMatrix& g = mappers_[static_cast<std::size_t>(i)]->graph();
        graphs.push_back(&g);
        std::vector<int32_t> rows, cols;
        std::vector<float> vals;
        graph_to_coo(g, rows, cols, vals);
        heads.push_back(std::move(rows));
        tails.push_back(std::move(cols));
        if (i == n_models_ - 1) {
            epochs_per_samples.push_back(
                make_epochs_per_sample(vals.data(), vals.size(), n_epochs));
        } else {
            // Old models frozen: np.full(n_samples, n_epochs + 1) -- note the
            // length is the SAMPLE count, not the edge count (upstream).
            epochs_per_samples.emplace_back(
                static_cast<std::size_t>(
                    mappers_[static_cast<std::size_t>(i)]->n_samples()),
                static_cast<double>(n_epochs + 1));
        }
    }

    const aligned::Tensor3f new_regularisation_weights =
        aligned::build_neighborhood_similarities(graphs, new_relations);

    const aligned::RelationVec inv_dict_relations =
        aligned::invert_relations(dict_relations_.back());
    embeddings_.push_back(aligned::init_from_existing(
        embeddings_.back(), params_.n_components, mappers_.back()->graph(),
        inv_dict_relations, np_rng_));

    nnd::NumpyRandomState rng_state_transform(params_.transform_seed);
    auto st = rng_state_transform.rng_state3();
    int64_t seed_triplet[3] = {st[0], st[1], st[2]};
    // move_other omitted upstream -> default False; no NaN masking either.
    run_aligned_optimization(new_relations, new_regularisation_weights, heads,
                             tails, epochs_per_samples, n_epochs, seed_triplet,
                             /*move_other=*/false);
}

}  // namespace umap
