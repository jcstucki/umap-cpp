// --- phase4 additions --------------------------------------------------------
// Model persistence: UMAP::save / UMAP::load and PreparedIndex::serialize /
// ::deserialize.
//
// File format "UMAPCPP1" version 1 (all integers/floats little-endian, C
// layouts; big-endian hosts are rejected at runtime):
//
//   bytes 0..7   magic "UMAPCPP1"
//   u32          byte-order sentinel 0x01020304 (readers verify)
//   u32          format version (1)
//   u32          section count
//   sections     { u32 id; u64 payload_bytes; payload }
//
// Sections (unknown ids are skipped on read, so later versions can append):
//   1  params           every UMAPParams field in declaration order
//                       (precomputed_knn stores its indices/dists and a
//                       search-index presence flag; the search index itself
//                       is re-linked to the fitted index on load)
//   2  fitted scalars   a_, b_, initial_alpha_, n_neighbors_, flags, sizes,
//                       input hash, disconnection bookkeeping, knn_cols_
//   3  dense raw data   raw_data_
//   4  sparse raw data  raw_indptr_ / raw_indices_ / raw_values_
//   5  kNN arrays       knn_indices_, knn_dists_
//   6  graph            graph_ CSR, sigmas_, rhos_
//   7  embedding        embedding_, fit_init_
//   8  extras           unique_inverse_, embedding_list_, graph_dists_,
//                       rad_orig_, rad_emb_
//   9  prepared index   full PreparedIndex state (present iff the fit kept a
//                       dense search index); the metric closure is rebuilt
//                       from params at load
//   10 rng bookkeeping  index_rs_ (the persisted numpy RandomState stream
//                       NNDescent.update() continues), as
//                       NumpyRandomState::save_state() text
//   11 sparse prepared  full SparsePreparedIndex state (hardening
//      index            additions; present iff the fit kept a sparse search
//                       index); the sparse metric closure is rebuilt from
//                       params (dim binds n_features metrics) at load
//
// Everything a fit can produce round-trips -- dense and sparse (small-data
// AND standard-branch), densmap/output_dens radii, n_epochs_list snapshots
// and composite-operator results -- except models with "custom"
// (std::function) metrics, which throw at save().
// -----------------------------------------------------------------------------
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "umap/nn_descent.hpp"
#include "umap/sparse_nn_descent.hpp"  // hardening additions
#include "umap/umap.hpp"

namespace umap {

namespace {

constexpr char kMagic[8] = {'U', 'M', 'A', 'P', 'C', 'P', 'P', '1'};
constexpr uint32_t kOrderSentinel = 0x01020304u;
constexpr uint32_t kFormatVersion = 1;

bool host_is_little_endian() {
    const uint32_t x = 1;
    uint8_t b;
    std::memcpy(&b, &x, 1);
    return b == 1;
}

void require_little_endian() {
    if (!host_is_little_endian())
        throw std::runtime_error(
            "UMAP model files are little-endian; big-endian hosts are not "
            "supported");
}

// ------------------------------------------------------------------- writers

template <typename T>
void w_pod(std::ostream& os, const T& v) {
    static_assert(std::is_trivially_copyable<T>::value, "POD only");
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

void w_u8(std::ostream& os, uint8_t v) { w_pod(os, v); }
void w_u32(std::ostream& os, uint32_t v) { w_pod(os, v); }
void w_u64(std::ostream& os, uint64_t v) { w_pod(os, v); }
void w_i32(std::ostream& os, int32_t v) { w_pod(os, v); }
void w_i64(std::ostream& os, int64_t v) { w_pod(os, v); }
void w_f32(std::ostream& os, float v) { w_pod(os, v); }
void w_f64(std::ostream& os, double v) { w_pod(os, v); }

void w_str(std::ostream& os, const std::string& s) {
    w_u64(os, s.size());
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

template <typename T>
void w_vec(std::ostream& os, const std::vector<T>& v) {
    static_assert(std::is_trivially_copyable<T>::value, "POD vectors only");
    w_u64(os, v.size());
    os.write(reinterpret_cast<const char*>(v.data()),
             static_cast<std::streamsize>(v.size() * sizeof(T)));
}

template <typename T, typename WriteFn>
void w_opt(std::ostream& os, const std::optional<T>& v, WriteFn wf) {
    w_u8(os, v.has_value() ? 1 : 0);
    if (v.has_value()) wf(os, *v);
}

void w_kwds(std::ostream& os, const MetricKwds& k) {
    w_f64(os, k.p);
    w_vec(os, k.v);
    w_vec(os, k.vinv);
    w_f64(os, k.z);
}

void w_csr(std::ostream& os, const SparseMatrix& m) {
    w_i32(os, m.n_rows);
    w_i32(os, m.n_cols);
    w_vec(os, m.indptr);
    w_vec(os, m.indices);
    w_vec(os, m.data);
}

// ------------------------------------------------------------------- readers

void r_bytes(std::istream& is, void* dst, std::size_t n) {
    is.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (static_cast<std::size_t>(is.gcount()) != n)
        throw std::runtime_error("UMAP model file: unexpected end of data");
}

template <typename T>
T r_pod(std::istream& is) {
    static_assert(std::is_trivially_copyable<T>::value, "POD only");
    T v;
    r_bytes(is, &v, sizeof(T));
    return v;
}

uint8_t r_u8(std::istream& is) { return r_pod<uint8_t>(is); }
uint32_t r_u32(std::istream& is) { return r_pod<uint32_t>(is); }
uint64_t r_u64(std::istream& is) { return r_pod<uint64_t>(is); }
int32_t r_i32(std::istream& is) { return r_pod<int32_t>(is); }
int64_t r_i64(std::istream& is) { return r_pod<int64_t>(is); }
float r_f32(std::istream& is) { return r_pod<float>(is); }
double r_f64(std::istream& is) { return r_pod<double>(is); }

std::string r_str(std::istream& is) {
    const uint64_t n = r_u64(is);
    std::string s(static_cast<std::size_t>(n), '\0');
    if (n > 0) r_bytes(is, &s[0], static_cast<std::size_t>(n));
    return s;
}

template <typename T>
std::vector<T> r_vec(std::istream& is) {
    const uint64_t n = r_u64(is);
    std::vector<T> v(static_cast<std::size_t>(n));
    if (n > 0) r_bytes(is, v.data(), static_cast<std::size_t>(n) * sizeof(T));
    return v;
}

template <typename T, typename ReadFn>
std::optional<T> r_opt(std::istream& is, ReadFn rf) {
    if (r_u8(is) == 0) return std::nullopt;
    return rf(is);
}

MetricKwds r_kwds(std::istream& is) {
    MetricKwds k;
    k.p = r_f64(is);
    k.v = r_vec<double>(is);
    k.vinv = r_vec<double>(is);
    k.z = r_f64(is);
    return k;
}

SparseMatrix r_csr(std::istream& is) {
    SparseMatrix m;
    m.n_rows = r_i32(is);
    m.n_cols = r_i32(is);
    m.indptr = r_vec<int32_t>(is);
    m.indices = r_vec<int32_t>(is);
    m.data = r_vec<float>(is);
    return m;
}

// Section ids.
enum : uint32_t {
    kSecParams = 1,
    kSecScalars = 2,
    kSecDenseRaw = 3,
    kSecSparseRaw = 4,
    kSecKnn = 5,
    kSecGraph = 6,
    kSecEmbedding = 7,
    kSecExtras = 8,
    kSecPreparedIndex = 9,
    kSecRng = 10,
    kSecSparsePreparedIndex = 11,  // hardening additions
};

}  // namespace

// ------------------------------------------------- PreparedIndex persistence

namespace nnd {

void PreparedIndex::serialize(std::ostream& os) const {
    w_i64(os, n_);
    w_i64(os, dim_);
    w_i64(os, n_neighbors_);
    w_u8(os, normalize_query_ ? 1 : 0);
    w_vec(os, data_);
    w_vec(os, vertex_order_);
    w_vec(os, search_graph_.indptr);
    w_vec(os, search_graph_.indices);
    w_f32(os, search_graph_.min_distance);
    w_i64(os, tree_.n_nodes);
    w_i64(os, tree_.dim);
    w_vec(os, tree_.hyperplanes);
    w_vec(os, tree_.offsets);
    w_vec(os, tree_.children);
    w_vec(os, tree_.indices);
    for (int64_t s : search_rng_state_) w_i64(os, s);
    w_vec(os, knn_indices_);
    w_vec(os, knn_dists_);
    for (int64_t s : rng_state_) w_i64(os, s);
    w_vec(os, knn_internal_dists_);
    w_i64(os, n_trees_after_update_);
    w_i64(os, n_iters_);
}

PreparedIndex PreparedIndex::deserialize(std::istream& is,
                                         const std::string& metric,
                                         const MetricKwds& metric_kwds) {
    PreparedIndex ix;
    ix.n_ = r_i64(is);
    ix.dim_ = r_i64(is);
    ix.n_neighbors_ = r_i64(is);
    ix.normalize_query_ = r_u8(is) != 0;
    ix.data_ = r_vec<float>(is);
    ix.vertex_order_ = r_vec<int32_t>(is);
    ix.search_graph_.indptr = r_vec<int32_t>(is);
    ix.search_graph_.indices = r_vec<int32_t>(is);
    ix.search_graph_.min_distance = r_f32(is);
    ix.tree_.n_nodes = r_i64(is);
    ix.tree_.dim = r_i64(is);
    ix.tree_.hyperplanes = r_vec<float>(is);
    ix.tree_.offsets = r_vec<float>(is);
    ix.tree_.children = r_vec<int32_t>(is);
    ix.tree_.indices = r_vec<int32_t>(is);
    for (int64_t& s : ix.search_rng_state_) s = r_i64(is);
    ix.knn_indices_ = r_vec<int32_t>(is);
    ix.knn_dists_ = r_vec<float>(is);
    for (int64_t& s : ix.rng_state_) s = r_i64(is);
    ix.knn_internal_dists_ = r_vec<float>(is);
    ix.n_trees_after_update_ = r_i64(is);
    ix.n_iters_ = r_i64(is);
    // The distance closure is not stored; rebuild it exactly as build() did.
    ix.metric_ = resolve_metric(metric, metric_kwds);
    return ix;
}

}  // namespace nnd

// --- hardening additions -----------------------------------------------------
// ------------------------------------- SparsePreparedIndex persistence

namespace spnnd {

void SparsePreparedIndex::serialize(std::ostream& os) const {
    w_i64(os, n_);
    w_i64(os, dim_);
    w_i64(os, n_neighbors_);
    w_vec(os, data_indptr_);
    w_vec(os, data_indices_);
    w_vec(os, data_data_);
    w_vec(os, vertex_order_);
    w_vec(os, search_graph_.indptr);
    w_vec(os, search_graph_.indices);
    w_f32(os, search_graph_.min_distance);
    // SparseFlatTree: per-node trimmed hyperplanes are vectors-of-vectors.
    w_i64(os, tree_.n_nodes);
    w_u64(os, tree_.hyperplane_inds.size());
    for (const auto& v : tree_.hyperplane_inds) w_vec(os, v);
    w_u64(os, tree_.hyperplane_data.size());
    for (const auto& v : tree_.hyperplane_data) w_vec(os, v);
    w_vec(os, tree_.offsets);
    w_vec(os, tree_.children);
    w_vec(os, tree_.indices);
    for (int64_t s : search_rng_state_) w_i64(os, s);
    w_vec(os, knn_indices_);
    w_vec(os, knn_dists_);
}

SparsePreparedIndex SparsePreparedIndex::deserialize(
    std::istream& is, const std::string& metric,
    const spdist::SparseMetricKwds& kwds) {
    SparsePreparedIndex ix;
    ix.n_ = r_i64(is);
    ix.dim_ = r_i64(is);
    ix.n_neighbors_ = r_i64(is);
    ix.data_indptr_ = r_vec<int32_t>(is);
    ix.data_indices_ = r_vec<int32_t>(is);
    ix.data_data_ = r_vec<float>(is);
    ix.vertex_order_ = r_vec<int32_t>(is);
    ix.search_graph_.indptr = r_vec<int32_t>(is);
    ix.search_graph_.indices = r_vec<int32_t>(is);
    ix.search_graph_.min_distance = r_f32(is);
    ix.tree_.n_nodes = r_i64(is);
    const uint64_t n_hi = r_u64(is);
    ix.tree_.hyperplane_inds.resize(static_cast<std::size_t>(n_hi));
    for (auto& v : ix.tree_.hyperplane_inds) v = r_vec<int32_t>(is);
    const uint64_t n_hd = r_u64(is);
    ix.tree_.hyperplane_data.resize(static_cast<std::size_t>(n_hd));
    for (auto& v : ix.tree_.hyperplane_data) v = r_vec<float>(is);
    ix.tree_.offsets = r_vec<float>(is);
    ix.tree_.children = r_vec<int32_t>(is);
    ix.tree_.indices = r_vec<int32_t>(is);
    for (int64_t& s : ix.search_rng_state_) s = r_i64(is);
    ix.knn_indices_ = r_vec<int32_t>(is);
    ix.knn_dists_ = r_vec<float>(is);
    // The distance closure is not stored; rebuild it exactly as build() did
    // (dim binds pynndescent's sparse_need_n_features metrics, kwds.p
    // minkowski).
    ix.metric_ = spdist::resolve_sparse_metric(metric, ix.dim_, kwds);
    return ix;
}

}  // namespace spnnd
// -----------------------------------------------------------------------------

// --------------------------------------------------------- UMAP persistence

namespace {

// One staged section: id + length-prefixed payload.
void emit_section(std::ostream& os, uint32_t id, const std::string& payload) {
    w_u32(os, id);
    w_u64(os, payload.size());
    os.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

}  // namespace

void UMAP::save(std::ostream& os) const {
    require_little_endian();
    if (!fitted_)
        throw std::runtime_error("UMAP::save: model is not fitted");
    // --- hardening additions ------------------------------------------------
    // std::function metrics cannot be persisted (like pickling a lambda).
    if (params_.metric == "custom" || params_.output_metric == "custom")
        throw std::runtime_error(
            "UMAP::save: models using \"custom\" (std::function) metrics are "
            "not serialisable");
    // ------------------------------------------------------------------------

    std::vector<std::pair<uint32_t, std::string>> sections;

    {  // params
        std::ostringstream s;
        const UMAPParams& p = params_;
        w_i32(s, p.n_neighbors);
        w_i32(s, p.n_components);
        w_str(s, p.metric);
        w_opt(s, p.n_epochs, [](std::ostream& o, int v) { w_i32(o, v); });
        w_f64(s, p.learning_rate);
        w_str(s, p.init);
        w_f64(s, p.min_dist);
        w_f64(s, p.spread);
        w_f64(s, p.set_op_mix_ratio);
        w_f64(s, p.local_connectivity);
        w_f64(s, p.repulsion_strength);
        w_i32(s, p.negative_sample_rate);
        w_f64(s, p.transform_queue_size);
        w_opt(s, p.a, [](std::ostream& o, double v) { w_f64(o, v); });
        w_opt(s, p.b, [](std::ostream& o, double v) { w_f64(o, v); });
        w_opt(s, p.random_state,
              [](std::ostream& o, uint32_t v) { w_u32(o, v); });
        w_u32(s, p.transform_seed);
        w_u8(s, p.angular_rp_forest ? 1 : 0);
        w_u8(s, p.force_approximation_algorithm ? 1 : 0);
        w_u8(s, p.verbose ? 1 : 0);
        w_opt(s, p.n_epochs_list,
              [](std::ostream& o, const std::vector<int>& v) {
                  w_vec(o, v);
              });
        w_opt(s, p.init_array,
              [](std::ostream& o, const std::vector<float>& v) {
                  w_vec(o, v);
              });
        w_i64(s, p.init_array_cols);
        w_u8(s, p.unique ? 1 : 0);
        w_opt(s, p.disconnection_distance,
              [](std::ostream& o, double v) { w_f64(o, v); });
        w_str(s, p.transform_mode);
        // precomputed_knn: arrays + a search-index presence flag (the index
        // itself is the fitted knn_index_, re-linked on load).
        w_u8(s, p.precomputed_knn.has_value() ? 1 : 0);
        if (p.precomputed_knn.has_value()) {
            w_vec(s, p.precomputed_knn->indices);
            w_vec(s, p.precomputed_knn->dists);
            w_i64(s, p.precomputed_knn->n_samples);
            w_i64(s, p.precomputed_knn->n_neighbors);
            w_u8(s, p.precomputed_knn->search_index ? 1 : 0);
        }
        w_kwds(s, p.metric_kwds);
        w_str(s, p.output_metric);
        w_kwds(s, p.output_metric_kwds);
        w_u8(s, p.densmap ? 1 : 0);
        w_f64(s, p.dens_lambda);
        w_f64(s, p.dens_frac);
        w_f64(s, p.dens_var_shift);
        w_u8(s, p.output_dens ? 1 : 0);
        w_str(s, p.target_metric);
        w_kwds(s, p.target_metric_kwds);
        w_i32(s, p.target_n_neighbors);
        w_f64(s, p.target_weight);
        w_str(s, p.ensure_all_finite);
        w_u8(s, p.device == Device::GPU ? 1 : 0);
        // hardening additions: n_jobs appended (readers treat trailing
        // fields as optional, so files written before it still load).
        w_i32(s, p.n_jobs);
        sections.emplace_back(kSecParams, s.str());
    }
    {  // fitted scalars
        std::ostringstream s;
        w_f64(s, a_);
        w_f64(s, b_);
        w_f64(s, initial_alpha_);
        w_i32(s, n_neighbors_);
        w_u8(s, small_data_ ? 1 : 0);
        w_i64(s, n_);
        w_i64(s, dim_);
        w_u8(s, supervised_ ? 1 : 0);
        w_u64(s, input_hash_);
        w_i64(s, input_hash_rows_);
        w_u8(s, sparse_data_ ? 1 : 0);
        w_i64(s, knn_cols_);
        w_u8(s, force_approximation_algorithm_ ? 1 : 0);
        w_f64(s, disconnection_distance_);
        w_i64(s, edges_removed_);
        w_i64(s, vertices_disconnected_);
        sections.emplace_back(kSecScalars, s.str());
    }
    {  // dense raw data
        std::ostringstream s;
        w_vec(s, raw_data_);
        sections.emplace_back(kSecDenseRaw, s.str());
    }
    if (sparse_data_) {  // sparse raw CSR
        std::ostringstream s;
        w_vec(s, raw_indptr_);
        w_vec(s, raw_indices_);
        w_vec(s, raw_values_);
        sections.emplace_back(kSecSparseRaw, s.str());
    }
    {  // knn arrays
        std::ostringstream s;
        w_vec(s, knn_indices_);
        w_vec(s, knn_dists_);
        sections.emplace_back(kSecKnn, s.str());
    }
    {  // graph + smooth-knn outputs
        std::ostringstream s;
        w_csr(s, graph_);
        w_vec(s, sigmas_);
        w_vec(s, rhos_);
        sections.emplace_back(kSecGraph, s.str());
    }
    {  // embedding + fit init
        std::ostringstream s;
        w_vec(s, embedding_);
        w_vec(s, fit_init_);
        sections.emplace_back(kSecEmbedding, s.str());
    }
    {  // extras
        std::ostringstream s;
        w_vec(s, unique_inverse_);
        w_u64(s, embedding_list_.size());
        for (const auto& e : embedding_list_) w_vec(s, e);
        w_csr(s, graph_dists_);
        w_vec(s, rad_orig_);
        w_vec(s, rad_emb_);
        sections.emplace_back(kSecExtras, s.str());
    }
    if (knn_index_) {  // prepared search index
        std::ostringstream s;
        knn_index_->serialize(s);
        sections.emplace_back(kSecPreparedIndex, s.str());
    }
    if (sparse_knn_index_) {  // hardening additions: sparse search index
        std::ostringstream s;
        sparse_knn_index_->serialize(s);
        sections.emplace_back(kSecSparsePreparedIndex, s.str());
    }
    if (index_rs_.has_value()) {  // persisted RandomState stream
        std::ostringstream s;
        w_str(s, index_rs_->save_state());
        sections.emplace_back(kSecRng, s.str());
    }

    os.write(kMagic, sizeof(kMagic));
    w_u32(os, kOrderSentinel);
    w_u32(os, kFormatVersion);
    w_u32(os, static_cast<uint32_t>(sections.size()));
    for (const auto& sec : sections) emit_section(os, sec.first, sec.second);
    if (!os)
        throw std::runtime_error("UMAP::save: stream write failed");
}

void UMAP::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("UMAP::save: cannot open '" + path +
                                 "' for writing");
    save(f);
    f.flush();
    if (!f)
        throw std::runtime_error("UMAP::save: write to '" + path + "' failed");
}

UMAP UMAP::load(std::istream& is) {
    require_little_endian();
    char magic[8];
    r_bytes(is, magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
        throw std::runtime_error("UMAP::load: not a UMAPCPP1 model file");
    if (r_u32(is) != kOrderSentinel)
        throw std::runtime_error(
            "UMAP::load: byte-order sentinel mismatch (file written on an "
            "incompatible platform)");
    const uint32_t version = r_u32(is);
    if (version != kFormatVersion)
        throw std::runtime_error("UMAP::load: unsupported model format "
                                 "version " + std::to_string(version));
    const uint32_t n_sections = r_u32(is);

    UMAP m;
    bool pknn_had_search_index = false;
    std::string prepared_payload;  // parsed after params (needs the metric)
    std::string sparse_prepared_payload;  // hardening additions: idem
    for (uint32_t si = 0; si < n_sections; ++si) {
        const uint32_t id = r_u32(is);
        const uint64_t nbytes = r_u64(is);
        std::string payload(static_cast<std::size_t>(nbytes), '\0');
        if (nbytes > 0)
            r_bytes(is, &payload[0], static_cast<std::size_t>(nbytes));
        std::istringstream s(payload);
        switch (id) {
            case kSecParams: {
                UMAPParams p;
                p.n_neighbors = r_i32(s);
                p.n_components = r_i32(s);
                p.metric = r_str(s);
                p.n_epochs =
                    r_opt<int>(s, [](std::istream& i) { return r_i32(i); });
                p.learning_rate = r_f64(s);
                p.init = r_str(s);
                p.min_dist = r_f64(s);
                p.spread = r_f64(s);
                p.set_op_mix_ratio = r_f64(s);
                p.local_connectivity = r_f64(s);
                p.repulsion_strength = r_f64(s);
                p.negative_sample_rate = r_i32(s);
                p.transform_queue_size = r_f64(s);
                p.a = r_opt<double>(s,
                                    [](std::istream& i) { return r_f64(i); });
                p.b = r_opt<double>(s,
                                    [](std::istream& i) { return r_f64(i); });
                p.random_state = r_opt<uint32_t>(
                    s, [](std::istream& i) { return r_u32(i); });
                p.transform_seed = r_u32(s);
                p.angular_rp_forest = r_u8(s) != 0;
                p.force_approximation_algorithm = r_u8(s) != 0;
                p.verbose = r_u8(s) != 0;
                p.n_epochs_list = r_opt<std::vector<int>>(
                    s, [](std::istream& i) { return r_vec<int>(i); });
                p.init_array = r_opt<std::vector<float>>(
                    s, [](std::istream& i) { return r_vec<float>(i); });
                p.init_array_cols = r_i64(s);
                p.unique = r_u8(s) != 0;
                p.disconnection_distance = r_opt<double>(
                    s, [](std::istream& i) { return r_f64(i); });
                p.transform_mode = r_str(s);
                if (r_u8(s) != 0) {
                    PrecomputedKnn pk;
                    pk.indices = r_vec<int32_t>(s);
                    pk.dists = r_vec<float>(s);
                    pk.n_samples = r_i64(s);
                    pk.n_neighbors = r_i64(s);
                    pknn_had_search_index = r_u8(s) != 0;
                    p.precomputed_knn = std::move(pk);
                }
                p.metric_kwds = r_kwds(s);
                p.output_metric = r_str(s);
                p.output_metric_kwds = r_kwds(s);
                p.densmap = r_u8(s) != 0;
                p.dens_lambda = r_f64(s);
                p.dens_frac = r_f64(s);
                p.dens_var_shift = r_f64(s);
                p.output_dens = r_u8(s) != 0;
                p.target_metric = r_str(s);
                p.target_metric_kwds = r_kwds(s);
                p.target_n_neighbors = r_i32(s);
                p.target_weight = r_f64(s);
                p.ensure_all_finite = r_str(s);
                p.device = r_u8(s) != 0 ? Device::GPU : Device::CPU;
                // hardening additions: optional trailing n_jobs.
                if (s.peek() != std::char_traits<char>::eof())
                    p.n_jobs = r_i32(s);
                m.params_ = std::move(p);
                break;
            }
            case kSecScalars:
                m.a_ = r_f64(s);
                m.b_ = r_f64(s);
                m.initial_alpha_ = r_f64(s);
                m.n_neighbors_ = r_i32(s);
                m.small_data_ = r_u8(s) != 0;
                m.n_ = r_i64(s);
                m.dim_ = r_i64(s);
                m.supervised_ = r_u8(s) != 0;
                m.input_hash_ = r_u64(s);
                m.input_hash_rows_ = r_i64(s);
                m.sparse_data_ = r_u8(s) != 0;
                m.knn_cols_ = r_i64(s);
                m.force_approximation_algorithm_ = r_u8(s) != 0;
                m.disconnection_distance_ = r_f64(s);
                m.edges_removed_ = r_i64(s);
                m.vertices_disconnected_ = r_i64(s);
                break;
            case kSecDenseRaw:
                m.raw_data_ = r_vec<float>(s);
                break;
            case kSecSparseRaw:
                m.raw_indptr_ = r_vec<int32_t>(s);
                m.raw_indices_ = r_vec<int32_t>(s);
                m.raw_values_ = r_vec<float>(s);
                break;
            case kSecKnn:
                m.knn_indices_ = r_vec<int32_t>(s);
                m.knn_dists_ = r_vec<float>(s);
                break;
            case kSecGraph:
                m.graph_ = r_csr(s);
                m.sigmas_ = r_vec<float>(s);
                m.rhos_ = r_vec<float>(s);
                break;
            case kSecEmbedding:
                m.embedding_ = r_vec<float>(s);
                m.fit_init_ = r_vec<float>(s);
                break;
            case kSecExtras: {
                m.unique_inverse_ = r_vec<int64_t>(s);
                const uint64_t n_snap = r_u64(s);
                m.embedding_list_.clear();
                m.embedding_list_.reserve(
                    static_cast<std::size_t>(n_snap));
                for (uint64_t i = 0; i < n_snap; ++i)
                    m.embedding_list_.push_back(r_vec<float>(s));
                m.graph_dists_ = r_csr(s);
                m.rad_orig_ = r_vec<float>(s);
                m.rad_emb_ = r_vec<float>(s);
                break;
            }
            case kSecPreparedIndex:
                // The metric name/kwds live in the params section; defer.
                prepared_payload = payload;
                break;
            case kSecRng:
                m.index_rs_ = nnd::NumpyRandomState::from_state(r_str(s));
                break;
            case kSecSparsePreparedIndex:
                // hardening additions: deferred like the dense index.
                sparse_prepared_payload = payload;
                break;
            default:
                break;  // forward compatibility: skip unknown sections
        }
        if (id != kSecPreparedIndex && id != kSecSparsePreparedIndex &&
            s.fail())
            throw std::runtime_error(
                "UMAP model file: truncated section " + std::to_string(id));
    }

    if (!prepared_payload.empty()) {
        std::istringstream s(prepared_payload);
        m.knn_index_ = std::make_shared<nnd::PreparedIndex>(
            nnd::PreparedIndex::deserialize(s, m.params_.metric,
                                            m.params_.metric_kwds));
        if (m.params_.precomputed_knn.has_value() && pknn_had_search_index)
            m.params_.precomputed_knn->search_index = m.knn_index_;
    }
    // --- hardening additions ------------------------------------------------
    if (!sparse_prepared_payload.empty()) {
        std::istringstream s(sparse_prepared_payload);
        spdist::SparseMetricKwds skwds;
        skwds.p = m.params_.metric_kwds.p;
        m.sparse_knn_index_ = std::make_shared<spnnd::SparsePreparedIndex>(
            spnnd::SparsePreparedIndex::deserialize(s, m.params_.metric,
                                                    skwds));
    }
    // ------------------------------------------------------------------------
    m.fitted_ = true;
    return m;
}

UMAP UMAP::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("UMAP::load: cannot open '" + path + "'");
    return load(f);
}

}  // namespace umap
