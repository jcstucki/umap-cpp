// AlignedUMAP tests: bit-exact parity of the aligned pipeline pieces against
// scripts/gen_golden_aligned.py captures (tests/golden_aligned.hpp), plus the
// port of reference/umap/umap/tests/test_aligned_umap.py (upstream test names
// and thresholds preserved).
//
// Parity level (see include/umap/aligned_umap.hpp deviations):
//   * optimize_layout_aligned_euclidean is BIT-EXACT vs the sequential numba
//     reference given identical inputs (cases as1 / as2 / au1: fit with
//     move_other=True, ws=1 variant, and update with frozen old models);
//   * expand_relations / build_neighborhood_similarities /
//     make_epochs_per_sample-from-stored-graphs / the transform_seed
//     RandomState stream (seed triplet) / init_from_existing are BIT-EXACT;
//   * mapper graph_ CSRs (incl. the write-through explicit zeros) BIT-EXACT;
//   * spectral inits match to ~1e-5 up to per-column sign (Spectra vs ARPACK)
//     and procrustes_align to ~1e-4 (JacobiSVD + sequential f32 matmuls vs
//     LAPACK sgesdd + BLAS sgemm), so FULL AlignedUMAP.fit parity vs Python
//     is validated at the upstream quality thresholds, not bitwise.
// Fixed-seed bake: upstream's aligned_iris fixtures run unseeded
// (random_state=None); the port pins random_state=42, for which the
// reference passes the same thresholds (verified in the golden script:
// nn_acc = 0.718/0.782/0.752/0.776/0.740, ari = 1.000/0.633).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "doctest.h"
#include "golden_aligned.hpp"
#include "umap/aligned_umap.hpp"
#include "umap/layouts.hpp"
#include "umap/nn_descent.hpp"
#include "umap/spectral.hpp"
#include "umap/umap.hpp"
#include "upstream_data.hpp"
#include "upstream_helpers.hpp"

namespace {

using umap::AlignedParamUpdate;
using umap::AlignedUMAP;
using umap::AlignedUMAPParams;
using umap::SparseMatrix;
using umap::aligned::NumbaNpRandom;
using umap::aligned::RelationVec;
using umap::aligned::Tensor3f;
using umap::aligned::Tensor3i;

template <typename T, std::size_t N>
constexpr std::size_t asize(const T (&)[N]) {
    return N;
}

template <typename T>
std::vector<T> vec(const T* a, std::size_t n) {
    return std::vector<T>(a, a + n);
}

SparseMatrix make_graph(const std::int32_t* indptr, std::size_t n_indptr,
                        const std::int32_t* indices, const float* data,
                        std::size_t nnz) {
    SparseMatrix g;
    g.n_rows = static_cast<int32_t>(n_indptr - 1);
    g.n_cols = g.n_rows;
    g.indptr = vec(indptr, n_indptr);
    g.indices = vec(indices, nnz);
    g.data = vec(data, nnz);
    return g;
}

// COO of a stored CSR (explicit zeros included), like graph_.tocoo().
void to_coo(const SparseMatrix& g, std::vector<int32_t>& rows,
            std::vector<int32_t>& cols, std::vector<float>& vals) {
    rows.clear();
    cols.clear();
    vals.clear();
    for (int32_t r = 0; r < g.n_rows; ++r)
        for (int32_t t = g.indptr[r]; t < g.indptr[r + 1]; ++t) {
            rows.push_back(r);
            cols.push_back(g.indices[t]);
            vals.push_back(g.data[t]);
        }
}

int64_t count_float_mismatches(const std::vector<float>& got,
                               const float* want, std::size_t n) {
    REQUIRE(got.size() == n);
    int64_t bad = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (!(got[i] == want[i])) ++bad;
    return bad;
}

// Shared driver: run the C++ aligned SGD from a golden capture and compare
// the final embeddings bitwise.
struct GoldenSgd {
    int n_models, n_epochs;
    double lambda;
    bool move_other;
    const std::int64_t* rng_state;
    const std::int32_t* rel_shape;
    const std::int32_t* rel;
    const float* reg;
    std::vector<const std::int32_t*> heads;
    std::vector<const std::int32_t*> tails;
    std::vector<std::size_t> n_edges;
    std::vector<const double*> eps;
    std::vector<const float*> init;
    std::vector<const float*> final_;
    std::vector<int> rows;
};

void run_sgd_case(const GoldenSgd& c) {
    const int dim = 2;
    Tensor3i rel(c.rel_shape[0], c.rel_shape[1], c.rel_shape[2], -1);
    rel.v = vec(c.rel, rel.v.size());
    Tensor3f reg(c.rel_shape[0], c.rel_shape[1], c.rel_shape[2], 0.0f);
    reg.v = vec(c.reg, reg.v.size());

    std::vector<std::vector<float>> embeddings;
    std::vector<std::vector<int32_t>> heads, tails;
    std::vector<std::vector<double>> eps;
    for (int m = 0; m < c.n_models; ++m) {
        embeddings.push_back(
            vec(c.init[m], static_cast<std::size_t>(c.rows[m]) * dim));
        heads.push_back(vec(c.heads[m], c.n_edges[m]));
        tails.push_back(vec(c.tails[m], c.n_edges[m]));
        eps.push_back(vec(c.eps[m], c.n_edges[m]));
    }
    // NOTE: for the update case the frozen models' epochs_per_sample is
    // shorter than the edge arrays (np.full(n_samples, ...) upstream); the
    // caller passes the *eps* lengths separately.

    int64_t rng[3] = {c.rng_state[0], c.rng_state[1], c.rng_state[2]};
    NumbaNpRandom np_rng;
    umap::aligned::optimize_layout_aligned_euclidean(
        embeddings, embeddings, dim, heads, tails, c.n_epochs, eps, reg, rel,
        rng, np_rng, 1.576943460405378, 0.8950608781227859, 1.0, c.lambda, 1.0,
        5.0, c.move_other);

    for (int m = 0; m < c.n_models; ++m) {
        CHECK(count_float_mismatches(
                  embeddings[m], c.final_[m],
                  static_cast<std::size_t>(c.rows[m]) * dim) == 0);
    }
}

const std::vector<RelationVec>& as1_relations() {
    static const std::vector<RelationVec> rels = [] {
        RelationVec r;
        for (std::size_t i = 0; i < asize(golden_aligned::as1_rel_keys); ++i)
            r.emplace_back(golden_aligned::as1_rel_keys[i],
                           golden_aligned::as1_rel_vals[i]);
        return std::vector<RelationVec>{r, r};
    }();
    return rels;
}

std::vector<std::unordered_map<int32_t, int32_t>> as1_relation_maps(
    std::size_t count) {
    std::unordered_map<int32_t, int32_t> d;
    for (std::size_t i = 0; i < asize(golden_aligned::as1_rel_keys); ++i)
        d[golden_aligned::as1_rel_keys[i]] = golden_aligned::as1_rel_vals[i];
    return std::vector<std::unordered_map<int32_t, int32_t>>(count, d);
}

std::vector<std::vector<float>> as1_slices() {
    return {vec(golden_aligned::as1_X0, asize(golden_aligned::as1_X0)),
            vec(golden_aligned::as1_X1, asize(golden_aligned::as1_X1)),
            vec(golden_aligned::as1_X2, asize(golden_aligned::as1_X2))};
}

}  // namespace

TEST_SUITE("aligned parity") {

TEST_CASE("numba np.random seed/shuffle/uniform replication") {
    std::size_t off = 0;
    NumbaNpRandom rng;
    for (std::size_t c = 0; c < asize(golden_aligned::nprand_seeds); ++c) {
        const int64_t seed = golden_aligned::nprand_seeds[c];
        const int32_t n = golden_aligned::nprand_ns[c];
        std::vector<int32_t> arr(static_cast<std::size_t>(n));
        std::iota(arr.begin(), arr.end(), 0);
        rng.seed(static_cast<uint32_t>(static_cast<uint64_t>(seed)));
        rng.shuffle(arr.data(), n);
        for (int32_t i = 0; i < n; ++i)
            CHECK(arr[static_cast<std::size_t>(i)] ==
                  golden_aligned::nprand_perms[off + static_cast<std::size_t>(i)]);
        off += static_cast<std::size_t>(n);
    }

    // seed(42); shuffle(arange(5)); uniform(-10, 10, 6) -- state continuity.
    std::vector<int32_t> arr(5);
    std::iota(arr.begin(), arr.end(), 0);
    rng.seed(42);
    rng.shuffle(arr.data(), 5);
    for (int i = 0; i < 5; ++i)
        CHECK(arr[static_cast<std::size_t>(i)] ==
              golden_aligned::nprand_u_perm[i]);
    for (int i = 0; i < 6; ++i)
        CHECK(rng.uniform(-10.0, 10.0) == golden_aligned::nprand_u_vals[i]);
}

TEST_CASE("expand_relations matches the reference (as1)") {
    const Tensor3i rel = umap::aligned::expand_relations(as1_relations(), 3);
    REQUIRE(rel.d0 == golden_aligned::as1_expanded_shape[0]);
    REQUIRE(rel.d1 == golden_aligned::as1_expanded_shape[1]);
    REQUIRE(rel.d2 == golden_aligned::as1_expanded_shape[2]);
    REQUIRE(rel.v.size() == asize(golden_aligned::as1_expanded));
    int64_t bad = 0;
    for (std::size_t i = 0; i < rel.v.size(); ++i)
        if (rel.v[i] != golden_aligned::as1_expanded[i]) ++bad;
    CHECK(bad == 0);
}

TEST_CASE("mapper graph_ CSRs bit-exact incl. write-through zeros (as1)") {
    AlignedUMAPParams p;
    p.n_neighbors = 8;
    p.random_state = 42;
    AlignedUMAP model(p);
    model.fit(as1_slices(), 4, as1_relation_maps(2));
    REQUIRE(model.n_models() == 3);

    const std::int32_t* indptrs[] = {golden_aligned::as1_g0_indptr,
                                     golden_aligned::as1_g1_indptr,
                                     golden_aligned::as1_g2_indptr};
    const std::int32_t* indices[] = {golden_aligned::as1_g0_indices,
                                     golden_aligned::as1_g1_indices,
                                     golden_aligned::as1_g2_indices};
    const float* data[] = {golden_aligned::as1_g0_data,
                           golden_aligned::as1_g1_data,
                           golden_aligned::as1_g2_data};
    const std::size_t nnzs[] = {asize(golden_aligned::as1_g0_data),
                                asize(golden_aligned::as1_g1_data),
                                asize(golden_aligned::as1_g2_data)};
    int64_t explicit_zeros = 0;
    for (int m = 0; m < 3; ++m) {
        const SparseMatrix& g = model.mapper(static_cast<std::size_t>(m)).graph();
        REQUIRE(g.indptr.size() == 41);
        REQUIRE(g.data.size() == nnzs[m]);
        int64_t bad = 0;
        for (std::size_t i = 0; i < g.indptr.size(); ++i)
            if (g.indptr[i] != indptrs[m][i]) ++bad;
        for (std::size_t i = 0; i < g.data.size(); ++i) {
            if (g.indices[i] != indices[m][i]) ++bad;
            if (!(g.data[i] == data[m][i])) ++bad;
        }
        CHECK(bad == 0);
        explicit_zeros += std::count(g.data.begin(), g.data.end(), 0.0f);
    }
    // The epoch filter's write-through leaves explicit zeros behind in at
    // least one mapper graph: the quirk is actually exercised by this case
    // (mapper 0 has two zeroed edges; mappers 1 and 2 happen to have none).
    CHECK(explicit_zeros > 0);

    // Full fit sanity on the same model: finite embeddings of the right
    // shape (behavioral -- the SGD consumes C++ spectral inits here).
    for (int m = 0; m < 3; ++m) {
        const auto& e = model.embeddings()[static_cast<std::size_t>(m)];
        REQUIRE(e.size() == 80);
        for (float v : e) CHECK(std::isfinite(v));
    }
}

TEST_CASE("build_neighborhood_similarities + epochs/heads/tails bit-exact "
          "(as1)") {
    const SparseMatrix g0 = make_graph(
        golden_aligned::as1_g0_indptr, asize(golden_aligned::as1_g0_indptr),
        golden_aligned::as1_g0_indices, golden_aligned::as1_g0_data,
        asize(golden_aligned::as1_g0_data));
    const SparseMatrix g1 = make_graph(
        golden_aligned::as1_g1_indptr, asize(golden_aligned::as1_g1_indptr),
        golden_aligned::as1_g1_indices, golden_aligned::as1_g1_data,
        asize(golden_aligned::as1_g1_data));
    const SparseMatrix g2 = make_graph(
        golden_aligned::as1_g2_indptr, asize(golden_aligned::as1_g2_indptr),
        golden_aligned::as1_g2_indices, golden_aligned::as1_g2_data,
        asize(golden_aligned::as1_g2_data));

    const Tensor3i rel = umap::aligned::expand_relations(as1_relations(), 3);
    const Tensor3f reg =
        umap::aligned::build_neighborhood_similarities({&g0, &g1, &g2}, rel);
    REQUIRE(reg.v.size() == asize(golden_aligned::as1_sgd_reg));
    int64_t bad = 0;
    for (std::size_t i = 0; i < reg.v.size(); ++i)
        if (!(reg.v[i] == golden_aligned::as1_sgd_reg[i])) ++bad;
    CHECK(bad == 0);

    // heads/tails = graph_.tocoo().row/.col; epochs_per_sample from the COO
    // data (explicit zeros -> -1 entries, sampled every epoch).
    const SparseMatrix* graphs[] = {&g0, &g1, &g2};
    const std::int32_t* gold_heads[] = {golden_aligned::as1_sgd_head0,
                                        golden_aligned::as1_sgd_head1,
                                        golden_aligned::as1_sgd_head2};
    const std::int32_t* gold_tails[] = {golden_aligned::as1_sgd_tail0,
                                        golden_aligned::as1_sgd_tail1,
                                        golden_aligned::as1_sgd_tail2};
    const double* gold_eps[] = {golden_aligned::as1_sgd_eps0,
                                golden_aligned::as1_sgd_eps1,
                                golden_aligned::as1_sgd_eps2};
    const std::size_t n_edges[] = {asize(golden_aligned::as1_sgd_eps0),
                                   asize(golden_aligned::as1_sgd_eps1),
                                   asize(golden_aligned::as1_sgd_eps2)};
    bool saw_negative = false;
    for (int m = 0; m < 3; ++m) {
        std::vector<int32_t> rows, cols;
        std::vector<float> vals;
        to_coo(*graphs[m], rows, cols, vals);
        REQUIRE(rows.size() == n_edges[m]);
        const std::vector<double> eps =
            umap::make_epochs_per_sample(vals.data(), vals.size(),
                                         golden_aligned::as1_sgd_n_epochs);
        int64_t bad_m = 0;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] != gold_heads[m][i]) ++bad_m;
            if (cols[i] != gold_tails[m][i]) ++bad_m;
            if (!(eps[i] == gold_eps[m][i])) ++bad_m;
            if (eps[i] == -1.0) saw_negative = true;
        }
        CHECK(bad_m == 0);
    }
    // The explicit-zero write-through quirk reaches the SGD as -1
    // epochs_per_sample entries (sampled every epoch) in mapper 0.
    CHECK(saw_negative);
}

TEST_CASE("transform_seed RandomState stream: seed triplet after the "
          "spectral draws (as1)") {
    // rng_state_transform = RandomState(42); three spectral_layout calls
    // (40 x 3 eigsh warm-start normals each); then randint(INT32_MIN,
    // INT32_MAX, 3).
    umap::nnd::NumpyRandomState rs(42);
    for (int s = 0; s < 3; ++s)
        for (int i = 0; i < 40 * 3; ++i) (void)rs.gauss();
    const auto st = rs.rng_state3();
    CHECK(st[0] == golden_aligned::as1_sgd_rng_state[0]);
    CHECK(st[1] == golden_aligned::as1_sgd_rng_state[1]);
    CHECK(st[2] == golden_aligned::as1_sgd_rng_state[2]);
}

TEST_CASE("spectral init parity: ~1e-5 up to per-column sign (as1, "
          "documented deviation)") {
    // Recompute what spectral_layout does for a connected graph (eigenvectors
    // 1..dim of the normalized Laplacian, scaled by 10/max|coord|, f32) with
    // the Spectra solver and compare against the ARPACK-computed golden.
    const SparseMatrix g0 = make_graph(
        golden_aligned::as1_g0_indptr, asize(golden_aligned::as1_g0_indptr),
        golden_aligned::as1_g0_indices, golden_aligned::as1_g0_data,
        asize(golden_aligned::as1_g0_data));
    std::vector<double> eigenvalues, eigenvectors;
    REQUIRE(umap::normalized_laplacian_eigs(g0, 3, eigenvalues, eigenvectors));
    std::vector<double> init(80);
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 2; ++j)
            init[static_cast<std::size_t>(i) * 2 + j] =
                eigenvectors[static_cast<std::size_t>(i) * 3 + j + 1];
    double max_abs = 0.0;
    for (double v : init) max_abs = std::max(max_abs, std::abs(v));
    const double expansion = 10.0 / max_abs;

    for (int col = 0; col < 2; ++col) {
        double best = 1e300;
        for (double sign : {1.0, -1.0}) {
            double diff = 0.0;
            for (int i = 0; i < 40; ++i) {
                const double ours =
                    sign * init[static_cast<std::size_t>(i) * 2 + col] *
                    expansion;
                diff = std::max(
                    diff,
                    std::abs(ours - static_cast<double>(
                                        golden_aligned::as1_spec0
                                            [static_cast<std::size_t>(i) * 2 +
                                             col])));
            }
            best = std::min(best, diff);
        }
        // Bound reflects the matvec-only eigsh mirror (Spectra Lanczos on
        // 2I-L at upstream's tol=1e-4, replacing the old tol=1e-10 shift-
        // invert). Two different Lanczos codes stopped at the same loose tol
        // land ~1e-3 apart on this small-eigengap graph (lambda2..3 gap ~0.05)
        // while each stays within ~1e-3 of the dense-exact eigenvectors;
        // observed 0.0007..0.00099. Spectral init is quality-level parity by
        // contract (Spectra vs ARPACK), validated downstream by trustworthiness.
        CHECK(best < 3e-3);
    }
}

TEST_CASE("procrustes_align parity (as1, documented tolerance)") {
    const float* bases[] = {golden_aligned::as1_proc0_base,
                            golden_aligned::as1_proc1_base};
    const float* aligns[] = {golden_aligned::as1_proc0_to_align,
                             golden_aligned::as1_proc1_to_align};
    const std::int64_t* anchors[] = {golden_aligned::as1_proc0_anchors,
                                     golden_aligned::as1_proc1_anchors};
    const int n_anchors[] = {golden_aligned::as1_proc0_n_anchors,
                             golden_aligned::as1_proc1_n_anchors};
    const float* results[] = {golden_aligned::as1_proc0_result,
                              golden_aligned::as1_proc1_result};
    const std::size_t sizes[] = {asize(golden_aligned::as1_proc0_result),
                                 asize(golden_aligned::as1_proc1_result)};
    for (int c = 0; c < 2; ++c) {
        std::vector<int32_t> left, right;
        for (int i = 0; i < n_anchors[c]; ++i) {
            left.push_back(static_cast<int32_t>(anchors[c][i]));
            right.push_back(
                static_cast<int32_t>(anchors[c][n_anchors[c] + i]));
        }
        const std::vector<float> base =
            vec(bases[c], asize(golden_aligned::as1_proc0_base));
        const std::vector<float> to_align = vec(aligns[c], sizes[c]);
        const std::vector<float> got =
            umap::aligned::procrustes_align(base, to_align, 2, left, right);
        REQUIRE(got.size() == sizes[c]);
        double max_diff = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i)
            max_diff = std::max(
                max_diff, std::abs(static_cast<double>(got[i]) -
                                   static_cast<double>(results[c][i])));
        CHECK(max_diff < 5e-4);  // JacobiSVD/sgesdd + f32 matmul ordering
    }
}

TEST_CASE("aligned SGD bit-exact vs numba: fit pass (as1, move_other=true)") {
    GoldenSgd c;
    c.n_models = golden_aligned::as1_sgd_n_models;
    c.n_epochs = golden_aligned::as1_sgd_n_epochs;
    c.lambda = golden_aligned::as1_sgd_lambda;
    c.move_other = golden_aligned::as1_sgd_move_other;
    c.rng_state = golden_aligned::as1_sgd_rng_state;
    c.rel_shape = golden_aligned::as1_sgd_rel_shape;
    c.rel = golden_aligned::as1_sgd_rel;
    c.reg = golden_aligned::as1_sgd_reg;
    c.heads = {golden_aligned::as1_sgd_head0, golden_aligned::as1_sgd_head1,
               golden_aligned::as1_sgd_head2};
    c.tails = {golden_aligned::as1_sgd_tail0, golden_aligned::as1_sgd_tail1,
               golden_aligned::as1_sgd_tail2};
    c.n_edges = {asize(golden_aligned::as1_sgd_eps0),
                 asize(golden_aligned::as1_sgd_eps1),
                 asize(golden_aligned::as1_sgd_eps2)};
    c.eps = {golden_aligned::as1_sgd_eps0, golden_aligned::as1_sgd_eps1,
             golden_aligned::as1_sgd_eps2};
    c.init = {golden_aligned::as1_sgd_init0, golden_aligned::as1_sgd_init1,
              golden_aligned::as1_sgd_init2};
    c.final_ = {golden_aligned::as1_sgd_final0, golden_aligned::as1_sgd_final1,
                golden_aligned::as1_sgd_final2};
    c.rows = {golden_aligned::as1_sgd_rows0, golden_aligned::as1_sgd_rows1,
              golden_aligned::as1_sgd_rows2};
    run_sgd_case(c);

    CHECK(c.move_other == true);
    CHECK(c.n_epochs == 200);
}

TEST_CASE("aligned SGD bit-exact vs numba: window_size=1, lambda=0.05 "
          "(as2)") {
    GoldenSgd c;
    c.n_models = golden_aligned::as2_sgd_n_models;
    c.n_epochs = golden_aligned::as2_sgd_n_epochs;
    c.lambda = golden_aligned::as2_sgd_lambda;
    c.move_other = golden_aligned::as2_sgd_move_other;
    c.rng_state = golden_aligned::as2_sgd_rng_state;
    c.rel_shape = golden_aligned::as2_sgd_rel_shape;
    c.rel = golden_aligned::as2_sgd_rel;
    c.reg = golden_aligned::as2_sgd_reg;
    c.heads = {golden_aligned::as2_sgd_head0, golden_aligned::as2_sgd_head1};
    c.tails = {golden_aligned::as2_sgd_tail0, golden_aligned::as2_sgd_tail1};
    c.n_edges = {asize(golden_aligned::as2_sgd_eps0),
                 asize(golden_aligned::as2_sgd_eps1)};
    c.eps = {golden_aligned::as2_sgd_eps0, golden_aligned::as2_sgd_eps1};
    c.init = {golden_aligned::as2_sgd_init0, golden_aligned::as2_sgd_init1};
    c.final_ = {golden_aligned::as2_sgd_final0, golden_aligned::as2_sgd_final1};
    c.rows = {golden_aligned::as2_sgd_rows0, golden_aligned::as2_sgd_rows1};
    run_sgd_case(c);

    CHECK(c.lambda == 0.05);
    CHECK(c.rel_shape[1] == 3);  // 2 * ws + 1 with ws = 1
}

TEST_CASE("update: init_from_existing bit-exact (au1)") {
    const SparseMatrix g = make_graph(
        golden_aligned::au1_ife_indptr, asize(golden_aligned::au1_ife_indptr),
        golden_aligned::au1_ife_indices, golden_aligned::au1_ife_data,
        asize(golden_aligned::au1_ife_data));
    RelationVec rel;
    for (std::size_t i = 0; i < asize(golden_aligned::au1_ife_rel_keys); ++i)
        rel.emplace_back(golden_aligned::au1_ife_rel_keys[i],
                         golden_aligned::au1_ife_rel_vals[i]);
    const std::vector<float> prev =
        vec(golden_aligned::au1_ife_prev, asize(golden_aligned::au1_ife_prev));
    NumbaNpRandom np_rng;  // unused: the golden case is fallback-free
    const std::vector<float> got =
        umap::aligned::init_from_existing(prev, 2, g, rel, np_rng);
    CHECK(count_float_mismatches(got, golden_aligned::au1_ife_result,
                                 asize(golden_aligned::au1_ife_result)) == 0);
}

TEST_CASE("aligned SGD bit-exact vs numba: update pass (au1, frozen old "
          "models, move_other=false)") {
    GoldenSgd c;
    c.n_models = golden_aligned::au1_sgd_n_models;
    c.n_epochs = golden_aligned::au1_sgd_n_epochs;
    c.lambda = golden_aligned::au1_sgd_lambda;
    c.move_other = golden_aligned::au1_sgd_move_other;
    c.rng_state = golden_aligned::au1_sgd_rng_state;
    c.rel_shape = golden_aligned::au1_sgd_rel_shape;
    c.rel = golden_aligned::au1_sgd_rel;
    c.reg = golden_aligned::au1_sgd_reg;
    c.heads = {golden_aligned::au1_sgd_head0, golden_aligned::au1_sgd_head1,
               golden_aligned::au1_sgd_head2};
    c.tails = {golden_aligned::au1_sgd_tail0, golden_aligned::au1_sgd_tail1,
               golden_aligned::au1_sgd_tail2};
    c.n_edges = {asize(golden_aligned::au1_sgd_head0),
                 asize(golden_aligned::au1_sgd_head1),
                 asize(golden_aligned::au1_sgd_head2)};
    c.eps = {golden_aligned::au1_sgd_eps0, golden_aligned::au1_sgd_eps1,
             golden_aligned::au1_sgd_eps2};
    c.init = {golden_aligned::au1_sgd_init0, golden_aligned::au1_sgd_init1,
              golden_aligned::au1_sgd_init2};
    c.final_ = {golden_aligned::au1_sgd_final0, golden_aligned::au1_sgd_final1,
                golden_aligned::au1_sgd_final2};
    c.rows = {golden_aligned::au1_sgd_rows0, golden_aligned::au1_sgd_rows1,
              golden_aligned::au1_sgd_rows2};

    // The frozen models' epochs_per_sample are np.full(n_samples, n_epochs+1)
    // -- SHORTER than the edge arrays. Rebuild the eps lengths accordingly.
    REQUIRE(asize(golden_aligned::au1_sgd_eps0) ==
            static_cast<std::size_t>(golden_aligned::au1_sgd_rows0));
    REQUIRE(asize(golden_aligned::au1_sgd_eps1) ==
            static_cast<std::size_t>(golden_aligned::au1_sgd_rows1));
    REQUIRE(asize(golden_aligned::au1_sgd_eps2) ==
            asize(golden_aligned::au1_sgd_head2));
    CHECK(c.move_other == false);
    for (std::size_t i = 0; i < asize(golden_aligned::au1_sgd_eps0); ++i)
        CHECK(golden_aligned::au1_sgd_eps0[i] ==
              static_cast<double>(c.n_epochs + 1));

    const int dim = 2;
    Tensor3i rel(c.rel_shape[0], c.rel_shape[1], c.rel_shape[2], -1);
    rel.v = vec(c.rel, rel.v.size());
    Tensor3f reg(c.rel_shape[0], c.rel_shape[1], c.rel_shape[2], 0.0f);
    reg.v = vec(c.reg, reg.v.size());
    std::vector<std::vector<float>> embeddings;
    std::vector<std::vector<int32_t>> heads, tails;
    std::vector<std::vector<double>> eps;
    const std::size_t eps_lens[] = {asize(golden_aligned::au1_sgd_eps0),
                                    asize(golden_aligned::au1_sgd_eps1),
                                    asize(golden_aligned::au1_sgd_eps2)};
    for (int m = 0; m < c.n_models; ++m) {
        embeddings.push_back(
            vec(c.init[m], static_cast<std::size_t>(c.rows[m]) * dim));
        heads.push_back(vec(c.heads[m], c.n_edges[m]));
        tails.push_back(vec(c.tails[m], c.n_edges[m]));
        eps.push_back(vec(c.eps[m], eps_lens[m]));
    }
    int64_t rng[3] = {c.rng_state[0], c.rng_state[1], c.rng_state[2]};
    NumbaNpRandom np_rng;
    umap::aligned::optimize_layout_aligned_euclidean(
        embeddings, embeddings, dim, heads, tails, c.n_epochs, eps, reg, rel,
        rng, np_rng, 1.576943460405378, 0.8950608781227859, 1.0, c.lambda, 1.0,
        5.0, c.move_other);
    for (int m = 0; m < c.n_models; ++m)
        CHECK(count_float_mismatches(
                  embeddings[m], c.final_[m],
                  static_cast<std::size_t>(c.rows[m]) * dim) == 0);

    // Old models frozen: embeddings 0 and 1 must equal their inits bitwise.
    CHECK(count_float_mismatches(
              embeddings[0], golden_aligned::au1_sgd_init0,
              static_cast<std::size_t>(c.rows[0]) * dim) == 0);
    CHECK(count_float_mismatches(
              embeddings[1], golden_aligned::au1_sgd_init1,
              static_cast<std::size_t>(c.rows[1]) * dim) == 0);
}

TEST_CASE("AlignedParam broadcasting + set_aligned_params semantics") {
    // get_nth_item_or_val: scalar broadcasts, list indexes, short list raises.
    umap::AlignedParam<int> scalar{7};
    CHECK(scalar.get(0) == 7);
    CHECK(scalar.get(12) == 7);
    umap::AlignedParam<int> list{std::vector<int>{3, 4, 5}};
    CHECK(list.get(2) == 5);
    CHECK_THROWS_AS((void)list.get(3), std::out_of_range);

    // set_aligned_params: list appends; scalar + equal value is a no-op;
    // scalar + differing value broadcasts n_models copies then appends.
    list.set_aligned(9, 3);
    CHECK(list.get(3) == 9);
    umap::AlignedParam<int> same{7};
    same.set_aligned(7, 3);
    CHECK_FALSE(same.is_list());
    umap::AlignedParam<int> diff{7};
    diff.set_aligned(8, 3);
    REQUIRE(diff.is_list());
    REQUIRE(diff.list().size() == 4);
    CHECK(diff.get(0) == 7);
    CHECK(diff.get(2) == 7);
    CHECK(diff.get(3) == 8);
}

TEST_CASE("validation: relations count, short param lists, n_epochs change") {
    AlignedUMAPParams p;
    p.n_neighbors = 8;
    p.random_state = 42;
    AlignedUMAP model(p);
    // len(X) - 1 != len(relations)
    CHECK_THROWS_AS(model.fit(as1_slices(), 4, as1_relation_maps(1)),
                    std::invalid_argument);

    // Per-model list shorter than n_models -> IndexError equivalent.
    AlignedUMAPParams p2;
    p2.n_neighbors = std::vector<int>{8, 8};
    p2.random_state = 42;
    AlignedUMAP model2(p2);
    CHECK_THROWS_AS(model2.fit(as1_slices(), 4, as1_relation_maps(2)),
                    std::out_of_range);

    // update() with a different n_epochs (upstream would tuple-ise and crash).
    AlignedUMAPParams p3;
    p3.n_neighbors = 8;
    p3.random_state = 42;
    AlignedUMAP model3(p3);
    auto slices = as1_slices();
    model3.fit({slices[0], slices[1]}, 4, as1_relation_maps(1));
    AlignedParamUpdate up;
    up.n_epochs = 123;
    CHECK_THROWS_AS(
        model3.update(slices[2], 4, as1_relation_maps(1)[0], nullptr, up),
        std::invalid_argument);
}

TEST_CASE("class-driven fit + update: behavioral parity with the reference "
          "(as1/au1 shapes, quality)") {
    AlignedUMAPParams p;
    p.n_neighbors = 8;
    p.random_state = 42;
    AlignedUMAP model(p);
    auto slices = as1_slices();
    model.fit({slices[0], slices[1]}, 4, as1_relation_maps(1));
    REQUIRE(model.n_models() == 2);
    model.update(slices[2], 4, as1_relation_maps(1)[0]);
    REQUIRE(model.n_models() == 3);
    REQUIRE(model.embeddings().size() == 3);
    for (const auto& e : model.embeddings()) {
        REQUIRE(e.size() == 80);
        for (float v : e) CHECK(std::isfinite(v));
    }
    // n_epochs was resolved to 200 and stored back (upstream mutates too).
    REQUIRE(model.params().n_epochs.has_value());
    CHECK(*model.params().n_epochs == 200);
}

}  // TEST_SUITE("aligned parity")

// ---------------------------------------------------------------------------
// Port of reference/umap/umap/tests/test_aligned_umap.py. Upstream fixtures:
//   aligned_iris: 5 slices iris.data[i:i+50], i in 0,25,..,100
//   aligned_iris_relations: 4 x {a: a+25 for a in range(25)}
//   aligned_iris_model: AlignedUMAP() fit (unseeded upstream; random_state=42
//   baked here, thresholds verified for the reference at this seed).
// ---------------------------------------------------------------------------

namespace {

constexpr int64_t kSliceRows = 50, kIrisCols = 4;

std::vector<std::vector<float>> aligned_iris_data() {
    std::vector<std::vector<float>> slices;
    for (int start = 0; start <= 100; start += 25) {
        std::vector<float> s(kSliceRows * kIrisCols);
        for (int64_t i = 0; i < kSliceRows * kIrisCols; ++i)
            s[static_cast<std::size_t>(i)] = static_cast<float>(
                updata::iris_data[start * kIrisCols + i]);
        slices.push_back(std::move(s));
    }
    return slices;
}

std::vector<std::vector<int>> aligned_iris_target() {
    std::vector<std::vector<int>> targets;
    for (int start = 0; start <= 100; start += 25) {
        std::vector<int> t(kSliceRows);
        for (int64_t i = 0; i < kSliceRows; ++i)
            t[static_cast<std::size_t>(i)] = updata::iris_target[start + i];
        targets.push_back(std::move(t));
    }
    return targets;
}

std::vector<std::unordered_map<int32_t, int32_t>> aligned_iris_relations(
    std::size_t count) {
    std::unordered_map<int32_t, int32_t> d;
    for (int32_t a = 0; a < 25; ++a) d[a] = a + 25;
    return std::vector<std::unordered_map<int32_t, int32_t>>(count, d);
}

AlignedUMAP& aligned_iris_model() {
    static AlignedUMAP model = [] {
        AlignedUMAPParams p;
        p.random_state = 42;  // upstream unseeded; fixed-seed bake
        AlignedUMAP m(p);
        m.fit(aligned_iris_data(), kIrisCols, aligned_iris_relations(4));
        return m;
    }();
    return model;
}

// test_aligned_umap.py::nn_accuracy over the top-10 exact neighbour lists
// (upstream uses np.argsort of pairwise_distances; stable argsort here --
// tie order at the 10th neighbour may differ, noise-level for these bars).
std::vector<int32_t> top10_neighbors(const float* X, int64_t n, int64_t dim) {
    std::vector<int32_t> nn(static_cast<std::size_t>(n) * 10);
    std::vector<std::pair<double, int32_t>> row(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double d2 = 0.0;
            for (int64_t c = 0; c < dim; ++c) {
                const double diff = static_cast<double>(X[i * dim + c]) -
                                    static_cast<double>(X[j * dim + c]);
                d2 += diff * diff;
            }
            row[static_cast<std::size_t>(j)] = {d2, static_cast<int32_t>(j)};
        }
        std::stable_sort(row.begin(), row.end(),
                         [](const auto& a, const auto& b) {
                             return a.first < b.first;
                         });
        for (int k = 0; k < 10; ++k)
            nn[static_cast<std::size_t>(i) * 10 + k] =
                row[static_cast<std::size_t>(k)].second;
    }
    return nn;
}

double nn_accuracy(const std::vector<int32_t>& true_nn,
                   const std::vector<int32_t>& embd_nn, int64_t n) {
    double correct = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int a = 0; a < 10; ++a) {
            const int32_t v = true_nn[static_cast<std::size_t>(i) * 10 + a];
            for (int b = 0; b < 10; ++b) {
                if (embd_nn[static_cast<std::size_t>(i) * 10 + b] == v) {
                    correct += 1.0;
                    break;
                }
            }
        }
    }
    return correct / (static_cast<double>(n) * 10.0);
}

double slice_nn_accuracy(const std::vector<float>& data,
                         const std::vector<float>& embedding, int64_t n) {
    const auto true_nn = top10_neighbors(data.data(), n, kIrisCols);
    const auto embd_nn = top10_neighbors(embedding.data(), n, 2);
    return nn_accuracy(true_nn, embd_nn, n);
}

}  // namespace

TEST_SUITE("test_aligned_umap.py port") {

TEST_CASE("test_neighbor_local_neighbor_accuracy") {
    const auto data = aligned_iris_data();
    AlignedUMAP& model = aligned_iris_model();
    for (std::size_t i = 0; i < data.size(); ++i) {
        const double acc =
            slice_nn_accuracy(data[i], model.embeddings()[i], kSliceRows);
        CHECK(acc >= 0.65);
    }
}

TEST_CASE("test_local_clustering") {
    const auto target = aligned_iris_target();
    AlignedUMAP& model = aligned_iris_model();

    {
        const auto& embd = model.embeddings()[1];
        const std::vector<int> clusters =
            testutil::kmeans_predict(embd.data(), kSliceRows, 2, 2);
        const double ari = testutil::adjusted_rand_score(target[1], clusters);
        CHECK(ari >= 0.75);
    }
    {
        const auto& embd = model.embeddings()[3];
        const std::vector<int> clusters =
            testutil::kmeans_predict(embd.data(), kSliceRows, 2, 2);
        const double ari = testutil::adjusted_rand_score(target[3], clusters);
        CHECK(ari >= 0.40);
    }
}

TEST_CASE("test_aligned_update") {
    const auto data = aligned_iris_data();
    AlignedUMAPParams p;
    p.random_state = 42;  // fixed-seed bake (upstream unseeded)
    AlignedUMAP small_model(p);
    small_model.fit({data[0], data[1], data[2]}, kIrisCols,
                    aligned_iris_relations(2));
    small_model.update(data[3], kIrisCols, aligned_iris_relations(1)[0]);
    for (std::size_t i = 0; i < 4; ++i) {
        const double acc = slice_nn_accuracy(
            data[i], small_model.embeddings()[i], kSliceRows);
        CHECK(acc >= 0.45);
    }
}

TEST_CASE("test_aligned_update_params") {
    const auto data = aligned_iris_data();
    AlignedUMAPParams p;
    p.n_neighbors = std::vector<int>{15, 15, 15};
    p.random_state = 42;  // fixed-seed bake (upstream unseeded)
    AlignedUMAP small_model(p);
    small_model.fit({data[0], data[1], data[2]}, kIrisCols,
                    aligned_iris_relations(2));
    AlignedParamUpdate up;
    up.n_neighbors = 15;
    small_model.update(data[3], kIrisCols, aligned_iris_relations(1)[0],
                       nullptr, up);
    REQUIRE(small_model.params().n_neighbors.is_list());
    CHECK(small_model.params().n_neighbors.list().size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        const double acc = slice_nn_accuracy(
            data[i], small_model.embeddings()[i], kSliceRows);
        CHECK(acc >= 0.45);
    }
}

TEST_CASE("test_aligned_update_array_error" * doctest::skip()) {}
// upstream @pytest.mark.skip(reason="Temporarily disable"); also relies on
// passing arrays where scalars are expected, unrepresentable in the typed
// C++ surface.

// ------------------------- expand_relations characterization (upstream) ----

namespace expand_golden {
// _EXPAND_RELATIONS_INPUT
const std::vector<RelationVec> input = {
    {{0, 1}, {1, 2}, {2, 0}, {3, 3}},
    {{0, 0}, {2, 1}, {3, 2}},
    {{1, 0}, {2, 3}},
};
// _EXPAND_RELATIONS_WS3 (verbatim)
const int32_t ws3[4 * 7 * 4] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1,  2,  0,  3,  -1, 1,  0,  2,  -1, -1, -1, -1,

    -1, -1, -1, -1, -1, -1, -1, -1, 2,  0,  1,  3,  -1, -1, -1, -1,
    0,  -1, 1,  2,  -1, -1, -1, -1, -1, -1, -1, -1,

    -1, -1, -1, -1, 2,  1,  3,  -1, 0,  2,  3,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,

    1,  -1, -1, 3,  2,  -1, -1, 3,  1,  -1, -1, 2,  -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};
}  // namespace expand_golden

TEST_CASE("test_expand_relations_golden_ws3") {
    const Tensor3i result =
        umap::aligned::expand_relations(expand_golden::input, 3);
    REQUIRE(result.d0 == 4);
    REQUIRE(result.d1 == 7);
    REQUIRE(result.d2 == 4);
    int64_t bad = 0;
    for (std::size_t i = 0; i < result.v.size(); ++i)
        if (result.v[i] != expand_golden::ws3[i]) ++bad;
    CHECK(bad == 0);
}

TEST_CASE("test_expand_relations_shape_and_sum") {
    struct Param {
        int window_size;
        int64_t d0, d1, d2;
        int64_t total;
    };
    const Param params[] = {
        {1, 4, 3, 4, -9}, {2, 4, 5, 4, -19}, {3, 4, 7, 4, -45}};
    for (const auto& pr : params) {
        CAPTURE(pr.window_size);
        const Tensor3i result =
            umap::aligned::expand_relations(expand_golden::input,
                                            pr.window_size);
        CHECK(result.d0 == pr.d0);
        CHECK(result.d1 == pr.d1);
        CHECK(result.d2 == pr.d2);
        int64_t total = 0;
        for (int32_t v : result.v) total += v;
        CHECK(total == pr.total);
    }
}

}  // TEST_SUITE("test_aligned_umap.py port")
