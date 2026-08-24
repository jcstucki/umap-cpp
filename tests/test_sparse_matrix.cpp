// Unit tests for the minimal CSR sparse matrix (scipy-equivalent semantics).
#include <vector>

#include "doctest.h"
#include "umap/sparse_matrix.hpp"

using umap::SparseMatrix;

namespace {
SparseMatrix small() {
    // [[1, 0, 2],
    //  [0, 0, 3],
    //  [4, 5, 0]]
    return SparseMatrix::from_coo(3, 3, {0, 0, 1, 2, 2}, {0, 2, 2, 0, 1},
                                  {1.f, 2.f, 3.f, 4.f, 5.f});
}
}  // namespace

TEST_CASE("from_coo sums duplicates and sorts columns") {
    // unsorted triplets with a duplicate at (1, 2)
    SparseMatrix m = SparseMatrix::from_coo(2, 3, {1, 0, 1, 1}, {2, 1, 0, 2},
                                            {1.5f, 7.f, 2.f, 0.5f});
    std::vector<float> dense = m.to_dense();
    std::vector<float> expected = {0.f, 7.f, 0.f, 2.f, 0.f, 2.f};
    CHECK(dense == expected);
    CHECK(m.nnz() == 3);
    // canonical: columns sorted within rows
    for (int32_t r = 0; r < m.n_rows; ++r) {
        for (int32_t k = m.indptr[r] + 1; k < m.indptr[r + 1]; ++k) {
            CHECK(m.indices[k - 1] < m.indices[k]);
        }
    }
}

TEST_CASE("transpose") {
    SparseMatrix t = small().transpose();
    std::vector<float> expected = {1.f, 0.f, 4.f, 0.f, 0.f, 5.f, 2.f, 3.f, 0.f};
    CHECK(t.to_dense() == expected);
    // double transpose round-trips
    CHECK(t.transpose().to_dense() == small().to_dense());
}

TEST_CASE("add / subtract / multiply / maximum") {
    SparseMatrix a = small();
    SparseMatrix b = a.transpose();
    std::vector<float> da = a.to_dense();
    std::vector<float> db = b.to_dense();

    std::vector<float> sum = a.add(b).to_dense();
    std::vector<float> diff = a.subtract(b).to_dense();
    std::vector<float> prod = a.multiply(b).to_dense();
    std::vector<float> mx = a.maximum(b).to_dense();
    for (std::size_t i = 0; i < da.size(); ++i) {
        CHECK(sum[i] == da[i] + db[i]);
        CHECK(diff[i] == da[i] - db[i]);
        CHECK(prod[i] == da[i] * db[i]);
        CHECK(mx[i] == (da[i] > db[i] ? da[i] : db[i]));
    }
    // multiply keeps only the intersection pattern
    CHECK(a.multiply(b).nnz() <= a.nnz());
}

TEST_CASE("scaled and eliminate_zeros") {
    SparseMatrix a = small();
    std::vector<float> d2 = a.scaled(2.0f).to_dense();
    std::vector<float> d1 = a.to_dense();
    for (std::size_t i = 0; i < d1.size(); ++i) CHECK(d2[i] == 2.0f * d1[i]);

    // a - a has explicit zeros everywhere in the union pattern
    SparseMatrix z = a.subtract(a);
    CHECK(z.nnz() == a.nnz());
    z.eliminate_zeros();
    CHECK(z.nnz() == 0);
    std::vector<float> zeros(9, 0.0f);
    CHECK(z.to_dense() == zeros);
}
