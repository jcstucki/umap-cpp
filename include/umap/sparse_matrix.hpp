// Minimal CSR sparse matrix with the operations needed by fuzzy_simplicial_set.
// Follows scipy defaults: float32 data, int32 indices/indptr, canonical format
// (per-row sorted column indices, duplicates summed).
#pragma once

#include <cstdint>
#include <vector>

namespace umap {

class SparseMatrix {
public:
    int32_t n_rows = 0;
    int32_t n_cols = 0;
    std::vector<int32_t> indptr;   // size n_rows + 1
    std::vector<int32_t> indices;  // column indices, sorted within each row
    std::vector<float> data;

    SparseMatrix() : indptr(1, 0) {}
    SparseMatrix(int32_t rows, int32_t cols)
        : n_rows(rows), n_cols(cols), indptr(static_cast<std::size_t>(rows) + 1, 0) {}

    // Build from COO triplets (like scipy.sparse.coo_matrix((vals,(rows,cols))).tocsr()).
    // Duplicate entries are summed; result is canonical.
    static SparseMatrix from_coo(int32_t n_rows, int32_t n_cols,
                                 const std::vector<int32_t>& rows,
                                 const std::vector<int32_t>& cols,
                                 const std::vector<float>& vals);

    std::size_t nnz() const { return data.size(); }

    SparseMatrix transpose() const;

    // Elementwise binary operations over the union of sparsity patterns,
    // with scipy csr_binop_csr semantics: exact-zero results are NOT stored
    // (scipy prunes them even where both operands store the position).
    SparseMatrix add(const SparseMatrix& other) const;
    SparseMatrix subtract(const SparseMatrix& other) const;
    // Elementwise (Hadamard) product, like scipy's A.multiply(B); the result
    // pattern is the intersection of the operand patterns minus exact-zero
    // products.
    SparseMatrix multiply(const SparseMatrix& other) const;
    // Elementwise maximum, like scipy's A.maximum(B) (zero maxima pruned).
    SparseMatrix maximum(const SparseMatrix& other) const;

    SparseMatrix scaled(float s) const;  // scalar * matrix

    void eliminate_zeros();  // drop explicitly-stored zeros

    // Dense row-major (n_rows * n_cols) copy, for tests.
    std::vector<float> to_dense() const;

private:
    // Merge rows of two canonical matrices with a combiner; keep_union controls
    // whether entries present in only one operand survive (with 0 for the other).
    template <typename Op>
    SparseMatrix binary_op(const SparseMatrix& other, Op op, bool keep_union) const;
};

}  // namespace umap
