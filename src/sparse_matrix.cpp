#include "umap/sparse_matrix.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace umap {

SparseMatrix SparseMatrix::from_coo(int32_t n_rows, int32_t n_cols,
                                    const std::vector<int32_t>& rows,
                                    const std::vector<int32_t>& cols,
                                    const std::vector<float>& vals) {
    assert(rows.size() == cols.size() && rows.size() == vals.size());
    const std::size_t nnz_in = vals.size();

    // Sort triplet order by (row, col), stable so duplicates sum in input order.
    std::vector<std::size_t> order(nnz_in);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                         if (rows[a] != rows[b]) return rows[a] < rows[b];
                         return cols[a] < cols[b];
                     });

    SparseMatrix out(n_rows, n_cols);
    out.indices.reserve(nnz_in);
    out.data.reserve(nnz_in);

    std::vector<int32_t> row_counts(static_cast<std::size_t>(n_rows), 0);
    int32_t prev_r = -1;
    int32_t prev_c = -1;
    for (std::size_t k = 0; k < nnz_in; ++k) {
        std::size_t idx = order[k];
        int32_t r = rows[idx];
        int32_t c = cols[idx];
        assert(r >= 0 && r < n_rows && c >= 0 && c < n_cols);
        if (r == prev_r && c == prev_c) {
            out.data.back() += vals[idx];  // sum duplicates
        } else {
            out.indices.push_back(c);
            out.data.push_back(vals[idx]);
            row_counts[static_cast<std::size_t>(r)]++;
            prev_r = r;
            prev_c = c;
        }
    }
    for (int32_t r = 0; r < n_rows; ++r) {
        out.indptr[static_cast<std::size_t>(r) + 1] =
            out.indptr[static_cast<std::size_t>(r)] + row_counts[static_cast<std::size_t>(r)];
    }
    return out;
}

SparseMatrix SparseMatrix::transpose() const {
    SparseMatrix out(n_cols, n_rows);
    out.indices.resize(nnz());
    out.data.resize(nnz());

    // Count entries per column.
    std::vector<int32_t> counts(static_cast<std::size_t>(n_cols) + 1, 0);
    for (int32_t c : indices) counts[static_cast<std::size_t>(c) + 1]++;
    for (std::size_t i = 1; i < counts.size(); ++i) counts[i] += counts[i - 1];
    out.indptr.assign(counts.begin(), counts.end());

    std::vector<int32_t> next(counts.begin(), counts.end() - 1);
    for (int32_t r = 0; r < n_rows; ++r) {
        for (int32_t k = indptr[r]; k < indptr[static_cast<std::size_t>(r) + 1]; ++k) {
            int32_t c = indices[static_cast<std::size_t>(k)];
            int32_t dst = next[static_cast<std::size_t>(c)]++;
            out.indices[static_cast<std::size_t>(dst)] = r;
            out.data[static_cast<std::size_t>(dst)] = data[static_cast<std::size_t>(k)];
        }
    }
    // Rows of the output were filled in increasing source-row order, so column
    // indices are already sorted; result is canonical.
    return out;
}

template <typename Op>
SparseMatrix SparseMatrix::binary_op(const SparseMatrix& other, Op op,
                                     bool keep_union) const {
    assert(n_rows == other.n_rows && n_cols == other.n_cols);
    SparseMatrix out(n_rows, n_cols);
    out.indices.reserve(nnz() + other.nnz());
    out.data.reserve(nnz() + other.nnz());

    // scipy csr_binop_csr(_canonical) semantics: evaluate the op over the
    // union of the sparsity patterns (zero fill for the absent side) and
    // store an entry ONLY when the result is nonzero. Exact-zero results are
    // pruned even where both operands store the position (explicit zeros
    // summing to 0, equal values subtracting to 0, max(0, 0), 0-products) --
    // matters once operand matrices carry explicit zeros, e.g. a fit graph_
    // after the epoch filter's write-through.
    const auto push_nonzero = [&out](int32_t col, float v) {
        if (v != 0.0f) {
            out.indices.push_back(col);
            out.data.push_back(v);
        }
    };
    for (int32_t r = 0; r < n_rows; ++r) {
        int32_t ka = indptr[r];
        const int32_t ea = indptr[static_cast<std::size_t>(r) + 1];
        int32_t kb = other.indptr[r];
        const int32_t eb = other.indptr[static_cast<std::size_t>(r) + 1];
        while (ka < ea || kb < eb) {
            int32_t ca = ka < ea ? indices[static_cast<std::size_t>(ka)] : n_cols;
            int32_t cb = kb < eb ? other.indices[static_cast<std::size_t>(kb)] : n_cols;
            if (ca == cb) {
                push_nonzero(ca, op(data[static_cast<std::size_t>(ka)],
                                    other.data[static_cast<std::size_t>(kb)]));
                ++ka;
                ++kb;
            } else if (ca < cb) {
                if (keep_union) {
                    push_nonzero(ca,
                                 op(data[static_cast<std::size_t>(ka)], 0.0f));
                }
                ++ka;
            } else {
                if (keep_union) {
                    push_nonzero(
                        cb, op(0.0f, other.data[static_cast<std::size_t>(kb)]));
                }
                ++kb;
            }
        }
        out.indptr[static_cast<std::size_t>(r) + 1] =
            static_cast<int32_t>(out.indices.size());
    }
    return out;
}

SparseMatrix SparseMatrix::add(const SparseMatrix& other) const {
    return binary_op(other, [](float a, float b) { return a + b; }, true);
}

SparseMatrix SparseMatrix::subtract(const SparseMatrix& other) const {
    return binary_op(other, [](float a, float b) { return a - b; }, true);
}

SparseMatrix SparseMatrix::multiply(const SparseMatrix& other) const {
    return binary_op(other, [](float a, float b) { return a * b; }, false);
}

SparseMatrix SparseMatrix::maximum(const SparseMatrix& other) const {
    return binary_op(other, [](float a, float b) { return a > b ? a : b; }, true);
}

SparseMatrix SparseMatrix::scaled(float s) const {
    SparseMatrix out = *this;
    for (float& v : out.data) v *= s;
    return out;
}

void SparseMatrix::eliminate_zeros() {
    std::size_t out_k = 0;
    int32_t prev_end = 0;
    std::vector<int32_t> new_indptr(indptr.size(), 0);
    for (int32_t r = 0; r < n_rows; ++r) {
        for (int32_t k = prev_end; k < indptr[static_cast<std::size_t>(r) + 1]; ++k) {
            if (data[static_cast<std::size_t>(k)] != 0.0f) {
                indices[out_k] = indices[static_cast<std::size_t>(k)];
                data[out_k] = data[static_cast<std::size_t>(k)];
                ++out_k;
            }
        }
        prev_end = indptr[static_cast<std::size_t>(r) + 1];
        new_indptr[static_cast<std::size_t>(r) + 1] = static_cast<int32_t>(out_k);
    }
    indptr = std::move(new_indptr);
    indices.resize(out_k);
    data.resize(out_k);
}

std::vector<float> SparseMatrix::to_dense() const {
    std::vector<float> dense(static_cast<std::size_t>(n_rows) *
                                 static_cast<std::size_t>(n_cols),
                             0.0f);
    for (int32_t r = 0; r < n_rows; ++r) {
        for (int32_t k = indptr[r]; k < indptr[static_cast<std::size_t>(r) + 1]; ++k) {
            dense[static_cast<std::size_t>(r) * n_cols +
                  indices[static_cast<std::size_t>(k)]] +=
                data[static_cast<std::size_t>(k)];
        }
    }
    return dense;
}

}  // namespace umap
