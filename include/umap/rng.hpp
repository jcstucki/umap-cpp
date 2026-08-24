// Faithful C++ port of tau_rand_int / tau_rand from reference/umap/umap/utils.py
// (lmcinnes/umap, BSD 3 clause).
//
// The Python version operates on an int64 state array of shape (3,) under
// numba's i8 semantics (64-bit wrapping integer arithmetic, arithmetic right
// shift) and returns the xor of the three state words truncated to int32.
// Seeds may be negative (upstream draws them from randint(INT32_MIN, INT32_MAX)),
// so shifts are performed on the unsigned representation to get well-defined
// two's-complement wrapping, matching numba exactly.
#pragma once

#include <cmath>
#include <cstdint>

namespace umap {

namespace detail {
// Left shift with int64 wrap-around (numba i8 << semantics).
inline int64_t shl64(int64_t v, unsigned s) {
    return static_cast<int64_t>(static_cast<uint64_t>(v) << s);
}
// Arithmetic right shift (numba i8 >> semantics; sign-extending).
inline int64_t sar64(int64_t v, unsigned s) {
    return v >> s;  // arithmetic for signed operands on all supported targets
}
}  // namespace detail

// A fast (pseudo)-random number generator. `state` is the 3-word internal
// state, updated in place. Returns a (pseudo)-random int32 value.
inline int32_t tau_rand_int(int64_t state[3]) {
    using detail::sar64;
    using detail::shl64;
    state[0] = (shl64(state[0] & 4294967294LL, 12) & 0xFFFFFFFFLL) ^
               sar64((shl64(state[0], 13) & 0xFFFFFFFFLL) ^ state[0], 19);
    state[1] = (shl64(state[1] & 4294967288LL, 4) & 0xFFFFFFFFLL) ^
               sar64((shl64(state[1], 2) & 0xFFFFFFFFLL) ^ state[1], 25);
    state[2] = (shl64(state[2] & 4294967280LL, 17) & 0xFFFFFFFFLL) ^
               sar64((shl64(state[2], 3) & 0xFFFFFFFFLL) ^ state[2], 11);
    return static_cast<int32_t>(
        static_cast<uint32_t>(static_cast<uint64_t>(state[0] ^ state[1] ^ state[2])));
}

// A fast (pseudo)-random number generator for floats in the range [0, 1].
// Mirrors: abs(float(integer) / 0x7FFFFFFF) computed in double, cast to f4.
inline float tau_rand(int64_t state[3]) {
    int32_t integer = tau_rand_int(state);
    return static_cast<float>(
        std::fabs(static_cast<double>(integer) / 2147483647.0));
}

}  // namespace umap
