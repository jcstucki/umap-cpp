// tau_prng vs golden sequences generated from the reference numba
// implementation (umap/utils.py tau_rand_int), including negative seeds and
// the degenerate all-zero fixed point.
#include <cmath>
#include <cstdint>

#include "doctest.h"
#include "golden_data.hpp"
#include "umap/rng.hpp"

TEST_CASE("tau_rand_int matches reference bit-for-bit") {
    struct Case {
        const int64_t* seed;
        const int32_t* expected;
    };
    const Case cases[] = {
        {golden::tau_seed_a, golden::tau_out_a},
        {golden::tau_seed_b, golden::tau_out_b},
        {golden::tau_seed_c, golden::tau_out_c},  // degenerate seed -> all zeros
    };
    for (const auto& c : cases) {
        int64_t state[3] = {c.seed[0], c.seed[1], c.seed[2]};
        for (int i = 0; i < 10; ++i) {
            CHECK(umap::tau_rand_int(state) == c.expected[i]);
        }
    }
}

TEST_CASE("tau_rand maps to [0, 1]") {
    int64_t state[3] = {golden::tau_seed_a[0], golden::tau_seed_a[1],
                        golden::tau_seed_a[2]};
    for (int i = 0; i < 100; ++i) {
        float v = umap::tau_rand(state);
        CHECK(v >= 0.0f);
        CHECK(v <= 1.0f);
    }
    // first value equals |int32 / 0x7FFFFFFF| of the first golden int
    int64_t state2[3] = {golden::tau_seed_a[0], golden::tau_seed_a[1],
                         golden::tau_seed_a[2]};
    float first = umap::tau_rand(state2);
    float expected = static_cast<float>(
        std::fabs(static_cast<double>(golden::tau_out_a[0]) / 2147483647.0));
    CHECK(first == expected);
}
