// testAlgebra.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "../../math/functions/Functions.h"
#include "../../math/types/Types.h"
#include "../../math/helpers/Utils.h"

using Real = double;
using VecReal = std::vector<Real>;

// ----------------------------------------
// Invariant functions
// ----------------------------------------
inline bool is_non_negative(Real v) { return v >= 0; }
inline bool is_finite(Real v) { return std::isfinite(v); }

// ----------------------------------------
// Function descriptor
// ----------------------------------------
struct AlgebraicFn {
    const char* name;
    std::function<Real(int)> fn_int;
    std::function<Real(Real)> fn_real;
    std::vector<std::function<bool(Real)>> invariants;
};

// ----------------------------------------
// Test suite
// ----------------------------------------
class DynamicTest : public ::testing::TestWithParam<AlgebraicFn> {};

TEST_P(DynamicTest, InvariantsHold) {
    const auto& fdesc = GetParam();

    if (fdesc.fn_int) {
        for (int n = 0; n <= 15; ++n) {
            Real v = fdesc.fn_int(n);
            for (auto& inv : fdesc.invariants) {
                EXPECT_TRUE(inv(v)) << "Invariant failed for " << fdesc.name << ", n=" << n;
            }
        }
    }

    if (fdesc.fn_real) {
        std::uniform_real_distribution<Real> dist(-10.0, 10.0);

        for (int i = 0; i < 100; ++i) {
            Real x = dist(Utils::rng);
            Real v = fdesc.fn_real(x);

            if (std::string(fdesc.name) == "sqrt" && x < 0) {
                EXPECT_TRUE(std::isnan(v)) << "Expected NaN for sqrt(" << x << ")";
                continue;
            }

            for (auto& inv : fdesc.invariants) {
                EXPECT_TRUE(inv(v)) << "Invariant failed for " << fdesc.name << ", x=" << x;
            }
        }
    }
}

// ----------------------------------------
// Instantiate
// ----------------------------------------
INSTANTIATE_TEST_SUITE_P(
    AlgebraicFunctions,
    DynamicTest,
    ::testing::Values(
        AlgebraicFn{"factorial",
            [](int n){ return Functions::factorial(n); },
            nullptr,
            {is_non_negative}},
        AlgebraicFn{"binomial",
            [](int n){ return Functions::binomial(n, n/2); },
            nullptr,
            {is_non_negative}},
        AlgebraicFn{"permutation",
            [](int n){ return Functions::permutation(n, n/2); },
            nullptr,
            {is_non_negative}},
        AlgebraicFn{"sqrt", nullptr,
            [](Real x){ return Functions::sqrt(x); },
            {is_finite}},
        AlgebraicFn{"cbrt", nullptr,
            [](Real x){ return Functions::cbrt(x); },
            {is_finite}},
        AlgebraicFn{"sign", nullptr,
            [](Real x){ return Functions::sign(x); },
            {is_finite}},
        AlgebraicFn{"abs", nullptr,
            [](Real x){ return Functions::abs(x); },
            {is_non_negative, is_finite}}
    )
);
