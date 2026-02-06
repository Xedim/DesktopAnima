#include <gtest/gtest.h>
#include <cmath>
#include "../../math/functions/Functions.h"
#include "../../math/types/Types.h"

// ---------------- x_pow_y ----------------
struct PowTestCase {
    Real x;
    Real y;
    Real expected;
    bool expect_nan = false;
};

class Hybrid_XPowY : public ::testing::TestWithParam<PowTestCase> {};

TEST_P(Hybrid_XPowY, CoreValues) {
    const auto& tc = GetParam();
    Real result = Functions::x_pow_y(tc.x, tc.y);

    if (tc.expect_nan) {
        EXPECT_TRUE(std::isnan(result));
    } else {
        EXPECT_DOUBLE_EQ(result, tc.expected);
    }
}

INSTANTIATE_TEST_SUITE_P(
    XPowYTests,
    Hybrid_XPowY,
    ::testing::Values(
        PowTestCase{0.0, 2.0, 0.0},
        PowTestCase{0.0, -1.0, 0.0, true},
        PowTestCase{1.0, 5.0, 1.0},
        PowTestCase{2.0, 0.0, 1.0},
        PowTestCase{2.0, 1.0, 2.0},
        PowTestCase{2.0, 3.0, 8.0},
        PowTestCase{-2.0, 2.0, 0.0, true}  // negative base
    )
);

// ---------------- sqrt1pm1 ----------------
struct Sqrt1pm1TestCase {
    Real input;
    Real expected;
    bool expect_nan = false;
};

class Hybrid_Sqrt1pm1 : public ::testing::TestWithParam<Sqrt1pm1TestCase> {};

TEST_P(Hybrid_Sqrt1pm1, CoreValues) {
    const auto& tc = GetParam();
    Real result = Functions::sqrt1pm1(tc.input);

    if (tc.expect_nan) {
        EXPECT_TRUE(std::isnan(result));
    } else {
        EXPECT_NEAR(result, tc.expected, 1e-12);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Sqrt1pm1Tests,
    Hybrid_Sqrt1pm1,
    ::testing::Values(
        Sqrt1pm1TestCase{-2.0, 0.0, true},
        Sqrt1pm1TestCase{-0.5, std::sqrt(1-0.5)-1},
        Sqrt1pm1TestCase{0.0, 0.0},
        Sqrt1pm1TestCase{1e-9, 5e-10},
        Sqrt1pm1TestCase{1.0, std::sqrt(2.0)-1}
    )
);

// ---------------- Heaviside ----------------
struct HeavisideTestCase {
    Real input;
    Real expected;
};

class Hybrid_Heaviside : public ::testing::TestWithParam<HeavisideTestCase> {};

TEST_P(Hybrid_Heaviside, CoreValues) {
    const auto& tc = GetParam();
    EXPECT_DOUBLE_EQ(Functions::heaviside(tc.input), tc.expected);
}

INSTANTIATE_TEST_SUITE_P(
    HeavisideTests,
    Hybrid_Heaviside,
    ::testing::Values(
        HeavisideTestCase{-1.0, 0.0},
        HeavisideTestCase{0.0, 1.0},
        HeavisideTestCase{0.5, 1.0},
        HeavisideTestCase{10.0, 1.0}
    )
);
