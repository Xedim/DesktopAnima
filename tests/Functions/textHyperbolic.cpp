#include <gtest/gtest.h>
#include <cmath>
#include "../../math/functions/Functions.h"
#include "../../math/types/Types.h"
#include <limits>

// ---------------- Valid core values ----------------
struct HyperbolicTestCase {
    Real input;
    Real expected_sinh;
    Real expected_cosh;
    Real expected_tanh;
    Real expected_asinh;
    Real expected_acosh;
    Real expected_atanh;
};

class CoreValues_HyperbolicDynamic : public ::testing::TestWithParam<HyperbolicTestCase> {};

TEST_P(CoreValues_HyperbolicDynamic, CoreValues) {
    const auto& tc = GetParam();

    // sinh / cosh / tanh
    EXPECT_DOUBLE_EQ(sinh(tc.input), tc.expected_sinh);
    EXPECT_DOUBLE_EQ(cosh(tc.input), tc.expected_cosh);
    EXPECT_DOUBLE_EQ(tanh(tc.input), tc.expected_tanh);

    // inverse
    EXPECT_DOUBLE_EQ(asinh(tc.input), tc.expected_asinh);

    if (tc.input >= 1.0) {
        EXPECT_DOUBLE_EQ(acosh(tc.input), tc.expected_acosh);
    }

    if (tc.input > -1.0 && tc.input < 1.0) {
        EXPECT_DOUBLE_EQ(atanh(tc.input), tc.expected_atanh);
    }
}

// ---------------- Invalid / extreme values ----------------
class HyperbolicInvalid : public ::testing::TestWithParam<Real> {};

TEST_P(HyperbolicInvalid, NaNValues) {
    Real x = GetParam();

    // sinh / cosh overflow threshold
    if (!std::isfinite(x) || std::abs(x) > 700) {
        EXPECT_TRUE(std::isnan(sinh(x)) || std::isinf(sinh(x)));
        EXPECT_TRUE(std::isnan(cosh(x)) || std::isinf(cosh(x)));
    }

    // tanh large numbers saturate
    if (std::abs(x) > 20.0) {
        EXPECT_DOUBLE_EQ(tanh(x), (x > 0 ? 1.0 : -1.0));
    }

    // acosh invalid
    if (x < 1.0 || !std::isfinite(x)) {
        EXPECT_TRUE(std::isnan(acosh(x)));
    }

    // atanh invalid
    if (x <= -1.0 || x >= 1.0) {
        EXPECT_TRUE(std::isnan(atanh(x)));
    }
}

// ---------------- Test data ----------------
INSTANTIATE_TEST_SUITE_P(
    CoreValues,
    CoreValues_HyperbolicDynamic,
    ::testing::Values(
        HyperbolicTestCase{0.0,
             0.0,
             1.0,
             0.0,
           0.0,
           0.0,
           0.0},
        HyperbolicTestCase{1.0, std::sinh(1.0), std::cosh(1.0), std::tanh(1.0),
                           std::asinh(1.0), std::acosh(1.0), std::atanh(0.5)}
    )
);

INSTANTIATE_TEST_SUITE_P(
    InvalidValues,
    HyperbolicInvalid,
    ::testing::Values(
        1e308, -1.5, 1.5, -1.5
    )
);
