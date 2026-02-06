// testStatShape.cpp
#include <gtest/gtest.h>
#include "../../math/functions/Functions.h"
#include "../../math/types/Types.h"
#include <vector>
#include <cmath>

// ----------------- raw_moment -----------------
struct RawMomentCase {
    VecReal x;
    int k;
    Real expected;
};

class RawMomentTests : public ::testing::TestWithParam<RawMomentCase> {};

TEST_P(RawMomentTests, RawMoment) {
    const auto& tc = GetParam();
    Real r = Functions::raw_moment(tc.x, tc.k);

    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
    }
}

INSTANTIATE_TEST_SUITE_P(
    StatShape, RawMomentTests,
    ::testing::Values(
        RawMomentCase{{}, 2, std::numeric_limits<Real>::quiet_NaN()},
        RawMomentCase{{1,2,3}, -1, std::numeric_limits<Real>::quiet_NaN()},
        RawMomentCase{{1,2,3}, 0, 1.0},
        RawMomentCase{{1,2,3}, 2, 14.0/3.0},
        RawMomentCase{{-1,0,1}, 2, 2.0/3.0}
    )
);

// ----------------- moment -----------------
struct MomentCase {
    VecReal x;
    int k;
    Real expected;
};

class MomentTests : public ::testing::TestWithParam<MomentCase> {};

TEST_P(MomentTests, CentralMoment) {
    const auto& tc = GetParam();
    Real r = Functions::moment(tc.x, tc.k);

    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
    }
}

INSTANTIATE_TEST_SUITE_P(
    StatShape, MomentTests,
    ::testing::Values(
        MomentCase{{}, 2, std::numeric_limits<Real>::quiet_NaN()},
        MomentCase{{1,2,3}, -1, std::numeric_limits<Real>::quiet_NaN()},
        MomentCase{{1,2,3}, 0, 1.0},
        MomentCase{{1,2,3}, 1, 0.0},
        MomentCase{{1,2,3}, 2, 2.0/3.0}
    )
);

// ----------------- skewness -----------------
struct SkewnessCase {
    VecReal x;
    Real expected;
};

class SkewnessTests : public ::testing::TestWithParam<SkewnessCase> {};

TEST_P(SkewnessTests, Skewness) {
    const auto& tc = GetParam();
    Real r = Functions::skewness(tc.x);

    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_03);
    }
}

INSTANTIATE_TEST_SUITE_P(
    StatShape, SkewnessTests,
    ::testing::Values(
        SkewnessCase{{}, std::numeric_limits<Real>::quiet_NaN()},
        SkewnessCase{{1,1}, std::numeric_limits<Real>::quiet_NaN()},
        SkewnessCase{{1,2,3}, 0.0},
        SkewnessCase{{1,2,4}, 0.381}
    )
);

// ----------------- kurtosis -----------------
struct KurtosisCase {
    VecReal x;
    Real expected;
};

class KurtosisTests : public ::testing::TestWithParam<KurtosisCase> {};

TEST_P(KurtosisTests, Kurtosis) {
    const auto& tc = GetParam();
    Real r = Functions::kurtosis(tc.x);

    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
    }
}

INSTANTIATE_TEST_SUITE_P(
    StatShape, KurtosisTests,
    ::testing::Values(
        KurtosisCase{{}, std::numeric_limits<Real>::quiet_NaN()},
        KurtosisCase{{1,1,1}, std::numeric_limits<Real>::quiet_NaN()},
        KurtosisCase{{1,2,3,4}, -1.36}, // рассчитано вручную
        KurtosisCase{{1,2,3,4,5}, -1.3} // приближенно
    )
);
