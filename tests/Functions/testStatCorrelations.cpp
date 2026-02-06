#include "../../math/functions/Functions.h"
#include "../../math/types/Types.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// ----------------- Covariance -----------------
struct CovCase {
    VecReal x, y;
    Real expected;
};

class CovTests : public ::testing::TestWithParam<CovCase> {};

TEST_P(CovTests, Covariance) {
    const auto& tc = GetParam();
    Real r = Functions::covariance(tc.x, tc.y);
    if (std::isnan(tc.expected))
        EXPECT_TRUE(std::isnan(r));
    else
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
}

std::vector<CovCase> make_cov_cases() {
    return {
        {{}, {}, NaN()},
        {{1}, {1}, NaN()},
        {{1,2}, {1,2}, 0.5},
        {{1,2,3}, {3,2,1}, -1.0},
        {{1,2,3,4}, {4,3,2,1}, -1.6666666666666667} // (ручная проверка)
    };
}

// ----------------- Pearson -----------------
struct PearsonCase {
    VecReal x, y;
    Real expected;
};

class PearsonTests : public ::testing::TestWithParam<PearsonCase> {};

TEST_P(PearsonTests, PearsonCorrelation) {
    const auto& tc = GetParam();
    Real r = Functions::correlation_pearson(tc.x, tc.y);
    if (std::isnan(tc.expected))
        EXPECT_TRUE(std::isnan(r));
    else
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
}

std::vector<PearsonCase> make_pearson_cases() {
    return {
        {{}, {}, NaN()},
        {{1}, {1}, NaN()},
        {{1,2,3}, {1,2,3}, 1.0},
        {{1,2,3}, {3,2,1}, -1.0},
        {{1,2,3,4}, {4,3,2,1}, -1.0}
    };
}

// ----------------- Spearman -----------------
struct SpearmanCase {
    VecReal x, y;
    Real expected;
};

class SpearmanTests : public ::testing::TestWithParam<SpearmanCase> {};

TEST_P(SpearmanTests, SpearmanCorrelation) {
    const auto& tc = GetParam();
    Real r = Functions::correlation_spearman(tc.x, tc.y);
    if (std::isnan(tc.expected))
        EXPECT_TRUE(std::isnan(r));
    else
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
}

std::vector<SpearmanCase> make_spearman_cases() {
    return {
        {{1,2,3}, {3,2,1}, -1.0},
        {{1,2,3}, {1,2,3}, 1.0},
        {{1,2,3,4}, {4,3,2,1}, -1.0}
    };
}

// ----------------- Kendall -----------------
struct KendallCase {
    VecReal x, y;
    Real expected;
};

class KendallTests : public ::testing::TestWithParam<KendallCase> {};

TEST_P(KendallTests, KendallCorrelation) {
    const auto& tc = GetParam();
    Real r = Functions::correlation_kendall(tc.x, tc.y);
    if (std::isnan(tc.expected))
        EXPECT_TRUE(std::isnan(r));
    else
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
}

std::vector<KendallCase> make_kendall_cases() {
    return {
        {{1,2,3}, {3,2,1}, -1.0},
        {{1,2,3}, {1,2,3}, 1.0},
        {{1,2,3,4}, {4,3,2,1}, -1.0}
    };
}

// ----------------- Autocovariance -----------------
struct AutoCase {
    VecReal x;
    int lag;
    Real expected;
};

class AutoCovTests : public ::testing::TestWithParam<AutoCase> {};

TEST_P(AutoCovTests, Autocovariance) {
    const auto& tc = GetParam();
    Real r = Functions::autocovariance(tc.x, tc.lag);
    if (std::isnan(tc.expected))
        EXPECT_TRUE(std::isnan(r));
    else
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
}

std::vector<AutoCase> make_autocov_cases() {
    return {
            {{}, 0, NaN()},
            {{1}, 0, NaN()},
            {{1,2,3}, 0, 2.0/3.0},
            {{1,2,3}, 1, 0.0},
            {{1,2,3,4}, 2, -0.75}
    };
}

// ----------------- Cross-correlation -----------------
struct CrossCase {
    VecReal x, y;
    int lag;
    Real expected;
};

class CrossCorrTests : public ::testing::TestWithParam<CrossCase> {};

TEST_P(CrossCorrTests, CrossCorrelation) {
    const auto& tc = GetParam();
    Real r = Functions::cross_correlation(tc.x, tc.y, tc.lag);
    if (std::isnan(tc.expected))
        EXPECT_TRUE(std::isnan(r));
    else
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
}

std::vector<CrossCase> make_cross_cases() {
    return {
            {{1,2,3}, {3,2,1}, 0, -2.0/3.0}, // lag=0
            {{1,2,3}, {3,2,1}, 1, 0.0},      // lag=1
            {{1,2,3}, {1,2,3}, -1, 0.0}      // lag=-1
    };
}

// ----------------- INSTANTIATE -----------------
INSTANTIATE_TEST_SUITE_P(Stats, CovTests, ::testing::ValuesIn(make_cov_cases()));
INSTANTIATE_TEST_SUITE_P(Stats, PearsonTests, ::testing::ValuesIn(make_pearson_cases()));
INSTANTIATE_TEST_SUITE_P(Stats, SpearmanTests, ::testing::ValuesIn(make_spearman_cases()));
INSTANTIATE_TEST_SUITE_P(Stats, KendallTests, ::testing::ValuesIn(make_kendall_cases()));
INSTANTIATE_TEST_SUITE_P(Stats, AutoCovTests, ::testing::ValuesIn(make_autocov_cases()));
INSTANTIATE_TEST_SUITE_P(Stats, CrossCorrTests, ::testing::ValuesIn(make_cross_cases()));
