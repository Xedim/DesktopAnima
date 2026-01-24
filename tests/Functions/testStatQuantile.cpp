#include "../math/pattern/Functions.h"
#include "../math/common/Types.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cmath>

// ----------------- quantile / percentile -----------------
struct QuantileCase {
    VecReal x;
    Real q;
    Real expected;
};

class QuantileTests : public ::testing::TestWithParam<QuantileCase> {};

TEST_P(QuantileTests, Quantile) {
    const auto& tc = GetParam();
    Real r = Functions::quantile(tc.x, tc.q);
    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
    }
}

// ----------------- quartiles / iqr -----------------
struct QuartileCase {
    VecReal x;
    Real q1, q2, q3, iqr_val;
};

class QuartileTests : public ::testing::TestWithParam<QuartileCase> {};

TEST_P(QuartileTests, QuartilesAndIQR) {
    const auto& tc = GetParam();
    auto q = Functions::quartiles(tc.x);
    if (std::isnan(tc.q1)) {
        EXPECT_TRUE(std::isnan(q.q1));
        EXPECT_TRUE(std::isnan(q.q2));
        EXPECT_TRUE(std::isnan(q.q3));
        EXPECT_TRUE(std::isnan(Functions::iqr(tc.x)));
    } else {
        EXPECT_NEAR(q.q1, tc.q1, Constants::EPS_09);
        EXPECT_NEAR(q.q2, tc.q2, Constants::EPS_09);
        EXPECT_NEAR(q.q3, tc.q3, Constants::EPS_09);
        EXPECT_NEAR(Functions::iqr(tc.x), tc.iqr_val, Constants::EPS_09);
    }
}

// ----------------- trimmed_mean -----------------
struct TrimmedMeanCase {
    VecReal x;
    Real alpha;
    Real expected;
};

class TrimmedMeanTests : public ::testing::TestWithParam<TrimmedMeanCase> {};

TEST_P(TrimmedMeanTests, TrimmedMean) {
    const auto& tc = GetParam();
    Real r = Functions::trimmed_mean(tc.x, tc.alpha);
    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
    }
}

// ----------------- генерация динамических кейсов -----------------
std::vector<QuantileCase> make_quantile_cases() {
    std::vector<QuantileCase> cases;
    cases.push_back({{}, 0.5, std::numeric_limits<Real>::quiet_NaN()});
    cases.push_back({{1}, 0.0, 1.0});
    cases.push_back({{1}, 1.0, 1.0});
    cases.push_back({{1,2,3,4}, 0.25, 1.75});
    cases.push_back({{1,2,3,4}, 0.5, 2.5});
    cases.push_back({{1,2,3,4}, 0.75, 3.25});
    return cases;
}

std::vector<QuartileCase> make_quartile_cases() {
    std::vector<QuartileCase> cases;
    cases.push_back({{}, NaN(), NaN(), NaN(), NaN()});
    cases.push_back({{1}, 1, 1, 1, 0});
    cases.push_back({{1,2,3,4}, 1.75, 2.5, 3.25, 1.5});
    return cases;
}


std::vector<TrimmedMeanCase> make_trimmed_cases() {
    std::vector<TrimmedMeanCase> cases;
    cases.push_back({{}, 0.1, NaN()});
    cases.push_back({{1,2,3,4,5}, 0.0, 3.0});
    cases.push_back({{1,2,3,4,5}, 0.2, 3.0}); // усечены 1 и 5
    return cases;
}

// ----------------- инстанцирование -----------------
INSTANTIATE_TEST_SUITE_P(StatShape, QuantileTests, ::testing::ValuesIn(make_quantile_cases()));
INSTANTIATE_TEST_SUITE_P(StatShape, QuartileTests, ::testing::ValuesIn(make_quartile_cases()));
INSTANTIATE_TEST_SUITE_P(StatShape, TrimmedMeanTests, ::testing::ValuesIn(make_trimmed_cases()));
