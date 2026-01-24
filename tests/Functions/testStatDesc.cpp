#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <functional>
#include "../../math/pattern/Functions.h"
#include "../../math/common/Types.h"

// ---------------- Dynamic Test ----------------

struct StatTestCase {
    VecReal data;
    Real expected;
    std::function<Real(const VecReal&)> func;
    std::string name;
};

class StatDescTests : public ::testing::TestWithParam<StatTestCase> {};

TEST_P(StatDescTests, RunDynamicTests) {
    const auto& tc = GetParam();
    Real res = tc.func(tc.data);
    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(res)) << "Function: " << tc.name;
    } else {
        EXPECT_NEAR(res, tc.expected, Constants::EPS_09) << "Function: " << tc.name;
    }
}

// ---------------- Instantiate ----------------

INSTANTIATE_TEST_SUITE_P(
    StatDesc,
    StatDescTests,
    ::testing::Values(
        // Sum
        StatTestCase{{}, NAN, Functions::sum, "sum_empty"},
        StatTestCase{{1,2,3,4}, 10, Functions::sum, "sum_1_2_3_4"},

        // Mean
        StatTestCase{{}, NAN, Functions::mean, "mean_empty"},
        StatTestCase{{1,2,3,4}, 2.5, Functions::mean, "mean_1_2_3_4"},

        // Median
        StatTestCase{{}, NAN, Functions::median, "median_empty"},
        StatTestCase{{1,3,2}, 2, Functions::median, "median_1_3_2"},
        StatTestCase{{1,2,3,4}, 2.5, Functions::median, "median_even"},

        // Mode
        StatTestCase{{}, NAN, Functions::mode, "mode_empty"},
        StatTestCase{{1,1,2,2,2,3}, 2, Functions::mode, "mode_1_1_2_2_2_3"},

        // Min / Max / Range
        StatTestCase{{}, NAN, Functions::min, "min_empty"},
        StatTestCase{{1,2,3}, 1, Functions::min, "min_1_2_3"},
        StatTestCase{{}, NAN, Functions::max, "max_empty"},
        StatTestCase{{1,2,3}, 3, Functions::max, "max_1_2_3"},
        StatTestCase{{1,2,3}, 2, Functions::range, "range_1_2_3"},

        // Variance / Stddev
        StatTestCase{{}, NAN, Functions::variance, "variance_empty"},
        StatTestCase{{1,1}, 0, Functions::variance, "variance_1_1"},
        StatTestCase{{1,2,3}, 2.0/3.0, Functions::variance, "variance_1_2_3"},
        StatTestCase{{}, NAN, Functions::stddev, "stddev_empty"},
        StatTestCase{{1,2,3}, std::sqrt(2.0/3.0), Functions::stddev, "stddev_1_2_3"},

        // Mean Absolute Deviation
        StatTestCase{{}, NAN, Functions::mean_absolute_deviation, "mad_empty"},
        StatTestCase{{1,2,3}, 2.0/3.0, Functions::mean_absolute_deviation, "mad_1_2_3"}
    )
);
