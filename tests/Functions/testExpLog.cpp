// tests/Functions/testExpLogUnified.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <random>
#include "../../math/pattern/Functions.h"
#include "../math/common/Utils.h"

// ---------------- Random generator -----------------
std::vector<Real> generate_random(Real min, Real max, int n=100) {
    std::vector<Real> vals(n);
    std::uniform_real_distribution<Real> dist(min, max);
    for (auto &v : vals) v = dist(Utils::rng);
    return vals;
}

// ----------------- Unified function descriptor -----------------
struct FunctionTest {
    std::string name;
    int nargs;
    std::function<Real(Real)> f1;
    std::function<Real(Real, Real)> f2;
    std::function<bool(Real, Real)> invalid_case;
    std::function<Real(Real, Real)> reference;
};

// ----------------- Test suite -----------------
class ExpLogDynamicTest : public ::testing::TestWithParam<FunctionTest> {};

TEST_P(ExpLogDynamicTest, RandomizedCheck) {
    const auto &f = GetParam();

    if (f.nargs == 1) {
        for (auto x : generate_random(-10.0, 10.0)) {
            SCOPED_TRACE(f.name + " x=" + std::to_string(x));
            if (f.invalid_case && f.invalid_case(x, 0.0)) {
                EXPECT_TRUE(std::isnan(f.f1(x)));
            } else {
                EXPECT_NEAR(f.f1(x), f.reference(x, 0.0), Constants::EPS_12);
            }
        }
    } else if (f.nargs == 2) {
        for (auto x : generate_random(-10.0, 10.0)) {
            for (auto y : generate_random(-10.0, 10.0)) {
                SCOPED_TRACE(f.name + " x=" + std::to_string(x) + " y=" + std::to_string(y));
                if (f.invalid_case && f.invalid_case(x, y)) {
                    EXPECT_TRUE(std::isnan(f.f2(x,y)));
                } else {
                    EXPECT_NEAR(f.f2(x,y), f.reference(x,y), Constants::EPS_12);
                }
            }
        }
    }
}

// ----------------- Instantiate tests -----------------
INSTANTIATE_TEST_SUITE_P(
    ExpLogFunctions,
    ExpLogDynamicTest,
    ::testing::Values(
        FunctionTest{
            "exp", 1,
            Functions::exp, {},
            nullptr,
            [](Real x, Real){ return std::exp(x); }
        },
        FunctionTest{
            "exp2", 1,
            Functions::exp2, {},
            nullptr,
            [](Real x, Real){ return std::exp2(x); }
        },
        FunctionTest{
            "expm1_safe", 1,
            Functions::expm1_safe, {},
            nullptr,
            [](Real x, Real){ return std::expm1(x); }
        },
        FunctionTest{
            "log", 1,
            Functions::log, {},
            [](Real x, Real){ return x <= 0.0; },
            [](Real x, Real){ return std::log(x); }
        },
        FunctionTest{
            "log2", 1,
            Functions::log2, {},
            [](Real x, Real){ return x <= 0.0; },
            [](Real x, Real){ return std::log2(x); }
        },
        FunctionTest{
            "log10", 1,
            Functions::log10, {},
            [](Real x, Real){ return x <= 0.0; },
            [](Real x, Real){ return std::log10(x); }
        },
        FunctionTest{
            "log1p", 1,
            Functions::log1p, {},
            [](Real x, Real){ return x <= -1.0; },
            [](Real x, Real){ return std::log1p(x); }
        },
        FunctionTest{
            "pow", 2,
            {}, Functions::pow,
            [](Real x, Real y){ return x < 0.0 && std::floor(y) != y; },
            [](Real x, Real y){ return std::pow(x, y); }
        },
        FunctionTest{
            "log_a", 2,
            {}, Functions::log_a,
            [](Real x, Real a){ return x <= 0.0 || a <= 0.0 || a == 1.0; },
            [](Real x, Real a){ return std::log(x)/std::log(a); }
        }
    )
);
