// testExpLog.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <random>
#include <limits>
#include "../../math/pattern/Functions.h"

using Real = double;

// ---------------- Random generator -----------------
std::vector<Real> generate_random(Real min, Real max, int n=50) {
    std::vector<Real> vals(n);
    std::uniform_real_distribution<Real> dist(min, max);
    for (auto &v : vals) v = dist(rng);
    return vals;
}

// ---------------- Edge values -----------------
std::vector<Real> edge_values() {
    return {
        0.0, -0.0,
        1.0, -1.0,
        std::numeric_limits<Real>::infinity(),
        -std::numeric_limits<Real>::infinity(),
        std::numeric_limits<Real>::quiet_NaN()
    };
}

// ---------------- Unified function descriptor -----------------
struct FunctionTest {
    std::string name;
    int nargs; // 1 или 2

    std::function<Real(Real)> f1 = nullptr;
    std::function<Real(Real, Real)> f2 = nullptr;

    std::function<bool(Real, Real)> invalid_case = nullptr; // x или x,y недопустимы
    std::function<Real(Real, Real)> reference = nullptr;    // эталон

    std::vector<std::function<bool(Real)>> invariants_1;
    std::vector<std::function<bool(Real)>> invariants_2;
};

// ---------------- Test suite -----------------
class UnifiedDynamicTest : public ::testing::TestWithParam<FunctionTest> {};

TEST_P(UnifiedDynamicTest, RandomAndEdgeCheck) {
    const auto &f = GetParam();

    std::vector<Real> x_vals = generate_random(-10.0, 10.0);
    std::vector<Real> y_vals = generate_random(-10.0, 10.0);

    auto edges = edge_values();
    x_vals.insert(x_vals.end(), edges.begin(), edges.end());
    y_vals.insert(y_vals.end(), edges.begin(), edges.end());

    if (f.nargs == 1) {
        for (auto x : x_vals) {
            SCOPED_TRACE(f.name + " x=" + std::to_string(x));
            Real v = f.f1(x);

            // invalid case
            if (f.invalid_case && f.invalid_case(x, 0.0)) {
                EXPECT_TRUE(std::isnan(v));
                continue;
            }

            // invariants
            for (auto &inv : f.invariants_1) {
                EXPECT_TRUE(inv(v)) << "Invariant failed for " << f.name << ", x=" << x;
            }

            // reference
            if (f.reference) {
                Real ref = f.reference(x, 0.0);
                if (std::isnan(ref)) EXPECT_TRUE(std::isnan(v));
                else EXPECT_NEAR(v, ref, 1e-12);
            }
        }
    } else if (f.nargs == 2) {
        for (auto x : x_vals) {
            for (auto y : y_vals) {
                SCOPED_TRACE(f.name + " x=" + std::to_string(x) + " y=" + std::to_string(y));
                Real v = f.f2(x, y);

                if (f.invalid_case && f.invalid_case(x, y)) {
                    EXPECT_TRUE(std::isnan(v));
                    continue;
                }

                for (auto &inv : f.invariants_2) {
                    EXPECT_TRUE(inv(v)) << "Invariant failed for " << f.name << ", x=" << x << ", y=" << y;
                }

                if (f.reference) {
                    Real ref = f.reference(x, y);
                    if (std::isnan(ref)) EXPECT_TRUE(std::isnan(v));
                    else EXPECT_NEAR(v, ref, 1e-12);
                }
            }
        }
    }
}

// ---------------- Instantiate -----------------
INSTANTIATE_TEST_SUITE_P(
    MathFunctions,
    UnifiedDynamicTest,
    ::testing::Values(
        FunctionTest{
            "sqrt", 1,
            Functions::sqrt, nullptr,
            [](Real x, Real){ return x < 0; },
            [](Real x, Real){ return std::sqrt(x); },
            {[](Real v){ return std::isfinite(v); }, [](Real v){ return v >= 0; }},
            {}
        },
        FunctionTest{
            "log", 1,
            Functions::log, nullptr,
            [](Real x, Real){ return x <= 0; },
            [](Real x, Real){ return std::log(x); },
            {[](Real v){ return std::isfinite(v); }},
            {}
        },
        FunctionTest{
            "pow", 2,
            nullptr, Functions::pow,
            [](Real x, Real y){ return x < 0 && std::floor(y)!=y; },
            [](Real x, Real y){ return std::pow(x, y); },
            {},
            {[](Real v){ return std::isfinite(v); }}
        },
        FunctionTest{
            "log_a", 2,
            nullptr, Functions::log_a,
            [](Real x, Real a){ return x <= 0.0 || a <= 0.0 || a == 1.0; },
            [](Real x, Real a){ return std::log(x)/std::log(a); },
            {},
            {[](Real v){ return std::isfinite(v); }}
        }
    )
);
