// uniTest.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <random>
#include "../../math/pattern/Functions.h"
#include "../../math/analytics/Analytics.h"

// ---------------- Helpers -----------------
std::vector<Real> generate_random(Real min, Real max, int n=50) {
    std::vector<Real> vals(n);
    std::uniform_real_distribution<Real> dist(min, max);
    for (auto &v : vals) v = dist(rng);
    return vals;
}
std::vector<Real> all_test_values(const std::vector<Real>& fixed) {
    auto random_vals = generate_random(-10.0, 10.0, 50);
    std::vector<Real> combined = fixed;
    combined.insert(combined.end(), random_vals.begin(), random_vals.end());
    return combined;
}

// ----------------- Invariants -----------------
inline bool is_non_negative(Real v) { return v >= 0; }
inline bool is_finite(Real v) { return std::isfinite(v); }

// ----------------- Unified function descriptor -----------------
struct UnifiedFn {
    std::string name;
    int nargs;
    std::function<Real(Real)> f1 = nullptr;
    std::function<Real(Real, Real)> f2 = nullptr;
    std::function<bool(Real, Real)> invalid_case = nullptr;
    std::function<Real(Real, Real)> reference = nullptr;

    std::vector<std::function<bool(Real)>> invariants_1;
    std::vector<std::function<bool(Real)>> invariants_2;

    std::vector<std::function<bool(const Function1D&)>> properties_1;
    std::vector<std::function<bool(const Function2D&)>> properties_2;

    std::vector<Real> fixed_values_1;
    std::vector<Real> fixed_values_2;
};

// ----------------- Test suite -----------------
class UnifiedFunctionTest : public ::testing::TestWithParam<UnifiedFn> {};

TEST_P(UnifiedFunctionTest, RandomAndEdgeCheckWithProperties) {
    const auto &f = GetParam();

    if (f.nargs == 1 && f.f1) {
        for (auto x : all_test_values(f.fixed_values_1)) {
            SCOPED_TRACE(f.name + " x=" + std::to_string(x));
            Real v = f.f1(x);

            if (f.invalid_case && f.invalid_case(x, 0.0)) {
                EXPECT_TRUE(std::isnan(v));
                continue;
            }

            for (auto &inv : f.invariants_1)
                EXPECT_TRUE(inv(v)) << "Invariant failed for " << f.name << ", x=" << x;

            if (f.reference)
                EXPECT_NEAR(v, f.reference(x, 0.0), Constants::EPS_12);

            // Проверка свойств через Analytics
            for (auto &prop : f.properties_1)
                EXPECT_TRUE(prop(f.f1)) << "Property failed for " << f.name;
        }
    }

    if (f.nargs == 2 && f.f2) {
        for (auto x : all_test_values(f.fixed_values_1)) {
            for (auto y : all_test_values(f.fixed_values_2)) {
                SCOPED_TRACE(f.name + " x=" + std::to_string(x) + " y=" + std::to_string(y));
                Real v = f.f2(x,y);

                if (f.invalid_case && f.invalid_case(x, y)) {
                    EXPECT_TRUE(std::isnan(v));
                    continue;
                }

                for (auto &inv : f.invariants_2)
                    EXPECT_TRUE(inv(v)) << "Invariant failed for " << f.name;

                if (f.reference)
                    EXPECT_NEAR(v, f.reference(x, y), Constants::EPS_12);

                for (auto &prop : f.properties_2)
                    EXPECT_TRUE(prop(f.f2)) << "Property failed for " << f.name;
            }
        }
    }
}

// ----------------- Instantiate -----------------
INSTANTIATE_TEST_SUITE_P(
    AllMathFunctions,
    UnifiedFunctionTest,
    ::testing::Values(
        // ---------------- Integer functions ----------------
        UnifiedFn{
            "factorial", 1,
            [](Real x){ return Functions::factorial(int(x)); }, {}, nullptr,
            [](Real x, Real){ return Functions::factorial(int(x)); },
            {is_non_negative}, {},
            {
                [](const Function1D& f){
                    return Analytics::isLocallyIncreasing(f, 0, 5, 1e-12, StabPolicy::Reject);
                }
            },
            {},
            {0,1,2,5,10}, {}
        },
        UnifiedFn{
            "binomial", 1,
            [](Real x){ return Functions::binomial(int(x), int(x/2)); }, {}, nullptr,
            [](Real x, Real){ return Functions::binomial(int(x), int(x/2)); },
            {is_non_negative}, {},
            {},
            {},
            {0,1,5,10}, {}
        },
        UnifiedFn{
            "permutation", 1,
            [](Real x){ return Functions::permutation(int(x), int(x/2)); }, {}, nullptr,
            [](Real x, Real){ return Functions::permutation(int(x), int(x/2)); },
            {is_non_negative}, {},
            {},
            {},
            {0,1,5,10}, {}
        },

        // ---------------- Real functions, 1 arg ----------------
        UnifiedFn{
            "sqrt", 1,
            Functions::sqrt, {}, [](Real x, Real){ return x < 0; },
            [](Real x, Real){ return std::sqrt(x); },
            {is_finite}, {},
            {
                [](const Function1D& f){
                    return Analytics::isLocallyConvex(f, 1.0, 1e-6, StabPolicy::Reject);
                },
                [](const Function1D& f){
                    return Analytics::isNonNegative(f, 1.0, StabPolicy::Reject);
                }
            },
            {},
            {0,1,4,9,16}, {}
        },
        UnifiedFn{
            "cbrt", 1,
            Functions::cbrt, {}, nullptr,
            [](Real x, Real){ return std::cbrt(x); },
            {is_finite}, {},
            {},
            {},
            {-10,-1,0,1,10}, {}
        },
        UnifiedFn{
            "abs", 1,
            Functions::abs, {}, nullptr,
            [](Real x, Real){ return std::abs(x); },
            {is_non_negative, is_finite}, {},
            {},
            {},
            {-10,-1,0,1,10}, {}
        },
        UnifiedFn{
            "sign", 1,
            Functions::sign, {}, nullptr,
            [](Real x, Real){ return (x>0) - (x<0); },
            {is_finite}, {},
            {},
            {},
            {-10,-1,0,1,10}, {}
        },

        // ---------------- Real functions, 2 args ----------------
        UnifiedFn{
            "pow", 2,
            {}, Functions::pow,
            [](Real x, Real y){ return x < 0 && std::floor(y) != y; },
            [](Real x, Real y){ return std::pow(x, y); },
            {}, {},
            {},
            {},
            {-2,0,2}, {-1,0,0.5,1,2}
        },
        UnifiedFn{
            "log_a", 2,
            {}, Functions::log_a,
            [](Real x, Real a){ return x <= 0.0 || a <= 0.0 || a == 1.0; },
            [](Real x, Real a){ return std::log(x)/std::log(a); },
            {}, {},
            {},
            {},
            {0.1,1,2,10}, {0.1,0.5,1,2,10}
        },
        UnifiedFn{
            "exp", 1,
            Functions::exp, {}, nullptr,
            [](Real x, Real){ return std::exp(x); },
            {is_finite}, {},
            {
                [](const Function1D& f){ return Analytics::isLocallyIncreasing(f, -1, 1, Constants::EPS_09, StabPolicy::Reject); },
                [](const Function1D& f){ return Analytics::isNonNegative(f, 0.0, StabPolicy::Reject); }
            },
            {},
            {-700, -1,0,1,700}, {}
        },
        UnifiedFn{
            "exp2", 1,
            Functions::exp2, {}, nullptr,
            [](Real x, Real){ return std::exp2(x); },
            {is_finite}, {},
            {
                [](const Function1D& f){ return Analytics::isLocallyIncreasing(f, -1, 1, Constants::EPS_09, StabPolicy::Reject); },
                [](const Function1D& f){ return Analytics::isNonNegative(f, 0.0, StabPolicy::Reject); }
            },
            {},
            {-700, -1,0,1,700}, {}
        },
        UnifiedFn{
            "expm1_safe", 1,
            Functions::expm1_safe, {}, nullptr,
            [](Real x, Real){ return std::expm1(x); },
            {is_finite}, {},
            {
                [](const Function1D& f){ return Analytics::isLocallyIncreasing(f, -1e-6, 1e-6, Constants::EPS_09, StabPolicy::Reject); }
            },
            {},
            {-1e-6,0,1e-6}, {}
        },
        UnifiedFn{
            "log", 1,
            Functions::log, {}, [](Real x, Real){ return x <= 0; },
            [](Real x, Real){ return std::log(x); },
            {is_finite}, {},
            {},
            {},
            {1e-12,1,10,100}, {}
        },
        UnifiedFn{
            "log2", 1,
            Functions::log2, {}, [](Real x, Real){ return x <= 0; },
            [](Real x, Real){ return std::log2(x); },
            {is_finite}, {},
            {},
            {},
            {1e-12,1,10,100}, {}
        },
        UnifiedFn{
            "log10", 1,
            Functions::log10, {}, [](Real x, Real){ return x <= 0; },
            [](Real x, Real){ return std::log10(x); },
            {is_finite}, {},
            {},
            {},
            {1e-12,1,10,100}, {}
        },
        UnifiedFn{
            "log1p", 1,
            Functions::log1p, {}, [](Real x, Real){ return x <= -1; },
            [](Real x, Real){ return std::log1p(x); },
            {is_finite}, {},
            {},
            {},
            {-0.9999, -0.5,0,1e-6,1}, {}
        }
    )
);
