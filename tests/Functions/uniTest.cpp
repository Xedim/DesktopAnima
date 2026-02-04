/*// uniTest.cpp
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
struct FunctionIdentityDescriptor {
    bool invertible;
    bool has_closed_form;
};

struct DomainDescriptor {
    bool defined_on_all_reals;
    bool defined_on_positive;
    bool defined_on_negative;
    bool defined_on_zero;

    bool has_discontinuities;
    std::vector<double> excluded_points;
    std::vector<std::pair<double, double>> allowed_intervals;

    bool allows_nan_input;
    bool allows_inf_input;
};

struct RangeDescriptor {
    bool finite_only;
    bool can_return_nan;
    bool can_return_inf;

    std::optional<double> min_value;
    std::optional<double> max_value;
};

struct ValueCorrectnessDescriptor {
    bool nan_is_valid_on_domain_violation;
    bool inf_is_valid_on_domain_violation;

    double max_relative_error;
    double max_absolute_error;
};

struct AnalyticProperties {
    bool even;
    bool odd;

    bool monotonic_increasing;
    bool monotonic_decreasing;

    bool continuous;
    bool periodic;
    std::optional<double> period;
};


struct DifferentialDescriptor {
    bool differentiable;
    bool twice_differentiable;

    bool derivative_continuous;
    bool second_derivative_exists;

    bool convex;
    bool concave;

    bool has_inflection_points;
    std::vector<Real> known_inflection_points;
};

struct NumericalProperties {
    bool stable_near_zero;
    bool stable_for_large_values;

    bool catastrophic_cancellation_possible;
    bool sensitive_to_rounding;

    bool requires_high_precision;
};

struct BehavioralProperties {
    bool throws_on_domain_error;
    bool returns_nan_on_domain_error;

    bool deterministic;
    bool pure_function;

    bool thread_safe;
};

struct LimitBehaviorDescriptor {
    bool limit_at_plus_inf_exists;
    bool limit_at_minus_inf_exists;

    std::optional<Real> limit_at_plus_inf;
    std::optional<Real> limit_at_minus_inf;

    bool limit_at_zero_exists;
    std::optional<Real> limit_at_zero;

    bool left_limit_equals_right_limit;
};

struct SingularityDescriptor {
    enum class Type {
        Removable,
        Jump,
        Essential,
        Pole
    };

    struct Singularity {
        Real x;
        Type type;
    };

    bool has_singularities;
    std::vector<Singularity> singularities;
};

struct CompositionDescriptor {
    bool closed_under_composition;

    bool preserves_monotonicity;
    bool preserves_evenness;
    bool preserves_sign;

    bool requires_domain_intersection_check;
};

struct SymmetryDescriptor {
    bool symmetric_about_y_axis;
    bool symmetric_about_origin;
    bool invariant_under_scaling;
    bool invariant_under_translation;
};

struct PerformanceDescriptor {
    enum class Complexity {
        O1,
        OLogN,
        OLinear,
        OPolynomial,
        OExponential
    };

    Complexity time_complexity;
    Complexity space_complexity;

    bool vectorizable;
    bool branch_heavy;
};

struct TestingStrategyDescriptor {
    bool requires_edge_case_testing;
    bool requires_randomized_testing;
    bool requires_property_based_testing;

    size_t recommended_random_samples;
};

struct FunctionDescriptor {
    std::string name;
    std::function<Real(Real)> f;

    FunctionIdentityDescriptor identity;
    DomainDescriptor domain;
    RangeDescriptor range;

    ValueCorrectnessDescriptor correctness;

    AnalyticProperties analytic;
    DifferentialDescriptor differential;
    LimitBehaviorDescriptor limits;
    SingularityDescriptor singularities;
    SymmetryDescriptor symmetry;

    NumericalProperties numerical;
    BehavioralProperties behavior;
    PerformanceDescriptor performance;
    CompositionDescriptor composition;

    TestingStrategyDescriptor testing;
};

void validateFunctionDescriptor(const FunctionDescriptor& fd) {
    using std::cout;
    using std::endl;

    cout << "=== Validating function: " << fd.name << " ===" << endl;

    // -------- Hard Constraints --------
    // Domain - Singularity
    if (fd.singularities.has_singularities) {
        for (const auto& s : fd.singularities.singularities) {
            bool in_domain = false;
            for (const auto& interval : fd.domain.allowed_intervals) {
                if (s.x >= interval.first && s.x <= interval.second) {
                    in_domain = true;
                    break;
                }
            }
            if (!in_domain) {
                cout << "[HC] Warning: Singularity at x=" << s.x
                     << " outside allowed domain intervals." << endl;
            }
        }
    }

    // Analytic - Symmetry
    if (fd.analytic.even && !fd.symmetry.symmetric_about_y_axis)
        cout << "[HC] Warning: Analytic property even=true but symmetry not set." << endl;
    if (fd.analytic.odd && !fd.symmetry.symmetric_about_origin)
        cout << "[HC] Warning: Analytic property odd=true but symmetry not set." << endl;

    // Differential - Analytic
    if (fd.differential.derivative_continuous && !fd.analytic.continuous)
        cout << "[HC] Warning: derivative_continuous=true but function not marked continuous." << endl;

    // Singularity - Differential
    if (fd.singularities.has_singularities) {
        for (const auto& s : fd.singularities.singularities) {
            if (fd.differential.differentiable && (s.type != SingularityDescriptor::Type::Removable))
                cout << "[HC] Warning: differentiable=true but singularity at x=" << s.x << endl;
        }
    }

    // -------- Soft Constraints --------
    // Analytic - Differential
    if (fd.differential.twice_differentiable && !fd.differential.derivative_continuous)
        cout << "[SC] Twice differentiable functions normally have continuous derivative." << endl;

    // Numerical - ValueCorrectness
    if (fd.numerical.requires_high_precision &&
        (fd.correctness.max_absolute_error > 1e-12 || fd.correctness.max_relative_error > 1e-12))
        cout << "[SC] High precision numerical property but correctness allows larger errors." << endl;

    // Symmetry - Testing
    if ((fd.symmetry.symmetric_about_y_axis || fd.symmetry.symmetric_about_origin) &&
        !fd.testing.requires_property_based_testing)
        cout << "[SC] Symmetric function; consider property-based testing." << endl;

    // CompositionDescriptor
    if (fd.composition.closed_under_composition && !fd.domain.defined_on_all_reals)
        cout << "[SC] Function claims closure under composition but domain is not all reals." << endl;

    if (fd.composition.preserves_monotonicity && !fd.analytic.monotonic_increasing && !fd.analytic.monotonic_decreasing)
        cout << "[SC] Monotonicity preservation flagged but function not monotonic." << endl;

    // Limits
    if (fd.limits.limit_at_plus_inf_exists && !fd.range.can_return_inf)
        cout << "[SC] Limit at +∞ exists but range allows infinite values?" << endl;

    if (fd.limits.limit_at_zero_exists && !fd.domain.defined_on_zero)
        cout << "[SC] Limit at 0 exists but 0 is not in domain." << endl;

    // Numerical Properties - ValueCorrectness
    if ((fd.numerical.catastrophic_cancellation_possible || fd.numerical.sensitive_to_rounding)
        && fd.correctness.max_relative_error > 1e-12)
        cout << "[SC] Potential numerical instability; consider stricter correctness limits." << endl;

    // Performance - Domain/Range
    if (fd.performance.vectorizable && fd.domain.has_discontinuities)
        cout << "[SC] Vectorization flagged but function has discontinuities." << endl;

    if (fd.performance.branch_heavy && fd.analytic.continuous)
        cout << "[SC] Branch-heavy function but marked continuous (check assumptions)." << endl;

    cout << "=== Validation finished ===" << endl;
}

void propagateFunctionDependencies(FunctionDescriptor& fd) {
    if (fd.analytic.even) fd.symmetry.symmetric_about_y_axis = true;
    if (fd.analytic.odd) fd.symmetry.symmetric_about_origin = true;

    if (fd.differential.derivative_continuous) fd.analytic.continuous = true;
    if (fd.differential.twice_differentiable) fd.differential.derivative_continuous = true;

    if (fd.limits.limit_at_plus_inf_exists && fd.limits.limit_at_plus_inf.has_value())
        fd.range.can_return_inf = false;
}

std::vector<Real> generateRandomPoints(const DomainDescriptor& domain, size_t n=50) {
    std::vector<Real> points;
    std::uniform_real_distribution<Real> dist(-10.0, 10.0);

    for (size_t i = 0; i < n; ++i) {
        Real val;
        do {
            val = dist(rng);
        } while (!domain.defined_on_all_reals &&
                 std::any_of(domain.excluded_points.begin(), domain.excluded_points.end(),
                             [&](double ex){ return std::abs(val - ex) < 1e-12; }));
        points.push_back(val);
    }
    return points;
}

std::vector<Real> generateEdgePoints(const DomainDescriptor& domain) {
    std::vector<Real> points;

    if (domain.defined_on_zero) points.push_back(0.0);
    if (domain.defined_on_positive) points.push_back(1.0);
    if (domain.defined_on_negative) points.push_back(-1.0);

    for (auto& ex : domain.excluded_points) points.push_back(ex);

    return points;
}

std::vector<Real> allTestPoints(const FunctionDescriptor& fd) {
    auto fixed = generateEdgePoints(fd.domain);
    auto random = generateRandomPoints(fd.domain, fd.testing.recommended_random_samples);
    fixed.insert(fixed.end(), random.begin(), random.end());
    return fixed;
}

// ----------------- Test suite -----------------
void testDomainAndRange(const FunctionDescriptor& fd,
                        const std::vector<Real>& points)
{
    for (auto x : points) {
        Real y = fd.f(x);

        if (fd.range.finite_only) {
            for (auto x : points)
                EXPECT_TRUE(Analytics::isFiniteAt(fd.f, x, StabPolicy::Reject));
        }

        if (fd.range.min_value)
            EXPECT_GE(y, *fd.range.min_value - fd.correctness.max_absolute_error);

        if (fd.range.max_value)
            EXPECT_LE(y, *fd.range.max_value + fd.correctness.max_absolute_error);

        if (!fd.range.can_return_nan)
            EXPECT_FALSE(std::isnan(y));

        if (!fd.range.can_return_inf)
            EXPECT_FALSE(std::isinf(y));
    }
}

void testDomainViolation(const FunctionDescriptor& fd) {
    std::vector<Real> invalid;

    if (!fd.domain.defined_on_zero)
        invalid.push_back(0.0);
    if (!fd.domain.defined_on_positive)
        invalid.push_back(1.0);
    if (!fd.domain.defined_on_negative)
        invalid.push_back(-1.0);

    for (Real x : invalid) {
        if (fd.behavior.throws_on_domain_error) {
            EXPECT_ANY_THROW(fd.f(x));
        } else {
            Real y = fd.f(x);
            if (fd.behavior.returns_nan_on_domain_error)
                EXPECT_TRUE(std::isnan(y));
        }
    }
}

void testAnalyticProperties(const FunctionDescriptor& fd,
                            const std::vector<Real>& points)
{
    constexpr Real eps = 1e-6;
    constexpr Real h   = 1e-4;
    auto f = fd.f;

    if (fd.analytic.even) {
        for (auto x : points)
            EXPECT_TRUE(Analytics::isLocallyEvenFunction(f, x, eps, StabPolicy::Reject));
    }

    if (fd.analytic.odd) {
        for (auto x : points)
            EXPECT_TRUE(Analytics::isLocallyOddFunction(f, x, eps, StabPolicy::Reject));
    }

    if (fd.analytic.periodic && fd.analytic.period) {
        for (auto x : points)
            EXPECT_TRUE(Analytics::isPeriodic(f, x, *fd.analytic.period, eps, StabPolicy::Reject));
    }

    if (fd.analytic.monotonic_increasing) {
        for (size_t i = 1; i < points.size(); ++i)
            EXPECT_TRUE(Analytics::isLocallyIncreasing(
                f, points[i-1], points[i], eps, StabPolicy::Reject));
    }

    if (fd.analytic.monotonic_decreasing) {
        for (size_t i = 1; i < points.size(); ++i)
            EXPECT_TRUE(Analytics::isLocallyDecreasing(
                f, points[i-1], points[i], eps, StabPolicy::Reject));
    }

    if (fd.analytic.continuous) {
        for (auto x : points)
            EXPECT_TRUE(Analytics::isContinuous(f, x, h, eps, StabPolicy::Reject));
    }
}



Real numericalDerivative(const std::function<Real(Real)>& f, Real x) {
    constexpr Real h = 1e-6;
    return (f(x + h) - f(x - h)) / (2 * h);
}

void testDifferentialProperties(const FunctionDescriptor& fd,
                                const std::vector<Real>& points)
{
    if (!fd.differential.differentiable) return;

    constexpr Real h = 1e-4;

    for (auto x : points) {
        Real d = Analytics::derivative(fd.f, x, h, StabPolicy::Reject);
        EXPECT_TRUE(std::isfinite(d))
            << fd.name << ": derivative not finite at x=" << x;
    }

    if (fd.differential.convex) {
        for (auto x : points)
            EXPECT_TRUE(Analytics::isLocallyConvex(fd.f, x, h, StabPolicy::Reject));
    }

    if (fd.differential.concave) {
        for (auto x : points)
            EXPECT_TRUE(Analytics::isLocallyConcave(fd.f, x, h, StabPolicy::Reject));
    }
}

void testNumericalProperties(const FunctionDescriptor& fd,
                             const std::vector<Real>& points)
{
    if (fd.numerical.stable_near_zero) {
        for (Real x : {1e-12, -1e-12}) {
            Real y = fd.f(x);
            EXPECT_TRUE(std::isfinite(y))
                << fd.name << ": unstable near zero";
        }
    }

    if (fd.numerical.stable_for_large_values) {
        for (Real x : {1e6, -1e6}) {
            Real y = fd.f(x);
            EXPECT_TRUE(std::isfinite(y))
                << fd.name << ": unstable for large x";
        }
    }
}

void testBehavioralProperties(const FunctionDescriptor& fd,
                              const std::vector<Real>& points)
{
    if (!fd.behavior.deterministic) return;

    for (auto x : points) {
        Real y1 = fd.f(x);
        Real y2 = fd.f(x);
        EXPECT_EQ(y1, y2)
            << fd.name << ": non-deterministic at x=" << x;
    }
}

void testLimitBehavior(const FunctionDescriptor& fd) {
    constexpr Real eps = 1e-6;

    if (fd.limits.limit_at_plus_inf_exists && fd.limits.limit_at_plus_inf) {
        for (Real x : {1e6, 1e7, 1e8}) {
            Real y = fd.f(x);
            EXPECT_NEAR(y, *fd.limits.limit_at_plus_inf, eps);
        }
    }

    if (fd.limits.limit_at_minus_inf_exists && fd.limits.limit_at_minus_inf) {
        for (Real x : {-1e6, -1e7, -1e8}) {
            Real y = fd.f(x);
            EXPECT_NEAR(y, *fd.limits.limit_at_minus_inf, eps);
        }
    }

    if (fd.limits.limit_at_zero_exists && fd.limits.limit_at_zero) {
        for (Real x : {1e-6, -1e-6}) {
            Real y = fd.f(x);
            EXPECT_NEAR(y, *fd.limits.limit_at_zero, eps);
        }
    }
}

void testSingularities(const FunctionDescriptor& fd) {
    constexpr Real h = 1e-6;

    for (const auto& s : fd.singularities.singularities) {
        Real xl = s.x - h;
        Real xr = s.x + h;

        Real yl = fd.f(xl);
        Real yr = fd.f(xr);

        if (s.type == SingularityDescriptor::Type::Pole) {
            EXPECT_TRUE(std::isinf(yl) || std::isinf(yr));
        }

        if (s.type == SingularityDescriptor::Type::Removable) {
            EXPECT_TRUE(std::isfinite(yl));
            EXPECT_TRUE(std::isfinite(yr));
        }

        if (fd.limits.left_limit_equals_right_limit) {
            EXPECT_NEAR(yl, yr, 1e-4);
        }
    }
}

void runFunctionTests(const FunctionDescriptor& fd) {
    propagateFunctionDependencies(const_cast<FunctionDescriptor&>(fd));
    validateFunctionDescriptor(fd);

    auto points = allTestPoints(fd);

    testDomainAndRange(fd, points);
    testAnalyticProperties(fd, points);
    testNumericalProperties(fd, points);
    testBehavioralProperties(fd, points);
    testLimitBehavior(fd);
    testSingularities(fd);
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
            {
                [](const Function2D& f){
                    return Analytics::isLocallyIncreasing([&](Real x){ return f(x, 2.0); }, 0.0, 2.0, Constants::EPS_09, StabPolicy::Reject);
                },
                [](const Function2D& f){
                    return Analytics::isLocallyIncreasing([&](Real y){ return f(2.0, y); }, 0.0, 2.0, Constants::EPS_09, StabPolicy::Reject);
                },
                [](const Function2D& f){
                    return Analytics::isNonNegative([&](Real x){ return f(x, 2.0); }, 1.0, StabPolicy::Reject);
                }
            },
            {-2,0,2}, {-1,0,0.5,1,2}
        },
        UnifiedFn{
            "log_a", 2,
            {}, Functions::log_a,
            [](Real x, Real a){ return x <= 0.0 || a <= 0.0 || a == 1.0; },
            [](Real x, Real a){ return std::log(x)/std::log(a); },
            {}, {},
            {},
            {
                [](const Function2D& f){
                    return Analytics::isLocallyIncreasing([&](Real x){ return f(x, 2.0); }, 0.1, 10.0, Constants::EPS_09, StabPolicy::Reject);
                },
                [](const Function2D& f){
                    return Analytics::isLocallyDecreasing([&](Real a){ return f(2.0, a); }, 1.1, 10.0, Constants::EPS_09, StabPolicy::Reject);
                }
            },
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
*/