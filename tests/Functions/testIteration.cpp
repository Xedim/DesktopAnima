// testFractal.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "../../math/pattern/Functions.h"
#include "../../math/common/Types.h"
#include <complex>


// ======================================================
// ==================== REAL TESTS =======================
// ======================================================

struct IterateRealCase {
    Real x0;
    int n;
    Real expected;
};


// ---------- Linear map: f(x) = x + 1 ----------
class IterateRealLinearTests
    : public ::testing::TestWithParam<IterateRealCase> {};

TEST_P(IterateRealLinearTests, LinearMap) {
    const auto& tc = GetParam();

    auto f = [](Real x) { return x + 1.0; };

    Real result = Functions::iterate(tc.x0, f, tc.n);

    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(result));
    } else {
        EXPECT_NEAR(result, tc.expected, Constants::EPS_09);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Fractal,
    IterateRealLinearTests,
    ::testing::Values(
        IterateRealCase{0.0, 0, 0.0},
        IterateRealCase{0.0, 1, 1.0},
        IterateRealCase{0.0, 5, 5.0},
        IterateRealCase{-2.0, 3, 1.0},
        IterateRealCase{10.0, 10, 20.0}
    )
);


// ---------- Quadratic map ----------
TEST(IterateRealStandalone, QuadraticMap) {
    auto f = [](Real x) { return x * x; };

    Real r = Functions::iterate(Real{2.0}, f, 3);
    // 2 → 4 → 16 → 256
    EXPECT_NEAR(r, 256.0, Constants::EPS_09);
}


// ---------- NaN propagation ----------
TEST(IterateRealStandalone, NaNPropagation) {
    auto f = [](Real) { return std::numeric_limits<Real>::quiet_NaN(); };

    Real r = Functions::iterate(Real{1.0}, f, 5);
    EXPECT_FALSE(std::isfinite(r));
}


// ======================================================
// ================== COMPLEX TESTS =====================
// ======================================================

struct IterateComplexCase {
    Complex z0;
    int n;
    Complex expected;
};


// ---------- Complex shift ----------
class IterateComplexShiftTests
    : public ::testing::TestWithParam<IterateComplexCase> {};

TEST_P(IterateComplexShiftTests, ShiftMap) {
    const auto& tc = GetParam();

    auto f = [](Complex z) {
        return z + Complex{1.0, -1.0};
    };

    Complex r = Functions::iterate(tc.z0, f, tc.n);

    EXPECT_NEAR(r.real(), tc.expected.real(), Constants::EPS_09);
    EXPECT_NEAR(r.imag(), tc.expected.imag(), Constants::EPS_09);
}

INSTANTIATE_TEST_SUITE_P(
    Fractal,
    IterateComplexShiftTests,
    ::testing::Values(
        IterateComplexCase{{0,0}, 0, {0,0}},
        IterateComplexCase{{0,0}, 1, {1,-1}},
        IterateComplexCase{{1,1}, 2, {3,-1}},
        IterateComplexCase{{-1,2}, 3, {2,-1}}
    )
);


// ---------- Julia-style quadratic ----------
TEST(IterateComplexStandalone, JuliaLikeIteration) {
    Complex c{0.3, 0.5};

    auto f = [c](Complex z) {
        return z * z + c;
    };

    Complex r = Functions::iterate(Complex{0,0}, f, 10);

    EXPECT_TRUE(std::isfinite(r.real()));
    EXPECT_TRUE(std::isfinite(r.imag()));
}


// ======================================================
// ================= ALGEBRAIC PROPERTY =================
// ======================================================

// f^(n+m)(x) == f^m(f^n(x))
TEST(IterateAlgebraic, Associativity) {
    auto f = [](Real x) { return x + 2.0; };

    Real x = 1.0;

    Real a = Functions::iterate(x, f, 3);
    Real b = Functions::iterate(a, f, 5);

    Real direct = Functions::iterate(x, f, 8);

    EXPECT_NEAR(b, direct, Constants::EPS_09);
}
