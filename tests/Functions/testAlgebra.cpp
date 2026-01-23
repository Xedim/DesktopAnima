// tests/Functions/testAlgebraDynamic.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "../../math/pattern/Functions.h"
#include "../../math/common/Constants.h"

constexpr Real EPS = 1e-9;

// генератор случайных чисел
std::mt19937 rng(42);
std::uniform_real_distribution<Real> dist_real(-100.0, 100.0);
std::uniform_int_distribution<int> dist_int(0, 20);

// ---------- Factorial / Binomial / Permutation ----------
TEST(AlgebraicDynamic, FactorialBinomialPermutation) {
    for (int n = 0; n <= 15; ++n) {
        Real f = Functions::factorial(n);
        if (n > 0) {
            EXPECT_DOUBLE_EQ(f, n * Functions::factorial(n - 1));
        }
        for (int k = 0; k <= n; ++k) {
            Real b = Functions::binomial(n, k);
            Real p = Functions::permutation(n, k);
            EXPECT_DOUBLE_EQ(b * Functions::factorial(k), p);
            EXPECT_DOUBLE_EQ(b, Functions::binomial(n, n - k)); // симметрия
        }
    }
}

// ---------- mod ----------
TEST(AlgebraicDynamic, Mod) {
    for (int i = 0; i < 100; ++i) {
        Real x = dist_real(rng);
        Real y = dist_real(rng) + 1e-3; // избегаем нуля
        Real r = Functions::mod(x, y);
        EXPECT_GE(r, 0);
        EXPECT_LT(r, std::abs(y));
        EXPECT_NEAR(r, Functions::mod(x + y, y), 1e-12); // периодичность
    }
}

// ---------- Polynomial / Rational ----------
TEST(AlgebraicDynamic, PolynomialRational) {
    for (int i = 0; i < 50; ++i) {
        int deg = dist_int(rng);
        VecReal coeffs(deg + 1);
        for (auto &c : coeffs) c = dist_real(rng);

        Real x = dist_real(rng);

        // naive polynomial evaluation
        Real naive = 0;
        for (int j = 0; j <= deg; ++j) {
            naive += coeffs[j] * std::pow(x, j);
        }

        EXPECT_NEAR(Functions::polynomial(x, coeffs), naive, 1e-12 * std::max(std::abs(naive), 1.0));

        // rational: numerator != denominator -> check consistency
        VecReal den = coeffs;
        den[0] += 1.0; // гарантируем не ноль
        Real r = Functions::rational(x, coeffs, den);
        EXPECT_NEAR(r * Functions::polynomial(x, den), Functions::polynomial(x, coeffs), EPS);
    }
}

// ---------- sqrt / cbrt / sign / abs ----------
TEST(AlgebraicDynamic, RootSignAbs) {
    for (int i = 0; i < 100; ++i) {
        Real x = dist_real(rng);

        // sqrt
        if (x < 0) {
            EXPECT_TRUE(std::isnan(Functions::sqrt(x)));
        } else {
            EXPECT_NEAR(std::pow(Functions::sqrt(x), 2), x, EPS);
        }

        // cbrt
        EXPECT_NEAR(std::pow(Functions::cbrt(x), 3), x, EPS);

        // sign
        if (x > 0) EXPECT_DOUBLE_EQ(Functions::sign(x), 1);
        else if (x < 0) EXPECT_DOUBLE_EQ(Functions::sign(x), -1);
        else EXPECT_DOUBLE_EQ(Functions::sign(x), 0);

        // abs
        EXPECT_GE(Functions::abs(x), 0);
        EXPECT_DOUBLE_EQ(Functions::abs(x), std::abs(x));
    }
}
