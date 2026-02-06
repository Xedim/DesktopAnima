// testStatRegression.cpp
#include <gtest/gtest.h>
#include "../../math/functions/Functions.h"

using namespace Functions;

// ---------------- Linear Regression ----------------

TEST(LinearRegression, EmptyOrMismatch) {
    VecReal x{1,2,3}, y{1,2};
    auto r = linear_regression(x, y);
    EXPECT_TRUE(std::isnan(r.slope));
    EXPECT_TRUE(std::isnan(r.intercept));
    EXPECT_TRUE(std::isnan(r.r2));
}

TEST(LinearRegression, ConstantX) {
    VecReal x{1,1,1}, y{2,4,6};
    auto r = linear_regression(x, y);
    EXPECT_DOUBLE_EQ(r.slope, 0.0);
    EXPECT_DOUBLE_EQ(r.intercept, mean(y));
}

TEST(LinearRegression, LinePassesThroughMean) {
    VecReal x{1,2,3}, y{2,4,6};
    auto r = linear_regression(x, y);
    EXPECT_NEAR(r.intercept + r.slope * mean(x), mean(y), 1e-12);
    EXPECT_GE(r.r2, 0);
    EXPECT_LE(r.r2, 1);
}

// ---------------- Polynomial Regression ----------------

TEST(PolynomialRegression, EmptyOrMismatch) {
    VecReal x{1,2}, y{1};
    auto coef = polynomial_regression(x, y, 2);
    EXPECT_TRUE(coef.empty());
}

TEST(PolynomialRegression, DegreeZero) {
    VecReal x{1,2,3}, y{2,4,6};
    auto coef = polynomial_regression(x, y, 0);
    ASSERT_EQ(coef.size(), 1);
    EXPECT_NEAR(coef[0], mean(y), 1e-12);
}

TEST(PolynomialRegression, LinearMatch) {
    VecReal x{1,2,3}, y{2,4,6};
    auto coef = polynomial_regression(x, y, 1);
    EXPECT_NEAR(coef[1], 2.0, 1e-12); // slope
    EXPECT_NEAR(coef[0], 0.0, 1e-12); // intercept
}

TEST(PolynomialRegression, SingularData) {
    VecReal x{1,1,1}, y{2,2,2};
    auto coef = polynomial_regression(x, y, 2);
    EXPECT_EQ(coef[1], 0);
    EXPECT_EQ(coef[2], 0);
    EXPECT_NEAR(coef[0], 2.0, 1e-12);
}

// ---------------- Least Squares ----------------

TEST(LeastSquares, Empty) {
    VecReal r;
    EXPECT_DOUBLE_EQ(least_squares(r), 0);
}

TEST(LeastSquares, SumSquares) {
    VecReal r{1,2,3};
    EXPECT_DOUBLE_EQ(least_squares(r), 1*1 + 2*2 + 3*3);
}

TEST(LeastSquares, NonNegative) {
    VecReal r{-1,-2,-3};
    EXPECT_GE(least_squares(r), 0);
}
