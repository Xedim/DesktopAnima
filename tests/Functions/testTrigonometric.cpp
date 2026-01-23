#include <gtest/gtest.h>
#include <cmath>
#include "../../math/pattern/Functions.h"
#include "../../math/common/Constants.h"

constexpr Real EPS = 1e-12;

TEST(TrigonometricDynamic, BasicValues) {
    // sin/cos
    EXPECT_NEAR(Functions::sin(0.0), 0.0, EPS);
    EXPECT_NEAR(Functions::cos(0.0), 1.0, EPS);

    // tan/cot
    EXPECT_NEAR(Functions::tan(0.0), 0.0, EPS);
    EXPECT_TRUE(std::isnan(Functions::cot(0.0)));

    // sec/csc
    EXPECT_NEAR(Functions::sec(0.0), 1.0, EPS);
    EXPECT_TRUE(std::isnan(Functions::csc(0.0)));

    // sinc
    EXPECT_NEAR(Functions::sinc(0.0), 1.0, EPS);
    EXPECT_NEAR(Functions::sinc(1e-6), 1.0 - 1e-12/6.0, 1e-12);
}

TEST(TrigonometricDynamic, ArcsinArccos) {
    EXPECT_NEAR(Functions::asin(0.0), 0.0, EPS);
    EXPECT_NEAR(Functions::asin(1.0), M_PI_2, EPS);
    EXPECT_NEAR(Functions::acos(1.0), 0.0, EPS);
    EXPECT_NEAR(Functions::acos(-1.0), M_PI, EPS);
    EXPECT_TRUE(std::isnan(Functions::asin(2.0)));
    EXPECT_TRUE(std::isnan(Functions::acos(-2.0)));
}

TEST(TrigonometricDynamic, Arctan) {
    EXPECT_NEAR(Functions::atan(0.0), 0.0, EPS);
    EXPECT_TRUE(std::isnan(Functions::atan(std::numeric_limits<Real>::infinity())));
}

TEST(TrigonometricDynamic, Arctan2Hypot) {
    EXPECT_NEAR(Functions::atan2(0.0, 1.0), 0.0, EPS);
    EXPECT_NEAR(Functions::atan2(1.0, 0.0), M_PI_2, EPS);
    EXPECT_TRUE(std::isnan(Functions::atan2(1.0, std::numeric_limits<Real>::infinity())));
    EXPECT_NEAR(Functions::hypot(3.0, 4.0), 5.0, EPS);
    EXPECT_NEAR(Functions::hypot(0.0, 0.0), 0.0, EPS);
    EXPECT_TRUE(std::isnan(Functions::hypot(1.0, std::numeric_limits<Real>::infinity())));
}

TEST(TrigonometricDynamic, PeriodicityAndSymmetry) {
    EXPECT_NEAR(Functions::sin(M_PI + 0.0), 0.0, EPS); // sin(pi) = 0
    EXPECT_NEAR(Functions::cos(-M_PI), -1.0, EPS);     // cos(-pi) = -1
    EXPECT_NEAR(Functions::sin(-M_PI_2), -1.0, EPS);   // sin(-pi/2) = -1
}
