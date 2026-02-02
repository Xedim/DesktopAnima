#include <gtest/gtest.h>
#include "../../math/pattern/Functions.h"

TEST(TimeSeries, RollingMeanEmpty) {
    VecReal x;
    EXPECT_TRUE(Functions::rolling_mean(x, 3).empty());
    EXPECT_TRUE(Functions::rolling_mean({1,2,3}, 0).empty());
}

TEST(TimeSeries, RollingMeanBasic) {
    VecReal x{1,2,3,4,5};
    auto res = Functions::rolling_mean(x, 3);
    EXPECT_EQ(res.size(), 3);
    EXPECT_NEAR(res[0], 2.0, 1e-12);
    EXPECT_NEAR(res[1], 3.0, 1e-12);
    EXPECT_NEAR(res[2], 4.0, 1e-12);
}

TEST(TimeSeries, RollingVarianceConst) {
    VecReal x(5, 10.0);
    auto res = Functions::rolling_variance(x, 3);
    for (auto v : res) EXPECT_NEAR(v, 0.0, 1e-12);
}

TEST(TimeSeries, EMAAlphaOne) {
    VecReal x{1,2,3,4};
    auto res = Functions::ema(x, 1.0);
    EXPECT_EQ(res, x);
}

TEST(TimeSeries, DetrendLinear) {
    VecReal x{0,1,2,3,4};
    auto y = Functions::detrend(x);
    double slope = (y.back() - y.front()) / (static_cast<Real>(x.size())-1);
    EXPECT_NEAR(slope, 0.0, 1e-12);
}

TEST(TimeSeries, DifferenceOrder1) {
    VecReal x{1,3,6,10};
    auto y = Functions::difference(x, 1);
    VecReal expected{2,3,4};
    EXPECT_EQ(y, expected);
}

TEST(TimeSeries, TakensMapBasic) {
    VecReal x{0,1,2,3,4,5,6};
    auto y = Functions::takens_map(x, 3, 2);
    VecReal expected{0,2,4,1,3,5,2,4,6};
    EXPECT_EQ(y, expected);
}

TEST(TimeSeries, HurstShort) {
    VecReal x{1,2,3};
    EXPECT_TRUE(std::isnan(Functions::hurst_exponent(x)));
}

TEST(TimeSeries, LyapunovShort) {
    VecReal x{1,2,3,4,5,6,7,8,9};
    EXPECT_TRUE(std::isnan(Functions::lyapunov_exponent(x)));
}

// partial_autocorrelation тесты делаем на маленьких известных рядах
TEST(TimeSeries, PartialAutocorrBasic) {
    VecReal x{1,2,1,2};
    EXPECT_NEAR(Functions::partial_autocorrelation(x, 1), -1.0, 1e-12);
    EXPECT_TRUE(std::isnan(Functions::partial_autocorrelation(x, 0)));
}
