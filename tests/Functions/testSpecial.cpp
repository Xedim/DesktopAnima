#include <gtest/gtest.h>
#include <cmath>
#include "../../math/pattern/Functions.h"
#include "../../math/common/Types.h"
#include <boost/math/special_functions/lambert_w.hpp> // used by Functions::lambert_w

// ---------------- erf / erfc ----------------
struct ErfTestCase {
    Real x;
    Real expected_erf;
    Real expected_erfc;
};

class Special_Erf : public ::testing::TestWithParam<ErfTestCase> {};

TEST_P(Special_Erf, Values) {
    const auto& tc = GetParam();
    EXPECT_NEAR(erf(tc.x), tc.expected_erf, 1e-12);
    EXPECT_NEAR(erfc(tc.x), tc.expected_erfc, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
    ErfTests,
    Special_Erf,
    ::testing::Values(
        ErfTestCase{0.0, 0.0, 1.0},
        ErfTestCase{0.5, std::erf(0.5), std::erfc(0.5)},
        ErfTestCase{-1.0, std::erf(-1.0), std::erfc(-1.0)},
        ErfTestCase{2.0, std::erf(2.0), std::erfc(2.0)}
    )
);

// ---------------- gamma / lgamma ----------------
struct GammaTestCase {
    Real x;
    Real expected;
    bool expect_nan = false;
};

class Special_TGamma : public ::testing::TestWithParam<GammaTestCase> {};

TEST_P(Special_TGamma, TgammaValues) {
    const auto& tc = GetParam();
    Real result = Functions::gamma(tc.x);
    if(tc.expect_nan) EXPECT_TRUE(std::isnan(result));
    else EXPECT_NEAR(result, tc.expected, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
    GammaTests_TGamma,
    Special_TGamma,
    ::testing::Values(
        GammaTestCase{0.5, std::tgamma(0.5)},
        GammaTestCase{1.0, std::tgamma(1.0)},
        GammaTestCase{0.0, 0.0, true},
        GammaTestCase{-1.0, 0.0, true},
        GammaTestCase{5.0, std::tgamma(5.0)}
    )
);

class Special_LGamma : public ::testing::TestWithParam<GammaTestCase> {};

TEST_P(Special_LGamma, LgammaValues) {
    const auto& tc = GetParam();
    Real result = Functions::lgamma(tc.x);
    if(tc.expect_nan) EXPECT_TRUE(std::isnan(result));
    else EXPECT_NEAR(result, tc.expected, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
    GammaTests_LGamma,
    Special_LGamma,
    ::testing::Values(
        GammaTestCase{0.5, std::log(std::tgamma(0.5))},
        GammaTestCase{1.0, std::log(std::tgamma(1.0))},
        GammaTestCase{0.0, 0.0, true},
        GammaTestCase{-1.0, 0.0, true},
        GammaTestCase{5.0, std::log(std::tgamma(5.0))}
    )
);

// ---------------- beta ----------------
struct BetaTestCase {
    Real x;
    Real y;
    Real expected;
    bool expect_nan = false;
};

class Special_Beta : public ::testing::TestWithParam<BetaTestCase> {};

TEST_P(Special_Beta, Values) {
    const auto& tc = GetParam();
    Real result = Functions::beta(tc.x, tc.y);
    if(tc.expect_nan) EXPECT_TRUE(std::isnan(result));
    else EXPECT_NEAR(result, tc.expected, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
    BetaTests,
    Special_Beta,
    ::testing::Values(
        BetaTestCase{0.5, 0.5, std::tgamma(0.5)*std::tgamma(0.5)/std::tgamma(1.0)},
        BetaTestCase{1.0, 2.0, std::tgamma(1.0)*std::tgamma(2.0)/std::tgamma(3.0)},
        BetaTestCase{-1.0, 2.0, 0.0, true},
        BetaTestCase{2.0, -3.0, 0.0, true}
    )
);

// ---------------- lambert_w ----------------
struct LambertWTestCase {
    Real x;
    Real expected;
    bool expect_nan = false;
};

class Special_LambertW : public ::testing::TestWithParam<LambertWTestCase> {};

TEST_P(Special_LambertW, Values) {
    const auto& tc = GetParam();
    Real result = Functions::lambert_w(tc.x);
    if(tc.expect_nan) EXPECT_TRUE(std::isnan(result));
    else EXPECT_NEAR(result, tc.expected, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
    LambertWTests,
    Special_LambertW,
    ::testing::Values(
        LambertWTestCase{0.0, 0.0},
        LambertWTestCase{1.0, boost::math::lambert_w0(1.0)},
        LambertWTestCase{-1.0/std::exp(1), boost::math::lambert_w0(-1.0/std::exp(1))},
        LambertWTestCase{-2.0, 0.0, true}
    )
);
