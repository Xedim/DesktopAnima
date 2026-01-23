#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../../math/pattern/Functions.h"

// ---------------- dirac_delta ----------------
struct DiracTestCase { double x; double eps; double expected; bool expect_nan = false; };

class General_DiracDelta : public ::testing::TestWithParam<DiracTestCase> {};

TEST_P(General_DiracDelta, Values) {
    const auto& tc = GetParam();
    double result = Functions::dirac_delta(tc.x, tc.eps);
    if(tc.expect_nan) EXPECT_TRUE(std::isnan(result));
    else EXPECT_NEAR(result, tc.expected, 1e-12);
}

// ---------------- geometric_sum ----------------
struct GeometricTestCase { double a; int N; double expected; };

class General_Geometric : public ::testing::TestWithParam<GeometricTestCase> {};

TEST_P(General_Geometric, Values) {
    const auto& tc = GetParam();
    double result = Functions::geometric_sum(tc.a, tc.N);
    EXPECT_NEAR(result, tc.expected, 1e-12);
}

// ---------------- algebraic_root ----------------
struct AlgebraicRootTestCase { double x; std::vector<double> coeffs; double expected; bool expect_nan = false; };

class General_AlgebraicRoot : public ::testing::TestWithParam<AlgebraicRootTestCase> {};

TEST_P(General_AlgebraicRoot, Values) {
    const auto& tc = GetParam();
    double result = Functions::algebraic_root(tc.x, tc.coeffs);
    if(tc.expect_nan) EXPECT_TRUE(std::isnan(result));
    else EXPECT_NEAR(result, tc.expected, 1e-12);
}

// ---------------- clamp ----------------
struct ClampTestCase { double x; double minVal; double maxVal; double expected; };

class General_Clamp : public ::testing::TestWithParam<ClampTestCase> {};

TEST_P(General_Clamp, Values) {
    const auto& tc = GetParam();
    EXPECT_DOUBLE_EQ(Functions::clamp(tc.x, tc.minVal, tc.maxVal), tc.expected);
}

// ---------------- lerp ----------------
struct LerpTestCase { double a; double b; double t; double expected; };

class General_Lerp : public ::testing::TestWithParam<LerpTestCase> {};

TEST_P(General_Lerp, Values) {
    const auto& tc = GetParam();
    EXPECT_NEAR(Functions::lerp(tc.a, tc.b, tc.t), tc.expected, 1e-12);
}

// ---------------- fma ----------------
struct FmaTestCase { double x; double y; double z; double expected; };

class General_Fma : public ::testing::TestWithParam<FmaTestCase> {};

TEST_P(General_Fma, Values) {
    const auto& tc = GetParam();
    EXPECT_NEAR(Functions::fma(tc.x, tc.y, tc.z), tc.expected, 1e-12);
}


INSTANTIATE_TEST_SUITE_P(
    DiracDeltaTests, General_DiracDelta,
    ::testing::Values(
        DiracTestCase{1.0, 0.1, std::exp(-1.0/0.1)/std::sqrt(M_PI*0.1)},
        DiracTestCase{-10.0, 0.1, 0.0},
        DiracTestCase{0.5, -1.0, 0.0, true}  // eps <=0
    )
);

INSTANTIATE_TEST_SUITE_P(
    GeometricSumTests, General_Geometric,
    ::testing::Values(
        GeometricTestCase{2.0, 4, 1+2+4+8},
        GeometricTestCase{1.0, 5, 5},
        GeometricTestCase{0.5, 3, 1+0.5+0.25},
        GeometricTestCase{3.0, 0, 0}
    )
);

INSTANTIATE_TEST_SUITE_P(
    AlgebraicRootTests, General_AlgebraicRoot,
    ::testing::Values(
        AlgebraicRootTestCase{4.0, {1,0,0}, 1},   // x^2
        AlgebraicRootTestCase{1.0, {1,1}, std::sqrt(2)},     // x+1
        AlgebraicRootTestCase{2.0, {-5,1}, 0.0, true}        // отрицательное
    )
);

INSTANTIATE_TEST_SUITE_P(
    ClampTests, General_Clamp,
    ::testing::Values(
        ClampTestCase{5.0, 0.0, 10.0, 5.0},
        ClampTestCase{-1.0, 0.0, 10.0, 0.0},
        ClampTestCase{11.0, 0.0, 10.0, 10.0}
    )
);

INSTANTIATE_TEST_SUITE_P(
    LerpTests, General_Lerp,
    ::testing::Values(
        LerpTestCase{0.0, 10.0, 0.0, 0.0},
        LerpTestCase{0.0, 10.0, 1.0, 10.0},
        LerpTestCase{5.0, 15.0, 0.5, 10.0}
    )
);

INSTANTIATE_TEST_SUITE_P(
    FmaTests, General_Fma,
    ::testing::Values(
        FmaTestCase{2.0, 3.0, 4.0, 10.0},
        FmaTestCase{-1.0, 5.0, 2.0, -3.0}
    )
);
