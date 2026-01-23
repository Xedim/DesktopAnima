// testFractal.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "../../math/pattern/Functions.h"
#include "../../math/common/Constants.h"

// ==============================================
// ================= Weierstrass =================
// ==============================================
struct WeierstrassTestCase {
    Real x;
    Real a;
    Real b;
    int N;
    StabPolicy policy;
    Real expected;
    bool expect_nan = false;
};

class WeierstrassTests : public ::testing::TestWithParam<WeierstrassTestCase> {};

TEST_P(WeierstrassTests, Values) {
    const auto& tc = GetParam();
    Real result = Functions::weierstrass(tc.x, tc.a, tc.b, tc.N, tc.policy);
    if (tc.expect_nan)
        EXPECT_TRUE(std::isnan(result));
    else
        EXPECT_NEAR(result, tc.expected, 2e-3);
}

INSTANTIATE_TEST_SUITE_P(
    Fractal,
    WeierstrassTests,
    ::testing::Values(
        // корректные параметры
        WeierstrassTestCase{0.5, 0.5, 2.0, 10, StabPolicy::Raw, 0.0, false},
        WeierstrassTestCase{0.0, 0.7, 3.0, 3, StabPolicy::Raw, 2.19, false},
        // некорректные a,b,x,N
        WeierstrassTestCase{0.0, 0.0, 2.0, 10, StabPolicy::Raw, 0.0, true},
        WeierstrassTestCase{0.0, 0.5, 0.0, 10, StabPolicy::Raw, 0.0, true},
        WeierstrassTestCase{Constants::WEIERSTRASS_X_MAX + 1.0, 0.5, 2.0, 10, StabPolicy::Raw, 0.0, true}
    )
);

// ==============================================
// ================= Cantor ====================
// ==============================================
struct CantorTestCase {
    Real x;
    int max_iter;
    StabPolicy policy;
    Real expected;
    bool expect_nan = false;
};

class CantorTests : public ::testing::TestWithParam<CantorTestCase> {};

TEST_P(CantorTests, Values) {
    const auto& tc = GetParam();
    Real result = Functions::cantor(tc.x, tc.max_iter, tc.policy);
    if (tc.expect_nan)
        EXPECT_TRUE(std::isnan(result));
    else
        EXPECT_NEAR(result, tc.expected, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
    Fractal,
    CantorTests,
    ::testing::Values(
        CantorTestCase{0.0, 5, StabPolicy::Reject, 0.0, false},
        CantorTestCase{0.5, 5, StabPolicy::Reject, 0.5, false},
        CantorTestCase{-0.1, 5, StabPolicy::Reject, 0.0, true}, // x < min
        CantorTestCase{1.1, 5, StabPolicy::Reject, 0.0, true},  // x > max
        CantorTestCase{0.3, 0, StabPolicy::Reject, 0.0, true}   // max_iter <= min
    )
);

// ==============================================
// ================= Logistic ===================
// ==============================================
struct LogisticTestCase {
    Real x;
    Real r;
    StabPolicy policy;
    Real expected;
    bool expect_nan = false;
};

class LogisticTests : public ::testing::TestWithParam<LogisticTestCase> {};

TEST_P(LogisticTests, Values) {
    const auto& tc = GetParam();
    Real result = Functions::logistic(tc.x, tc.r, tc.policy);
    if (tc.expect_nan)
        EXPECT_TRUE(std::isnan(result));
    else
        EXPECT_NEAR(result, tc.expected, 1e-3);
}

INSTANTIATE_TEST_SUITE_P(
    Fractal,
    LogisticTests,
    ::testing::Values(
        LogisticTestCase{0.5, 2.0, StabPolicy::Reject, 0.5, false},
        LogisticTestCase{0.0, 2.0, StabPolicy::Reject, 0.0, false},
        LogisticTestCase{1.0, 3.0, StabPolicy::Reject, 0.0, false},
        LogisticTestCase{-0.1, 2.0, StabPolicy::Reject, 0.0, true}, // x < min
        LogisticTestCase{0.5, 0.0, StabPolicy::Reject, 0.0, true}   // r < min
    )
);

// ==============================================
// ================= Tent =======================
// ==============================================
struct TentTestCase {
    Real x;
    StabPolicy policy;
    Real expected;
    bool expect_nan = false;
};

class TentTests : public ::testing::TestWithParam<TentTestCase> {};

TEST_P(TentTests, Values) {
    const auto& tc = GetParam();
    Real result = Functions::tent(tc.x, tc.policy);
    if (tc.expect_nan)
        EXPECT_TRUE(std::isnan(result));
    else
        EXPECT_NEAR(result, tc.expected, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
    Fractal,
    TentTests,
    ::testing::Values(
        TentTestCase{0.0, StabPolicy::Reject, 0.0, false},
        TentTestCase{Constants::TENT_PEAK, StabPolicy::Reject, Constants::TENT_SLOPE * Constants::TENT_PEAK, false},
        TentTestCase{1.0, StabPolicy::Reject, 0.0, false},
        TentTestCase{-0.1, StabPolicy::Reject, 0.0, true}, // x < min
        TentTestCase{1.1, StabPolicy::Reject, 0.0, true}   // x > max
    )
);

// ==============================================
// ================= Julia ======================
// ==============================================
struct JuliaTestCase {
    Complex z;
    Complex c;
    StabPolicy policy;
    Complex expected;
};

class JuliaTests : public ::testing::TestWithParam<JuliaTestCase> {};

TEST_P(JuliaTests, Values) {
    const auto& tc = GetParam();
    Complex result = Functions::julia(tc.z, tc.c, tc.policy);
    EXPECT_NEAR(result.real(), tc.expected.real(), 1e-2);
    EXPECT_NEAR(result.imag(), tc.expected.imag(), 1e-2);
}

INSTANTIATE_TEST_SUITE_P(
    Fractal,
    JuliaTests,
    ::testing::Values(
        JuliaTestCase{{0.0, 0.0}, {0.0, 0.0}, StabPolicy::Raw, {0.0, 0.0}},
        JuliaTestCase{{1.0, 1.0}, {0.0, 0.0}, StabPolicy::Raw, {-4.0, 0.0}},
        JuliaTestCase{{1.0, -1.0}, {0.5, 0.5}, StabPolicy::Raw, {-1.5, -1.0}}
    )
);

// ==============================================
// ================= Escapes ====================
// ==============================================
struct EscapesTestCase {
    Complex z0;
    Complex c;
    int max_iter;
    Real threshold;
    bool expected;
};

class EscapesTests : public ::testing::TestWithParam<EscapesTestCase> {};

TEST_P(EscapesTests, Values) {
    const auto& tc = GetParam();
    bool result = Functions::escapes(tc.z0, tc.c, tc.max_iter, tc.threshold);
    EXPECT_EQ(result, tc.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Fractal,
    EscapesTests,
    ::testing::Values(
        EscapesTestCase{{0.0,0.0}, {0.0,0.0}, 10, 2.0, false},
        EscapesTestCase{{1.0,1.0}, {0.0,0.0}, 10, 2.0, true},
        EscapesTestCase{{0.0,0.0}, {1.0,1.0}, 10, 2.0, true},
        EscapesTestCase{{0.0,0.0}, {0.0,0.0}, 0, 2.0, false},   // max_iter <=0
        EscapesTestCase{{0.0,0.0}, {0.0,0.0}, 10, -1.0, false}  // threshold <=0
    )
);
