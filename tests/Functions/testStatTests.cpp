#include <gtest/gtest.h>
#include "../math/pattern/Functions.h"
#include <vector>
#include <cmath>

using Real = double;
using VecReal = std::vector<Real>;

// ==================================================
// ================= Test Data ======================
// ==================================================

static const VecReal a{1,2,3,4,5};
static const VecReal b{2,3,4,5,6};
static const VecReal dummies{1,1,1,1,1};

// ==================================================
// ================= Test IDs =======================
// ==================================================

enum class StatTestId {
    ZTest,
    TTest,
    WelchTTest,
    MannWhitney,
    WilcoxonSigned,
    KSTest,
    ChiSquare,
    AndersonDarling
};

// ==================================================
// ================= Test Suite =====================
// ==================================================

class StatTest : public ::testing::TestWithParam<StatTestId> {};

TEST_P(StatTest, InvariantHolds) {
    Real v;

    switch (GetParam()) {

        case StatTestId::ZTest:
            v = Functions::z_test(a, 3, 1);
            EXPECT_TRUE(std::isfinite(v));
            break;

        case StatTestId::TTest:
            v = Functions::t_test(a, 3);
            EXPECT_TRUE(std::isfinite(v));
            break;

        case StatTestId::WelchTTest:
            v = Functions::welch_t_test(a, b);
            EXPECT_TRUE(std::isfinite(v));
            break;

        case StatTestId::MannWhitney:
            v = Functions::mann_whitney_u(a, b);
            EXPECT_GE(v, 0);
            EXPECT_LE(v, static_cast<Real>(a.size() * b.size()));
            break;

        case StatTestId::WilcoxonSigned:
            v = Functions::wilcoxon_signed_rank(a, b);
            EXPECT_TRUE(std::isfinite(v));
            break;

        case StatTestId::KSTest:
            v = Functions::ks_test(a, b);
            EXPECT_GE(v, 0);
            EXPECT_LE(v, 1);
            break;

        case StatTestId::ChiSquare:
            v = Functions::chi_square_test(a, dummies);
            EXPECT_GE(v, 0);
            break;

        case StatTestId::AndersonDarling:
            v = Functions::anderson_darling(a);
            EXPECT_TRUE(std::isfinite(v));
            break;
    }
}

// ==================================================
// ================= Instantiate ====================
// ==================================================

INSTANTIATE_TEST_SUITE_P(
    StatisticalFunctions,
    StatTest,
    ::testing::Values(
        StatTestId::ZTest,
        StatTestId::TTest,
        StatTestId::WelchTTest,
        StatTestId::MannWhitney,
        StatTestId::WilcoxonSigned,
        StatTestId::KSTest,
        StatTestId::ChiSquare,
        StatTestId::AndersonDarling
    )
);
