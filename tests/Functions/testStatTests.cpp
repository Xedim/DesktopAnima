#include <gtest/gtest.h>
#include "../math/pattern/Functions.h"
#include <functional>
#include <string>


struct StatTestCase {
    std::string name;
    std::function<Real()> fn;
    std::function<bool(Real)> invariant;
};

inline bool isfinite_real(Real v) {
    return std::isfinite(v);
}

inline bool non_negative(Real v) {
    return v >= 0;
}

inline bool in_unit_interval(Real v) {
    return v >= 0 && v <= 1;
}

class StatisticsTest : public ::testing::TestWithParam<StatTestCase> {};

TEST_P(StatisticsTest, Invariants) {
    const auto& c = GetParam();
    SCOPED_TRACE(c.name);
    Real v = c.fn();
    EXPECT_TRUE(c.invariant(v));
}

std::vector<StatTestCase> make_stat_tests() {
    VecReal a{1,2,3,4,5};
    VecReal b{2,3,4,5,6};

    return {
        {"z_test",          [&]{ return Functions::z_test(a, 3, 1); }, isfinite_real},
        {"t_test",          [&]{ return Functions::t_test(a, 3); }, isfinite_real},
        {"welch_t_test",    [&]{ return Functions::welch_t_test(a, b); }, isfinite_real},
        {"mann_whitney",    [&]{ return Functions::mann_whitney_u(a, b); },
                             [&](Real v){ return v >= 0 && v <= static_cast<Real>(a.size()*b.size()); }},
        {"wilcoxon_signed", [&]{ return Functions::wilcoxon_signed_rank(a, b); }, isfinite_real},
        {"ks_test",         [&]{ return Functions::ks_test(a, b); }, in_unit_interval},
        {"chi_square",      [&]{ return Functions::chi_square_test(a, VecReal{1,1,1,1,1}); }, non_negative},
        {"anderson_darling", [&]{ return Functions::anderson_darling(a); }, isfinite_real}
    };
}

INSTANTIATE_TEST_SUITE_P(
    StatisticalFunctions,
    StatisticsTest,
    ::testing::ValuesIn(make_stat_tests())
);
