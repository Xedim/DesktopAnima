#include "../math/pattern/Functions.h"
#include "../math/common/Types.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// ----------------- MAD -----------------
struct MadCase {
    VecReal x;
    Real expected;
};

class MadTests : public ::testing::TestWithParam<MadCase> {};

TEST_P(MadTests, MedianAbsoluteDeviation) {
    const auto& tc = GetParam();
    Real r = Functions::median_absolute_deviation(tc.x);
    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
    }
}

std::vector<MadCase> make_mad_cases() {
    return {
        {{}, NaN()},
        {{1}, 0.0},
        {{2,2,2,2}, 0.0},
        {{1,2,3,4,5}, 1.0}, // медиана=3, отклонения=[2,1,0,1,2], медиана=1
        {{1,3,5,7}, 2.0}    // медиана=4, отклонения=[3,1,1,3], медиана=2
    };
}

// ----------------- Winsorized Mean -----------------
struct WinsorCase {
    VecReal x;
    Real alpha;
    Real expected;
};

class WinsorTests : public ::testing::TestWithParam<WinsorCase> {};

TEST_P(WinsorTests, WinsorizedMean) {
    const auto& tc = GetParam();
    Real r = Functions::winsorized_mean(tc.x, tc.alpha);
    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
    }
}

std::vector<WinsorCase> make_winsor_cases() {
    return {
            {{}, 0.1, NaN()},
            {{1}, 0.0, 1.0},
            {{1,2,3,4,100}, 0.2, 3.0},
            {{1,2,3,4,5}, 0.0, 3.0}
    };
}

// ----------------- Huber Mean -----------------
struct HuberCase {
    VecReal x;
    Real delta;
    Real expected;
};

class HuberTests : public ::testing::TestWithParam<HuberCase> {};

TEST_P(HuberTests, HuberMean) {
    const auto& tc = GetParam();
    Real r = Functions::huber_mean(tc.x, tc.delta);
    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
    }
}

std::vector<HuberCase> make_huber_cases() {
    return {
                {{}, 1.0, NaN()},
                {{1}, 1.0, 1.0},
                {{1,2,3,4,5}, 10.0, 3.0},
                {{1,2,3,4,100}, 1.0, 21.4}
    };
}

// ----------------- Biweight Mean -----------------
struct BiweightCase {
    VecReal x;
    Real expected;
};

class BiweightTests : public ::testing::TestWithParam<BiweightCase> {};

TEST_P(BiweightTests, BiweightMean) {
    const auto& tc = GetParam();
    Real r = Functions::biweight_mean(tc.x);
    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
    }
}

std::vector<BiweightCase> make_biweight_cases() {
    return {
        {{}, NaN()},
        {{1}, 1.0},
        {{1,1,1,1}, 1.0},
        {{1,2,3,4,5}, 3.0} // медиана=3, MAD=1, простая проверка
    };
}

// ----------------- SNR -----------------
struct SnrCase {
    VecReal x;
    Real expected;
};

class SnrTests : public ::testing::TestWithParam<SnrCase> {};

TEST_P(SnrTests, SignalToNoise) {
    const auto& tc = GetParam();
    Real r = Functions::snr(tc.x);
    if (std::isnan(tc.expected)) {
        EXPECT_TRUE(std::isnan(r));
    } else {
        EXPECT_NEAR(r, tc.expected, Constants::EPS_09);
    }
}

std::vector<SnrCase> make_snr_cases() {
    return {
        {{}, NaN()},
        {{1}, NaN()},
        {{2,2,2}, NaN()},
        {{1,2,3,4}, 2.5 / std::sqrt(1.6666666666666667)} // mean=2.5, var=1.66666
    };
}

INSTANTIATE_TEST_SUITE_P(RobustStats, MadTests, ::testing::ValuesIn(make_mad_cases()));
INSTANTIATE_TEST_SUITE_P(RobustStats, WinsorTests, ::testing::ValuesIn(make_winsor_cases()));
INSTANTIATE_TEST_SUITE_P(RobustStats, HuberTests, ::testing::ValuesIn(make_huber_cases()));
INSTANTIATE_TEST_SUITE_P(RobustStats, BiweightTests, ::testing::ValuesIn(make_biweight_cases()));
INSTANTIATE_TEST_SUITE_P(RobustStats, SnrTests, ::testing::ValuesIn(make_snr_cases()));
