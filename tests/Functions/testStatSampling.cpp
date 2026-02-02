// testStatSampling.cpp
#include <gtest/gtest.h>
#include "../../math/pattern/Functions.h"
#include <cmath>
#include <algorithm>

// ============================
// ======== bootstrap_mean =====
// ============================
TEST(Sampling, BootstrapMean_EmptyInput) {
    VecReal x;
    EXPECT_TRUE(std::isnan(Functions::bootstrap_mean(x, 100)));
}

TEST(Sampling, BootstrapMean_NegativeTrials) {
    VecReal x{1,2,3};
    EXPECT_TRUE(std::isnan(Functions::bootstrap_mean(x, -5)));
}

TEST(Sampling, BootstrapMean_Correctness) {
    VecReal x{1,2,3,4,5};
    auto bm = Functions::bootstrap_mean(x, 500);
    // Среднее выборки должно быть близко к обычному среднему
    Real expected = 3.0;
    EXPECT_NEAR(bm, expected, 0.2); // ±0.2 для случайности
}

// =================================
// ======== bootstrap_ci ===========
// =================================
TEST(Sampling, BootstrapCI_EmptyInput) {
    VecReal x;
    auto ci = Functions::bootstrap_ci(x, 0.95, 1000);
    EXPECT_TRUE(std::isnan(ci.first));
    EXPECT_TRUE(std::isnan(ci.second));
}

TEST(Sampling, BootstrapCI_AlphaBounds) {
    VecReal x{1,2,3};
    auto ci1 = Functions::bootstrap_ci(x, 0, 1000);
    auto ci2 = Functions::bootstrap_ci(x, 1, 1000);
    EXPECT_TRUE(std::isnan(ci1.first));
    EXPECT_TRUE(std::isnan(ci2.first));
}

TEST(Sampling, BootstrapCI_ContainsMean) {
    VecReal x{1,2,3,4,5,6,7,8,9,10};
    auto ci = Functions::bootstrap_ci(x, 0.8, 1000);
    Real mean_x = std::accumulate(x.begin(), x.end(), Real{0}) / static_cast<Real>(x.size());
    EXPECT_GE(mean_x, ci.first);
    EXPECT_LE(mean_x, ci.second);
}

// =================================
// ============ jackknife ===========
// =================================
TEST(Sampling, Jackknife_EmptyInput) {
    VecReal x;
    auto out = Functions::jackknife(x);
    EXPECT_TRUE(out.empty());
}

TEST(Sampling, Jackknife_Correctness) {
    VecReal x{1,2,3,4};
    auto out = Functions::jackknife(x);
    // Каждое значение должно быть средним остальных элементов
    EXPECT_NEAR(out[0], (2+3+4)/3.0, 1e-12);
    EXPECT_NEAR(out[1], (1+3+4)/3.0, 1e-12);
    EXPECT_NEAR(out[2], (1+2+4)/3.0, 1e-12);
    EXPECT_NEAR(out[3], (1+2+3)/3.0, 1e-12);
}

// =================================
// ======= permutation_test =========
// =================================
TEST(Sampling, PermutationTest_EmptyInput) {
    VecReal x, y;
    EXPECT_TRUE(std::isnan(Functions::permutation_test(x, y, 1000)));
}

TEST(Sampling, PermutationTest_SizeMismatch) {
    VecReal x{1,2,3}, y{4,5};
    EXPECT_TRUE(std::isnan(Functions::permutation_test(x, y, 1000)));
}

TEST(Sampling, PermutationTest_Identical) {
    VecReal x{1,2,3,4}, y{1,2,3,4};
    auto p = Functions::permutation_test(x, y, 1000);
    // Рядки идентичны → p-value должен быть ~1
    EXPECT_NEAR(p, 1.0, 0.05);
}

TEST(Sampling, PermutationTest_Different) {
    VecReal x{1,1,1,1}, y{10,10,10,10};
    auto p = Functions::permutation_test(x, y, 1000);
    // Явно различны → p-value близко к 0
    EXPECT_NEAR(p, 0.0, 0.05);
}
