// testOutliers.cpp
#include <gtest/gtest.h>
#include "../../math/functions/Functions.h"

using namespace Functions;

// ---------------- z_score ----------------

TEST(ZScore, EmptyVector) {
    VecReal x;
    auto zs = z_score(x);
    EXPECT_TRUE(zs.empty());
}

TEST(ZScore, ConstantVector) {
    VecReal x{5,5,5,5};
    auto zs = z_score(x);
    for (auto v : zs) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST(ZScore, MeanAndStd) {
    VecReal x{1,2,3,4,5};
    auto zs = z_score(x);
    double mean_z = mean(zs);
    double std_z  = stddev_unbiased(zs);
    EXPECT_NEAR(mean_z, 0.0, 1e-12);
    EXPECT_NEAR(std_z, 1.0, 1e-12);
}

// ---------------- modified_z_score ----------------

TEST(ModifiedZScore, EmptyVector) {
    VecReal x;
    auto mz = modified_z_score(x);
    EXPECT_TRUE(mz.empty());
}

TEST(ModifiedZScore, ConstantVector) {
    VecReal x{7,7,7};
    auto mz = modified_z_score(x);
    for (auto v : mz) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST(ModifiedZScore, MedianZero) {
    VecReal x{1,2,3,4,5};
    auto mz = modified_z_score(x);
    double med = median(mz);
    EXPECT_NEAR(med, 0.0, 1e-12);
}

// ---------------- is_outlier ----------------

TEST(IsOutlier, ThresholdZeroOrNegative) {
    EXPECT_FALSE(is_outlier(5, 0, 1, 0));
    EXPECT_FALSE(is_outlier(5, 0, 1, -1));
}

TEST(IsOutlier, StdZero) {
    EXPECT_FALSE(is_outlier(5, 2, 0, 1));
}

TEST(IsOutlier, DetectOutlier) {
    EXPECT_TRUE(is_outlier(10, 0, 3, 3));
    EXPECT_FALSE(is_outlier(2, 0, 3, 3));
}

// ---------------- grubbs_test ----------------

TEST(GrubbsTest, TooSmallVector) {
    VecReal x{1,2};
    EXPECT_FALSE(grubbs_test(x, 0.05));
}

TEST(GrubbsTest, ConstantVector) {
    VecReal x{5,5,5,5};
    EXPECT_FALSE(grubbs_test(x, 0.05));
}

TEST(GrubbsTest, SingleOutlier) {
    VecReal x{1,1,1,10};
    EXPECT_TRUE(grubbs_test(x, 0.05));
}

TEST(GrubbsTest, NoOutlier) {
    VecReal x{1,2,3,4,5};
    EXPECT_FALSE(grubbs_test(x, 0.05));
}

// ---------------- chauvenet_criterion ----------------

TEST(Chauvenet, TooSmallVector) {
    VecReal x{1,2};
    EXPECT_FALSE(chauvenet_criterion(x));
}

TEST(Chauvenet, ConstantVector) {
    VecReal x{3,3,3,3};
    EXPECT_FALSE(chauvenet_criterion(x));
}

TEST(Chauvenet, SingleOutlier) {
    VecReal x{1,1,1,10};
    EXPECT_TRUE(chauvenet_criterion(x));
}

TEST(Chauvenet, NoOutlier) {
    VecReal x{1,2,3,4,5};
    EXPECT_FALSE(chauvenet_criterion(x));
}
