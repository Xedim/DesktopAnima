#include <gtest/gtest.h>
#include "../math/common/Types.h"
#include <cmath>

// -------------------------
// Тест NaN()
// -------------------------
TEST(TypesTest, NaNReturnsNaN) {
    Real x = NaN();
    EXPECT_TRUE(std::isnan(x));
}

// -------------------------
// Тест Interval
// -------------------------
TEST(TypesTest, IntervalContains) {
    Interval intv(0.0, 10.0);

    EXPECT_TRUE(intv.contains(0.0));
    EXPECT_TRUE(intv.contains(5.0));
    EXPECT_TRUE(intv.contains(10.0));
    EXPECT_FALSE(intv.contains(-0.1));
    EXPECT_FALSE(intv.contains(10.1));
}

TEST(TypesTest, IntervalConstructor) {
    Interval intv(-5.0, 5.0);
    EXPECT_DOUBLE_EQ(intv.min, -5.0);
    EXPECT_DOUBLE_EQ(intv.max, 5.0);
}

// -------------------------
// Тест Quartiles
// -------------------------
TEST(TypesTest, QuartilesInit) {
    Quartiles q{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(q.q1, 1.0);
    EXPECT_DOUBLE_EQ(q.q2, 2.0);
    EXPECT_DOUBLE_EQ(q.q3, 3.0);
}

// -------------------------
// Тест LinearRegressionResult
// -------------------------
TEST(TypesTest, LinearRegressionResultDefault) {
    LinearRegressionResult lr;
    EXPECT_TRUE(std::isnan(lr.slope));
    EXPECT_TRUE(std::isnan(lr.intercept));
    EXPECT_TRUE(std::isnan(lr.r2));
}

// -------------------------
// Тест PatternDescriptor
// -------------------------
TEST(TypesTest, PatternDescriptorInit) {
    Interval dom(0,1), rng(-1,1);
    PatternDescriptor pd{"test", dom, rng, PatternKind::Algebra};

    EXPECT_STREQ(pd.name, "test");
    EXPECT_DOUBLE_EQ(pd.domain.min, 0);
    EXPECT_DOUBLE_EQ(pd.domain.max, 1);
    EXPECT_DOUBLE_EQ(pd.range.min, -1);
    EXPECT_DOUBLE_EQ(pd.range.max, 1);
    EXPECT_EQ(pd.kind, PatternKind::Algebra);
}

// -------------------------
// Тест ArgVariant и ResultVariant (типизация)
// -------------------------
TEST(TypesTest, ArgVariantHolds) {
    ArgVariant a = 1.23;
    EXPECT_TRUE(std::holds_alternative<Real>(a));

    a = 42;
    EXPECT_TRUE(std::holds_alternative<int>(a));

    VecReal v{1.0,2.0};
    a = v;
    EXPECT_TRUE(std::holds_alternative<VecReal>(a));
}

TEST(TypesTest, ResultVariantHolds) {
    ResultVariant r = 3.14;
    EXPECT_TRUE(std::holds_alternative<Real>(r));

    Quartiles q{0,1,2};
    r = q;
    EXPECT_TRUE(std::holds_alternative<Quartiles>(r));

    LinearRegressionResult lr;
    r = lr;
    EXPECT_TRUE(std::holds_alternative<LinearRegressionResult>(r));
}
