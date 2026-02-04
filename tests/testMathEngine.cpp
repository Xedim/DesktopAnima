#include <gtest/gtest.h>
#include "../math/pattern/MathEngine.h"

TEST(MathEngineSmoke, FactorialRawCall) {
    MathEngine engine;

    ResultVariant result;

    ASSERT_NO_THROW({
        result = engine.compute(
            static_cast<size_t>(PatternID::Factorial),
            5
        );
    });

    // Проверяем, что вернулся Real
    ASSERT_TRUE(std::holds_alternative<Real>(result))
        << "Expected Real result from Factorial";

    auto value = std::get<Real>(result);

    // Минимальная sanity-проверка
    EXPECT_EQ(value, 120.0);
}