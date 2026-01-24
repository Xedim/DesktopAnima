// tests/Functions/testExpLogUnified.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include "../../math/pattern/Functions.h"
#include "../math/common/Types.h"

constexpr Real EPS = 1e-12;

// ----------------- Унифицированная структура функции -----------------
struct FunctionTest {
    std::string name;
    int nargs; // 1 или 2 аргумента
    Function1D f1;
    Function2D f2;
    std::vector<Real> x_vals;
    std::vector<Real> y_vals; // для 2-х аргументов
};

// ----------------- Список функций -----------------
std::vector<FunctionTest> getFunctions() {
    return {
        // pow(x, alpha) как 2-аргументная
        { "pow", 2, {}, [](Real x, Real a){ return Functions::pow(x, a); }, {0.0, 1.5, -2.0, 2.5}, {0.0, 0.5, 1.0, 3.7} },

        // exp(x)
        { "exp", 1, Functions::exp, {}, {0.0, -1.0, 1.0, 700.0, -700.0, 1000.0}, {} },

        // exp2(x)
        { "exp2", 1, Functions::exp2, {}, {0.0, -1.0, 1.0, 700.0, -700.0, 1000.0}, {} },

        // expm1_safe(x)
        { "expm1_safe", 1, Functions::expm1_safe, {}, {0.0, 1e-6, -1e-6, 1.0, -1.0}, {} },

        // log(x)
        { "log", 1, Functions::log, {}, {0.0, 1.0, 2.5, 10.0}, {} },

        // log2(x)
        { "log2", 1, Functions::log2, {}, {0.0, 1.0, 2.5, 10.0}, {} },

        // log10(x)
        { "log10", 1, Functions::log10, {}, {0.0, 1.0, 2.5, 10.0}, {} },

        // log_a(x, a)
        { "log_a", 2, {}, [](Real x, Real a){ return Functions::log_a(x, a); }, {0.0, 1.0, 2.5, 10.0}, {0.5, 1.0, 2.0, 3.0} },

        // log1p(x)
        { "log1p", 1, Functions::log1p, {}, {-1.0, -0.5, 0.0, 1e-6, 0.5, 1.0}, {} }
    };
}

// ----------------- Тест -----------------
TEST(ExpLogUnified, DynamicTests) {
    auto funcs = getFunctions();

    for (auto &f : funcs) {
        SCOPED_TRACE(f.name); // чтобы видеть имя функции в отчёте

        if (f.nargs == 1) {
            for (auto x : f.x_vals) {
                Real r = f.f1(x);

                // проверка NaN
                if (f.name == "log" || f.name == "log2" || f.name == "log10" || f.name == "log1p") {
                    if (x <= (f.name=="log1p" ? -1.0 : 0.0)) {
                        EXPECT_TRUE(std::isnan(r));
                        continue;
                    }
                }

                // проверка экстремальных значений exp
                if (f.name == "exp" || f.name == "exp2") {
                    if (x > 700.0) {
                        EXPECT_EQ(r, std::numeric_limits<Real>::infinity());
                        continue;
                    }
                    if (x < -700.0) {
                        EXPECT_EQ(r, 0.0);
                        continue;
                    }
                }

                // сравнение с std:: функцией
                Real ref = 0.0;
                if (f.name == "exp") ref = std::exp(x);
                else if (f.name == "exp2") ref = std::exp2(x);
                else if (f.name == "expm1_safe") ref = std::expm1(x);
                else if (f.name == "log") ref = std::log(x);
                else if (f.name == "log2") ref = std::log2(x);
                else if (f.name == "log10") ref = std::log10(x);
                else if (f.name == "log1p") ref = std::log1p(x);

                if (f.name != "expm1_safe" && f.name.substr(0,3)!="log") {
                    EXPECT_NEAR(r, ref, EPS);
                } else if (f.name.substr(0,3)=="log") {
                    EXPECT_NEAR(r, ref, EPS);
                } else if (f.name == "expm1_safe") {
                    EXPECT_NEAR(r, ref, EPS);
                }
            }
        } else if (f.nargs == 2) {
            for (auto x : f.x_vals) {
                for (auto y : f.y_vals) {
                    Real r = f.f2(x, y);

                    // NaN для pow: отрицательное основание + дробный показатель
                    if (f.name == "pow" && x < 0.0 && std::floor(y) != y) {
                        EXPECT_TRUE(std::isnan(r));
                        continue;
                    }

                    // NaN для log_a: x<=0, a<=0, a==1
                    if (f.name == "log_a") {
                        if (x <= 0.0 || y <= 0.0 || y == 1.0) {
                            EXPECT_TRUE(std::isnan(r));
                            continue;
                        }
                    }

                    // сравнение с std:: pow / log_a
                    Real ref = 0.0;
                    if (f.name == "pow") ref = std::pow(x, y);
                    else if (f.name == "log_a") ref = std::log(x)/std::log(y);

                    if (std::isnan(ref)) {
                        EXPECT_TRUE(std::isnan(r));
                    } else {
                        EXPECT_NEAR(r, ref, EPS);
                    }
                }
            }
        }
    }
}
