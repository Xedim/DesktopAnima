//Constants.h
#pragma once
#include "Types.h"

namespace Constants {

    // ---------- General ----------
    constexpr double PI = 3.14159265358979323846;
    constexpr double E  = 2.71828182845904523536;

    constexpr double H = 1e-5;                      // малое число для производных
    constexpr double EPS = 1e-9;                    // общая точность сравнения чисел
    constexpr double SMALL = 1e-6;                  // малое число для приближений

    // ---------- Weierstrass ----------
    constexpr double WEIERSTRASS_AMP_MIN = 0.0;
    constexpr double WEIERSTRASS_AMP_MAX = 1.0;
    constexpr double WEIERSTRASS_FREQ_MIN = 1.0;
    constexpr int WEIERSTRASS_ITER_MIN = 1;

    constexpr int WEIERSTRASS_ITER = 50;            // число членов суммы N
    constexpr double WEIERSTRASS_AMP_COEF = 0.5;    // коэффициент амплитуды a
    constexpr double WEIERSTRASS_FREQ_COEF = 3.0;   // коэффициент частоты b
    constexpr double WEIERSTRASS_X_MIN     = 0.0;
    constexpr double WEIERSTRASS_X_MAX     = 1.0;

    // ---------- Cantor Set ----------
    constexpr double CANTOR_X_MIN = 0.0;
    constexpr double CANTOR_X_MAX = 1.0;
    constexpr int CANTOR_ITER_MIN = 1;

    constexpr double CANTOR_LEFT = 1.0 / 3.0;       // левая граница интервала
    constexpr double CANTOR_RIGHT = 2.0 / 3.0;      // правая граница интервала
    constexpr double CANTOR_MID = 0.5;              // средняя точка сегмента
    constexpr double CANTOR_SCALE = 3.0;            // масштабирование x при переходе в левый сегмент
    constexpr double CANTOR_RIGHT_SCALE = 2.0;      // сдвиг для правого сегмента: x = 3*x - 2
    constexpr double CANTOR_FACTOR = 0.5;           // уменьшение "веса" на каждом уровне
    constexpr int CANTOR_ITER = 32;                 // число итераций

    // ---------- Tent Map ----------
    constexpr double TENT_PEAK  = 0.5;              // координата пика
    constexpr double TENT_SLOPE = 2.0;              // масштаб наклона линий
    constexpr double TENT_X_MIN = 0.0;
    constexpr double TENT_X_MAX = 1.0;

    // ---------- Logistic Map ----------
    constexpr double LOGISTIC_X_MIN = 0.0;
    constexpr double LOGISTIC_X_MAX = 1.0;
    constexpr double LOGISTIC_R_MIN = 0.0;
    constexpr double LOGISTIC_R_MAX = 4.0;

    constexpr double LOGISTIC_R_DEFAULT = 3.9;      // стандартное значение r для хаотического режима

    // ---------- Iterations / General ----------
    constexpr int MAP_ITER = 100;                   // стандартное число итераций для логистической карты
    constexpr int JULIA_ITER = 100;                 // стандартное число итераций для множества Жюли
    constexpr int L2_NORM_ITER = 10000;             // количество шагов для интегральной нормы

    // ---------- Output ranges ----------
    constexpr double WEIERSTRASS_Y_MAX = Utils::geometric_sum(WEIERSTRASS_AMP_COEF, WEIERSTRASS_ITER);
    constexpr double WEIERSTRASS_Y_MIN = -WEIERSTRASS_Y_MAX;

    constexpr double CANTOR_Y_MIN = 0.0;
    constexpr double CANTOR_Y_MAX = 1.0;

    constexpr double LOGISTIC_Y_MIN = 0.0;
    constexpr double LOGISTIC_Y_MAX = LOGISTIC_R_MAX / 4.0;

    constexpr double TENT_Y_MIN = 0.0;
    constexpr double TENT_Y_MAX = TENT_PEAK * TENT_SLOPE;
}
