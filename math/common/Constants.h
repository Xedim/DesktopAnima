//Constants.h
#pragma once
#include "Types.h"

namespace Constants {

    // ---------- General ----------
    constexpr Real PI = 3.14159265358979323846;
    constexpr Real SQRT2  = 1.41421356237309504880;
    constexpr Real SQRT_2PI = 2.50662827463100050242;
    constexpr Real E  = 2.71828182845904523536;

    constexpr Real H = 1e-5;                      // малое число для производных
    constexpr Real EPS = 1e-9;                    // общая точность сравнения чисел
    constexpr Real SMALL = 1e-6;                  // малое число для приближений

    inline constexpr Real POS_LIMIT = std::numeric_limits<Real>::max();
    inline constexpr Real NEG_LIMIT = -std::numeric_limits<Real>::max();

    // ---------- Weierstrass ----------
    constexpr Real WEIERSTRASS_AMP_MIN = 0.0;
    constexpr Real WEIERSTRASS_AMP_MAX = 1.0;
    constexpr Real WEIERSTRASS_FREQ_MIN = 1.0;
    constexpr int WEIERSTRASS_ITER_MIN = 1;

    constexpr int WEIERSTRASS_ITER = 50;          // число членов суммы N
    constexpr Real WEIERSTRASS_AMP_COEF = 0.5;    // коэффициент амплитуды a
    constexpr Real WEIERSTRASS_FREQ_COEF = 3.0;   // коэффициент частоты b
    constexpr Real WEIERSTRASS_X_MIN     = 0.0;
    constexpr Real WEIERSTRASS_X_MAX     = 1.0;

    // ---------- Cantor Set ----------
    constexpr Real CANTOR_X_MIN = 0.0;
    constexpr Real CANTOR_X_MAX = 1.0;
    constexpr int CANTOR_ITER_MIN = 1;

    constexpr Real CANTOR_LEFT = 1.0 / 3.0;       // левая граница интервала
    constexpr Real CANTOR_RIGHT = 2.0 / 3.0;      // правая граница интервала
    constexpr Real CANTOR_MID = 0.5;              // средняя точка сегмента
    constexpr Real CANTOR_SCALE = 3.0;            // масштабирование x при переходе в левый сегмент
    constexpr Real CANTOR_RIGHT_SCALE = 2.0;      // сдвиг для правого сегмента: x = 3*x - 2
    constexpr Real CANTOR_FACTOR = 0.5;           // уменьшение "веса" на каждом уровне
    constexpr int CANTOR_ITER = 32;               // число итераций

    // ---------- Tent Map ----------
    constexpr Real TENT_PEAK  = 0.5;              // координата пика
    constexpr Real TENT_SLOPE = 2.0;              // масштаб наклона линий
    constexpr Real TENT_X_MIN = 0.0;
    constexpr Real TENT_X_MAX = 1.0;

    // ---------- Logistic Map ----------
    constexpr Real LOGISTIC_X_MIN = 0.0;
    constexpr Real LOGISTIC_X_MAX = 1.0;
    constexpr Real LOGISTIC_R_MIN = 0.0;
    constexpr Real LOGISTIC_R_MAX = 4.0;

    constexpr Real LOGISTIC_R_DEFAULT = 3.9;      // стандартное значение r для хаотического режима

    // ---------- Escapes Map ----------
    constexpr Real ESC_THRESHOLD = 2.0;

    // ---------- Iterations / General ----------
    constexpr int MAP_ITER = 100;                   // стандартное число итераций для логистической карты
    constexpr int JULIA_ITER = 100;                 // стандартное число итераций для множества Жюли
    constexpr int L2_NORM_ITER = 10000;             // количество шагов для интегральной нормы

    // ---------- Output ranges ----------
    //constexpr Real WEIERSTRASS_Y_MAX = Functions::geometric_sum(WEIERSTRASS_AMP_COEF, WEIERSTRASS_ITER);
    constexpr Real WEIERSTRASS_Y_MAX = 100;
    constexpr Real WEIERSTRASS_Y_MIN = -WEIERSTRASS_Y_MAX;

    constexpr Real CANTOR_Y_MIN = 0.0;
    constexpr Real CANTOR_Y_MAX = 1.0;

    constexpr Real LOGISTIC_Y_MIN = 0.0;
    constexpr Real LOGISTIC_Y_MAX = LOGISTIC_R_MAX / 4.0;

    constexpr Real TENT_Y_MIN = 0.0;
    constexpr Real TENT_Y_MAX = TENT_PEAK * TENT_SLOPE;
}
