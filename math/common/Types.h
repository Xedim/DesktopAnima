//Types.h
#pragma once
#include <complex>
#include <vector>
#include <functional>
#include <limits>
#include <variant>



// -------------------------
// Базовые типы
// -------------------------
using Real = double;
using RealPair = std::pair<Real, Real>;
using Complex = std::complex<Real>;
using VecReal = std::vector<Real>;
using VecComplex = std::vector<Complex>;

// -------------------------
// Общие утилиты
// -------------------------

inline Real NaN() noexcept {
    return std::numeric_limits<Real>::quiet_NaN();
}
// -------------------------
// Общие POD-структуры
// -------------------------

struct Quartiles { Real q1, q2, q3; };

struct LinearRegressionResult {
    Real slope = NaN();
    Real intercept = NaN();
    Real r2 = NaN();
};

// -------------------------
// Категории функций
// -------------------------

enum class PatternKind {
    Algebra,
    Power,
    Logarithmic,
    Trigonometric,
    Hyperbolic,
    Hybrid,
    Special,
    Generalized,
    Numerical,
    Fractal,
    Iteration,
    Statistical,
    Distributional,
    Information,
    TimeSeries,
    Sampling,
    Regression,
    Outliers,
};

// -------------------------
// Политики стабильности
// -------------------------

enum class StabPolicy {
    Raw,
    Clamp,
    Reject
};
// -------------------------
// Интервалы
// -------------------------

struct Interval {
    Real min;
    Real max;

    constexpr Interval(Real a, Real b) : min(a), max(b) {}
    [[nodiscard]] constexpr bool contains(Real x) const { return x >= min && x <= max; }
};

// -------------------------
// Типы аргументов
// -------------------------

enum class ArgType {
    Real,
    Int,
    VecReal,
    Complex,
    VecComplex
};

// -------------------------
// Групповое разнообразие функций
// -------------------------

enum class FunctionGroup {
    UnaryAlgebra,           // sqrt, cbrt, abs, sign
    BinaryAlgebra,          // pow, mod, x_pow_y
    Trigonometric,          // sin, cos, tan ...
    Hyperbolic,             // sinh, cosh ...
    Logarithmic,            // log, log2, log10, log_a
    Exponential,            // exp, exp2, expm1_safe
    Statistical,            // sum, mean, median ...
    Fractal,                // weierstrass, cantor ...
    Special,                // erf, gamma, legendre ...
    ProbabilityDist,        // Normal, Beta, etc
    Regression,             // linear, polynomial ...
    TimeSeries,             // rolling_mean, ema ...
    OutlierDetection,       // z_score, grubbs_test ...
    Information,            // entropy, cross_entropy ...
    CharacteristicFunction, // normal_characteristic ...
    Mixed                   // гибридные или нестандартные сигнатуры
};

// -------------------------
// Метаданные функций
// -------------------------

struct PatternDescriptor {
    const char* name;
    Interval domain;
    Interval range;
    PatternKind kind;
};

// -------------------------
// Типы аргументов движка
// -------------------------

using ArgVariant = std::variant<
    Real,
    int,
    VecReal,
    Complex,
    VecComplex,
    StabPolicy
>;

using ResultVariant = std::variant<
    std::monostate,
    Real, bool, VecReal, Complex,
    RealPair, LinearRegressionResult, Quartiles
>;
