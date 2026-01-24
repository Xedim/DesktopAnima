//Types.h
#pragma once
#include <complex>
#include <vector>
#include <functional>
#include <limits>
#include <variant>
#include <random>


using Real = double;
using RealPair = std::pair<Real, Real>;
using Complex = std::complex<Real>;
using VecReal = std::vector<Real>;
using VecComplex = std::vector<Complex>;

using Function1D = std::function<Real(Real)>;
using Function2D = std::function<Real(Real, Real)>;

// -------------------------
// Генератор случайных чисел
// -------------------------
std::random_device inline rd;
std::mt19937 inline rng(rd());
std::uniform_real_distribution<Real> inline dist_real(-100.0, 100.0);
std::uniform_int_distribution<int> inline dist_int(0, 20);

// -------------------------
// Общие конструкты
// -------------------------

inline Real NaN() noexcept {
    return std::numeric_limits<Real>::quiet_NaN();
}

struct WeierState {
    Real sum;
    Real amp;
    Real freq;
};

struct CantorState {
    Real x;
    Real result;
    Real scale;
};

struct Normal {
    Real mu;
    Real sigma;
};

struct LogNormal {
    Real mu;
    Real sigma;
};

struct Exponential {
    Real lambda;
};

struct Gamma {
    Real k;
    Real theta;
};

struct Beta {
    Real alpha;
    Real beta;
};

struct Weibull {
    Real k;
    Real lambda;
};

struct Cauchy {
    Real x0;
    Real gamma;
};

struct StudentT {
    Real nu;
};

struct Quartiles { Real q1, q2, q3; };

struct LinearRegressionResult {
    Real slope = NaN();
    Real intercept = NaN();
    Real r2 = NaN();
};

// -------------------------
// Политики стабильности
// -------------------------

enum class StabPolicy {
    Raw,    // без ограничений
    Clamp,  // проекция в допустимый диапазон
    Reject  // NaN при выходе за диапазон
};

// -------------------------
// Статистические функции
// -------------------------

using PDF = std::function<Real(Real)>;
using CDF = std::function<Real(Real)>;

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
// Основные категории функций
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
// Возможный расширенный интерфейс для descriptor
// -------------------------

struct PatternDescriptor {
    const char* name;
    Interval domain;
    Interval range;
    PatternKind kind;
};

// -------------------------
// Функционные алиасы
// -------------------------

using ArgVariant = std::variant<
    std::monostate,
    std::size_t,
    Real,
    int,
    VecReal,
    RealPair,
    Complex,
    StabPolicy,
    LinearRegressionResult,
    Quartiles
    >;

template<typename R, typename... Args>
using Func = std::function<R(Args...)>;

template<typename... Args>
using ArgsTuple = std::tuple<Args...>;

template<typename R, typename Tuple>
struct WrapFn;

template<typename R, typename... Args>
struct WrapFn<R, std::tuple<Args...>> {
    using result_type = R;
    using args_tuple  = std::tuple<Args...>;

    std::function<R(Args...)> fn;
};

//=============================================

using UnaryFns = std::variant<
    WrapFn<Real, ArgsTuple<Real>>,
    WrapFn<Real, ArgsTuple<int>>,
    WrapFn<Real, ArgsTuple<VecReal>>
>;

using BinaryFns = std::variant<
    WrapFn<Real, ArgsTuple<Real, Real>>,
    WrapFn<Real, ArgsTuple<Real, int>>,
    WrapFn<Real, ArgsTuple<Real, VecReal>>,

    WrapFn<Real, ArgsTuple<int, Real>>,
    WrapFn<Real, ArgsTuple<int, int>>,
    WrapFn<Real, ArgsTuple<int, VecReal>>,

    WrapFn<Real, ArgsTuple<VecReal, Real>>,
    WrapFn<Real, ArgsTuple<VecReal, int>>,
    WrapFn<Real, ArgsTuple<VecReal, VecReal>>
>;

using TernaryFns = std::variant<
    WrapFn<Real, ArgsTuple<Real, Real, Real>>,
    WrapFn<Real, ArgsTuple<Real, Real, int>>,
    WrapFn<Real, ArgsTuple<Real, Real, VecReal>>,

    WrapFn<Real, ArgsTuple<Real, int, Real>>,
    WrapFn<Real, ArgsTuple<Real, int, int>>,
    WrapFn<Real, ArgsTuple<Real, int, VecReal>>,

    WrapFn<Real, ArgsTuple<Real, VecReal, Real>>,
    WrapFn<Real, ArgsTuple<Real, VecReal, int>>,
    WrapFn<Real, ArgsTuple<Real, VecReal, VecReal>>,

    WrapFn<Real, ArgsTuple<int, Real, Real>>,
    WrapFn<Real, ArgsTuple<int, Real, int>>,
    WrapFn<Real, ArgsTuple<int, Real, VecReal>>,

    WrapFn<Real, ArgsTuple<int, int, Real>>,
    WrapFn<Real, ArgsTuple<int, int, int>>,
    WrapFn<Real, ArgsTuple<int, int, VecReal>>,

    WrapFn<Real, ArgsTuple<int, VecReal, Real>>,
    WrapFn<Real, ArgsTuple<int, VecReal, int>>,
    WrapFn<Real, ArgsTuple<int, VecReal, VecReal>>,

    WrapFn<Real, ArgsTuple<VecReal, Real, Real>>,
    WrapFn<Real, ArgsTuple<VecReal, Real, int>>,
    WrapFn<Real, ArgsTuple<VecReal, Real, VecReal>>,

    WrapFn<Real, ArgsTuple<VecReal, int, Real>>,
    WrapFn<Real, ArgsTuple<VecReal, int, int>>,
    WrapFn<Real, ArgsTuple<VecReal, int, VecReal>>,

    WrapFn<Real, ArgsTuple<VecReal, VecReal, Real>>,
    WrapFn<Real, ArgsTuple<VecReal, VecReal, int>>,
    WrapFn<Real, ArgsTuple<VecReal, VecReal, VecReal>>
>;

using ComplexFns = std::variant<
    WrapFn<Complex, ArgsTuple<Complex, Complex>>,
    WrapFn<bool, ArgsTuple<Complex, Complex, Real, Real>>,
    WrapFn<bool, ArgsTuple<Real, Real, Real>>,
    WrapFn<bool, ArgsTuple<VecReal, Real>>
>;

using PolicyFns = std::variant<
    WrapFn<Real, ArgsTuple<Real, Real, Real, Real, int, StabPolicy>>,
    WrapFn<Real, ArgsTuple<Real, Real, int, StabPolicy>>,
    WrapFn<Real, ArgsTuple<Real, Real, Real, StabPolicy>>,
    WrapFn<Real, ArgsTuple<Real, StabPolicy>>,
    WrapFn<Real, ArgsTuple<Real, Real, int, StabPolicy>>
>;

using VectorFns = std::variant<
    WrapFn<VecReal, ArgsTuple<const VecReal&, std::size_t>>,
    WrapFn<VecReal, ArgsTuple<const VecReal&, Real>>,
    WrapFn<VecReal, ArgsTuple<const VecReal&, int>>,
    WrapFn<VecReal, ArgsTuple<const VecReal&>>,
    WrapFn<VecReal, ArgsTuple<int, int>>,
    WrapFn<VecReal, ArgsTuple<const VecReal&, const VecReal&, int>>
>;

using BoolFns = std::variant<
    WrapFn<bool, ArgsTuple<const VecReal&, Real>>,
    WrapFn<bool, ArgsTuple<const VecReal&>>,
    WrapFn<bool, ArgsTuple<Real, Real, Real, Real>>
>;

using SpecialFns = std::variant<
    WrapFn<RealPair, ArgsTuple<const VecReal&, Real>>,
    WrapFn<LinearRegressionResult, ArgsTuple<const VecReal&, const VecReal&>>,
    WrapFn<Quartiles, ArgsTuple<const VecReal&>>
>;

using DistFns = std::variant<
    WrapFn<Real, ArgsTuple<const Normal&, Real>>,
    WrapFn<Real, ArgsTuple<const LogNormal&, Real>>,
    WrapFn<Real, ArgsTuple<const Exponential&, Real>>,
    WrapFn<Real, ArgsTuple<const Gamma&, Real>>,
    WrapFn<Real, ArgsTuple<const Beta&, Real>>,
    WrapFn<Real, ArgsTuple<const Weibull&, Real>>,
    WrapFn<Real, ArgsTuple<const Cauchy&, Real>>,
    WrapFn<Real, ArgsTuple<const StudentT&, Real>>,

    WrapFn<Real, ArgsTuple<const Normal&, const VecReal&>>,
    WrapFn<Real, ArgsTuple<const LogNormal&, const VecReal&>>,
    WrapFn<Real, ArgsTuple<const Exponential&, const VecReal&>>,
    WrapFn<Real, ArgsTuple<const Gamma&, const VecReal&>>,
    WrapFn<Real, ArgsTuple<const Beta&, const VecReal&>>,
    WrapFn<Real, ArgsTuple<const Weibull&, const VecReal&>>,
    WrapFn<Real, ArgsTuple<const Cauchy&, const VecReal&>>,
    WrapFn<Real, ArgsTuple<const StudentT&, const VecReal&>>
>;

using AnyFnVariant = std::variant<
    UnaryFns, BinaryFns, TernaryFns, ComplexFns,
    VectorFns, BoolFns, SpecialFns, DistFns, PolicyFns
>;

using ResultVariant = std::variant<
    std::monostate, Real, bool, VecReal, Complex,
    RealPair, LinearRegressionResult, Quartiles
>;
