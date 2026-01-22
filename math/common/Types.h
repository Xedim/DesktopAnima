//Types.h
#pragma once
#include <complex>
#include <vector>
#include <functional>
#include <limits>
#include <variant>


using Real = double;
using RealPair = std::pair<Real, Real>;
using Complex = std::complex<Real>;
using VecReal = std::vector<Real>;
using VecComplex = std::vector<Complex>;

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
    Real slope;
    Real intercept;
    Real r2;
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
// Общие константы
// -------------------------

inline Real NaN() noexcept {
    return std::numeric_limits<Real>::quiet_NaN();
}

// -------------------------
// Интервалы
// -------------------------

struct Interval {
    Real min;
    Real max;

    constexpr bool contains(Real x) const {
        return x >= min && x <= max;
    }
};

// -------------------------
// Сигнатуры функций
// -------------------------

enum class PatternSignature {
    UnaryReal,   // f(x)
    VecReal,     // f(vec)
    IntInt       // f(int,int)
    // В дальнейшем можно добавить: Complex, VecComplex, Mixed
};

// -------------------------
// Основные категории функций
// -------------------------

enum class PatternKind {
    Algebra,
    Trigonometric,
    Logarithmic,
    Power,
    Hyperbolic,
    Statistical,
    Fractal,
    HybridNumerical,
    Special,
    Probability,
    Regression,
    TimeSeries,
    OutlierDetection,
    Information,
    CharacteristicFunction
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
    PatternSignature signature;
    PatternKind kind;
    FunctionGroup group;
    Interval domain;
    Interval range;
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

using RealFn            = Func<Real, Real>;
using IntFn             = Func<Real, Real>;
using VecFn             = Func<Real, const VecReal&>;

using UnaryFns = std::variant<
    RealFn,
    IntFn,
    VecFn
>;

using RealRealFn        = Func<Real, Real,            Real>;
using RealIntFn         = Func<Real, Real,            int>;
using RealVecFn         = Func<Real, Real,            const VecReal&>;
using IntRealFn         = Func<Real, int,             Real>;
using IntIntFn          = Func<Real, int,             int>;
using IntVecFn          = Func<Real, int,             const VecReal&>;
using VecRealFn         = Func<Real, const VecReal&,  Real>;
using VecIntFn          = Func<Real, const VecReal&,  int>;
using VecVecFn          = Func<Real, const VecReal&,  const VecReal&>;

using BinaryFns = std::variant<
    RealRealFn,
    RealIntFn,
    RealVecFn,
    IntRealFn,
    IntIntFn,
    IntVecFn,
    VecRealFn,
    VecIntFn,
    VecVecFn
    >;

using RealRealRealFn    = Func<Real, Real,            Real,            Real>;
using RealRealIntFn     = Func<Real, Real,            Real,            int>;
using RealRealVecFn     = Func<Real, Real,            Real,            const VecReal&>;
using RealIntRealFn     = Func<Real, Real,            int,             Real>;
using RealIntIntFn      = Func<Real, Real,            int,             int>;
using RealIntVecFn      = Func<Real, Real,            int,             const VecReal&>;
using RealVecRealFn     = Func<Real, Real,            const VecReal&,  Real>;
using RealVecIntFn      = Func<Real, Real,            const VecReal&,  int>;
using RealVecVecFn      = Func<Real, Real,            const VecReal&,  const VecReal&>;
using IntRealRealFn     = Func<Real, int,             Real,            Real>;
using IntRealIntFn      = Func<Real, int,             Real,            int>;
using IntRealVecFn      = Func<Real, int,             Real,            const VecReal&>;
using IntIntRealFn      = Func<Real, int,             int,             Real>;
using IntIntIntFn       = Func<Real, int,             int,             int>;
using IntIntVecFn       = Func<Real, int,             int,             const VecReal&>;
using IntVecRealFn      = Func<Real, int,             const VecReal&,  Real>;
using IntVecIntFn       = Func<Real, int,             const VecReal&,  int>;
using IntVecVecFn       = Func<Real, int,             const VecReal&,  const VecReal&>;
using VecRealRealFn     = Func<Real, const VecReal&,  Real,            Real>;
using VecRealIntFn      = Func<Real, const VecReal&,  Real,            int>;
using VecRealVecFn      = Func<Real, const VecReal&,  Real,            const VecReal&>;
using VecIntRealFn      = Func<Real, const VecReal&,  int,             Real>;
using VecIntIntFn      = Func<Real, const VecReal&,  int,             int>;
using VecIntVecFn       = Func<Real, const VecReal&,  int,             const VecReal&>;
using VecVecRealFn      = Func<Real, const VecReal&,  const VecReal&,  Real>;
using VecVecIntFn       = Func<Real, const VecReal&,  const VecReal&,  int>;
using VecVecVecFn       = Func<Real, const VecReal&,  const VecReal&,  const VecReal&>;

using TernaryFns = std::variant<
    RealRealRealFn,
    RealRealIntFn,
    RealRealVecFn,
    RealIntRealFn,
    RealIntIntFn,
    RealIntVecFn,
    RealVecRealFn,
    RealVecIntFn,
    RealVecVecFn,
    IntRealRealFn,
    IntRealIntFn,
    IntRealVecFn,
    IntIntRealFn,
    IntIntIntFn,
    IntIntVecFn,
    IntVecRealFn,
    IntVecIntFn,
    IntVecVecFn,
    VecRealRealFn,
    VecRealIntFn,
    VecRealVecFn,
    VecIntRealFn,
    VecIntIntFn,
    VecIntVecFn,
    VecVecRealFn,
    VecVecIntFn,
    VecVecVecFn
    >;

using ComplexComplexFn  = Func<Complex, Complex, Complex>;
using ComplexComplexRealRealFn = Func<bool, Complex, Complex, Real, Real>;
using Complex_3RealFn = Func<bool, Real, Real, Real>;
using Complex_VecRealFn = Func<bool, const VecReal&, Real>;

using ComplexFns = std::variant<
    ComplexComplexFn,
    ComplexComplexRealRealFn,
    Complex_3RealFn,
    Complex_VecRealFn
    >;

using RealRealRealIntPolicyFn = Func<Real, Real, Real, Real, int, StabPolicy>;
using RealIntPolicyFn = Func<Real, Real, int, StabPolicy>;
using RealRealPolicyFn = Func<Real, Real, Real, StabPolicy>;
using RealPolicyFn = Func<Real, Real, StabPolicy>;
using RealRealIntPolicyFn = Func<Real, Real, Real, int, StabPolicy>;

using PolicyFns = std::variant<
    RealRealRealIntPolicyFn,
    RealIntPolicyFn,
    RealRealPolicyFn,
    RealPolicyFn,
    RealRealIntPolicyFn
>;

using Vec_VecRealSizeTFn = Func<VecReal, const VecReal&, std::size_t>;
using Vec_VecRealFn = Func<VecReal, const VecReal&, Real>;
using Vec_VecIntFn = Func<VecReal, const VecReal&, int>;
using Vec_VecFn = Func<VecReal, const VecReal&>;
using Vec_IntIntFn = Func<VecReal, int, int>;
using Vec_VecVecIntFn = Func<VecReal, const VecReal&, const VecReal&, int>;

using VectorFns = std::variant<
    Vec_VecRealSizeTFn,
    Vec_VecRealFn,
    Vec_VecIntFn,
    Vec_VecFn,
    Vec_IntIntFn,
    Vec_VecVecIntFn
>;

using bool_VecRealFn = Func<bool, const VecReal&, Real>;
using bool_VecFn = Func<bool, const VecReal&>;
using Bool4RealFn = Func<bool, Real, Real, Real, Real>;

using BoolFns = std::variant<
    bool_VecRealFn,
    bool_VecFn,
    Bool4RealFn
>;

using Pair_VecRealFn = Func<RealPair, const VecReal&, Real>;
using LR_VecVecFn = Func<LinearRegressionResult, const VecReal&, const VecReal&>;
using Quartiles_VecFn = Func<Quartiles, const VecReal&>;

using SpecialFns = std::variant<
    Pair_VecRealFn,
    LR_VecVecFn,
    Quartiles_VecFn
>;

template<typename Dist>
using DistRealFnT = Func<Real, const Dist&, Real>;

template<typename Dist>
using DistVecRealFnT = Func<Real, const Dist&, const VecReal&>;

using DistFns = std::variant<Normal, LogNormal, Exponential, Gamma, Beta, Weibull, Cauchy, StudentT>;
