//Types.h
#pragma once
#include <complex>
#include <vector>
#include <limits>
#include <variant>
#include "../helpers/Constants.h"
#include <boost/math/distributions/gamma.hpp>

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


struct DistNormal {
    Real mu;
    Real sigma;

    // --- cached ---
    Real inv_sigma;
    Real inv_sigma_sqrt2;
    Real inv_sigma_sqrt2pi;
    Real log_norm;

    explicit DistNormal(Real mu_, Real sigma_)
        : mu(mu_), sigma(sigma_) {

        if (sigma_ > 0) {
            inv_sigma = Real{1} / sigma_;
            inv_sigma_sqrt2 = inv_sigma / Constants::SQRT2;
            inv_sigma_sqrt2pi = inv_sigma / Constants::SQRT_2PI;
            log_norm = std::log(sigma_ * Constants::SQRT_2PI);
        } else {
            inv_sigma = inv_sigma_sqrt2 = inv_sigma_sqrt2pi = log_norm = NaN();
        }
    }
};

struct DistLogNormal {
    Real mu;
    Real sigma;

    // cached
    Real inv_sigma;
    Real log_norm;

    explicit DistLogNormal(Real mu_, Real sigma_)
        : mu(mu_), sigma(sigma_) {

        if (sigma_ > 0) {
            inv_sigma = Real{1} / sigma_;
            log_norm = std::log(sigma_ * Constants::SQRT_2PI);
        } else {
            inv_sigma = log_norm = NaN();
        }
    }
};

struct DistExp {
    Real lambda;

    explicit DistExp(Real lambda_)
        : lambda(lambda_) {}
};

struct DistGamma {
    Real k;       // shape
    Real theta;   // scale

    // cached
    Real inv_theta;
    Real log_theta;
    Real log_gamma_k;

    explicit DistGamma(Real k_, Real theta_)
        : k(k_), theta(theta_) {

        if (k_ > 0 && theta_ > 0) {
            inv_theta = Real{1} / theta_;
            log_theta = std::log(theta_);
            log_gamma_k = boost::math::lgamma(k_);
        } else {
            inv_theta = log_theta = log_gamma_k = NaN();
        }
    }
};

struct DistBeta {
    Real alpha;
    Real beta;

    // cached
    Real log_beta_fn;   // log B(alpha, beta)

    explicit DistBeta(Real alpha_, Real beta_)
        : alpha(alpha_), beta(beta_) {

        if (alpha_ > 0 && beta_ > 0) {
            log_beta_fn =
                boost::math::lgamma(alpha_)
              + boost::math::lgamma(beta_)
              - boost::math::lgamma(alpha_ + beta_);
        } else {
            log_beta_fn = NaN();
        }
    }
};

struct DistWeibull {
    Real k;
    Real lambda;

    Real inv_lambda;
    Real log_lambda;

    explicit DistWeibull(Real k_, Real lambda_)
        : k(k_), lambda(lambda_) {

        inv_lambda = Real{1} / lambda_;
        log_lambda = std::log(lambda_);
    }
};

struct DistCauchy {
    Real x0;
    Real gamma;

    Real inv_gamma;
    Real log_norm;

    explicit DistCauchy(Real x0_, Real gamma_)
        : x0(x0_), gamma(gamma_) {

        inv_gamma = Real{1} / gamma_;
        log_norm = std::log(Constants::PI * gamma_);
    }
};

struct DistStudentT {
    Real nu;

    Real log_norm;

    explicit DistStudentT(Real nu_) : nu(nu_) {
        log_norm =
            boost::math::lgamma((nu + 1) / 2) -
            boost::math::lgamma(nu / 2) -
            Real{0.5} * std::log(nu * Constants::PI);
    }
};

// -------------------------
// Категории функций
// -------------------------

enum class PatternKind {
    Sign,
    Algebra,
    Power,
    ExpLog,
    Trigonometric,
    Hyperbolic,
    Special,
    Generalized,
    Numerical,
    Fractal,
    Iteration,
    Statistical,
    Distributional,
    StatTest,
    Characteristical,
    Information,
    TimeSeries,
    Resampling,
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

enum class ArgType {
    Real,
    Int,
    VecReal,
    Complex,
    VecComplex,
    DistNormal,
    DistLogNormal,
    DistExp,
    DistGamma,
    DistBeta,
    DistWeibull,
    DistCauchy,
    DistStudentT
};

using ArgVariant = std::variant<
    Real,
    RealPair,
    int,
    bool,
    VecReal,
    Complex,
    VecComplex,
    StabPolicy
>;

using ResultVariant = std::variant<
    std::monostate,
    Real,
    RealPair,
    int,
    bool,
    VecReal,
    Complex,
    LinearRegressionResult,
    Quartiles
>;
