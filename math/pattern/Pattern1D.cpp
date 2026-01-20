#include "Pattern1D.h"

namespace pattern1D {

    // -------------- Algebraic --------------
    double algebraic_root(double x, const std::vector<double>& coefficients) {
        double result = 0.0;
        for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
            result = result * x + *it;
        return std::sqrt(result);
    }

    double sqrt(double x) { return std::sqrt(x); }
    double cbrt(double x) { return std::cbrt(x); }

    // -------------- Power / Exponential --------------
    double pow(double x, double alpha) { return std::pow(x, alpha); }
    double abs(double x) { return std::abs(x); }
    double exp(double x) { return std::exp(x); }

    // -------------- Logarithmic --------------
    double log(double x) { return std::log(x); }
    double log10(double x) { return std::log10(x); }
    double log_a(double x, double a) { return std::log(x) / std::log(a); }

    // -------------- Trigonometric --------------
    double sin(double x) { return std::sin(x); }
    double cos(double x) { return std::cos(x); }
    double tan(double x) { return std::tan(x); }
    double cot(double x) { return 1.0 / std::tan(x); }

    double asin(double x) { return std::asin(x); }
    double acos(double x) { return std::acos(x); }
    double atan(double x) { return std::atan(x); }

    // -------------- Hyperbolic --------------
    double sinh(double x) { return std::sinh(x); }
    double cosh(double x) { return std::cosh(x); }
    double tanh(double x) { return std::tanh(x); }
    double coth(double x) { return 1.0 / std::tanh(x); }

    double asinh(double x) { return std::asinh(x); }
    double acosh(double x) { return std::acosh(x); }
    double atanh(double x) { return std::atanh(x); }

    // -------------- Hybrid --------------
    double x_pow_y(double x, double y) { return std::exp(y * std::log(x)); }

    // -------------- Generalised / Pattern --------------
    double dirac_delta(double x, double eps) {
        return std::exp(-x * x / eps) / std::sqrt(M_PI * eps);
    }

}

