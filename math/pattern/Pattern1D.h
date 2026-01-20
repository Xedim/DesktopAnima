#pragma once
#include <vector>
#include "cmath"

namespace Pattern1D {
    double polynomial(double x,
                      const std::vector<double>& coefficients);

    double rational(double x,
                    const std::vector<double>& p,
                    const std::vector<double>& q);

    double algebraic_root(double x, const std::vector<double>& coefficients);

    double weierstrass(double x, double a, double b, int N);

    double cantor(double x, int max_iter);

    double tent(double x);

    double heaviside(double x);
    double smooth_heaviside(double x, double eps);

    double sqrt(double x);
    double cbrt(double x);

    double pow(double x, double alpha);
    double abs(double x);
    double exp(double x);

    double log(double x);
    double log10(double x);
    double log_a(double x, double a);

    double sin(double x);
    double cos(double x);
    double tan(double x);
    double cot(double x);

    double asin(double x);
    double acos(double x);
    double atan(double x);

    double sinh(double x);
    double cosh(double x);
    double tanh(double x);
    double coth(double x);

    double asinh(double x);
    double acosh(double x);
    double atanh(double x);

    double x_pow_y(double x, double y);
    double dirac_delta(double x, double eps = 1e-3);

}