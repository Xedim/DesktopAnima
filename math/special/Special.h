#pragma once
#include <cmath>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/lambert_w.hpp>

namespace Special {

    // Gamma functions
    double tgamma(double x);
    double lgamma(double x);
    double beta(double x, double y);

    // Bessel functions
    double cyl_bessel_j(double nu, double x);
    double cyl_neumann(double nu, double x);
    double cyl_bessel_i(double nu, double x);
    double cyl_bessel_k(double nu, double x);

    // Lambert W
    double lambert_w(double x);

    // Legendre functions
    double legendre(int l, double x);
    double assoc_legendre(int l, int m, double x);

    // Error functions
    double erf(double x);
    double erfc(double x);

    // Zeta functions
    double riemann_zeta(double s);
    double zeta(double s); // boost::math::zeta alias

}
