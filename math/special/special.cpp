#include "Special.h"

namespace Special {

    // ---------------- Gamma functions ----------------
    double tgamma(double x) {
        return std::tgamma(x);
    }

    double lgamma(double x) {
        return std::lgamma(x);
    }

    double beta(double x, double y) {
        return std::tgamma(x) * std::tgamma(y) / std::tgamma(x + y);
    }

    // ---------------- Bessel functions ----------------
    double cyl_bessel_j(double nu, double x) {
        return boost::math::cyl_bessel_j(nu, x);
    }

    double cyl_neumann(double nu, double x) {
        return boost::math::cyl_neumann(nu, x);
    }

    double cyl_bessel_i(double nu, double x) {
        return boost::math::cyl_bessel_i(nu, x);
    }

    double cyl_bessel_k(double nu, double x) {
        return boost::math::cyl_bessel_k(nu, x);
    }

    // ---------------- Lambert W ----------------
    double lambert_w(double x) {
        return boost::math::lambert_w0(x);
    }

    // ---------------- Legendre functions ----------------
    double legendre(int l, double x) {
        return std::legendre(l, x);
    }

    double assoc_legendre(int l, int m, double x) {
        return std::assoc_legendre(l, m, x);
    }

    // ---------------- Error functions ----------------
    double erf(double x) {
        return std::erf(x);
    }

    double erfc(double x) {
        return std::erfc(x);
    }

    // ---------------- Zeta functions ----------------
    double riemann_zeta(double s) {
        return std::riemann_zeta(s);
    }

    double zeta(double s) {
        return boost::math::zeta(s);
    }

}
