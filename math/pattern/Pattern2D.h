#pragma once
#include <complex>
#include <cmath>

namespace pattern2D {

    // -------------- Trigonometric / Geometry --------------
    double atan2(double y, double x);
    double hypot(double x, double y);

    // -------------- Complex / Fractal --------------
    std::complex<double> julia(std::complex<double> z, std::complex<double> c);
    bool escapes(std::complex<double> z0, std::complex<double> c, int max_iter = 100);
    std::complex<double> clog(const std::complex<double>& z);

}