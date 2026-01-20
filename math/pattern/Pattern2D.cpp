#include "Pattern2D.h"

namespace pattern2D {

    // -------------- Trigonometric / Geometry --------------
    double atan2(double y, double x) { return std::atan2(y, x); }
    double hypot(double x, double y) { return std::hypot(x, y); }

    // -------------- Complex / Fractal --------------
    std::complex<double> julia(std::complex<double> z, std::complex<double> c) {
        return z * z + c;
    }

    bool escapes(std::complex<double> z0, std::complex<double> c, int max_iter) {
        std::complex<double> z = z0;
        for (int i = 0; i < max_iter; ++i) {
            z = z * z + c;
            if (std::abs(z) > 2.0)
                return true;
        }
        return false;
    }

    std::complex<double> clog(const std::complex<double>& z) {
        return std::log(z);
    }

}
