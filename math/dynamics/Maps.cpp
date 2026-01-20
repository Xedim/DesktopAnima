#include "Maps.h"

namespace Maps {

    // -------------- 1D Maps / Fractals --------------
    inline Real logistic(Real x, Real r) {
        return r * x * (1.0 - x);
    }

    Real iterate(Real x0, Real r, int n) {
        Real x = x0;
        for (int i = 0; i < n; ++i)
            x = logistic(x, r);
        return x;
    }

    inline Real tent(Real x) {
        return x < 0.5 ? 2.0 * x : 2.0 * (1.0 - x);
    }

    // -------------- 2D / Complex Maps --------------
    inline Complex julia(Complex z, Complex c) {
        return z * z + c;
    }

    bool escapes(Complex z0, Complex c, int max_iter) {
        Real threshold2 = 4.0;
        Complex z = z0;

        for (int i = 0; i < max_iter; ++i) {
            Real zr = z.real();
            Real zi = z.imag();
            Real zr2 = zr * zr;
            Real zi2 = zi * zi;

            if (zr2 + zi2 > threshold2)
                return true;

            // z = z*z + c
            z = Complex(zr2 - zi2 + c.real(), 2.0 * zr * zi + c.imag());
        }
        return false;
    }

    // -------------- Utilities --------------
    Real atan2(Real y, Real x) {
        return std::atan2(y, x);
    }

    Real hypot(Real x, Real y) {
        return std::hypot(x, y);
    }

}
