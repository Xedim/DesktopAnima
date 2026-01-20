//Maps.cpp

#include "Maps.h"
#include "../common/Utils.h"
#include "../fractal/Fractal.h"
#include <cmath>

namespace Maps {
    // -------------- 2D -> 1D / Maps --------------

    [[nodiscard]] bool escapes(Complex z0, Complex c, int max_iter, Real threshold = Real{2}) {
        if (max_iter <= 0 || threshold <= Real{0}) return false;
        if (!Utils::isFiniteNum(z0) || !Utils::isFiniteNum(c)) return false;
        Complex z = z0;
        Real threshold2 = threshold * threshold;

        for (int i = 0; i < max_iter; ++i) {
            Real zr = z.real();
            Real zi = z.imag();
            Real zr2 = zr * zr;
            Real zi2 = zi * zi;

            if (zr2 + zi2 > threshold2)
                return true;

            z = Complex(zr2 - zi2 + c.real(), Real{2} * zr * zi + c.imag());
            z = Utils::checkStability(z, Constants::NEG_LIMIT, Constants::POS_LIMIT, StabPolicy::Reject);
        }

        return false;
    }

    [[nodiscard]] Real atan2(Real x, Real y) {
        if (!Utils::isFiniteNum(x) || !Utils::isFiniteNum(y)) return NaN();
        Real val = std::atan2(y, x);
        return Utils::checkStability(val, -Constants::PI, Constants::PI, StabPolicy::Reject);
    }

    [[nodiscard]] Real hypot(Real x, Real y) {
        if (!Utils::isFiniteNum(x) || !Utils::isFiniteNum(y)) return NaN();
        Real val = std::hypot(x, y);
        return Utils::checkStability(val, Real{0}, Constants::POS_LIMIT, StabPolicy::Reject);
    }
}
