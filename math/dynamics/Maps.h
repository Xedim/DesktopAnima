#pragma once
#include <vector>
#include "../common/Constants.h"
#include "../common/Types.h"

namespace Maps {

    // ---------------- Complex / 2D maps ----------------
    bool escapes(Complex z0, Complex c, Real eps = Constants::JULIA_ITER, Real threshold = Constants::ESC_TRESHOLD);

    // ---------------- Utility ----------------
    Real atan2(Real y, Real x);
    Real hypot(Real x, Real y);

}