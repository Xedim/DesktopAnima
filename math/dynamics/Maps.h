#pragma once
#include <vector>
#include "../common/Constants.h"
#include "../common/Types.h"

namespace Maps {

    // ---------------- Logistic map ----------------
    Real logistic(Real x, Real r);
    Real iterate(Real x0, Real r, int n = Constants::MAP_ITER);

    // ---------------- Tent map ----------------
    Real tent(Real x);

    // ---------------- Complex / 2D maps ----------------
    Complex julia(Complex z, Complex c);
    bool escapes(Complex z0, Complex c, Real eps = Constants::JULIA_ITER);

    // ---------------- Utility ----------------
    Real atan2(Real y, Real x);
    Real hypot(Real x, Real y);

}