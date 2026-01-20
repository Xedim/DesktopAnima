//Fractal.h

#pragma once
#include "../common/Constants.h"
#include "../common/Types.h"

namespace Fractal {


    // ---------- 1D Fractals / Self-similar ----------

    Real weierstrass(Real x,
                     Real a = Constants::WEIERSTRASS_AMP_COEF,
                     Real b = Constants::WEIERSTRASS_FREQ_COEF,
                     int  N = Constants::WEIERSTRASS_ITER,
                     StabPolicy policy = StabPolicy::Reject);

    Real cantor(Real x,
                int max_iter = Constants::CANTOR_ITER,
                StabPolicy policy = StabPolicy::Reject);

    // ---------- 1D Maps ----------

    Real logistic(Real x,
                  Real r,
                  StabPolicy policy = StabPolicy::Reject);

    Real iterate(Real x,
                 Real r,
                 int n = Constants::MAP_ITER,
                 StabPolicy policy = StabPolicy::Reject);

    Real tent(Real x,
              StabPolicy policy = StabPolicy::Reject);

    // ---------- 2D / Complex ----------

    Complex julia(Complex z, Complex c);

}
