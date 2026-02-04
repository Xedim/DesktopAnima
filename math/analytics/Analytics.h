//Analytics.h

#pragma once
#include "../common/Constants.h"
#include "../common/Types.h"


namespace Analytics {

    // ---------------- Differential ----------------

    Real derivative(const Function1D& f, Real x, Real h = Constants::H,
                        StabPolicy policy = StabPolicy::Reject);

    Real derivativeX(const Function2D& f, Real x, Real y_fixed, Real h = Constants::H,
                        StabPolicy policy = StabPolicy::Reject);

    Real derivativeY(const Function2D& f, Real x_fixed, Real y, Real h = Constants::H,
                        StabPolicy policy = StabPolicy::Reject);
    // ---------------- Norm ----------------

    Real L2_norm(const Function1D& f, Real a, Real b, Real n = Constants::L2_NORM_ITER,
                        StabPolicy policy = StabPolicy::Reject);

    // ---------------- Properties ----------------

    bool isLocallyEvenFunction(const Function1D& f, Real x, Real eps = Constants::EPS_09,
                        StabPolicy policy = StabPolicy::Reject);

    bool isLocallyOddFunction(const Function1D& f, Real x, Real eps = Constants::EPS_09,
                        StabPolicy policy = StabPolicy::Reject);

    bool isPeriodic(const Function1D& f, Real x, Real T, Real eps = Constants::EPS_09,
                        StabPolicy policy = StabPolicy::Reject);

    bool isLocallyIncreasing(const Function1D& f, Real x1, Real x2, Real eps = Constants::EPS_09,
                        StabPolicy policy = StabPolicy::Reject);

    bool isLocallyDecreasing(const Function1D& f, Real x1, Real x2, Real eps = Constants::EPS_09,
                        StabPolicy policy = StabPolicy::Reject);

    bool isFiniteAt(const Function1D& f, Real x, StabPolicy policy = StabPolicy::Reject);

    bool isLocallyConvex(const Function1D& f, Real x, Real h = Constants::H,
                        StabPolicy policy = StabPolicy::Reject);

    bool isLocallyConcave(const Function1D& f, Real x, Real h = Constants::H,
                        StabPolicy policy = StabPolicy::Reject);

    bool isNonNegative(const Function1D& f, Real x,
                        StabPolicy policy = StabPolicy::Reject);

    bool isContinuous(const Function1D& f, Real x, Real h = Constants::H, Real eps = Constants::EPS_09,
                        StabPolicy policy = StabPolicy::Reject);
}