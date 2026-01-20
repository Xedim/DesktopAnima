//Analytics.h

#pragma once
#include "../common/Constants.h"
#include "../common/Types.h"

namespace Analytics {

    // ---------------- Differential ----------------

    Real derivative(Real (*f)(Real), Real x, Real h = Constants::H,
                        StabPolicy policy = StabPolicy::Reject);

    // ---------------- Norm ----------------

    Real L2_norm(Function1D f, Real a, Real b, Real n = Constants::L2_NORM_ITER,
                        StabPolicy policy = StabPolicy::Reject);

    // ---------------- Properties ----------------

    bool isEvenFunction(const Function1D& f, Real x, Real eps = Constants::EPS,
                        StabPolicy policy = StabPolicy::Reject);

    bool isOddFunction(const Function1D& f, Real x, Real eps = Constants::EPS,
                        StabPolicy policy = StabPolicy::Reject);

    bool isPeriodic(const Function1D& f, Real x, Real T, Real eps = Constants::EPS,
                        StabPolicy policy = StabPolicy::Reject);

    bool isIncreasing(const Function1D& f, Real x1, Real x2, Real eps = Constants::EPS,
                        StabPolicy policy = StabPolicy::Reject);

    bool isDecreasing(const Function1D& f, Real x1, Real x2, Real eps = Constants::EPS,
                        StabPolicy policy = StabPolicy::Reject);

    bool isBounded(const Function1D& f, Real x,
                        StabPolicy policy = StabPolicy::Reject);

    bool isConvex(const Function1D& f, Real x, Real h = Constants::H,
                        StabPolicy policy = StabPolicy::Reject);

    bool isConcave(const Function1D& f, Real x, Real h = Constants::H,
                        StabPolicy policy = StabPolicy::Reject);

    bool isNonNegative(const Function1D& f, Real x,
                        StabPolicy policy = StabPolicy::Reject);

    bool isContinuous(const Function1D& f, Real x, Real h = Constants::H,
                        StabPolicy policy = StabPolicy::Reject);
}