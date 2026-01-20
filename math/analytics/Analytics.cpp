// Analytics.cpp

#include "Analytics.h"
#include "../common/Utils.h"
#include <cmath>

namespace Analytics {
    inline constexpr Real LO = Constants::NEG_LIMIT;
    inline constexpr Real HI = Constants::POS_LIMIT;

    // ---------------- Differential ----------------
    [[nodiscard]] Real derivative(const Function1D& f, Real x, Real h, StabPolicy policy) {
        if (h <= Real{0}) return NaN();
        Real f_x_plus_h  = Utils::evalStable(f, x + h, LO, HI, policy);
        Real f_x_minus_h = Utils::evalStable(f, x - h, LO, HI, policy);
        if (!Utils::isFiniteNum(f_x_plus_h) || !Utils::isFiniteNum(f_x_minus_h)) return NaN();
        Real y = (f_x_plus_h - f_x_minus_h) / (Real{2} * h);
        return Utils::checkStability(y, LO, HI, policy);
    }

    [[nodiscard]] Real derivativeX(const Function2D& f, Real x, Real y_fixed, Real h, StabPolicy policy) {
        x = Utils::checkStability(x, LO, HI, policy);
        y_fixed = Utils::checkStability(y_fixed, LO, HI, policy);
        if (!Utils::isFiniteNum(x) || !Utils::isFiniteNum(y_fixed)) return NaN();
        auto g = [&](Real xx) { return f(xx, y_fixed); };
        return derivative(g, x, h, policy);
    }

    [[nodiscard]] Real derivativeY(const Function2D& f, Real x_fixed, Real y, Real h, StabPolicy policy) {
        x_fixed = Utils::checkStability(x_fixed, LO, HI, policy);
        y = Utils::checkStability(y, LO, HI, policy);
        if (!Utils::isFiniteNum(x_fixed) || !Utils::isFiniteNum(y)) return NaN();
        auto g = [&](Real yy) { return f(x_fixed, yy); };
        return derivative(g, y, h, policy);
    }

    // ---------------- Norm ----------------
    [[nodiscard]] Real L2_norm(const Function1D& f, Real a, Real b, int n, StabPolicy policy) {
        if (n <= 0 || a >= b) return NaN();
        Real h = (b - a) / n;
        Real sum = Real{0};
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < n; ++i) {
            Real xi = a + i * h;
            Real v = Utils::evalStable(f, xi, LO, HI, policy);
            if (!Utils::isFiniteNum(v)) v = Real{0};
            sum += v * v;
        }
        Real y = std::sqrt(sum * h);
        return Utils::checkStability(y, Real{0}, HI, policy);
    }

    // ---------------- Properties ----------------
    [[nodiscard]] bool isLocallyEvenFunction(const Function1D& f, Real x, Real eps, StabPolicy policy) {
        if (eps <= Real{0}) return false;
        Real fx  = Utils::evalStable(f, x, LO, HI, policy);
        Real fmx = Utils::evalStable(f, -x, LO, HI, policy);
        return Utils::isFiniteNum(fx) && Utils::isFiniteNum(fmx) && std::abs(fmx - fx) < eps;
    }
    [[nodiscard]] bool isLocallyOddFunction(const Function1D& f, Real x, Real eps, StabPolicy policy) {
        if (eps <= Real{0}) return false;
        Real fx  = Utils::evalStable(f, x, LO, HI, policy);
        Real fmx = Utils::evalStable(f, -x, LO, HI, policy);
        return Utils::isFiniteNum(fx) && Utils::isFiniteNum(fmx) && std::abs(fmx + fx) < eps;
    }

    [[nodiscard]] bool isPeriodic(const Function1D& f, Real x, Real T, Real eps, StabPolicy policy) {
        if (eps <= Real{0} || T <= Real{0}) return false;
        Real fx  = Utils::evalStable(f, x, LO, HI, policy);
        Real fxT = Utils::evalStable(f, x + T, LO, HI, policy);
        return Utils::isFiniteNum(fx) && Utils::isFiniteNum(fxT) && std::abs(fxT - fx) < eps;
    }

    [[nodiscard]] bool isLocallyIncreasing(const Function1D& f, Real x1, Real x2, Real eps, StabPolicy policy) {
        if (eps <= Real{0} || x2 < x1) return false;
        Real f1 = Utils::evalStable(f, x1, LO, HI, policy);
        Real f2 = Utils::evalStable(f, x2, LO, HI, policy);
        return Utils::isFiniteNum(f1) && Utils::isFiniteNum(f2) && (f2 - f1) >= -eps;
    }

    [[nodiscard]] bool isLocallyDecreasing(const Function1D& f, Real x1, Real x2, Real eps, StabPolicy policy) {
        if (eps <= Real{0} || x2 < x1) return false;
        Real f1 = Utils::evalStable(f, x1, LO, HI, policy);
        Real f2 = Utils::evalStable(f, x2, LO, HI, policy);
        return Utils::isFiniteNum(f1) && Utils::isFiniteNum(f2) && (f1 - f2) >= -eps;
    }

    [[nodiscard]] bool isFiniteAt(const Function1D& f, Real x, StabPolicy policy) {
        Real y = Utils::evalStable(f, x, LO, HI, policy);
        return Utils::isFiniteNum(y);
    }

    [[nodiscard]] bool isLocallyConvex(const Function1D& f, Real x, Real h, StabPolicy policy) {
        if (h <= Real{0}) return false;
        Real fxh       = Utils::evalStable(f, x + h, LO, HI, policy);
        Real f_x_minus_h = Utils::evalStable(f, x - h, LO, HI, policy);
        Real fx        = Utils::evalStable(f, x, LO, HI, policy);
        Real fpp = (fxh - Real{2} * fx + f_x_minus_h) / (h * h);
        return Utils::checkStability(fpp, LO, HI, policy) >= Real{0};
    }

    [[nodiscard]] bool isLocallyConcave(const Function1D& f, Real x, Real h, StabPolicy policy) {
        if (h <= Real{0}) return false;
        Real fxh       = Utils::evalStable(f, x + h, LO, HI, policy);
        Real f_x_minus_h = Utils::evalStable(f, x - h, LO, HI, policy);
        Real fx        = Utils::evalStable(f, x, LO, HI, policy);
        Real fpp = (fxh - Real{2} * fx + f_x_minus_h) / (h * h);
        return Utils::checkStability(fpp, LO, HI, policy) <= Real{0};
    }

    [[nodiscard]] bool isNonNegative(const Function1D& f, Real x, StabPolicy policy) {
        Real fx = Utils::evalStable(f, x, Real{0}, HI, policy);
        return Utils::isFiniteNum(fx) && fx >= Real{0};
    }

    [[nodiscard]] bool isContinuous(const Function1D& f, Real x, Real h, Real eps, StabPolicy policy) {
        if (h <= Real{0} || eps <= Real{0}) return false;
        Real fx  = Utils::evalStable(f, x, LO, HI, policy);
        Real fxh = Utils::evalStable(f, x + h, LO, HI, policy);
        return Utils::isFiniteNum(fx) && Utils::isFiniteNum(fxh) &&
               std::abs(fxh - fx) < eps * std::max(Real{1}, std::abs(fx));
    }
}