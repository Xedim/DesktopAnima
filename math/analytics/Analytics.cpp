// Analytics.cpp

#include "Analytics.h"

namespace Analytics {

    // ---------------- Differential ----------------
    template<typename F>
    [[nodiscard]] Real derivative(F&& f, Real x, Real h, StabPolicy policy) {
        if (h <= 0.0) return NaN();

        Real fxph = Utils::safeEval(f, x + h, -std::numeric_limits<Real>::max(),
                                               std::numeric_limits<Real>::max(), policy);
        Real fxmh = Utils::safeEval(f, x - h, -std::numeric_limits<Real>::max(),
                                               std::numeric_limits<Real>::max(), policy);

        Real y = (fxph - fxmh) / (2.0 * h);
        return applyPolicy(y, -std::numeric_limits<Real>::max(),
                              std::numeric_limits<Real>::max(),
                              policy);
    }

    // ---------------- Norm ----------------
    template<typename F>
    [[nodiscard]] Real L2_norm(F&& f, Real a, Real b, int n, StabPolicy policy) {
        if (n <= 0 || a >= b) return NaN();

        Real h = (b - a) / n;
        Real sum = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < n; ++i) {
            Real xi = a + i * h;
            Real v  = Utils::safeEval(f, xi, -std::numeric_limits<Real>::max(),
                                           std::numeric_limits<Real>::max(), policy);
            if (!std::isfinite(v)) v = 0.0;
            sum += v * v;
        }

        Real y = std::sqrt(sum * h);
        return applyPolicy(y, 0.0, std::numeric_limits<Real>::max(), policy);
    }

    // ---------------- Properties ----------------
    template<typename F>
    [[nodiscard]] bool isEvenFunction(F&& f, Real x, Real eps, StabPolicy policy) {
        if (eps <= 0.0) return false;

        Real fx  = Utils::safeEval(f, x, -std::numeric_limits<Real>::max(),
                                        std::numeric_limits<Real>::max(), policy);
        Real fmx = Utils::safeEval(f, -x, -std::numeric_limits<Real>::max(),
                                         std::numeric_limits<Real>::max(), policy);

        return std::isfinite(fx) && std::isfinite(fmx) && std::abs(fmx - fx) < eps;
    }

    template<typename F>
    [[nodiscard]] bool isOddFunction(F&& f, Real x, Real eps, StabPolicy policy) {
        if (eps <= 0.0) return false;

        Real fx  = Utils::safeEval(f, x, -std::numeric_limits<Real>::max(),
                                        std::numeric_limits<Real>::max(), policy);
        Real fmx = Utils::safeEval(f, -x, -std::numeric_limits<Real>::max(),
                                         std::numeric_limits<Real>::max(), policy);

        return std::isfinite(fx) && std::isfinite(fmx) && std::abs(fmx + fx) < eps;
    }

    template<typename F>
    [[nodiscard]] bool isPeriodic(F&& f, Real x, Real T, Real eps, StabPolicy policy) {
        if (eps <= 0.0 || T <= 0.0) return false;

        Real fx  = Utils::safeEval(f, x, -std::numeric_limits<Real>::max(),
                                        std::numeric_limits<Real>::max(), policy);
        Real fxT = Utils::safeEval(f, x + T, -std::numeric_limits<Real>::max(),
                                          std::numeric_limits<Real>::max(), policy);

        return std::isfinite(fx) && std::isfinite(fxT) && std::abs(fxT - fx) < eps;
    }

    template<typename F>
    [[nodiscard]] bool isIncreasing(F&& f, Real x1, Real x2, Real eps, StabPolicy policy) {
        if (eps <= 0.0 || x2 < x1) return false;

        Real f1 = Utils::safeEval(f, x1, -std::numeric_limits<Real>::max(),
                                         std::numeric_limits<Real>::max(), policy);
        Real f2 = Utils::safeEval(f, x2, -std::numeric_limits<Real>::max(),
                                         std::numeric_limits<Real>::max(), policy);

        return std::isfinite(f1) && std::isfinite(f2) && f2 - f1 >= -eps;
    }

    template<typename F>
    [[nodiscard]] bool isDecreasing(F&& f, Real x1, Real x2, Real eps, StabPolicy policy) {
        if (eps <= 0.0 || x2 < x1) return false;

        Real f1 = Utils::safeEval(f, x1, -std::numeric_limits<Real>::max(),
                                         std::numeric_limits<Real>::max(), policy);
        Real f2 = Utils::safeEval(f, x2, -std::numeric_limits<Real>::max(),
                                         std::numeric_limits<Real>::max(), policy);

        return std::isfinite(f1) && std::isfinite(f2) && f1 - f2 >= -eps;
    }

    template<typename F>
    [[nodiscard]] bool isBounded(F&& f, Real x, StabPolicy policy) {
        Real y = Utils::safeEval(f, x, -std::numeric_limits<Real>::max(),
                                      std::numeric_limits<Real>::max(), policy);
        return std::isfinite(y);
    }

    template<typename F>
    [[nodiscard]] bool isConvex(F&& f, Real x, Real h, StabPolicy policy) {
        if (h <= 0.0) return false;

        Real fxh  = Utils::safeEval(f, x + h, -std::numeric_limits<Real>::max(),
                                          std::numeric_limits<Real>::max(), policy);
        Real fxmh = Utils::safeEval(f, x - h, -std::numeric_limits<Real>::max(),
                                          std::numeric_limits<Real>::max(), policy);
        Real fx   = Utils::safeEval(f, x, -std::numeric_limits<Real>::max(),
                                          std::numeric_limits<Real>::max(), policy);

        Real fpp = (fxh - 2.0 * fx + fxmh) / (h * h);
        fpp = applyPolicy(fpp, -std::numeric_limits<Real>::max(),
                             std::numeric_limits<Real>::max(), policy);

        return fpp >= 0.0;
    }

    template<typename F>
    [[nodiscard]] bool isConcave(F&& f, Real x, Real h, StabPolicy policy) {
        if (h <= 0.0) return false;

        Real fxh  = Utils::safeEval(f, x + h, -std::numeric_limits<Real>::max(),
                                          std::numeric_limits<Real>::max(), policy);
        Real fxmh = Utils::safeEval(f, x - h, -std::numeric_limits<Real>::max(),
                                          std::numeric_limits<Real>::max(), policy);
        Real fx   = Utils::safeEval(f, x, -std::numeric_limits<Real>::max(),
                                          std::numeric_limits<Real>::max(), policy);

        Real fpp = (fxh - 2.0 * fx + fxmh) / (h * h);
        fpp = applyPolicy(fpp, -std::numeric_limits<Real>::max(),
                             std::numeric_limits<Real>::max(), policy);

        return fpp <= 0.0;
    }

    template<typename F>
    [[nodiscard]] bool isNonNegative(F&& f, Real x, StabPolicy policy) {
        Real fx = Utils::safeEval(f, x, 0.0, std::numeric_limits<Real>::max(), policy);
        return std::isfinite(fx) && fx >= 0.0;
    }

    template<typename F>
    [[nodiscard]] bool isContinuous(F&& f, Real x, Real h, Real eps, StabPolicy policy) {
        if (h <= 0.0 || eps <= 0.0) return false;

        Real fx  = Utils::safeEval(f, x, -std::numeric_limits<Real>::max(),
                                        std::numeric_limits<Real>::max(), policy);
        Real fxh = Utils::safeEval(f, x + h, -std::numeric_limits<Real>::max(),
                                          std::numeric_limits<Real>::max(), policy);

        return std::isfinite(fx) && std::isfinite(fxh) && std::abs(fxh - fx) < eps;
    }

}
