#include "Fractal.h"
#include <cmath>

namespace Fractal {

    // ---------- 1D Fractals / Self-similar ----------

    Real weierstrass(Real x, Real a, Real b, int N, StabPolicy policy) {
        if (a <= Constants::WEIERSTRASS_AMP_MIN ||
            a >= Constants::WEIERSTRASS_AMP_MAX ||
            b <= Constants::WEIERSTRASS_FREQ_MIN ||
            N <= Constants::WEIERSTRASS_ITER_MIN ||
            x <  Constants::WEIERSTRASS_X_MIN ||
            x >  Constants::WEIERSTRASS_X_MAX) {
            return NaN();
        }

        Real sum = 0.0;
        Real an = 1.0;
        Real bn = 1.0;
        const double pi_x = Constants::PI * x;

        for (int n = 0; n < N; ++n) {
            sum += an * std::cos(bn * pi_x);
            an *= a;
            bn *= b;
        }

        return applyPolicy(sum, Constants::WEIERSTRASS_Y_MIN, Constants::WEIERSTRASS_Y_MAX, policy);
    }

    Real cantor(Real x, int max_iter, StabPolicy policy) {
        if (x < Constants::CANTOR_X_MIN ||
            x > Constants::CANTOR_X_MAX ||
            max_iter <= Constants::CANTOR_ITER_MIN) {
            return NaN();
        }

        Real y = 0.0;
        Real factor = 1.0;

        for (int i = 0; i < max_iter; ++i) {
            if (x < Constants::CANTOR_LEFT) {
                x *= Constants::CANTOR_SCALE;
            } else if (x > Constants::CANTOR_RIGHT) {
                y += factor;
                x = Constants::CANTOR_SCALE * x - Constants::CANTOR_RIGHT_SCALE;
            } else {
                return applyPolicy(y + factor * Constants::CANTOR_MID,
                                  Constants::CANTOR_Y_MIN,
                                  Constants::CANTOR_Y_MAX,
                                     policy);
            }
            factor *= Constants::CANTOR_FACTOR;
        }

        return applyPolicy(y, Constants::CANTOR_Y_MIN, Constants::CANTOR_Y_MAX, policy);
    }

    // ---------- 1D Maps ----------

    Real logistic(Real x, Real r, StabPolicy policy) {
        if (x < Constants::LOGISTIC_X_MIN || x > Constants::LOGISTIC_X_MAX ||
            r < Constants::LOGISTIC_R_MIN || r > Constants::LOGISTIC_R_MAX) {
            return NaN();
        }

        Real y = r * x * (1.0 - x);
        return applyPolicy(y, Constants::LOGISTIC_Y_MIN, Constants::LOGISTIC_Y_MAX, policy);
    }

    Real iterate(Real x0, Real r, int n, StabPolicy policy) {
        if (r  < Constants::LOGISTIC_R_MIN ||
            r  > Constants::LOGISTIC_R_MAX ||
            n  <= 0) {
            return NaN();
        }

        Real x = x0;
        for (int i = 0; i < n; ++i) {
            if (policy == StabPolicy::Reject && (x < Constants::LOGISTIC_Y_MIN || x > Constants::LOGISTIC_Y_MAX)) {
                return NaN();
            }
            x = logistic(x, r, policy);

            if (!std::isfinite(x)) return NaN();
        }

        return x;
    }

    Real tent(Real x, StabPolicy policy) {
        if (x < Constants::TENT_X_MIN || x > Constants::TENT_X_MAX) return NaN();

        Real y = (x < Constants::TENT_PEAK)
               ? Constants::TENT_SLOPE * x
               : Constants::TENT_SLOPE * (1.0 - x);

        return applyPolicy(y, Constants::TENT_Y_MIN, Constants::TENT_Y_MAX, policy);
    }

    // ---------- 2D / Complex ----------

    Complex julia(Complex z, Complex c, StabPolicy policy) {
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag()) ||
            !std::isfinite(c.real()) || !std::isfinite(c.imag())) {
            return {NaN(), NaN()};
        }

        Complex y = z * z + c;
        if (policy == StabPolicy::Reject &&
            (!std::isfinite(y.real()) || !std::isfinite(y.imag()))) {
            return {NaN(), NaN()};
        }
        return y;
    }

}