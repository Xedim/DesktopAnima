// Fractal.cpp
#include "Fractal.h"
#include "../common/Utils.h"
#include "../analytics/Analytics.h"
#include <cmath>

namespace Fractal {

    // ---------- 1D -> 1D Fractals / Self-similar ----------

    [[nodiscard]] Real weierstrass(Real x, Real a, Real b, int N, StabPolicy policy) {
        if (a <= Constants::WEIERSTRASS_AMP_MIN || a >= Constants::WEIERSTRASS_AMP_MAX ||
            b <= Constants::WEIERSTRASS_FREQ_MIN ||
            N <= Constants::WEIERSTRASS_ITER_MIN ||
            x <  Constants::WEIERSTRASS_X_MIN ||
            x >  Constants::WEIERSTRASS_X_MAX)
        {
            return NaN();
        }

        Real sum = 0.0;
        Real amp_factor = 1.0;
        Real freq_factor = 1.0;
        const Real pi_x = Constants::PI * x;

        for (int n = 0; n < N; ++n) {
            Real term = amp_factor * std::cos(freq_factor * pi_x);
            term = Utils::checkStability(term, Constants::WEIERSTRASS_Y_MIN, Constants::WEIERSTRASS_Y_MAX, policy);
            if (!std::isfinite(term)) break;

            sum += term;
            amp_factor *= a;
            freq_factor *= b;

            amp_factor = Utils::checkStability(amp_factor, Constants::NEG_LIMIT, Constants::POS_LIMIT, policy);
            freq_factor = Utils::checkStability(freq_factor, Constants::NEG_LIMIT, Constants::POS_LIMIT, policy);
            if (!std::isfinite(amp_factor) || !std::isfinite(freq_factor)) break;
        }

        return Utils::checkStability(sum, Constants::WEIERSTRASS_Y_MIN, Constants::WEIERSTRASS_Y_MAX, policy);
    }

    [[nodiscard]] Real cantor(Real x, int max_iter, StabPolicy policy) {
        if (x < Constants::CANTOR_X_MIN || x > Constants::CANTOR_X_MAX ||
            max_iter <= Constants::CANTOR_ITER_MIN)
        {
            return NaN();
        }

        Real result = 0.0;
        Real scale_factor = 1.0;

        for (int i = 0; i < max_iter; ++i) {
            if (x < Constants::CANTOR_LEFT) {
                x *= Constants::CANTOR_SCALE;
            } else if (x > Constants::CANTOR_RIGHT) {
                result += scale_factor;
                x = Constants::CANTOR_SCALE * x - Constants::CANTOR_RIGHT_SCALE;
            } else {
                Real final_val = result + scale_factor * Constants::CANTOR_MID;
                return Utils::checkStability(final_val, Constants::CANTOR_Y_MIN, Constants::CANTOR_Y_MAX, policy);
            }

            scale_factor = Utils::checkStability(scale_factor * Constants::CANTOR_FACTOR, Constants::NEG_LIMIT, Constants::POS_LIMIT, policy);
            if (!std::isfinite(scale_factor)) break;
        }

        return Utils::checkStability(result, Constants::CANTOR_Y_MIN, Constants::CANTOR_Y_MAX, policy);
    }

    // ---------- 1D -> 1D Maps ----------

    [[nodiscard]] Real logistic(Real x, Real r, StabPolicy policy) {
        if (x < Constants::LOGISTIC_X_MIN || x > Constants::LOGISTIC_X_MAX ||
            r < Constants::LOGISTIC_R_MIN || r > Constants::LOGISTIC_R_MAX)
        {
            return NaN();
        }

        Real y = r * x * (1.0 - x);
        return Utils::checkStability(y, Constants::LOGISTIC_Y_MIN, Constants::LOGISTIC_Y_MAX, policy);
    }

    [[nodiscard]] Real tent(Real x, StabPolicy policy) {
        if (x < Constants::TENT_X_MIN || x > Constants::TENT_X_MAX) return NaN();

        Real result = (x < Constants::TENT_PEAK)
                    ? Constants::TENT_SLOPE * x
                    : Constants::TENT_SLOPE * (1.0 - x);

        return Utils::checkStability(result, Constants::TENT_Y_MIN, Constants::TENT_Y_MAX, policy);
    }

    // ---------- 2D -> 1D / Complex Fractal ----------

    [[nodiscard]] Complex julia(const Complex& z, const Complex& c, StabPolicy policy) {
        Complex y = z * z + c;
        return Utils::checkStability(y, Constants::NEG_LIMIT, Constants::POS_LIMIT, policy);
    }

    // ---------- Generic iterate for Real and Complex ----------
    template<typename T, typename MapFunc>
    [[nodiscard]] T iterate(T x, MapFunc&& f, int n,
                            Real lo = Constants::NEG_LIMIT,
                            Real hi = Constants::POS_LIMIT,
                            StabPolicy policy = StabPolicy::Reject)
    {
        for (int i = 0; i < n; ++i) {
            x = f(x);
            x = Utils::checkStability(x, lo, hi, policy);
            if constexpr (std::is_same_v<T, Real>) {
                if (!std::isfinite(x)) return NaN();
            } else if constexpr (std::is_same_v<T, Complex>) {
                if (!std::isfinite(x.real()) || !std::isfinite(x.imag()))
                    return {NaN(), NaN()};
            }
        }
        return x;
    }

} // namespace Fractal

