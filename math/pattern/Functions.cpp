//Functions.cpp

#include "Pattern1D.h"
#include "../common/Utils.h"
#include "../analytics/Analytics.h"
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include <random>
#include <utility>

namespace functions {

    constexpr int FACTORIAL_CACHE_SIZE = 21;
    constexpr inline Real factorial_cache[FACTORIAL_CACHE_SIZE] = {
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0,
        362880.0, 3628800.0, 39916800.0, 479001600.0, 6227020800.0,
        87178291200.0, 1307674368000.0, 20922789888000.0, 355687428096000.0,
        6402373705728000.0, 121645100408832000.0, 2432902008176640000.0
    };

    // =============================================
    // ================= Algebraic =================
    // =============================================

    [[nodiscard]] Real factorial(int n) {
        if (n < 0) return NaN();
        if (n < FACTORIAL_CACHE_SIZE) return factorial_cache[n];
        return std::tgamma(n + Real{1});
    }

    [[nodiscard]] Real binomial(int n, int k) {
        if (k < 0 || k > n) return Real{0};
        if (k > n - k) k = n - k;

        if (n < 30) {
            Real result = Real{1};
            for (int i = 1; i <= k; ++i)
                result *= static_cast<Real>(n - k + i) / i;
            return result;
        }

        return std::round(std::tgamma(n + Real{1}) / (std::tgamma(k + Real{1}) * std::tgamma(n - k + Real{1})));
    }

    [[nodiscard]] Real combination(int n, int k) { return binomial(n, k); }
    [[nodiscard]] Real permutation(int n, int k) {
        if (k < 0 || k > n) return Real{0};

        if (n < 30) {
            Real result = Real{1};
            for (int i = 0; i < k; ++i) result *= n - i;
            return result;
        }

        return std::tgamma(n + Real{1}) / std::tgamma(n - k + Real{1});
    }

    [[nodiscard]] Real mod(Real x, Real y) {
        if (y == Real{0}) return NaN();
        Real r = std::fmod(x, y);
        if (r < 0) r += std::abs(y);
        return r;
    }


    [[nodiscard]] Real geometric_sum(Real a, int N) noexcept {
        if (N <= 0) return Real{0};
        if (a == Real{1}) return static_cast<Real>(N);

        Real sum = Real{0};
        Real term = Real{1};
        for (int i = 0; i < N; ++i) {
            sum += term;
            term *= a;
        }
        return sum;
    }

    [[nodiscard]] Real algebraic_root(Real x, const VecReal& coefficients) {
        Real poly = Real{0};
        for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
            poly = poly * x + *it;

        if (poly < 0) return NaN();
        return std::sqrt(poly);
    }

    [[nodiscard]] Real sqrt(Real x) { return x < Real{0} ? NaN() : std::sqrt(x); }
    [[nodiscard]] Real cbrt(Real x) { return std::cbrt(x); }

    // =======================================================
    // ================= Power / Exponential =================
    // =======================================================

    struct PowCache {
        std::array<Real, 21> values{};
        Real base = NaN();
        std::mutex mtx;

        Real get(Real x, int n) {
            if (n < 0 || n > 20) return NaN();
            std::lock_guard<std::mutex> lock(mtx);
            if (x != base) {
                base = x;
                values[0] = 1.0;
                for (int i = 1; i <= 20; ++i)
                    values[i] = values[i - 1] * x;
            }
            return values[n];
        }
    };

    inline PowCache powCache;

    [[nodiscard]] Real pow(Real x, Real alpha) {
        if (x == Real{0}) return alpha > Real{0} ? Real{0} : NaN();
        if (alpha == Real{0}) return Real{1};
        if (alpha == Real{0.5}) return std::sqrt(x);
        if (alpha == Real{1}) return x;
        if (alpha == Real{2}) return x * x;
        if (x > 0.0 && std::floor(alpha) == alpha && alpha >= 0 && alpha <= 20) {
            return powCache.get(x, static_cast<int>(alpha));
        }
        if (x < 0.0 && std::floor(alpha) != alpha) return NaN();
        return std::pow(x, alpha);
    }

    [[nodiscard]] inline Real sign(Real x) {
        return (x > 0) ? Real{1} : (x < 0 ? Real{-1} : Real{0});
    }

    [[nodiscard]] Real abs(Real x) { return std::abs(x); }

    [[nodiscard]] Real exp(Real x) {
        if (x > 700.0) return std::numeric_limits<Real>::infinity();
        if (x < -700.0) return 0.0;
        return std::exp(x);
    }

    [[nodiscard]] Real exp2(Real x) {
        if (x > 700.0) return std::numeric_limits<Real>::infinity();
        if (x < -700.0) return 0.0;
        return std::exp2(x); }

    [[nodiscard]] Real expm1_safe(Real x) {
        if (std::abs(x) < 1e-5) return x + 0.5*x*x; // Тейлор для маленьких x
        return std::expm1(x);
    }

    // ===============================================
    // ================= Logarithmic =================
    // ===============================================

    struct LogBaseCache {
        Real base = NaN();
        Real ln_base = NaN();
        std::mutex mtx;

        Real get(Real a) {
            std::lock_guard<std::mutex> lock(mtx);
            if (a != base) {
                base = a;
                ln_base = std::log(a);
            }
            return ln_base;
        }
    };

    inline LogBaseCache logBaseCache;

    [[nodiscard]] Real log2(Real x) {
        if (x <= 0.0) return NaN();
        return std::log(x) / 0.6931471805599453;
    }
    [[nodiscard]] Real log(Real x) {
        if (x <= Real{0}) return NaN();
        return std::log(x);
    }

    [[nodiscard]] Real log10(Real x) {
        if (x <= Real{0}) return NaN();
        return std::log(x) / 2.302585092994046;
    }

    [[nodiscard]] Real log_a(Real x, Real a) {
        if (x <= Real{0} || a <= Real{0} || a == Real{1}) return NaN();
        Real ln_a = logBaseCache.get(a);
        return std::log(x) / ln_a;
    }

    [[nodiscard]] Real log1p(Real x) {
        if (x <= -1.0) return NaN();
        if (std::abs(x) < 1e-5) return x - 0.5 * x * x + (1/3) * x * x * x;
        return std::log1p(x);
    }

    // =================================================
    // ================= Trigonometric =================
    // =================================================

    [[nodiscard]] Real sin(Real x) {
        if (!std::isfinite(x)) return NaN();
        return std::sin(x);
    }

    [[nodiscard]] Real cos(Real x) {
        if (!std::isfinite(x)) return NaN();
        return std::cos(x);
    }

    [[nodiscard]] Real sec(Real x) {
        Real c = cos(x);
        if (c == 0.0) return NaN();
        return 1.0 / c;
    }

    [[nodiscard]] Real csc(Real x) {
        Real s = sin(x);
        if (s == 0.0) return NaN();
        return 1.0 / s;
    }

    [[nodiscard]] Real sinc(Real x) {
        if (x == 0.0) return 1.0;
        if (std::abs(x) < 1e-5) {
            Real x2 = x * x;
            return 1.0 - x2 / 6.0 + x2 * x2 / 120.0;
        }
        return sin(x) / x;
    }

    [[nodiscard]] Real tan(Real x) {
        Real t = std::tan(x);
        return std::isfinite(t) ? t : NaN();
    }

    [[nodiscard]] Real cot(Real x) {
        Real t = tan(x);
        return t != 0.0 ? 1.0 / t : NaN();
    }

    [[nodiscard]] Real asin(Real x) {
        if (x < -1.0 || x > 1.0) return NaN();
        return std::asin(x);
    }

    [[nodiscard]] Real acos(Real x) {
        if (x < -1.0 || x > 1.0) return NaN();
        return std::acos(x);
    }

    [[nodiscard]] Real atan(Real x) {
        if (!std::isfinite(x)) return NaN();
        return std::atan(x);
    }

    [[nodiscard]] Real atan2(Real y, Real x) {
        if (!std::isfinite(x) || !std::isfinite(y)) return NaN();
        return std::atan2(y, x);
    }

    [[nodiscard]] Real hypot(Real x, Real y) {
        if (!std::isfinite(x) || !std::isfinite(y)) return NaN();
        return std::hypot(x, y);
    }

    // ==============================================
    // ================= Hyperbolic =================
    // ==============================================

    [[nodiscard]] Real sinh(Real x) {
        if (!std::isfinite(x)) return NaN();
        if (std::abs(x) > Real{700}) return NaN();
        return std::sinh(x);
    }

    [[nodiscard]] Real cosh(Real x) {
        if (!std::isfinite(x)) return NaN();
        if (std::abs(x) > Real{700}) return NaN();
        return std::cosh(x);
    }

    [[nodiscard]] Real sech(Real x) {
        Real c = cosh(x);
        if (c == 0.0) return NaN();
        return 1.0 / c;
    }

    [[nodiscard]] Real csch(Real x) {
        Real s = sinh(x);
        if (s == 0.0) return NaN();
        return 1.0 / s;
    }

    [[nodiscard]] Real tanh(Real x) {
        if (!std::isfinite(x)) return NaN();
        if (x > 20.0) return 1.0;
        if (x < -20.0) return -1.0;
        return std::tanh(x);
    }

    [[nodiscard]] Real coth(Real x) {
        Real t = tanh(x);
        if (t == 0.0) return NaN();
        return 1.0 / t;
    }

    [[nodiscard]] Real asinh(Real x) {
        if (!std::isfinite(x)) return NaN();
        return std::asinh(x);
    }

    [[nodiscard]] Real acosh(Real x) {
        if (x < 1.0) return NaN();
        return std::acosh(x);
    }

    [[nodiscard]] Real atanh(Real x) {
        if (x <= -1.0 || x >= 1.0) return NaN();
        return std::atanh(x);
    }

    // ======================================================
    // ================= Hybrid / Numerical =================
    // ======================================================

    [[nodiscard]] Real x_pow_y(Real x, Real y) {
        if (x == 0.0) return (y > 0.0) ? 0.0 : NaN(); // 0^y
        if (x == 1.0 || y == 0.0) return 1.0;         // 1^y, x^0
        if (y == 1.0) return x;                        // x^1
        if (x > 0.0) return std::exp(y * std::log(x));
        return NaN();
    }

    [[nodiscard]] Real sqrt1pm1(Real x) {
        if (x < -1.0) return NaN();
        if (std::abs(x) < 1e-8) return x / 2.0;
        return std::sqrt(1.0 + x) - 1.0;
    }

    [[nodiscard]] Real heaviside(Real x) { return x < 0.0 ? 0.0 : 1.0; }

    // ===========================================
    // ================= Special =================
    // ===========================================

    [[nodiscard]] Real erf(Real x) { return std::erf(x); }
    [[nodiscard]] Real erfc(Real x) { return std::erfc(x); }

    struct GammaCache {
        static constexpr int SIZE = 21;
        std::array<Real, SIZE> values{};
        Real last_x = NaN();
        std::mutex mtx;

        Real get(Real x) {
            if (x <= 0.0 && std::floor(x) == x) return NaN();
            if (x >= 0 && x < SIZE && std::floor(x) == x) {
                std::lock_guard<std::mutex> lock(mtx);
                if (x != last_x) {
                    last_x = x;
                    for (int i = 0; i < SIZE; ++i)
                        values[i] = std::tgamma(static_cast<Real>(i));
                }
                return values[static_cast<int>(x)];
            }
            return std::tgamma(x);
        }
    };

    inline GammaCache gammaCache;

    [[nodiscard]] Real gamma(Real x) { return gammaCache.get(x); }

    [[nodiscard]] Real lgamma(Real x) {
        if (x <= Real{0} && std::floor(x) == x) return NaN();
        return std::lgamma(x);
    }

    [[nodiscard]] Real beta(Real x, Real y) {
        if (x <= 0.0 || y <= 0.0) return NaN();
        if (x >= 0 && x < 21 && y >= 0 && y < 21) {
            return gammaCache.get(x) * gammaCache.get(y) / gammaCache.get(x + y);
        }
        return std::tgamma(x) * std::tgamma(y) / std::tgamma(x + y);
    }

    [[nodiscard]] Real cyl_bessel_j(Real nu, Real x) {
        if (x < 0.0) return NaN();
        return boost::math::cyl_bessel_j(nu, x);
    }

    [[nodiscard]] Real cyl_neumann(Real nu, Real x) {
        if (x <= 0.0) return NaN();
        return boost::math::cyl_neumann(nu, x);
    }

    [[nodiscard]] Real cyl_bessel_i(Real nu, Real x) {
        if (x < 0.0) return NaN();
        return boost::math::cyl_bessel_i(nu, x);
    }

    [[nodiscard]] Real cyl_bessel_k(Real nu, Real x) {
        if (x <= 0.0) return NaN();
        return boost::math::cyl_bessel_k(nu, x);
    }

    [[nodiscard]] Real lambert_w(Real x) {
        if (x < -1.0 / std::exp(1)) return NaN();
        return boost::math::lambert_w0(x);
    }

    [[nodiscard]] Real legendre(int l, Real x) {
        if (l < 0 || x < -1.0 || x > 1.0) return NaN();
        return std::legendre(l, x);
    }

    [[nodiscard]] Real assoc_legendre(int l, int m, Real x) {
        if (l < 0 || std::abs(m) > l || x < -1.0 || x > 1.0) return NaN();
        return std::assoc_legendre(l, m, x);
    }

    [[nodiscard]] Real riemann_zeta(Real s) {
        if (s <= 1.0) return NaN();
        return std::riemann_zeta(s);
    }

    [[nodiscard]] Real zeta_func(Real s) {
        if (s <= 1.0) return NaN();
        return boost::math::zeta(s);
    }

    // ===============================================
    // ================= Generalized =================
    // ===============================================

    [[nodiscard]] Real dirac_delta(Real x, Real eps) {
        if (eps <= Real{0}) return NaN();
        const Real threshold = Real{50} * eps;
        if (x * x > threshold) return Real{0};
        const Real invSqrtPiEps = 1.0 / std::sqrt(M_PI * eps);
        return std::exp(-x * x / eps) * invSqrtPiEps;
    }

    // ====================================================
    // ================= Numerical / Misc =================
    // ====================================================

    [[nodiscard]] Real round(Real x) { return std::round(x); }
    [[nodiscard]] Real floor(Real x) { return std::floor(x); }
    [[nodiscard]] Real ceil(Real x) { return std::ceil(x); }
    [[nodiscard]] Real trunc(Real x) { return std::trunc(x); }

    [[nodiscard]] Real clamp(Real x, Real minVal, Real maxVal) {
        #if __cplusplus >= 201703L
                return std::clamp(x, minVal, maxVal);
        #else
                return std::max(minVal, std::min(maxVal, x));
        #endif
    }

    [[nodiscard]] Real lerp(Real a, Real b, Real t) {
        return std::fma(t, b - a, a);
    }

    [[nodiscard]] Real fma(Real x, Real y, Real z) { return std::fma(x, y, z); }

    // ==============================================
    // ================= Fractals ===================
    // ==============================================

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
            term = Utils::checkStability(term,
                                        Constants::WEIERSTRASS_Y_MIN,
                                        Constants::WEIERSTRASS_Y_MAX,
                                           policy);
            if (!std::isfinite(term)) break;

            sum += term;
            amp_factor *= a;
            freq_factor *= b;

            amp_factor = Utils::checkStability(amp_factor,
                                              Constants::NEG_LIMIT,
                                              Constants::POS_LIMIT,
                                                 policy);
            freq_factor = Utils::checkStability(freq_factor,
                                               Constants::NEG_LIMIT,
                                               Constants::POS_LIMIT,
                                                  policy);
            if (!std::isfinite(amp_factor) || !std::isfinite(freq_factor)) break;
        }

        return Utils::checkStability(sum,
                                    Constants::WEIERSTRASS_Y_MIN,
                                    Constants::WEIERSTRASS_Y_MAX,
                                       policy);
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

            scale_factor = Utils::checkStability(scale_factor * Constants::CANTOR_FACTOR,
                                                Constants::NEG_LIMIT,
                                                Constants::POS_LIMIT,
                                                   policy);
            if (!std::isfinite(scale_factor)) break;
        }

        return Utils::checkStability(result, Constants::CANTOR_Y_MIN, Constants::CANTOR_Y_MAX, policy);
    }

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

    [[nodiscard]] Complex julia(const Complex& z, const Complex& c, StabPolicy policy) {
        Complex y = z * z + c;
        return Utils::checkStability(y, Constants::NEG_LIMIT, Constants::POS_LIMIT, policy);
    }

    [[nodiscard]] bool escapes(Complex z0, Complex c, int max_iter, Real threshold = Constants::ESC_TRESHOLD) {
        if (max_iter <= 0 || threshold <= Real{0}) return false;
        if (!Utils::isFiniteNum(z0) || !Utils::isFiniteNum(c)) return false;

        Complex z = z0;
        const Real threshold2 = threshold * threshold;

        for (int i = 0; i < max_iter; ++i) {
            const Real zr = z.real();
            const Real zi = z.imag();
            const Real r2 = zr * zr + zi * zi;

            if (r2 > threshold2)
                return true;

            z = { zr * zr - zi * zi + c.real(),
                  Real{2} * zr * zi + c.imag() };

            // защита от NaN / Inf
            if (!Utils::isFiniteNum(z))
                return true;
        }

        return false;
    }

    // =========================================
    // ================ Iterate ================
    // =========================================

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

    // ==========================================================
    // ================= Descriptive Statistics =================
    // ==========================================================

    Real sum(const VecReal& x) {
        return std::accumulate(x.begin(), x.end(), Real{0});
    }

    Real mean(const VecReal& x) {
        return x.empty() ? NaN() : sum(x) / x.size();
    }

    Real median(VecReal x) {
        if (x.empty()) return NaN();
        auto mid = x.begin() + x.size() / 2;
        std::nth_element(x.begin(), mid, x.end());
        Real m = *mid;
        if (x.size() % 2 == 0) {
            auto max_it = std::max_element(x.begin(), mid);
            m = (*max_it + m) * Real{0.5};
        }
        return m;
    }

    Real mode(const VecReal& x) {
        if (x.empty()) return NaN();
        std::unordered_map<Real, std::size_t> freq;
        Real mode_val = x[0];
        std::size_t max_count = 0;

        for (Real v : x) {
            std::size_t count = ++freq[v];
            if (count > max_count) {
                max_count = count;
                mode_val = v;
            }
        }
        return mode_val;
    }

    Real min(const VecReal& x) {
        return x.empty() ? NaN() : *std::min_element(x.begin(), x.end());
    }

    Real max(const VecReal& x) {
        return x.empty() ? NaN() : *std::max_element(x.begin(), x.end());
    }

    Real range(const VecReal& x) {
        if (x.empty()) return NaN();
        auto [mn, mx] = std::minmax_element(x.begin(), x.end());
        return *mx - *mn;
    }

    struct StatsCache {
        Real sum = 0;
        Real sumsq = 0;
        Real min_val = std::numeric_limits<Real>::max();
        Real max_val = std::numeric_limits<Real>::lowest();
        std::size_t n = 0;
    };

    inline StatsCache compute_stats(const VecReal& x) {
        StatsCache cache;
        for (Real v : x) {
            cache.sum += v;
            cache.sumsq += v * v;
            cache.min_val = std::min(cache.min_val, v);
            cache.max_val = std::max(cache.max_val, v);
            ++cache.n;
        }
        return cache;
    }

    Real variance(const VecReal& x) {
        if (x.size() < 2) return NaN();
        auto stats = compute_stats(x);
        Real m = stats.sum / stats.n;
        return (stats.sumsq / stats.n) - (m * m);
    }

    Real variance_unbiased(const VecReal& x) {
        if (x.size() < 2) return NaN();
        auto stats = compute_stats(x);
        Real m = stats.sum / stats.n;
        return ((stats.sumsq / stats.n) - m * m) * (stats.n / (stats.n - 1.0));
    }

    Real stddev(const VecReal& x) {
        Real v = variance(x);
        return std::isfinite(v) ? std::sqrt(v) : NaN();
    }

    Real stddev_unbiased(const VecReal& x) {
        Real v = variance_unbiased(x);
        return std::isfinite(v) ? std::sqrt(v) : NaN();
    }

    Real mean_absolute_deviation(const VecReal& x) {
        if (x.empty()) return NaN();
        auto stats = compute_stats(x);
        Real mean_val = stats.sum / stats.n;
        Real acc = 0;
        for (Real v : x) acc += std::abs(v - mean_val);
        return acc / stats.n;
    }

    // ==========================================================
    // ================= Shape Statistics =======================
    // ==========================================================

    Real raw_moment(const VecReal& x, int k) {
        if (x.empty() || k < 0) return NaN();
        Real acc = 0;
        for (Real v : x) acc += std::pow(v, k);
        return acc / x.size();
    }

    Real moment(const VecReal& x, int k) {
        if (x.empty() || k < 0) return NaN();
        Real m = mean(x);
        Real acc = 0;
        for (Real v : x) acc += std::pow(v - m, k);
        return acc / x.size();
    }

    Real skewness(const VecReal& x) {
        if (x.size() < 3) return NaN();
        Real m = mean(x);
        Real s = stddev(x);
        if (s == 0) return NaN();
        Real acc = 0;
        for (Real v : x) acc += std::pow((v - m) / s, 3);
        return acc / x.size();
    }

    Real kurtosis(const VecReal& x) {
        if (x.size() < 4) return NaN();
        Real m = mean(x);
        Real s = stddev(x);
        if (s == 0) return NaN();
        Real acc = 0;
        for (Real v : x) acc += std::pow((v - m) / s, 4);
        return acc / x.size() - 3.0;
    }

    // ==========================================================
    // ================= Order & Quantiles ======================
    // ==========================================================

    Real quantile(VecReal x, Real q) {
        if (x.empty() || q < 0 || q > 1) return NaN();
        std::sort(x.begin(), x.end());
        Real pos = q * (x.size() - 1);
        std::size_t i = static_cast<std::size_t>(pos);
        Real frac = pos - i;
        if (i + 1 < x.size())
            return x[i] * (1 - frac) + x[i + 1] * frac;
        return x[i];
    }

    Real percentile(VecReal x, Real p) {
        return quantile(std::move(x), p / 100.0);
    }

    struct Quartiles { Real q1, q2, q3; };

    Quartiles quartiles(VecReal x) {
        return {
            quantile(x, 0.25),
            quantile(x, 0.50),
            quantile(x, 0.75)
        };
    }

    Real iqr(VecReal x) {
        auto q = quartiles(std::move(x));
        return q.q3 - q.q1;
    }

    Real trimmed_mean(VecReal x, Real alpha) {
        if (x.empty() || alpha < 0 || alpha >= 0.5) return NaN();
        std::sort(x.begin(), x.end());
        std::size_t k = static_cast<std::size_t>(alpha * x.size());
        Real acc = 0;
        for (std::size_t i = k; i < x.size() - k; ++i)
            acc += x[i];
        return acc / (x.size() - 2 * k);
    }

    // ==========================================================
    // ================= Robust Statistics ======================
    // ==========================================================

    Real median_absolute_deviation(VecReal x) {
        Real m = median(x);
        for (Real& v : x) v = std::abs(v - m);
        return median(x);
    }

    Real winsorized_mean(VecReal x, Real alpha) {
        if (x.empty() || alpha < 0 || alpha >= 0.5) return NaN();
        std::sort(x.begin(), x.end());
        std::size_t k = static_cast<std::size_t>(alpha * x.size());
        for (std::size_t i = 0; i < k; ++i)
            x[i] = x[k];
        for (std::size_t i = x.size() - k; i < x.size(); ++i)
            x[i] = x[x.size() - k - 1];
        return mean(x);
    }

    Real huber_mean(const VecReal& x, Real delta) {
        if (x.empty() || delta <= 0) return NaN();
        Real m = mean(x);
        Real acc = 0;
        for (Real v : x) {
            Real d = v - m;
            acc += std::abs(d) <= delta ? v : m + delta * (d > 0 ? 1 : -1);
        }
        return acc / x.size();
    }

    Real biweight_mean(const VecReal& x) {
        if (x.empty()) return NaN();
        Real m = median(x);
        Real mad = median_absolute_deviation(x);
        if (mad == 0) return m;

        Real acc = 0, wsum = 0;
        for (Real v : x) {
            Real u = (v - m) / (9 * mad);
            if (std::abs(u) < 1) {
                Real w = std::pow(1 - u * u, 2);
                acc += v * w;
                wsum += w;
            }
        }
        return wsum > 0 ? acc / wsum : NaN();
    }

    Real snr(const VecReal& signal) {
        Real m = mean(signal);
        Real s = stddev(signal);
        return s != 0 ? m / s : NaN();
    }

    // ==========================================================
    // ================= Correlation & Dependence ===============
    // ==========================================================

    Real covariance(const VecReal& x, const VecReal& y) {
        if (x.size() != y.size() || x.size() < 2) return NaN();
        Real mx = mean(x), my = mean(y);
        Real acc = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            acc += (x[i] - mx) * (y[i] - my);
        return acc / (x.size() - 1);
    }

    Real correlation_pearson(const VecReal& x, const VecReal& y) {
        Real c = covariance(x, y);
        Real sx = stddev_unbiased(x);
        Real sy = stddev_unbiased(y);
        return (sx > 0 && sy > 0) ? c / (sx * sy) : NaN();
    }

    Real correlation_spearman(const VecReal& x, const VecReal& y) {
        VecReal rx = x, ry = y;
        auto rank = [](VecReal& v) {
            VecReal r(v.size());
            std::vector<std::size_t> idx(v.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                [&](auto a, auto b) { return v[a] < v[b]; });
            for (std::size_t i = 0; i < idx.size(); ++i)
                r[idx[i]] = i;
            v = r;
        };
        rank(rx);
        rank(ry);
        return correlation_pearson(rx, ry);
    }

    Real correlation_kendall(const VecReal& x, const VecReal& y) {
        if (x.size() != y.size()) return NaN();
        int concordant = 0, discordant = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            for (std::size_t j = i + 1; j < x.size(); ++j) {
                Real dx = x[i] - x[j];
                Real dy = y[i] - y[j];
                if (dx * dy > 0) ++concordant;
                else if (dx * dy < 0) ++discordant;
            }
        int total = concordant + discordant;
        return total > 0 ? Real(concordant - discordant) / total : NaN();
    }

    Real autocovariance(const VecReal& x, int lag) {
        if (lag < 0 || lag >= static_cast<int>(x.size())) return NaN();
        Real m = mean(x);
        Real acc = 0;
        for (std::size_t i = 0; i + lag < x.size(); ++i)
            acc += (x[i] - m) * (x[i + lag] - m);
        return acc / (x.size() - lag);
    }

    Real autocorrelation(const VecReal& x, int lag) {
        return autocovariance(x, lag) / variance(x);
    }

    Real cross_correlation(const VecReal& x, const VecReal& y, int lag) {
        if (x.size() != y.size() || x.empty()) return NaN();
        if (std::abs(lag) >= static_cast<int>(x.size())) return NaN();

        Real mx = mean(x);
        Real my = mean(y);

        Real acc = 0;
        std::size_t n = x.size();

        if (lag >= 0) {
            for (std::size_t i = 0; i + lag < n; ++i)
                acc += (x[i] - mx) * (y[i + lag] - my);
            return acc / (n - lag);
        } else {
            lag = -lag;
            for (std::size_t i = 0; i + lag < n; ++i)
                acc += (x[i + lag] - mx) * (y[i] - my);
            return acc / (n - lag);
        }
    }
    // ==========================================================
    // ================= Probability Distributions ==============
    // ==========================================================

    namespace dist {

        struct Normal {
            Real mu;
            Real sigma;
        };

        struct LogNormal {
            Real mu;
            Real sigma;
        };

        struct Exponential {
            Real lambda;
        };

        struct Gamma {
            Real k;
            Real theta;
        };

        struct Beta {
            Real alpha;
            Real beta;
        };

        struct Weibull {
            Real k;
            Real lambda;
        };

        struct Cauchy {
            Real x0;
            Real gamma;
        };

        struct StudentT {
            Real nu;
        };

        // ---- common interface ----

        Real pdf(const Normal& d, Real x) {
            if (d.sigma <= 0) return NaN();
            Real z = (x - d.mu) / d.sigma;
            return std::exp(-0.5 * z * z) / (d.sigma * std::sqrt(2 * M_PI));
        }

        Real cdf(const Normal& d, Real x) {
            if (d.sigma <= 0) return NaN();
            Real z = (x - d.mu) / (d.sigma * std::sqrt(2));
            return 0.5 * (1 + std::erf(z));
        }

        Real quantile(const Normal& d, Real p) {
            if (d.sigma <= 0 || p <= 0 || p >= 1) return NaN();
            return d.mu + d.sigma * std::sqrt(2) * boost::math::erf_inv(2 * p - 1);
        }

        Real log_likelihood(const Normal& d, const VecReal& data) {
            if (d.sigma <= 0 || data.empty()) return NaN();
            Real ll = 0;
            Real log_norm = std::log(d.sigma * std::sqrt(2 * M_PI));
            for (Real x : data) {
                Real z = (x - d.mu) / d.sigma;
                ll -= log_norm + 0.5 * z * z;
            }
            return ll;
        }

    } // namespace dist

    // ==========================================================
    // ================= Statistical Tests ======================
    // ==========================================================

    Real z_test(const VecReal& x, Real mu, Real sigma) {
        if (x.empty() || sigma <= 0) return NaN();
        Real z = (mean(x) - mu) / (sigma / std::sqrt(x.size()));
        return z;
    }

    Real t_test(const VecReal& x, Real mu) {
        if (x.size() < 2) return NaN();
        Real s = stddev_unbiased(x);
        if (s == 0) return NaN();
        return (mean(x) - mu) / (s / std::sqrt(x.size()));
    }

    Real welch_t_test(const VecReal& x, const VecReal& y) {
        if (x.size() < 2 || y.size() < 2) return NaN();
        Real mx = mean(x), my = mean(y);
        Real vx = variance_unbiased(x), vy = variance_unbiased(y);
        return (mx - my) / std::sqrt(vx / x.size() + vy / y.size());
    }

    Real mann_whitney_u(const VecReal& x, const VecReal& y) {
        if (x.empty() || y.empty()) return NaN();
        int u = 0;
        for (Real xi : x)
            for (Real yj : y)
                if (xi > yj) ++u;
        return static_cast<Real>(u);
    }

    Real wilcoxon_signed_rank(const VecReal& x, const VecReal& y) {
        if (x.size() != y.size()) return NaN();
        VecReal d;
        for (std::size_t i = 0; i < x.size(); ++i) {
            Real diff = x[i] - y[i];
            if (diff != 0) d.push_back(std::abs(diff));
        }
        std::sort(d.begin(), d.end());
        Real sum = 0;
        for (std::size_t i = 0; i < d.size(); ++i)
            sum += i + 1;
        return sum;
    }

    Real ks_test(const VecReal& x, const VecReal& y) {
        if (x.empty() || y.empty()) return NaN();
        VecReal xs = x, ys = y;
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());

        std::size_t i = 0, j = 0;
        Real d = 0;

        while (i < xs.size() && j < ys.size()) {
            Real v = std::min(xs[i], ys[j]);
            while (i < xs.size() && xs[i] <= v) ++i;
            while (j < ys.size() && ys[j] <= v) ++j;
            d = std::max(d,
                std::abs(Real(i) / xs.size() - Real(j) / ys.size()));
        }
        return d;
    }

    Real chi_square_test(const VecReal& observed, const VecReal& expected) {
        if (observed.size() != expected.size()) return NaN();
        Real chi2 = 0;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            if (expected[i] <= 0) return NaN();
            Real diff = observed[i] - expected[i];
            chi2 += diff * diff / expected[i];
        }
        return chi2;
    }

    Real anderson_darling(const VecReal& x) {
        if (x.size() < 2) return NaN();
        VecReal xs = x;
        std::sort(xs.begin(), xs.end());
        Real m = mean(xs);
        Real s = stddev(xs);

        Real A2 = 0;
        for (std::size_t i = 0; i < xs.size(); ++i) {
            Real Fi = 0.5 * (1 + std::erf((xs[i] - m) / (s * std::sqrt(2))));
            Real Fj = 0.5 * (1 + std::erf((xs[xs.size() - 1 - i] - m) / (s * std::sqrt(2))));
            A2 += (2 * i + 1) * (std::log(Fi) + std::log(1 - Fj));
        }
        return -xs.size() - A2 / xs.size();
    }

    // ==========================================================
    // ================= Entropy & Information ==================
    // ==========================================================

    Real entropy(const VecReal& p) {
        Real h = 0;
        for (Real v : p)
            if (v > 0)
                h -= v * std::log(v);
        return h;
    }

    Real cross_entropy(const VecReal& p, const VecReal& q) {
        if (p.size() != q.size()) return NaN();
        Real h = 0;
        for (std::size_t i = 0; i < p.size(); ++i)
            if (p[i] > 0 && q[i] > 0)
                h -= p[i] * std::log(q[i]);
        return h;
    }

    Real kl_divergence(const VecReal& p, const VecReal& q) {
        if (p.size() != q.size()) return NaN();
        Real d = 0;
        for (std::size_t i = 0; i < p.size(); ++i)
            if (p[i] > 0 && q[i] > 0)
                d += p[i] * std::log(p[i] / q[i]);
        return d;
    }

    Real js_divergence(const VecReal& p, const VecReal& q) {
        if (p.size() != q.size()) return NaN();
        VecReal m(p.size());
        for (std::size_t i = 0; i < p.size(); ++i)
            m[i] = 0.5 * (p[i] + q[i]);
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m);
    }

    Real mutual_information(const VecReal& x, const VecReal& y) {
        if (x.size() != y.size()) return NaN();
        return entropy(x) + entropy(y) - entropy(x); // placeholder для дискретных joint
    }

    Real conditional_entropy(const VecReal& x, const VecReal& y) {
        return entropy(x) - mutual_information(x, y);
    }

    // ==========================================================
    // ================= Characteristic Functions ==============
    // ==========================================================

    std::complex<double> normal_characteristic(double t, double mu, double sigma) {
        double real = -0.5 * sigma * sigma * t * t;
        double imag = mu * t;
        return std::exp(std::complex<double>(real, imag));
    }

    std::complex<double> characteristic_from_samples(const std::vector<double>& samples,
                                                     double t) {
        std::complex<double> sum(0.0, 0.0);
        for (double x : samples)
            sum += std::exp(std::complex<double>(0.0, t * x));
        return sum / static_cast<double>(samples.size());
    }

    // ==========================================================
    // ================= Time Series / Chaos ====================
    // ==========================================================

    VecReal rolling_mean(const VecReal& x, std::size_t window) {
        VecReal out;
        if (window == 0 || x.size() < window) return out;

        out.resize(x.size() - window + 1);
        Real sum = std::accumulate(x.begin(), x.begin() + window, Real{0});

        out[0] = sum / window;
        for (std::size_t i = window; i < x.size(); ++i) {
            sum += x[i] - x[i - window];
            out[i - window + 1] = sum / window;
        }
        return out;
    }

    VecReal rolling_variance(const VecReal& x, std::size_t window) {
        VecReal out;
        if (window == 0 || x.size() < window) return out;

        out.resize(x.size() - window + 1);

        Real mean = 0, M2 = 0;
        for (std::size_t i = 0; i < window; ++i) {
            Real delta = x[i] - mean;
            mean += delta / (i + 1);
            M2 += delta * (x[i] - mean);
        }
        out[0] = M2 / window;

        for (std::size_t i = window; i < x.size(); ++i) {
            Real old = x[i - window];
            Real neu = x[i];

            Real delta = neu - old;
            mean += delta / window;
            M2 += delta * (neu - mean + old - mean);

            out[i - window + 1] = M2 / window;
        }
        return out;
    }

    VecReal ema(const VecReal& x, Real alpha) {
        VecReal out;
        if (x.empty() || alpha <= 0 || alpha > 1) return out;

        out.resize(x.size());
        out[0] = x[0];
        for (std::size_t i = 1; i < x.size(); ++i)
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1];

        return out;
    }

    Real partial_autocorrelation(const VecReal& x, int lag) {
        if (lag <= 0 || lag >= static_cast<int>(x.size())) return NAN;

        // Yule–Walker (наивно, O(n²))
        std::vector<Real> r(lag + 1);
        for (int i = 0; i <= lag; ++i)
            r[i] = autocovariance(x, i);

        std::vector<Real> phi(lag + 1);
        phi[1] = r[1] / r[0];

        for (int k = 2; k <= lag; ++k) {
            Real num = r[k];
            Real den = r[0];
            for (int j = 1; j < k; ++j)
                num -= phi[j] * r[k - j];
            for (int j = 1; j < k; ++j)
                den -= phi[j] * r[j];

            phi[k] = num / den;
        }
        return phi[lag];
    }

    Real hurst_exponent(const VecReal& x) {
        if (x.size() < 20) return NAN;

        VecReal y(x.size());
        std::partial_sum(x.begin(), x.end(), y.begin());

        Real mean = y.back() / y.size();
        for (auto& v : y) v -= mean;

        Real R = *std::max_element(y.begin(), y.end()) -
                 *std::min_element(y.begin(), y.end());

        Real S = std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), Real{0}) / x.size());
        return std::log(R / S) / std::log(x.size());
    }

    VecReal detrend(const VecReal& x) {
        VecReal out(x.size());
        if (x.empty()) return out;

        Real n = static_cast<Real>(x.size());
        Real sumX = (n - 1) * n / 2;
        Real sumY = std::accumulate(x.begin(), x.end(), Real{0});
        Real sumXY = 0;
        Real sumXX = (n - 1) * n * (2 * n - 1) / 6;

        for (std::size_t i = 0; i < x.size(); ++i)
            sumXY += i * x[i];

        Real slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        Real intercept = (sumY - slope * sumX) / n;

        for (std::size_t i = 0; i < x.size(); ++i)
            out[i] = x[i] - (slope * i + intercept);

        return out;
    }

    VecReal difference(const VecReal& x, int order) {
        if (order <= 0) return x;
        if (x.size() <= static_cast<std::size_t>(order)) return {};

        VecReal out = x;
        for (int k = 0; k < order; ++k) {
            VecReal tmp(out.size() - 1);
            for (std::size_t i = 1; i < out.size(); ++i)
                tmp[i - 1] = out[i] - out[i - 1];
            out.swap(tmp);
        }
        return out;
    }

    Real lyapunov_exponent(const VecReal& x) {
        if (x.size() < 10) return NAN;

        Real sum = 0;
        for (std::size_t i = 1; i < x.size(); ++i)
            sum += std::log(std::abs(x[i] / x[i - 1]));

        return sum / (x.size() - 1);
    }

    [[nodiscard]] VecReal takens_map(const VecReal& signal, int dim, int tau) {
        if (dim <= 0 || tau <= 0 || signal.empty()) return {};
        size_t N = signal.size();
        size_t n_embedded = N < static_cast<size_t>((dim - 1) * tau) ? 0 : N - (dim - 1) * tau;
        VecReal embedded;
        embedded.reserve(n_embedded * dim);

        for (size_t i = 0; i < n_embedded; ++i) {
            for (int j = 0; j < dim; ++j) {
                embedded.push_back(signal[i + j * tau]);
            }
        }
        return embedded;
    }

    // ==========================================================
    // ================= Sampling & Resampling ==================
    // ==========================================================

    Real bootstrap_mean(const VecReal& x, int n) {
        if (x.empty() || n <= 0) return NAN;

        std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<> d(0, x.size() - 1);

        Real acc = 0;
        for (int i = 0; i < n; ++i) {
            Real sum = 0;
            for (std::size_t j = 0; j < x.size(); ++j)
                sum += x[d(gen)];
            acc += sum / x.size();
        }
        return acc / n;
    }

    std::pair<Real, Real> bootstrap_ci(const VecReal& x, Real alpha) {
        VecReal samples;
        int n = 1000;

        std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<> d(0, x.size() - 1);

        for (int i = 0; i < n; ++i) {
            Real sum = 0;
            for (std::size_t j = 0; j < x.size(); ++j)
                sum += x[d(gen)];
            samples.push_back(sum / x.size());
        }

        std::sort(samples.begin(), samples.end());
        std::size_t lo = static_cast<std::size_t>((1 - alpha) / 2 * n);
        std::size_t hi = static_cast<std::size_t>((1 + alpha) / 2 * n);

        return {samples[lo], samples[hi]};
    }

    VecReal jackknife(const VecReal& x) {
        VecReal out(x.size());
        Real total = std::accumulate(x.begin(), x.end(), Real{0});

        for (std::size_t i = 0; i < x.size(); ++i)
            out[i] = (total - x[i]) / (x.size() - 1);

        return out;
    }

    Real permutation_test(const VecReal& x, const VecReal& y) {
        if (x.size() != y.size()) return NAN;

        VecReal z = x;
        z.insert(z.end(), y.begin(), y.end());

        Real obs = std::abs(
            std::accumulate(x.begin(), x.end(), Real{0}) / x.size() -
            std::accumulate(y.begin(), y.end(), Real{0}) / y.size()
        );

        int count = 0;
        int trials = 1000;

        std::mt19937 gen{std::random_device{}()};
        for (int i = 0; i < trials; ++i) {
            std::shuffle(z.begin(), z.end(), gen);
            Real m1 = std::accumulate(z.begin(), z.begin() + x.size(), Real{0}) / x.size();
            Real m2 = std::accumulate(z.begin() + x.size(), z.end(), Real{0}) / y.size();
            if (std::abs(m1 - m2) >= obs) ++count;
        }
        return static_cast<Real>(count) / trials;
    }

    // ==========================================================
    // ================= Regression & Estimation ================
    // ==========================================================

    struct LinearRegressionResult {
        Real slope;
        Real intercept;
        Real r2;
    };

    LinearRegressionResult linear_regression(const VecReal& x, const VecReal& y) {
        LinearRegressionResult r{};
        if (x.size() != y.size() || x.empty()) return r;

        Real mx = std::accumulate(x.begin(), x.end(), Real{0}) / x.size();
        Real my = std::accumulate(y.begin(), y.end(), Real{0}) / y.size();

        Real num = 0, den = 0, ss = 0, ssr = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            num += (x[i] - mx) * (y[i] - my);
            den += (x[i] - mx) * (x[i] - mx);
        }

        r.slope = num / den;
        r.intercept = my - r.slope * mx;

        for (std::size_t i = 0; i < x.size(); ++i) {
            Real yi = r.slope * x[i] + r.intercept;
            ss += (y[i] - my) * (y[i] - my);
            ssr += (yi - my) * (yi - my);
        }
        r.r2 = ssr / ss;
        return r;
    }

    VecReal polynomial_regression(const VecReal& x, const VecReal& y, int degree) {
        // Заглушка: предполагается нормальное уравнение / QR
        return {};
    }

    Real least_squares(const VecReal& residuals) {
        return std::inner_product(
            residuals.begin(), residuals.end(),
            residuals.begin(), Real{0}
        );
    }

    // ==========================================================
    // ================= Outliers ===============================
    // ==========================================================

    VecReal z_score(const VecReal& x) {
        if (x.empty()) return {};

        Real mean = std::accumulate(x.begin(), x.end(), Real{0}) / x.size();
        Real sq_sum = 0;
        for (auto v : x) sq_sum += (v - mean) * (v - mean);
        Real stddev = std::sqrt(sq_sum / x.size());

        VecReal out(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            out[i] = stddev == 0 ? 0 : (x[i] - mean) / stddev;

        return out;
    }

    VecReal modified_z_score(const VecReal& x) {
        if (x.empty()) return {};

        VecReal sorted = x;
        std::sort(sorted.begin(), sorted.end());
        Real median = sorted[sorted.size() / 2];

        VecReal deviations(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            deviations[i] = std::abs(x[i] - median);

        std::sort(deviations.begin(), deviations.end());
        Real mad = deviations[deviations.size() / 2];
        if (mad == 0) mad = 1e-12; // избегаем деления на ноль

        VecReal out(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            out[i] = 0.6745 * (x[i] - median) / mad;

        return out;
    }

    bool is_outlier(Real x, Real mean, Real stddev, Real threshold) {
        if (stddev == 0) return false;
        return std::abs(x - mean) > threshold * stddev;
    }

    // Груббс-тест (односторонний)
    bool grubbs_test(const VecReal& x, Real alpha) {
        if (x.size() < 3) return false;

        Real mean = std::accumulate(x.begin(), x.end(), Real{0}) / x.size();
        Real sq_sum = 0;
        for (auto v : x) sq_sum += (v - mean) * (v - mean);
        Real stddev = std::sqrt(sq_sum / x.size());

        auto it = std::max_element(x.begin(), x.end(), [mean](Real a, Real b){
            return std::abs(a - mean) < std::abs(b - mean);
        });
        Real G = std::abs(*it - mean) / stddev;

        // Критическое значение для α=0.05 (приблизительно)
        Real n = x.size();
        Real t = 1.96; // грубая аппроксимация
        Real Gcrit = (n - 1) / std::sqrt(n) * std::sqrt(t * t / (n - 2 + t * t));

        return G > Gcrit;
    }

    // Критерий Шевенета
    bool chauvenet_criterion(const VecReal& x) {
        if (x.empty()) return false;

        Real mean = std::accumulate(x.begin(), x.end(), Real{0}) / x.size();
        Real sq_sum = 0;
        for (auto v : x) sq_sum += (v - mean) * (v - mean);
        Real stddev = std::sqrt(sq_sum / x.size());

        for (auto v : x) {
            Real z = std::abs(v - mean) / stddev;
            Real prob = std::erfc(z / std::sqrt(2));
            if (prob * x.size() < 0.5) return true; // выброс
        }
        return false;
    }

} // namespace functions
