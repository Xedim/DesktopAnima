//Functions.cpp

#include "Pattern1D.h"
#include "../common/Utils.h"
#include "../analytics/Analytics.h"
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include <random>
#include <utility>

namespace Functions {

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
        if (std::abs(x) < 1e-5) return x - 0.5 * x * x + (1.0/3.0) * x * x * x;
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
        const Real invSqrtPiEps = 1.0 / std::sqrt(Constants::PI * eps);
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
        return x.empty() ? NaN() : sum(x) / static_cast<Real>(x.size());
    }

    Real median(VecReal x) {
        if (x.empty()) return NaN();
        auto mid = x.begin() + static_cast<std::ptrdiff_t>(x.size() / 2);
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
        Real stats_n = static_cast<Real>(stats.n);
        Real m = stats.sum / stats_n;
        return (stats.sumsq / stats_n) - (m * m);
    }

    Real variance_unbiased(const VecReal& x) {
        if (x.size() < 2) return NaN();
        auto stats = compute_stats(x);
        Real stats_n = static_cast<Real>(stats.n);
        Real m = stats.sum / stats_n;
        return ((stats.sumsq / stats_n) - m * m) * (stats_n / (stats_n - Real{1}));
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
        Real stats_n = static_cast<Real>(stats.n);
        Real mean_val = stats.sum / stats_n;
        Real acc = 0;
        for (Real v : x) acc += std::abs(v - mean_val);
        return acc / stats_n;
    }

    // ==========================================================
    // ================= Shape Statistics =======================
    // ==========================================================

    Real raw_moment(const VecReal& x, int k) {
        if (x.empty() || k < 0) return NaN();

        const std::size_t n = x.size();
        Real acc = 0;

        if (k == 0) return Real{1};

        for (Real v : x) {
            Real p = 1;
            for (int i = 0; i < k; ++i)
                p *= v;
            acc += p;
        }
        return acc / static_cast<Real>(n);
    }

    Real moment(const VecReal& x, int k) {
        if (x.empty() || k < 0) return NaN();

        const std::size_t n = x.size();
        const Real m = mean(x);

        if (k == 0) return Real{1};

        Real acc = 0;
        for (Real v : x) {
            Real d = v - m;
            Real p = 1;
            for (int i = 0; i < k; ++i)
                p *= d;
            acc += p;
        }
        return acc / static_cast<Real>(n);
    }

    Real skewness(const VecReal& x) {
        const std::size_t n = x.size();
        if (n < 3) return NaN();

        const Real m = mean(x);
        const Real s = stddev(x);
        if (s == 0) return NaN();

        const Real inv_s = Real{1} / s;
        Real acc = 0;

        for (Real v : x) {
            Real z = (v - m) * inv_s;
            acc += z * z * z;
        }
        return acc / static_cast<Real>(n);
    }

    Real kurtosis(const VecReal& x) {
        const std::size_t n = x.size();
        if (n < 4) return NaN();

        const Real m = mean(x);
        const Real s = stddev(x);
        if (s == 0) return NaN();

        const Real inv_s = Real{1} / s;
        Real acc = 0;

        for (Real v : x) {
            Real z = (v - m) * inv_s;
            Real z2 = z * z;
            acc += z2 * z2;
        }
        return acc / static_cast<Real>(n) - Real{3};
    }

    // ==========================================================
    // ================= Order & Quantiles ======================
    // ==========================================================

    Real quantile(VecReal x, Real q) {
        const std::size_t n = x.size();
        if (n == 0 || q < 0 || q > 1) return NaN();

        const Real pos = q * static_cast<Real>(n - 1);
        auto size_t i = static_cast<std::size_t>(pos);
        const Real frac = pos - static_cast<Real>(i);

        // приведение к signed difference type
        auto it_i = x.begin() + static_cast<std::vector<Real>::difference_type>(i);
        std::nth_element(x.begin(), it_i, x.end());
        const Real a = *it_i;

        if (frac == 0 || i + 1 == n)
            return a;

        auto it_i1 = x.begin() + static_cast<std::vector<Real>::difference_type>(i + 1);
        std::nth_element(x.begin(), it_i1, x.end());
        const Real b = *it_i1;

        return a + frac * (b - a);
    }


    Real percentile(VecReal x, Real p) {
        return quantile(std::move(x), p * Real(0.01));
    }

    struct Quartiles { Real q1, q2, q3; };

    Quartiles quartiles(VecReal x) {
        const std::size_t n = x.size();
        if (n == 0) return { NaN(), NaN(), NaN() };

        const std::size_t i1 = (n - 1) / 4;
        const std::size_t i2 = (n - 1) / 2;
        const std::size_t i3 = 3 * (n - 1) / 4;

        std::nth_element(x.begin(), x.begin() + i2, x.end());
        const Real q2 = x[i2];

        std::nth_element(x.begin(), x.begin() + i1, x.begin() + i2);
        const Real q1 = x[i1];

        std::nth_element(x.begin() + i2 + 1, x.begin() + i3, x.end());
        const Real q3 = x[i3];

        return { q1, q2, q3 };
    }


    Real iqr(VecReal x) {
        const auto q = quartiles(std::move(x));
        return q.q3 - q.q1;
    }

    Real trimmed_mean(VecReal x, Real alpha) {
        const std::size_t n = x.size();
        if (n == 0 || alpha < 0 || alpha >= Real{0.5}) return NaN();

        const std::size_t k = static_cast<std::size_t>(alpha * n);
        const std::size_t lo = k;
        const std::size_t hi = n - k;

        std::nth_element(x.begin(), x.begin() + lo, x.end());
        std::nth_element(x.begin() + lo, x.begin() + hi, x.end());

        Real acc = 0;
        for (std::size_t i = lo; i < hi; ++i)
            acc += x[i];

        return acc / (hi - lo);
    }


    // ==========================================================
    // ================= Robust Statistics ======================
    // ==========================================================

    Real median_absolute_deviation(VecReal x) {
        if (x.empty()) return NaN();
        const Real m = median(x);
        for (Real& v : x)
            v = std::abs(v - m);
        return median(x);
    }


    Real winsorized_mean(VecReal x, Real alpha) {
        if (x.empty() || alpha < 0 || alpha >= Real{0.5}) return NaN();

        const std::size_t n = x.size();
        const std::size_t k = static_cast<std::size_t>(alpha * n);

        std::nth_element(x.begin(), x.begin() + k, x.end());
        const Real lo = x[k];

        std::nth_element(x.begin(), x.end() - k - 1, x.end());
        const Real hi = x[n - k - 1];

        Real acc = 0;
        for (Real v : x) {
            if (v < lo) v = lo;
            else if (v > hi) v = hi;
            acc += v;
        }
        return acc / n;
    }


    Real huber_mean(const VecReal& x, Real delta) {
        if (x.empty() || delta <= 0) return NaN();

        const Real m = mean(x);
        Real acc = 0;

        for (Real v : x) {
            const Real d = v - m;
            if (std::abs(d) <= delta)
                acc += v;
            else
                acc += m + delta * (d > 0 ? 1 : -1);
        }
        return acc / x.size();
    }

    Real biweight_mean(const VecReal& x) {
        if (x.empty()) return NaN();

        const Real m = median(x);
        const Real mad = median_absolute_deviation(x);
        if (mad == 0) return m;

        const Real c = 9 * mad;
        Real acc = 0;
        Real wsum = 0;

        for (Real v : x) {
            const Real u = (v - m) / c;
            if (std::abs(u) < 1) {
                const Real t = 1 - u * u;
                const Real w = t * t;
                acc += v * w;
                wsum += w;
            }
        }
        return wsum > 0 ? acc / wsum : NaN();
    }

    Real snr(const VecReal& signal) {
        if (signal.empty()) return NaN();

        Real mean = 0;
        Real m2 = 0;
        std::size_t n = 0;

        for (Real x : signal) {
            ++n;
            const Real delta = x - mean;
            mean += delta / n;
            m2 += delta * (x - mean);
        }

        if (n < 2) return NaN();

        const Real variance = m2 / (n - 1);
        return variance > 0 ? mean / std::sqrt(variance) : NaN();
    }

    // ==========================================================
    // ================= Correlation & Dependence ===============
    // ==========================================================

    Real covariance(const VecReal& x, const VecReal& y) {
        const std::size_t n = x.size();
        if (n != y.size() || n < 2) return NaN();

        Real mean_x = 0, mean_y = 0;
        Real C = 0;

        for (std::size_t i = 0; i < n; ++i) {
            const Real dx = x[i] - mean_x;
            const Real dy = y[i] - mean_y;
            mean_x += dx / (i + 1);
            mean_y += dy / (i + 1);
            C += dx * (y[i] - mean_y);
        }
        return C / (n - 1);
    }

    Real correlation_pearson(const VecReal& x, const VecReal& y) {
        const std::size_t n = x.size();
        if (n != y.size() || n < 2) return NaN();

        Real mx = 0, my = 0;
        Real Sx = 0, Sy = 0, Sxy = 0;

        for (std::size_t i = 0; i < n; ++i) {
            const Real dx = x[i] - mx;
            const Real dy = y[i] - my;
            mx += dx / (i + 1);
            my += dy / (i + 1);
            Sx += dx * (x[i] - mx);
            Sy += dy * (y[i] - my);
            Sxy += dx * (y[i] - my);
        }

        if (Sx <= 0 || Sy <= 0) return NaN();
        return Sxy / std::sqrt(Sx * Sy);
    }

    Real correlation_spearman(const VecReal& x, const VecReal& y) {
        const std::size_t n = x.size();
        if (n != y.size() || n < 2) return NaN();

        std::vector<std::size_t> ix(n), iy(n);
        std::iota(ix.begin(), ix.end(), 0);
        std::iota(iy.begin(), iy.end(), 0);

        std::sort(ix.begin(), ix.end(), [&](auto a, auto b) { return x[a] < x[b]; });
        std::sort(iy.begin(), iy.end(), [&](auto a, auto b) { return y[a] < y[b]; });

        VecReal rx(n), ry(n);
        for (std::size_t i = 0; i < n; ++i) {
            rx[ix[i]] = static_cast<Real>(i);
            ry[iy[i]] = static_cast<Real>(i);
        }

        return correlation_pearson(rx, ry);
    }

    Real correlation_kendall(const VecReal& x, const VecReal& y) {
        const std::size_t n = x.size();
        if (n != y.size() || n < 2) return NaN();

        int concordant = 0, discordant = 0;

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const Real dx = x[i] - x[j];
                const Real dy = y[i] - y[j];
                const Real p = dx * dy;
                concordant += (p > 0);
                discordant += (p < 0);
            }
        }

        const int total = concordant + discordant;
        return total ? Real(concordant - discordant) / total : NaN();
    }

    Real autocovariance(const VecReal& x, int lag) {
        const std::size_t n = x.size();
        if (lag < 0 || lag >= static_cast<int>(n)) return NaN();

        const Real m = mean(x);
        Real acc = 0;

        for (std::size_t i = 0; i + lag < n; ++i)
            acc += (x[i] - m) * (x[i + lag] - m);

        return acc / (n - lag);
    }

    Real autocorrelation(const VecReal& x, int lag) {
        const std::size_t n = x.size();
        if (lag < 0 || lag >= static_cast<int>(n)) return NaN();

        Real mean = 0, var = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const Real d = x[i] - mean;
            mean += d / (i + 1);
            var += d * (x[i] - mean);
        }
        if (var <= 0) return NaN();

        Real acc = 0;
        for (std::size_t i = 0; i + lag < n; ++i)
            acc += (x[i] - mean) * (x[i + lag] - mean);

        return acc / var;
    }

    Real cross_correlation(const VecReal& x, const VecReal& y, int lag) {
        const std::size_t n = x.size();
        if (n != y.size() || n == 0) return NaN();
        if (std::abs(lag) >= static_cast<int>(n)) return NaN();

        const Real mx = mean(x);
        const Real my = mean(y);

        Real acc = 0;
        const std::size_t L = std::abs(lag);

        if (lag >= 0) {
            for (std::size_t i = 0; i + L < n; ++i)
                acc += (x[i] - mx) * (y[i + L] - my);
        } else {
            for (std::size_t i = 0; i + L < n; ++i)
                acc += (x[i + L] - mx) * (y[i] - my);
        }

        return acc / (n - L);
    }

    // ==========================================================
    // ================= Probability Distributions ==============
    // ==========================================================

    namespace dist {

        struct Normal {
            Real mu;
            Real sigma;

            // --- cached ---
            Real inv_sigma;
            Real inv_sigma_sqrt2;
            Real inv_sigma_sqrt2pi;
            Real log_norm;

            explicit Normal(Real mu_, Real sigma_)
                : mu(mu_), sigma(sigma_) {

                if (sigma_ > 0) {
                    inv_sigma = Real{1} / sigma_;
                    inv_sigma_sqrt2 = inv_sigma / Constants::SQRT2;
                    inv_sigma_sqrt2pi = inv_sigma / Constants::SQRT_2PI;
                    log_norm = std::log(sigma_ * Constants::SQRT_2PI);
                } else {
                    inv_sigma = inv_sigma_sqrt2 = inv_sigma_sqrt2pi = log_norm = NaN();
                }
            }
        };

        inline Real pdf(const Normal& d, Real x) {
            const Real z = (x - d.mu) * d.inv_sigma;
            return std::exp(-Real{0.5} * z * z) * d.inv_sigma_sqrt2pi;
        }

        inline Real cdf(const Normal& d, Real x) {
            const Real z = (x - d.mu) * d.inv_sigma_sqrt2;
            return Real{0.5} * (Real{1} + std::erf(z));
        }

        inline Real quantile(const Normal& d, Real p) {
            if (p <= 0 || p >= 1) return NaN();
            return d.mu + d.sigma * Constants::SQRT2 *
                   boost::math::erf_inv(Real{2} * p - Real{1});
        }

        Real log_likelihood(const Normal& d, const VecReal& data) {
            if (data.empty()) return NaN();

            Real acc = 0;
            const Real inv_sigma = d.inv_sigma;
            const Real log_norm = d.log_norm;

            // tight loop, SIMD-friendly
            for (Real x : data) {
                const Real z = (x - d.mu) * inv_sigma;
                acc += z * z;
            }

            return -Real{0.5} * acc - Real{data.size()} * log_norm;
        }

        struct LogNormal {
            Real mu;
            Real sigma;

            // cached
            Real inv_sigma;
            Real log_norm;

            explicit LogNormal(Real mu_, Real sigma_)
                : mu(mu_), sigma(sigma_) {

                if (sigma_ > 0) {
                    inv_sigma = Real(1) / sigma_;
                    log_norm = std::log(sigma_ * Constants::SQRT_2PI);
                } else {
                    inv_sigma = log_norm = NaN();
                }
            }
        };

        inline Real pdf(const LogNormal& d, Real x) {
            if (x <= 0) return 0;
            Real lx = std::log(x);
            Real z = (lx - d.mu) * d.inv_sigma;
            return std::exp(-Real(0.5) * z * z) / (x * d.sigma * Constants::SQRT_2PI);
        }

        inline Real cdf(const LogNormal& d, Real x) {
            if (x <= 0) return 0;
            Real z = (std::log(x) - d.mu) * d.inv_sigma / Constants::SQRT2;
            return Real(0.5) * (Real(1) + std::erf(z));
        }

        inline Real quantile(const LogNormal& d, Real p) {
            if (p <= 0 || p >= 1) return NaN();
            return std::exp(d.mu + d.sigma * Constants::SQRT2 *
                            boost::math::erf_inv(Real(2) * p - Real(1)));
        }

        Real log_likelihood(const LogNormal& d, const VecReal& data) {
            Real acc = 0;
            Real sq = 0;

            for (Real x : data) {
                if (x <= 0) return NaN();
                Real lx = std::log(x);
                Real z = (lx - d.mu) * d.inv_sigma;
                sq += z * z;
                acc += lx;
            }

            return -Real(0.5) * sq
                   - Real(data.size()) * d.log_norm
                   - acc;
        }

        struct Exponential {
            Real lambda;

            explicit Exponential(Real lambda_)
                : lambda(lambda_) {}
        };

        inline Real pdf(const Exponential& d, Real x) {
            return (x < 0 || d.lambda <= 0) ? 0
                   : d.lambda * std::exp(-d.lambda * x);
        }

        inline Real cdf(const Exponential& d, Real x) {
            return (x < 0 || d.lambda <= 0) ? 0
                   : Real(1) - std::exp(-d.lambda * x);
        }

        inline Real quantile(const Exponential& d, Real p) {
            return (p <= 0 || p >= 1 || d.lambda <= 0)
                   ? NaN()
                   : -std::log(Real(1) - p) / d.lambda;
        }

        Real log_likelihood(const Exponential& d, const VecReal& data) {
            if (d.lambda <= 0) return NaN();

            Real sum = 0;
            for (Real x : data) {
                if (x < 0) return NaN();
                sum += x;
            }

            return Real(data.size()) * std::log(d.lambda)
                   - d.lambda * sum;
        }

        struct Weibull {
            Real k;
            Real lambda;

            Real inv_lambda;
            Real log_lambda;

            explicit Weibull(Real k_, Real lambda_)
                : k(k_), lambda(lambda_) {

                inv_lambda = Real(1) / lambda_;
                log_lambda = std::log(lambda_);
            }
        };

        inline Real pdf(const Weibull& d, Real x) {
            if (x < 0) return 0;
            Real t = x * d.inv_lambda;
            return (d.k / d.lambda) *
                   std::pow(t, d.k - 1) *
                   std::exp(-std::pow(t, d.k));
        }

        inline Real cdf(const Weibull& d, Real x) {
            if (x < 0) return 0;
            return Real(1) - std::exp(-std::pow(x * d.inv_lambda, d.k));
        }

        inline Real quantile(const Weibull& d, Real p) {
            if (p <= 0 || p >= 1) return NaN();
            return d.lambda * std::pow(-std::log(Real(1) - p), Real(1) / d.k);
        }

        Real log_likelihood(const Weibull& d, const VecReal& data) {
            Real sum_log = 0;
            Real sum_pow = 0;

            for (Real x : data) {
                if (x <= 0) return NaN();
                Real t = x * d.inv_lambda;
                sum_log += std::log(t);
                sum_pow += std::pow(t, d.k);
            }

            return Real(data.size()) * (std::log(d.k) - d.k * d.log_lambda)
                   + (d.k - 1) * sum_log
                   - sum_pow;
        }

        struct Cauchy {
            Real x0;
            Real gamma;

            Real inv_gamma;
            Real log_norm;

            explicit Cauchy(Real x0_, Real gamma_)
                : x0(x0_), gamma(gamma_) {

                inv_gamma = Real(1) / gamma_;
                log_norm = std::log(Constants::PI * gamma_);
            }
        };

        inline Real pdf(const Cauchy& d, Real x) {
            Real z = (x - d.x0) * d.inv_gamma;
            return Real(1) / (Constants::PI * d.gamma * (Real(1) + z * z));
        }

        inline Real cdf(const Cauchy& d, Real x) {
            return Real(0.5) + std::atan((x - d.x0) * d.inv_gamma) / Constants::PI;
        }

        inline Real quantile(const Cauchy& d, Real p) {
            if (p <= 0 || p >= 1) return NaN();
            return d.x0 + d.gamma * std::tan(Constants::PI * (p - Real(0.5)));
        }

        Real log_likelihood(const Cauchy& d, const VecReal& data) {
            Real acc = 0;
            for (Real x : data) {
                Real z = (x - d.x0) * d.inv_gamma;
                acc += std::log(Real(1) + z * z);
            }
            return -Real(data.size()) * d.log_norm - acc;
        }

        struct StudentT {
            Real nu;

            Real log_norm;

            explicit StudentT(Real nu_) : nu(nu_) {
                log_norm =
                    boost::math::lgamma((nu + 1) / 2) -
                    boost::math::lgamma(nu / 2) -
                    Real(0.5) * std::log(nu * Constants::PI);
            }
        };

        inline Real pdf(const StudentT& d, Real x) {
            Real t = Real(1) + (x * x) / d.nu;
            return std::exp(d.log_norm) * std::pow(t, -(d.nu + 1) / 2);
        }

        inline Real cdf(const StudentT& d, Real x) {
            if (d.nu <= 0) return NaN();
            boost::math::students_t_distribution<Real> t(d.nu);
            return boost::math::cdf(t, x);
        }

        inline Real quantile(const StudentT& d, Real p) {
            if (d.nu <= 0 || p <= 0 || p >= 1) return NaN();
            boost::math::students_t_distribution<Real> t(d.nu);
            return boost::math::quantile(t, p);
        }

        Real log_likelihood(const StudentT& d, const VecReal& data) {
            Real acc = 0;
            for (Real x : data)
                acc += std::log(Real(1) + (x * x) / d.nu);

            return Real(data.size()) * d.log_norm
                   - Real(0.5) * (d.nu + 1) * acc;
        }

    } // namespace dist

    // ==========================================================
    // ================= Statistical Tests ======================
    // ==========================================================

    inline Real z_test(const VecReal& x, Real mu, Real sigma) {
        if (x.empty() || sigma <= 0) return NaN();
        Real m = mean(x);
        return (m - mu) / (sigma / std::sqrt(x.size()));
    }

    inline Real t_test(const VecReal& x, Real mu) {
        if (x.size() < 2) return NaN();
        Real s = stddev_unbiased(x);
        if (s == 0) return NaN();
        Real m = mean(x);
        return (m - mu) / (s / std::sqrt(x.size()));
    }

    inline Real welch_t_test(const VecReal& x, const VecReal& y) {
        if (x.size() < 2 || y.size() < 2) return NaN();
        Real mx = mean(x), my = mean(y);
        Real vx = variance_unbiased(x), vy = variance_unbiased(y);
        return (mx - my) / std::sqrt(vx / x.size() + vy / y.size());
    }

    // Mann-Whitney U оптимизирован через вложенные циклы без временных массивов
    inline Real mann_whitney_u(const VecReal& x, const VecReal& y) {
        if (x.empty() || y.empty()) return NaN();
        Real u = 0;
        for (Real xi : x)
            for (Real yj : y)
                u += xi > yj ? 1 : 0;
        return u;
    }

    // Wilcoxon Signed-Rank
    inline Real wilcoxon_signed_rank(const VecReal& x, const VecReal& y) {
        if (x.size() != y.size()) return NaN();
        VecReal d;
        d.reserve(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            Real diff = x[i] - y[i];
            if (diff != 0) d.push_back(std::abs(diff));
        }
        std::sort(d.begin(), d.end());
        Real sum = 0;
        for (std::size_t i = 0; i < d.size(); ++i)
            sum += i + 1; // ранги
        return sum;
    }

    // Kolmogorov-Smirnov
    inline Real ks_test(const VecReal& x, const VecReal& y) {
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
            d = std::max(d, std::abs(Real(i)/xs.size() - Real(j)/ys.size()));
        }
        return d;
    }

    // Chi-square
    inline Real chi_square_test(const VecReal& observed, const VecReal& expected) {
        if (observed.size() != expected.size()) return NaN();
        Real chi2 = 0;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            if (expected[i] <= 0) return NaN();
            Real diff = observed[i] - expected[i];
            chi2 += diff * diff / expected[i];
        }
        return chi2;
    }

    // Anderson-Darling (для нормального распределения)
    inline Real anderson_darling(const VecReal& x) {
        if (x.size() < 2) return NaN();
        VecReal xs = x;
        std::sort(xs.begin(), xs.end());
        Real m = mean(xs);
        Real s = stddev(xs);

        Real A2 = 0;
        for (std::size_t i = 0; i < xs.size(); ++i) {
            Real Fi = 0.5 * (1 + std::erf((xs[i] - m) / (s * std::sqrt(2))));
            Real Fj = 0.5 * (1 + std::erf((xs[xs.size() - 1 - i] - m) / (s * std::sqrt(2))));
            A2 += (2*i + 1) * (std::log(Fi) + std::log(1 - Fj));
        }
        return -xs.size() - A2 / xs.size();
    }

    // ==========================================================
    // ================= Entropy & Information ==================
    // ==========================================================

    inline Real entropy(const VecReal& p) {
        Real h = 0;
        for (Real v : p)
            if (v > 0) h -= v * std::log(v);
        return h;
    }

    inline Real cross_entropy(const VecReal& p, const VecReal& q) {
        if (p.size() != q.size()) return NaN();
        Real h = 0;
        for (std::size_t i = 0; i < p.size(); ++i)
            if (p[i] > 0 && q[i] > 0) h -= p[i] * std::log(q[i]);
        return h;
    }

    inline Real kl_divergence(const VecReal& p, const VecReal& q) {
        if (p.size() != q.size()) return NaN();
        Real d = 0;
        for (std::size_t i = 0; i < p.size(); ++i)
            if (p[i] > 0 && q[i] > 0) d += p[i] * std::log(p[i] / q[i]);
        return d;
    }

    inline Real js_divergence(const VecReal& p, const VecReal& q) {
        if (p.size() != q.size()) return NaN();
        VecReal m(p.size());
        for (std::size_t i = 0; i < p.size(); ++i)
            m[i] = 0.5 * (p[i] + q[i]);
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m);
    }

    inline Real mutual_information(const VecReal& x, const VecReal& y) {
        if (x.size() != y.size()) return NaN();
        return entropy(x) + entropy(y) - entropy(x);
    }

    inline Real conditional_entropy(const VecReal& x, const VecReal& y) {
        return entropy(x) - mutual_information(x, y);
    }

    // ==========================================================
    // ================= Characteristic Functions ==============
    // ==========================================================

    inline Complex normal_characteristic(double t, double mu, double sigma) {
        // exp(i*mu*t - 0.5*sigma^2*t^2)
        return std::exp(Complex(-0.5 * sigma * sigma * t * t, mu * t));
    }

    inline Complex characteristic_from_samples(const std::vector<double>& samples, double t) {
        Complex sum{0.0, 0.0};
        for (double x : samples)
            sum += std::exp(Complex{0.0, t * x});
        return sum / static_cast<double>(samples.size());
    }

    // ==========================================================
    // ================= Time Series / Chaos ====================
    // ==========================================================

    // Rolling mean (O(n))
    inline VecReal rolling_mean(const VecReal& x, std::size_t window) {
        if (window == 0 || x.size() < window) return {};
        VecReal out(x.size() - window + 1);
        Real sum = std::accumulate(x.begin(), x.begin() + window, Real{0});
        out[0] = sum / window;
        for (std::size_t i = window; i < x.size(); ++i) {
            sum += x[i] - x[i - window];
            out[i - window + 1] = sum / window;
        }
        return out;
    }

    // Rolling variance (Welford)
    inline VecReal rolling_variance(const VecReal& x, std::size_t window) {
        if (window == 0 || x.size() < window) return {};
        VecReal out(x.size() - window + 1);

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

    // Exponential Moving Average
    inline VecReal ema(const VecReal& x, Real alpha) {
        if (x.empty() || alpha <= 0 || alpha > 1) return {};
        VecReal out(x.size());
        out[0] = x[0];
        for (std::size_t i = 1; i < x.size(); ++i)
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1];
        return out;
    }

    // Partial Autocorrelation (Yule-Walker, naive)
    inline Real partial_autocorrelation(const VecReal& x, int lag) {
        if (lag <= 0 || lag >= static_cast<int>(x.size())) return NaN();
        std::vector<Real> r(lag + 1);
        for (int i = 0; i <= lag; ++i)
            r[i] = autocovariance(x, i);

        std::vector<Real> phi(lag + 1, 0);
        phi[1] = r[1] / r[0];
        for (int k = 2; k <= lag; ++k) {
            Real num = r[k], den = r[0];
            for (int j = 1; j < k; ++j) {
                num -= phi[j] * r[k - j];
                den -= phi[j] * r[j];
            }
            phi[k] = num / den;
        }
        return phi[lag];
    }

    // Hurst Exponent (R/S method simplified)
    inline Real hurst_exponent(const VecReal& x) {
        if (x.size() < 20) return NaN();
        VecReal y(x.size());
        std::partial_sum(x.begin(), x.end(), y.begin());

        Real mean = y.back() / y.size();
        for (auto& v : y) v -= mean;

        Real R = *std::max_element(y.begin(), y.end()) - *std::min_element(y.begin(), y.end());
        Real S = std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), Real{0}) / x.size());
        return std::log(R / S) / std::log(x.size());
    }

    // Detrend via linear regression
    inline VecReal detrend(const VecReal& x) {
        if (x.empty()) return {};
        const Real n = static_cast<Real>(x.size());
        const Real sumX = (n - 1) * n / 2;
        const Real sumXX = (n - 1) * n * (2 * n - 1) / 6;
        Real sumY = std::accumulate(x.begin(), x.end(), Real{0});
        Real sumXY = 0;
        for (std::size_t i = 0; i < x.size(); ++i) sumXY += i * x[i];

        Real slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        Real intercept = (sumY - slope * sumX) / n;

        VecReal out(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            out[i] = x[i] - (slope * i + intercept);
        return out;
    }

    // Differencing
    inline VecReal difference(const VecReal& x, int order) {
        if (order <= 0) return x;
        if (x.size() <= static_cast<size_t>(order)) return {};
        VecReal out = x;
        for (int k = 0; k < order; ++k) {
            for (size_t i = 1; i < out.size(); ++i)
                out[i-1] = out[i] - out[i-1];
            out.resize(out.size() - 1);
        }
        return out;
    }

    // Lyapunov Exponent (naive)
    inline Real lyapunov_exponent(const VecReal& x) {
        if (x.size() < 10) return NaN();
        Real sum = 0;
        for (std::size_t i = 1; i < x.size(); ++i)
            sum += std::log(std::abs(x[i] / x[i - 1]));
        return sum / (x.size() - 1);
    }

    // Takens Embedding
    [[nodiscard]] inline VecReal takens_map(const VecReal& signal, int dim, int tau) {
        if (dim <= 0 || tau <= 0 || signal.empty()) return {};
        size_t N = signal.size();
        if (N < static_cast<size_t>((dim - 1) * tau + 1)) return {};
        size_t n_embedded = N - (dim - 1) * tau;
        VecReal embedded;
        embedded.reserve(n_embedded * dim);

        for (size_t i = 0; i < n_embedded; ++i)
            for (int j = 0; j < dim; ++j)
                embedded.push_back(signal[i + j * tau]);

        return embedded;
    }

    // ==========================================================
    // ================= Sampling & Resampling ==================
    // ==========================================================

    inline Real bootstrap_mean(const VecReal& x, int n) {
        if (x.empty() || n <= 0) return NaN();

        std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<std::size_t> d(0, x.size() - 1);

        Real acc = 0;
        for (int i = 0; i < n; ++i) {
            Real sum = 0;
            for (std::size_t j = 0; j < x.size(); ++j)
                sum += x[d(gen)];
            acc += sum / x.size();
        }
        return acc / n;
    }

    inline RealPair bootstrap_ci(const VecReal& x, Real alpha, int n = 1000) {
        if (x.empty() || alpha <= 0 || alpha >= 1) return {NaN(), NaN()};

        std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<std::size_t> d(0, x.size() - 1);
        VecReal samples;
        samples.reserve(n);

        for (int i = 0; i < n; ++i) {
            Real sum = 0;
            for (std::size_t j = 0; j < x.size(); ++j)
                sum += x[d(gen)];
            samples.push_back(sum / x.size());
        }

        std::nth_element(samples.begin(), samples.begin() + static_cast<std::size_t>((1 - alpha) / 2 * n), samples.end());
        std::nth_element(samples.begin(), samples.begin() + static_cast<std::size_t>((1 + alpha) / 2 * n), samples.end());

        std::size_t lo = static_cast<std::size_t>((1 - alpha) / 2 * n);
        std::size_t hi = static_cast<std::size_t>((1 + alpha) / 2 * n);
        return {samples[lo], samples[hi]};
    }

    inline VecReal jackknife(const VecReal& x) {
        if (x.empty()) return {};
        Real total = std::accumulate(x.begin(), x.end(), Real{0});
        VecReal out(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            out[i] = (total - x[i]) / (x.size() - 1);
        return out;
    }

    inline Real permutation_test(const VecReal& x, const VecReal& y, int trials = 1000) {
        if (x.size() != y.size() || x.empty()) return NaN();
        VecReal z;
        z.reserve(x.size() + y.size());
        z.insert(z.end(), x.begin(), x.end());
        z.insert(z.end(), y.begin(), y.end());

        Real obs = std::abs(mean(x) - mean(y));

        std::mt19937 gen{std::random_device{}()};
        int count = 0;

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
        Real slope = NaN();
        Real intercept = NaN();
        Real r2 = NaN();
    };

    inline LinearRegressionResult linear_regression(const VecReal& x, const VecReal& y) {
        if (x.size() != y.size() || x.empty()) return {};
        const std::size_t n = x.size();

        Real mx = mean(x);
        Real my = mean(y);

        Real num = 0, den = 0;
        for (std::size_t i = 0; i < n; ++i) {
            Real dx = x[i] - mx;
            num += dx * (y[i] - my);
            den += dx * dx;
        }

        LinearRegressionResult r;
        if (den == 0) return r; // вырождённый случай
        r.slope = num / den;
        r.intercept = my - r.slope * mx;

        Real ss_tot = 0, ss_reg = 0;
        for (std::size_t i = 0; i < n; ++i) {
            Real yi_hat = r.slope * x[i] + r.intercept;
            ss_tot += (y[i] - my) * (y[i] - my);
            ss_reg += (yi_hat - my) * (yi_hat - my);
        }
        r.r2 = (ss_tot != 0) ? ss_reg / ss_tot : NaN();

        return r;
    }

    inline VecReal polynomial_regression(const VecReal& x, const VecReal& y, int degree) {
        if (x.size() != y.size() || x.empty() || degree < 0) return {};

        const std::size_t n = x.size();
        const int m = degree + 1;

        // --- 1. Построение матрицы Вандермонда ---
        std::vector<VecReal> X(n, VecReal(m, 1.0));
        for (std::size_t i = 0; i < n; ++i) {
            for (int j = 1; j < m; ++j) {
                X[i][j] = X[i][j - 1] * x[i]; // x^j
            }
        }

        // --- 2. Формируем нормальные уравнения: A * coef = b ---
        std::vector<VecReal> A(m, VecReal(m, 0.0));
        VecReal b(m, 0.0);

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                for (std::size_t k = 0; k < n; ++k)
                    A[i][j] += X[k][i] * X[k][j];
            }
            for (std::size_t k = 0; k < n; ++k)
                b[i] += X[k][i] * y[k];
        }

        // --- 3. Решаем систему методом Гаусса с частичным выбором главного элемента ---
        VecReal coef(m, 0.0);
        std::vector<int> idx(m);
        for (int i = 0; i < m; ++i) idx[i] = i;

        for (int i = 0; i < m; ++i) {
            // Поиск максимального элемента по столбцу для устойчивости
            int pivot = i;
            Real max_val = std::abs(A[i][i]);
            for (int j = i + 1; j < m; ++j) {
                if (std::abs(A[j][i]) > max_val) {
                    pivot = j;
                    max_val = std::abs(A[j][i]);
                }
            }
            if (pivot != i) {
                std::swap(A[i], A[pivot]);
                std::swap(b[i], b[pivot]);
            }

            // Прямой ход
            for (int j = i + 1; j < m; ++j) {
                Real factor = A[j][i] / A[i][i];
                for (int k = i; k < m; ++k)
                    A[j][k] -= factor * A[i][k];
                b[j] -= factor * b[i];
            }
        }

        // Обратный ход
        for (int i = m - 1; i >= 0; --i) {
            Real sum = b[i];
            for (int j = i + 1; j < m; ++j)
                sum -= A[i][j] * coef[j];
            coef[i] = sum / A[i][i];
        }

        return coef; // coef[0] + coef[1]*x + coef[2]*x^2 + ...
    }

    inline Real least_squares(const VecReal& residuals) {
        return std::inner_product(residuals.begin(), residuals.end(), residuals.begin(), Real{0});
    }

    // ==========================================================
    // ================= Outliers ===============================
    // ==========================================================

    inline VecReal z_score(const VecReal& x) {
        if (x.empty()) return {};
        const std::size_t n = x.size();

        Real mean = std::accumulate(x.begin(), x.end(), Real{0}) / n;

        Real sq_sum = 0;
        for (Real v : x) {
            Real d = v - mean;
            sq_sum += d * d;
        }
        Real stddev = std::sqrt(sq_sum / n);
        if (stddev == 0) stddev = 1; // защита от деления на ноль

        VecReal out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = (x[i] - mean) / stddev;

        return out;
    }

    inline VecReal modified_z_score(const VecReal& x) {
        if (x.empty()) return {};
        const std::size_t n = x.size();

        VecReal sorted = x;
        std::nth_element(sorted.begin(), sorted.begin() + n / 2, sorted.end());
        Real median = sorted[n / 2];

        VecReal deviations(n);
        for (std::size_t i = 0; i < n; ++i)
            deviations[i] = std::abs(x[i] - median);

        std::nth_element(deviations.begin(), deviations.begin() + n / 2, deviations.end());
        Real mad = deviations[n / 2];
        if (mad == 0) mad = 1e-12; // защита от деления на ноль

        VecReal out(n);
        const Real scale = 0.6745; // стандартная константа для модифицированного Z
        for (std::size_t i = 0; i < n; ++i)
            out[i] = scale * (x[i] - median) / mad;

        return out;
    }

    inline bool is_outlier(Real x, Real mean, Real stddev, Real threshold) {
        return stddev > 0 && std::abs(x - mean) > threshold * stddev;
    }

    // Грубс-тест (односторонний)
    inline bool grubbs_test(const VecReal& x, Real alpha = 0.05) {
        const std::size_t n = x.size();
        if (n < 3) return false;

        Real mean = std::accumulate(x.begin(), x.end(), Real{0}) / n;
        Real sq_sum = 0;
        Real max_dev = 0;
        for (Real v : x) {
            Real d = v - mean;
            sq_sum += d * d;
            max_dev = std::max(max_dev, std::abs(d));
        }
        Real stddev = std::sqrt(sq_sum / n);
        if (stddev == 0) return false;

        Real G = max_dev / stddev;

        // Критическое значение Грубса (приблизительно)
        Real t = 1.96; // для alpha=0.05
        Real Gcrit = (n - 1) / std::sqrt(n) * std::sqrt(t * t / (n - 2 + t * t));

        return G > Gcrit;
    }

    // Критерий Шевенета
    inline bool chauvenet_criterion(const VecReal& x) {
        const std::size_t n = x.size();
        if (n == 0) return false;

        Real mean = std::accumulate(x.begin(), x.end(), Real{0}) / n;
        Real sq_sum = 0;
        for (Real v : x) {
            Real d = v - mean;
            sq_sum += d * d;
        }
        Real stddev = std::sqrt(sq_sum / n);
        if (stddev == 0) return false;

        const Real inv_sqrt2 = 1.0 / std::sqrt(2.0);
        for (Real v : x) {
            Real z = std::abs(v - mean) / stddev;
            Real prob = std::erfc(z * inv_sqrt2);
            if (prob * n < 0.5) return true;
        }
        return false;
    }

} // namespace functions
