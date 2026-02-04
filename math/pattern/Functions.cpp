//Functions.cpp

#include "Functions.h"

#include "../common/Utils.h"
#include "../common/Constants.h"
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/exponential.hpp>
#include <boost/math/distributions/weibull.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/cauchy.hpp>
#include <mutex>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include <random>
#include <utility>

namespace Functions {

    // ===========================================
    // ================= Sign ====================
    // ===========================================

    [[nodiscard]] Real sign(Real x) {
        return (x > 0) ? Real{1} : (x < 0 ? Real{-1} : Real{0});
    }

    [[nodiscard]] Real abs(Real x) { return std::abs(x); }

    [[nodiscard]] Real heaviside(Real x) { return x < 0.0 ? 0.0 : 1.0; }

    // =============================================
    // ================= Algebraic =================
    // =============================================

    [[nodiscard]] Real factorial(int n) {
        if (n < 0) return NaN();
        if (n < Constants::FACTORIAL_CACHE_SIZE) return Constants::factorial_cache[n];
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
        if (r < 0) r += Functions::abs(y);
        return r;
    }

    [[nodiscard]] Real polynomial(Real x, const VecReal& coefficients) {
        Real result = 0;
        for (const Real& c : std::views::reverse(coefficients)) {
            result = result * x + c;
        }
        return result;
    }

    [[nodiscard]] Real rational(Real x, const VecReal& p, const VecReal& q) {
        Real num = polynomial(x, p);
        Real den = polynomial(x, q);

        if (den == 0) return NaN();

        return num / den;
    }

    // ================================================
    // ================= Power / Root =================
    // ================================================

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
        if (x == Real{0}) {
            if (alpha == 0.0) return 1.0;
            if (alpha > 0.0) return 0.0;
            return std::numeric_limits<Real>::infinity();
        }
        if (alpha == 0.0) return 1.0;
        if (alpha == 0.5) return std::sqrt(x);
        if (alpha == 1.0) return x;
        if (alpha == 2.0) return x*x;
        if (x > 0.0 && std::floor(alpha) == alpha && alpha >= 0 && alpha <= 20) {
            return powCache.get(x, static_cast<int>(alpha));
        }
        if (x < 0.0 && std::floor(alpha) != alpha) return NaN();
        return std::pow(x, alpha);
    }

    [[nodiscard]] Real x_pow_y(Real x, Real y) {
        if (x == 0.0) return (y > 0.0) ? 0.0 : NaN(); // 0^y
        if (x == 1.0 || y == 0.0) return 1.0;         // 1^y, x^0
        if (y == 1.0) return x;                        // x^1
        if (x > 0.0) return std::exp(y * std::log(x));
        return NaN();
    }

    [[nodiscard]] Real algebraic_root(Real x, const VecReal& coefficients) {
        Real poly = Real{0};
        for (const Real& c : std::views::reverse(coefficients)) {
            poly = poly * x + c;
        }
        if (poly < 0) return NaN();
        return std::sqrt(poly);
    }

    [[nodiscard]] Real sqrt(Real x) { return x < Real{0} ? NaN() : std::sqrt(x); }

    [[nodiscard]] Real sqrt1pm1(Real x) {
        if (x < -1.0) return NaN();
        if (abs(x) < 1e-8) return x / 2.0;
        return std::sqrt(1.0 + x) - 1.0;
    }

    [[nodiscard]] Real cbrt(Real x) { return std::cbrt(x); }

    // =============================================================
    // ================= Exponential /Logarithmic ==================
    // =============================================================

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
        if (abs(x) < 1e-5) return x + 0.5*x*x; // Тейлор для маленьких x
        return std::expm1(x);
    }

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

    [[nodiscard]] Real log(Real x) {
        if (x <= Real{0}) return NaN();
        return std::log(x);
    }

    [[nodiscard]] Real log_a(Real x, Real a) {
        if (x <= Real{0} || a <= Real{0} || a == Real{1}) return NaN();
        Real ln_a = logBaseCache.get(a);
        return std::log(x) / ln_a;
    }

    [[nodiscard]] Real log2(Real x) { return log_a(x, 2.0); }
    [[nodiscard]] Real log10(Real x) { return log_a(x, 10.0); }

    [[nodiscard]] Real log1p(Real x) {
        if (x <= -1.0) return NaN();
        if (abs(x) < 1e-5) return x - 0.5 * x * x + (1.0 / 3.0) * x * x * x;
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
        if (abs(x) < 1e-5) {
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
        if (abs(x) > Real{700}) return NaN();
        return std::sinh(x);
    }

    [[nodiscard]] Real cosh(Real x) {
        if (!std::isfinite(x)) return NaN();
        if (abs(x) > Real{700}) return NaN();
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
            if (x <= 0.0 && std::floor(x) == x) return NaN(); // отрицательные целые
            if (x >= 1 && x < SIZE && std::floor(x) == x) {
                std::lock_guard<std::mutex> lock(mtx);
                if (last_x != x) {
                    last_x = x;
                    for (int i = 0; i < SIZE; ++i)
                        values[i] = std::tgamma(static_cast<Real>(i));
                }
                return values[static_cast<int>(x)];
            }
            return std::tgamma(x); // дробные и x >= SIZE
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
        if (l < 0 || abs(m) > l || x < -1.0 || x > 1.0) return NaN();
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

    [[nodiscard]] Real geometric_sum(Real a, int N) {
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

    [[nodiscard]] Real weierstrass(Real x, Real a, Real b, int N, StabPolicy policy)
    {
        if (a <= Constants::WEIERSTRASS_AMP_MIN || a >= Constants::WEIERSTRASS_AMP_MAX ||
            b <= Constants::WEIERSTRASS_FREQ_MIN ||
            N <= Constants::WEIERSTRASS_ITER_MIN ||
            x < Constants::WEIERSTRASS_X_MIN ||
            x > Constants::WEIERSTRASS_X_MAX)
            return NaN();

        const Real pi_x = Constants::PI * x;

        auto f = [=](const WeierState& s) -> WeierState {
            Real term = s.amp * std::cos(s.freq * pi_x);

            term = Utils::checkStability(
                term,
                Constants::WEIERSTRASS_Y_MIN,
                Constants::WEIERSTRASS_Y_MAX,
                policy
            );

            if (!std::isfinite(term))
                return WeierState{NaN(), NaN(), NaN()};

            return {
                s.sum + term,
                s.amp * a,
                s.freq * b
            };
        };

        WeierState init{0.0, 1.0, 1.0};

        auto res = Functions::iterate<WeierState>(
            init,
            f,
            N
        );

        return Utils::checkStability(
            res.sum,
            Constants::WEIERSTRASS_Y_MIN,
            Constants::WEIERSTRASS_Y_MAX,
            policy
        );
    }

    [[nodiscard]] Real cantor(Real x, int max_iter, StabPolicy policy)
    {
        if (x < Constants::CANTOR_X_MIN || x > Constants::CANTOR_X_MAX ||
            max_iter <= Constants::CANTOR_ITER_MIN)
            return NaN();

        auto f = [](const CantorState& s) {
            if (s.x < Constants::CANTOR_LEFT) {
                return CantorState{
                    s.x * Constants::CANTOR_SCALE,
                    s.result,
                    s.scale * Constants::CANTOR_FACTOR
                };
            }

            if (s.x > Constants::CANTOR_RIGHT) {
                return CantorState{
                    Constants::CANTOR_SCALE * s.x - Constants::CANTOR_RIGHT_SCALE,
                    s.result + s.scale,
                    s.scale * Constants::CANTOR_FACTOR
                };
            }

            return CantorState{
                s.x,
                s.result + s.scale * 0.5,
                0.0
            };
        };

        CantorState init{x, 0.0, 1.0};

        auto res = Functions::iterate<CantorState>(
            init,
            f,
            max_iter
        );

        return Utils::checkStability(
            res.result,
            Constants::CANTOR_Y_MIN,
            Constants::CANTOR_Y_MAX,
            policy
        );
    }

    [[nodiscard]] Real logistic(Real x, Real r, int n, StabPolicy policy) {
        if (x < Constants::LOGISTIC_X_MIN || x > Constants::LOGISTIC_X_MAX ||
            r <= Constants::LOGISTIC_R_MIN || r >= Constants::LOGISTIC_R_MAX)
            return NaN();

        auto f = [r](Real x) {
            return r * x * (Real{1} - x);
        };

        return Functions::iterate(
            x,
            f,
            n
        );
    }

    [[nodiscard]] Real tent(Real x, int n, StabPolicy policy) {
        if (x < Constants::TENT_X_MIN || x > Constants::TENT_X_MAX)
            return NaN();

        auto f = [](Real x) {
            return (x < Constants::TENT_PEAK)
                 ? Constants::TENT_SLOPE * x
                 : Constants::TENT_SLOPE * (Real{1} - x);
        };

        return Functions::iterate<Real>(
            x,
            f,
            n
        );
    }

    [[nodiscard]] Complex julia(const Complex& z0,
                                const Complex& c,
                                int n,
                                StabPolicy policy)
    {
        auto f = [c](Complex z) {
            return z * z + c;
        };

        return Functions::iterate<Complex>(
            z0,
            f,
            n
        );
    }

    [[nodiscard]] bool escapes(Complex z0, Complex c, int max_iter, Real threshold) {
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

    // ==========================================================
    // ================= Descriptive Statistics =================
    // ==========================================================

    [[nodiscard]] Real sum(const VecReal& x) {
        return x.empty() ? NaN() : std::accumulate(x.begin(), x.end(), Real{0});
    }

    [[nodiscard]] Real mean(const VecReal& x) {
        return sum(x) / static_cast<Real>(x.size());
    }

    [[nodiscard]] Real median(VecReal x) {
        if (x.empty()) return NaN();
        auto mid = x.begin() + static_cast<std::ptrdiff_t>(x.size() / 2);
        std::ranges::nth_element(x, mid);
        Real m = *mid;
        if (x.size() % 2 == 0) {
            auto max_it = std::max_element(x.begin(), mid);
            m = (*max_it + m) * Real{0.5};
        }
        return m;
    }

    [[nodiscard]] Real mode(const VecReal& x) {
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

    [[nodiscard]] Real min(const VecReal& x) {
        return x.empty() ? NaN() : *std::ranges::min_element(x);
    }

    [[nodiscard]] Real max(const VecReal& x) {
        return x.empty() ? NaN() : *std::ranges::max_element(x);
    }

    [[nodiscard]] Real range(const VecReal& x) {
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

    [[nodiscard]] Real variance(const VecReal& x) {
        if (x.size() < 2) return NaN();
        auto stats = compute_stats(x);
        Real stats_n = static_cast<Real>(stats.n);
        Real m = stats.sum / stats_n;
        return (stats.sumsq / stats_n) - (m * m);
    }

    [[nodiscard]] Real variance_unbiased(const VecReal& x) {
        if (x.size() < 2) return NaN();
        auto stats = compute_stats(x);
        Real stats_n = static_cast<Real>(stats.n);
        Real m = stats.sum / stats_n;
        return ((stats.sumsq / stats_n) - m * m) * (stats_n / (stats_n - Real{1}));
    }

    [[nodiscard]] Real stddev(const VecReal& x) {
        Real v = variance(x);
        return std::isfinite(v) ? std::sqrt(v) : NaN();
    }

    [[nodiscard]] Real stddev_unbiased(const VecReal& x) {
        Real v = variance_unbiased(x);
        return std::isfinite(v) ? std::sqrt(v) : NaN();
    }

    [[nodiscard]] Real mean_absolute_deviation(const VecReal& x) {
        if (x.empty()) return NaN();
        auto stats = compute_stats(x);
        Real stats_n = static_cast<Real>(stats.n);
        Real mean_val = stats.sum / stats_n;
        Real acc = 0;
        for (Real v : x) acc += abs(v - mean_val);
        return acc / stats_n;
    }

    // ==========================================================
    // ================= Shape Statistics =======================
    // ==========================================================

    [[nodiscard]] Real raw_moment(const VecReal& x, int k) {
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

    [[nodiscard]] Real moment(const VecReal& x, int k) {
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

    [[nodiscard]] Real skewness(const VecReal& x) {
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

    [[nodiscard]] Real kurtosis(const VecReal& x) {
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

    [[nodiscard]] Real quantile(VecReal x, Real q) {
        const std::size_t n = x.size();
        if (n == 0 || q < 0 || q > 1) return std::numeric_limits<Real>::quiet_NaN();

        std::ranges::sort(x.begin(), x.end());
        const Real pos = q * (static_cast<Real>(n) - 1);
        const auto i = static_cast<std::size_t>(pos);
        const Real frac = pos - static_cast<Real>(i);

        if (frac == 0.0 || i + 1 == n)
            return x[i];

        return x[i] + frac * (x[i+1] - x[i]);
    }

    [[nodiscard]] Real percentile(VecReal x, Real p) {
        return quantile(std::move(x), p * Real{0.01});
    }

    [[nodiscard]] Quartiles quartiles(const VecReal& x) {
        if (x.empty()) return { NaN(), NaN(), NaN() };
        return { quantile(x, 0.25), quantile(x, 0.5), quantile(x, 0.75) };
    }

    [[nodiscard]] Real iqr(const VecReal& x) {
        const auto q = quartiles(x);
        return q.q3 - q.q1;
    }

    [[nodiscard]] Real trimmed_mean(VecReal x, Real alpha) {
        const std::size_t n = x.size();
        if (n == 0 || alpha < 0 || alpha >= 0.5) return std::numeric_limits<Real>::quiet_NaN();

        std::ranges::sort(x.begin(), x.end());

        const auto k = static_cast<std::size_t>(static_cast<Real>(n) * alpha);
        const std::size_t lo = k;
        const std::size_t hi = n - k;

        if (hi <= lo) return std::numeric_limits<Real>::quiet_NaN();

        using Diff = VecReal::difference_type;
        Real acc = std::accumulate(x.begin() + static_cast<Diff>(lo), x.begin() + static_cast<Diff>(hi), Real{0});
        return acc / static_cast<Real>(hi - lo);
    }

    // ==========================================================
    // ================= Robust Statistics ======================
    // ==========================================================

    [[nodiscard]] Real median_absolute_deviation(VecReal x) {
        if (x.empty()) return NaN();
        const Real m = median(x);
        for (Real& v : x)
            v = abs(v - m);
        return median(x);
    }

    [[nodiscard]] Real winsorized_mean(VecReal x, Real alpha) {
        if (x.empty() || alpha < 0 || alpha >= Real{0.5}) return NaN();

        const std::size_t n = x.size();
        const auto k = static_cast<std::size_t>(static_cast<Real>(n) * alpha);

        using Diff = VecReal::difference_type;
        std::ranges::nth_element(x.begin(), x.begin() + static_cast<Diff>(k), x.end());
        const Real lo = x[k];

        std::ranges::nth_element(x.begin(), x.end() - static_cast<Diff>(k) - 1, x.end());
        const Real hi = x[n - k - 1];

        Real acc = 0;
        for (Real v : x) {
            if (v < lo) v = lo;
            else if (v > hi) v = hi;
            acc += v;
        }
        return acc / static_cast<Real>(n);
    }

    [[nodiscard]] Real huber_mean(const VecReal& x, Real delta) {
        if (x.empty() || delta <= 0) return NaN();

        const Real m = mean(x);
        Real acc = 0;

        for (Real v : x) {
            const Real d = v - m;
            if (abs(d) <= delta)
                acc += v;
            else
                acc += m + delta * (d > 0 ? 1 : -1);
        }
        return acc / static_cast<Real>(x.size());
    }

    [[nodiscard]] Real biweight_mean(const VecReal& x) {
        if (x.empty()) return NaN();

        const Real m = median(x);
        const Real mad = median_absolute_deviation(x);
        if (mad == 0) return m;

        const Real c = 9 * mad;
        Real acc = 0;
        Real wsum = 0;

        for (Real v : x) {
            const Real u = (v - m) / c;
            if (abs(u) < 1) {
                const Real t = 1 - u * u;
                const Real w = t * t;
                acc += v * w;
                wsum += w;
            }
        }
        return wsum > 0 ? acc / wsum : NaN();
    }

    [[nodiscard]] Real snr(const VecReal& signal) {
        if (signal.empty()) return NaN();

        Real mean = 0;
        Real m2 = 0;
        std::size_t n = 0;

        for (Real x : signal) {
            ++n;
            const Real delta = x - mean;
            mean += delta / static_cast<Real>(n);
            m2 += delta * (x - mean);
        }

        if (n < 2) return NaN();

        const Real variance = m2 / (static_cast<Real>(n) - 1);
        return variance > 0 ? mean / std::sqrt(variance) : NaN();
    }

    // ==========================================================
    // ================= Correlation & Dependence ===============
    // ==========================================================

    [[nodiscard]] Real covariance(const VecReal& x, const VecReal& y) {
        const std::size_t n = x.size();
        if (n != y.size() || n < 2) return NaN();

        Real mean_x = 0, mean_y = 0;
        Real C = 0;

        for (std::size_t i = 0; i < n; ++i) {
            const Real dx = x[i] - mean_x;
            const Real dy = y[i] - mean_y;
            mean_x += dx / (static_cast<Real>(i) + 1);
            mean_y += dy / (static_cast<Real>(i) + 1);
            C += dx * (y[i] - mean_y);
        }
        return C / (static_cast<Real>(n) - 1);
    }

    [[nodiscard]] Real correlation_pearson(const VecReal& x, const VecReal& y) {
        const std::size_t n = x.size();
        if (n != y.size() || n < 2) return NaN();

        Real mx = 0, my = 0;
        Real Sx = 0, Sy = 0, Sxy = 0;

        for (std::size_t i = 0; i < n; ++i) {
            const Real dx = x[i] - mx;
            const Real dy = y[i] - my;
            mx += dx / (static_cast<Real>(i) + 1);
            my += dy / (static_cast<Real>(i) + 1);
            Sx += dx * (x[i] - mx);
            Sy += dy * (y[i] - my);
            Sxy += dx * (y[i] - my);
        }

        if (Sx <= 0 || Sy <= 0) return NaN();
        return Sxy / std::sqrt(Sx * Sy);
    }

    [[nodiscard]] Real correlation_spearman(const VecReal& x, const VecReal& y) {
        const std::size_t n = x.size();
        if (n != y.size() || n < 2) return NaN();

        auto rank = [&](const VecReal& v) {
            std::vector<std::size_t> idx(n);
            std::iota(idx.begin(), idx.end(), 0);

            std::ranges::sort(idx, [&](auto a, auto b) {
                return v[a] < v[b];
            });

            VecReal r(n);
            for (std::size_t i = 0; i < n;) {
                std::size_t j = i;
                while (j < n && v[idx[i]] == v[idx[j]]) ++j;

                const Real avg = (static_cast<Real>(i) + static_cast<Real>(j - 1)) / 2;
                for (std::size_t k = i; k < j; ++k)
                    r[idx[k]] = avg;

                i = j;
            }
            return r;
        };

        return correlation_pearson(rank(x), rank(y));
    }

    [[nodiscard]] Real correlation_kendall(const VecReal& x, const VecReal& y) {
        const std::size_t n = x.size();
        if (n != y.size() || n < 2) return NaN();

        Real concordant = 0, discordant = 0;

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const Real dx = x[i] - x[j];
                const Real dy = y[i] - y[j];
                const Real p = dx * dy;

                concordant += (p > 0);
                discordant += (p < 0);
            }
        }

        const Real total = concordant + discordant;
        return total > 0 ? (concordant - discordant) / total : NaN();
    }

    [[nodiscard]] Real autocovariance(const VecReal& x, int lag) {
        const std::size_t n = x.size();
        if (n < 2) return NaN();
        if (abs(lag) >= static_cast<int>(n)) return NaN();

        const Real m = mean(x);
        Real acc = 0;
        std::size_t count = 0;
        int L = static_cast<int>(abs(lag));

        for (std::size_t i = 0; i + L < n; ++i) {
            if (lag >= 0)
                acc += (x[i] - m) * (x[i + L] - m);
            else
                acc += (x[i + L] - m) * (x[i] - m);
            ++count;
        }

        return count > 0 ? acc / static_cast<Real>(count) : NaN();
    }

    [[nodiscard]] Real autocorrelation(const VecReal& x, int lag) {
        const std::size_t n = x.size();
        if (lag < 0 || lag >= static_cast<int>(n)) return NaN();

        Real mean = 0, var = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const Real d = x[i] - mean;
            mean += d / (static_cast<Real>(i) + 1);
            var += d * (x[i] - mean);
        }
        if (var <= 0) return NaN();

        Real acc = 0;
        for (std::size_t i = 0; i + lag < n; ++i)
            acc += (x[i] - mean) * (x[i + lag] - mean);

        return acc / var;
    }

    [[nodiscard]] Real cross_correlation(const VecReal& x, const VecReal& y, int lag) {
        const std::size_t n = x.size();
        if (n < 2 || n != y.size()) return NaN();
        if (abs(lag) >= static_cast<int>(n)) return NaN();

        const Real mx = mean(x);
        const Real my = mean(y);
        Real acc = 0;
        std::size_t count = 0;
        int L = static_cast<int>(abs(lag));

        for (std::size_t i = 0; i + L < n; ++i) {
            if (lag >= 0)
                acc += (x[i] - mx) * (y[i + L] - my);
            else
                acc += (x[i + L] - mx) * (y[i] - my);
            ++count;
        }

        return count > 0 ? acc / static_cast<Real>(count) : NaN();
    }

    // ==========================================================
    // ================= Probability Distributions ==============
    // ==========================================================

    namespace dist {

        [[nodiscard]] Real pdf(const Normal& d, Real x) {
            const Real z = (x - d.mu) * d.inv_sigma;
            return std::exp(-Real{0.5} * z * z) * d.inv_sigma_sqrt2pi;
        }

        [[nodiscard]] Real cdf(const Normal& d, Real x) {
            boost::math::normal_distribution n(d.mu, d.sigma);
            return boost::math::cdf(n, x);
        }

        [[nodiscard]] Real quantile(const Normal& d, Real p) {
            if (d.sigma <= 0) return NaN();
            boost::math::normal_distribution n(d.mu, d.sigma);
            return boost::math::quantile(n, Utils::safeProb(p));
        }

        [[nodiscard]] Real log_likelihood(const Normal& d, const VecReal& data) {
            if (data.empty()) return NaN();

            Real acc = 0;
            const Real inv_sigma = d.inv_sigma;
            const Real log_norm  = d.log_norm;
            const Real n         = static_cast<Real>(data.size());

            for (Real x : data) {
                const Real z = (x - d.mu) * inv_sigma;
                acc += z * z;
            }

            return -Real{0.5} * acc - n * log_norm;
        }

        [[nodiscard]] Real pdf(const LogNormal& d, Real x) {
            if (x <= 0) return 0;
            Real lx = std::log(x);
            Real z = (lx - d.mu) * d.inv_sigma;
            return std::exp(-Real{0.5} * z * z) / (x * d.sigma * Constants::SQRT_2PI);
        }

        [[nodiscard]] Real cdf(const LogNormal& d, Real x) {
            if (x <= 0) return Real{0};
            boost::math::lognormal_distribution ln(d.mu, d.sigma);
            return boost::math::cdf(ln, x);
        }

        [[nodiscard]] Real quantile(const LogNormal& d, Real p) {
            if (d.sigma <= 0) return NaN();
            boost::math::lognormal_distribution ln(d.mu, d.sigma);
            return boost::math::quantile(ln, Utils::safeProb(p));
        }

        [[nodiscard]] Real log_likelihood(const LogNormal& d, const VecReal& data) {
            if (data.empty()) return NaN();
            Real acc = 0;
            Real sq = 0;

            for (Real x : data) {
                if (x <= 0) return NaN();
                Real lx = std::log(x);
                Real z = (lx - d.mu) * d.inv_sigma;
                sq += z * z;
                acc += lx;
            }

            return -Real{0.5} * sq
                   - static_cast<Real>(data.size()) * d.log_norm
                   - acc;
        }

        [[nodiscard]] Real pdf(const Exponential& d, Real x) {
            return (x < 0 || d.lambda <= 0) ? 0
                   : d.lambda * std::exp(-d.lambda * x);
        }

        [[nodiscard]] Real cdf(const Exponential& d, Real x) {
            return (x < 0 || d.lambda <= 0) ? 0
                   : Real{1} - std::exp(-d.lambda * x);
        }

        [[nodiscard]] Real quantile(const Exponential& d, Real p) {
            if (d.lambda <= 0) return NaN();
            boost::math::exponential_distribution e(d.lambda);
            return boost::math::quantile(e, Utils::safeProb(p));
        }

        [[nodiscard]] Real log_likelihood(const Exponential& d, const VecReal& data) {
            if (d.lambda <= 0 || data.empty()) return NaN();

            Real sum = 0;
            for (Real x : data) {
                if (x < 0) return NaN();
                sum += x;
            }

            return static_cast<Real>(data.size()) * std::log(d.lambda)
                   - d.lambda * sum;
        }

        [[nodiscard]] Real pdf(const Gamma& d, Real x) {
            if (x <= 0 || d.k <= 0 || d.theta <= 0) return 0;

            return std::exp(
                (d.k - 1) * std::log(x)
                - x * d.inv_theta
                - d.k * d.log_theta
                - d.log_gamma_k
            );
        }

        [[nodiscard]] Real cdf(const Gamma& d, Real x) {
            if (x <= 0 || d.k <= 0 || d.theta <= 0) return 0;
            boost::math::gamma_distribution g(d.k, d.theta);
            return boost::math::cdf(g, x);
        }

        [[nodiscard]] Real quantile(const Gamma& d, Real p) {
            if (d.k <= 0 || d.theta <= 0) return NaN();
            boost::math::gamma_distribution g(d.k, d.theta);
            return boost::math::quantile(g, Utils::safeProb(p));
        }

        [[nodiscard]] Real log_likelihood(const Gamma& d, const VecReal& data) {
            if (d.k <= 0 || d.theta <= 0 || data.empty()) return NaN();

            Real sum_log = 0;
            Real sum_x   = 0;

            for (Real x : data) {
                if (x <= 0) return NaN();
                sum_log += std::log(x);
                sum_x   += x;
            }

            const Real n = static_cast<Real>(data.size());

            return (d.k - 1) * sum_log
                 - sum_x * d.inv_theta
                 - n * (d.k * d.log_theta + d.log_gamma_k);
        }

        [[nodiscard]] Real pdf(const Beta& d, Real x) {
            if (x <= 0 || x >= 1 || d.alpha <= 0 || d.beta <= 0)
                return 0;

            return std::exp(
                (d.alpha - 1) * std::log(x)
              + (d.beta  - 1) * std::log(Real{1} - x)
              - d.log_beta_fn
            );
        }

        [[nodiscard]] Real cdf(const Beta& d, Real x) {
            if (x <= 0) return 0;
            if (x >= 1) return 1;
            if (d.alpha <= 0 || d.beta <= 0) return NaN();
            boost::math::beta_distribution b(d.alpha, d.beta);
            return boost::math::cdf(b, x);
        }

        [[nodiscard]] Real quantile(const Beta& d, Real p) {
            if (d.alpha <= 0 || d.beta <= 0)
                return NaN();

            boost::math::beta_distribution b(d.alpha, d.beta);
            return boost::math::quantile(b, Utils::clampProb(p));
        }

        [[nodiscard]] Real log_likelihood(const Beta& d, const VecReal& data) {
            if (d.alpha <= 0 || d.beta <= 0 || data.empty())
                return NaN();

            Real sum_log_x  = 0;
            Real sum_log_1x = 0;

            for (Real x : data) {
                if (x <= 0 || x >= 1) return NaN();
                sum_log_x  += std::log(x);
                sum_log_1x += std::log(Real{1} - x);
            }

            const Real n = static_cast<Real>(data.size());

            return (d.alpha - 1) * sum_log_x
                 + (d.beta  - 1) * sum_log_1x
                 - n * d.log_beta_fn;
        }

        [[nodiscard]] Real pdf(const Weibull& d, Real x) {
            if (x < 0) return 0;
            Real t = x * d.inv_lambda;
            return (d.k / d.lambda) *
                   std::pow(t, d.k - 1) *
                   std::exp(-std::pow(t, d.k));
        }

        [[nodiscard]] Real cdf(const Weibull& d, Real x) {
            if (x < 0) return 0;
            return Real{1} - std::exp(-std::pow(x * d.inv_lambda, d.k));
        }

        [[nodiscard]] Real quantile(const Weibull& d, Real p) {
            boost::math::weibull_distribution w(d.k, d.lambda);
            return boost::math::quantile(w, Utils::safeProb(p));
        }

        [[nodiscard]] Real log_likelihood(const Weibull& d, const VecReal& data) {
            if (data.empty()) return NaN();
            Real sum_log = 0;
            Real sum_pow = 0;

            for (Real x : data) {
                if (x <= 0) return NaN();
                Real t = x * d.inv_lambda;
                sum_log += std::log(t);
                sum_pow += std::pow(t, d.k);
            }

            return static_cast<Real>(data.size()) * (std::log(d.k) - d.k * d.log_lambda)
                   + (d.k - 1) * sum_log
                   - sum_pow;
        }

        [[nodiscard]] Real pdf(const Cauchy& d, Real x) {
            Real z = (x - d.x0) * d.inv_gamma;
            return Real{1} / (Constants::PI * d.gamma * (Real{1} + z * z));
        }

        [[nodiscard]] Real cdf(const Cauchy& d, Real x) {
            boost::math::cauchy_distribution c(d.x0, d.gamma);
            return boost::math::cdf(c, x);
        }

        [[nodiscard]] Real quantile(const Cauchy& d, Real p) {
            if (p <= 0 || p >= 1) return NaN();
            boost::math::cauchy_distribution c(d.x0, d.gamma);
            return boost::math::quantile(c, Utils::safeProb(p));
        }

        [[nodiscard]] Real log_likelihood(const Cauchy& d, const VecReal& data) {
            if (data.empty()) return NaN();
            Real acc = 0;
            for (Real x : data) {
                Real z = (x - d.x0) * d.inv_gamma;
                acc += std::log(Real{1} + z * z);
            }
            return -static_cast<Real>(data.size()) * d.log_norm - acc;
        }

        [[nodiscard]] Real pdf(const StudentT& d, Real x) {
            Real t = Real{1} + (x * x) / d.nu;
            return std::exp(d.log_norm) * std::pow(t, -(d.nu + 1) / 2);
        }

        [[nodiscard]] Real cdf(const StudentT& d, Real x) {
            if (d.nu <= 0) return NaN();
            boost::math::students_t_distribution t(d.nu);
            return boost::math::cdf(t, x);
        }

        [[nodiscard]] Real quantile(const StudentT& d, Real p) {
            if (d.nu <= 0) return NaN();
            boost::math::students_t_distribution<Real> t(d.nu);
            return boost::math::quantile(t, Utils::safeProb(p));
        }

        [[nodiscard]] Real log_likelihood(const StudentT& d, const VecReal& data) {
            if (data.empty()) return NaN();
            Real acc = 0;
            for (Real x : data)
                acc += std::log(Real{1} + (x * x) / d.nu);

            return static_cast<Real>(data.size()) * d.log_norm
                   - Real{0.5} * (d.nu + 1) * acc;
        }

    } // namespace dist

    // ==========================================================
    // ================= Statistical Tests ======================
    // ==========================================================

    [[nodiscard]] Real z_test(const VecReal& x, Real mu, Real sigma) {
        if (x.empty() || sigma <= 0) return NaN();
        return (mean(x) - mu) / (sigma / std::sqrt(x.size()));
    }

    [[nodiscard]] Real t_test(const VecReal& x, Real mu) {
        if (x.size() < 2) return NaN();
        for (Real xi : x) if (!std::isfinite(xi)) return NaN();
        Real s = stddev_unbiased(x);
        if (s <= 0 || !std::isfinite(s)) return NaN();

        return (mean(x) - mu) / (s / std::sqrt(x.size()));
    }

    [[nodiscard]] Real welch_t_test(const VecReal& x, const VecReal& y) {
        if (x.size() < 2 || y.size() < 2) return NaN();
        for (Real xi : x) if (!std::isfinite(xi)) return NaN();
        for (Real yi : y) if (!std::isfinite(yi)) return NaN();

        Real vx = variance_unbiased(x);
        Real vy = variance_unbiased(y);
        if (vx <= 0 || vy <= 0 || !std::isfinite(vx) || !std::isfinite(vy)) return NaN();

        return (mean(x) - mean(y)) /
               std::sqrt(vx / static_cast<Real>(x.size()) + vy / static_cast<Real>(y.size()));
    }

    [[nodiscard]] Real mann_whitney_u(const VecReal& x, const VecReal& y) {
        if (x.empty() || y.empty()) return NaN();
        for (Real xi : x) if (!std::isfinite(xi)) return NaN();
        for (Real yi : y) if (!std::isfinite(yi)) return NaN();

        Real u = 0;
        for (Real xi : x)
            for (Real yj : y)
                if (xi > yj) u += 1;
        return u;
    }

    [[nodiscard]] Real wilcoxon_signed_rank(const VecReal& x, const VecReal& y) {
        if (x.size() != y.size() || x.empty()) return NaN();
        for (size_t i = 0; i < x.size(); ++i)
            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) return NaN();


        struct Diff { Real abs; Real sign; };
        std::vector<Diff> d;
        if (static_cast<Real>(d.size()) > 1e7) return NaN();

        for (size_t i = 0; i < x.size(); ++i) {
            Real v = x[i] - y[i];
            if (v != 0) d.push_back({abs(v), static_cast<Real>(v > 0 ? 1 : -1)});
        }

        if (d.empty()) return 0;

        std::ranges::sort(d, std::less<>(), &Diff::abs);

        Real W = 0;
        for (size_t i = 0; i < d.size(); ++i)
            W += (static_cast<Real>(i) + 1) * d[i].sign;

        return W;
    }

    [[nodiscard]] Real ks_test(const VecReal& x, const VecReal& y) {
        if (x.empty() || y.empty()) return NaN();
        for (Real xi : x) if (!std::isfinite(xi)) return NaN();
        for (Real yi : y) if (!std::isfinite(yi)) return NaN();

        VecReal xs = x, ys = y;
        std::ranges::sort(xs);
        std::ranges::sort(ys);

        size_t i = 0, j = 0;
        Real d = 0;

        while (i < xs.size() && j < ys.size()) {
            Real v = std::min(xs[i], ys[j]);
            while (i < xs.size() && xs[i] <= v) ++i;
            while (j < ys.size() && ys[j] <= v) ++j;
            d = std::max(d, abs(static_cast<Real>(i) / static_cast<Real>(xs.size()) -
                static_cast<Real>(j) / static_cast<Real>(ys.size())));
        }
        return d;
    }

    [[nodiscard]] Real chi_square_test(const VecReal& o, const VecReal& e) {
        if (o.size() != e.size() || o.empty()) return NaN();
        for (size_t i = 0; i < o.size(); ++i)
            if (!std::isfinite(o[i]) || !std::isfinite(e[i]) || e[i] <= 0) return NaN();

        Real chi2 = 0;
        for (size_t i = 0; i < o.size(); ++i) {
            if (e[i] <= 0) return NaN();
            Real d = o[i] - e[i];
            chi2 += d * d / e[i];
        }
        return chi2;
    }

    [[nodiscard]] Real anderson_darling(const VecReal& x) {
        if (x.size() < 2) return NaN();
        for (Real xi : x) if (!std::isfinite(xi)) return NaN();

        VecReal xs = x;
        std::ranges::sort(xs);

        Real m = mean(xs);
        Real s = stddev(xs); // MLE
        if (s <= 0 || !std::isfinite(s)) return NaN();

        size_t n = xs.size();
        Real sum = 0;

        for (size_t i = 0; i < n; ++i) {
            Real Fi = 0.5 * (1 + std::erf((xs[i] - m) / (s * std::sqrt(2))));
            Real Fj = 0.5 * (1 + std::erf((xs[n - 1 - i] - m) / (s * std::sqrt(2))));
            Fi = std::clamp(Fi, Constants::EPS_12, 1 - Constants::EPS_12);
            Fj = std::clamp(Fj, Constants::EPS_12, 1 - Constants::EPS_12);
            sum += (2 * static_cast<Real>(i) + 1) * (std::log(Fi) + std::log(1 - Fj));
        }

        return -static_cast<Real>(n) - sum / static_cast<Real>(n);
    }

    // ==========================================================
    // ================= Entropy & Information ==================
    // ==========================================================

    [[nodiscard]] Real entropy(const VecReal& p) {
        if (p.empty()) return NaN();

        Real h = 0;
        for (Real v : p)
            if (v > 0) h -= v * std::log(v);
        return h;
    }

    [[nodiscard]] Real cross_entropy(const VecReal& p, const VecReal& q) {
        if (p.empty() || q.empty() || p.size() != q.size()) return NaN();
        Real h = 0;
        for (std::size_t i = 0; i < p.size(); ++i)
            if (p[i] > 0 && q[i] > 0) h -= p[i] * std::log(q[i]);
        return h;
    }

    [[nodiscard]] Real kl_divergence(const VecReal& p, const VecReal& q) {
        if (p.empty() || q.empty() || p.size() != q.size()) return NaN();
        Real d = 0;
        for (std::size_t i = 0; i < p.size(); ++i)
            if (p[i] > 0 && q[i] > 0) d += p[i] * std::log(p[i] / q[i]);
        return d;
    }

    [[nodiscard]] Real js_divergence(const VecReal& p, const VecReal& q) {
        if (p.empty() || q.empty() || p.size() != q.size()) return NaN();
        VecReal m(p.size());
        for (std::size_t i = 0; i < p.size(); ++i)
            m[i] = 0.5 * (p[i] + q[i]);
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m);
    }

    [[nodiscard]] Real joint_entropy(const VecReal& pxy) {
        Real h = 0;
        for (Real v : pxy)
            if (v > 0) h -= v * std::log(v);
        return h;
    }

    [[nodiscard]] Real mutual_information(
        const VecReal& px,
        const VecReal& py,
        const VecReal& pxy
    ) {
        if (px.empty() || py.empty() || pxy.empty() || pxy.size() != px.size() * py.size()) return NaN();
        return entropy(px) + entropy(py) - joint_entropy(pxy);
    }

    [[nodiscard]] Real conditional_entropy(
        const VecReal& py,
        const VecReal& pxy
    ) {
        if (py.empty() || pxy.empty()) return NaN();
        return joint_entropy(pxy) - entropy(py);
    }

    // ==========================================================
    // ================= Characteristic Functions ==============
    // ==========================================================

    [[nodiscard]] Complex normal_characteristic(Real t, Real mu, Real sigma) {
        // exp(i*mu*t - 0.5*sigma^2*t^2)
        return std::exp(Complex(-0.5 * sigma * sigma * t * t, mu * t));
    }

    [[nodiscard]] Complex samples_characteristic(const VecReal& samples, Real t) {
        if(samples.empty()) return Complex{0.0, 0.0};
        Complex sum{0.0, 0.0};
        for (Real x : samples)
            sum += std::exp(Complex{0.0, t * x});
        return sum / static_cast<double>(samples.size());
    }

    // ==========================================================
    // ================= Time Series / Chaos ====================
    // ==========================================================

    // Rolling mean
    [[nodiscard]] VecReal rolling_mean(const VecReal& x, std::size_t window) {
        if (window == 0 || x.size() < window) return {};
        VecReal out(x.size() - window + 1);
        using Diff = VecReal::difference_type;

        Real sum = std::accumulate(x.begin(), x.begin() + static_cast<Diff>(window), Real{0});
        out[0] = sum / static_cast<Real>(window);
        for (std::size_t i = window; i < x.size(); ++i) {
            sum += x[i] - x[i - window];
            out[i - window + 1] = sum / static_cast<Real>(window);
        }
        return out;
    }

    // Rolling variance (Welford)
    [[nodiscard]] VecReal rolling_variance(const VecReal& x, std::size_t window) {
        if (window == 0 || x.size() < window) return {};
        VecReal out(x.size() - window + 1);
        VecReal mean = rolling_mean(x, window);

        for (std::size_t i = 0; i <= x.size() - window; ++i) {
            Real sum_sq = 0;
            for (std::size_t j = 0; j < window; ++j) {
                Real diff = x[i + j] - mean[i];
                sum_sq += diff * diff;
            }
            out[i] = sum_sq / static_cast<Real>(window);
        }
        return out;
    }

    // Exponential Moving Average
    [[nodiscard]] VecReal ema(const VecReal& x, Real alpha) {
        if (x.empty() || alpha <= 0 || alpha > 1) return {};
        VecReal out(x.size());
        out[0] = x[0];
        for (std::size_t i = 1; i < x.size(); ++i)
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1];
        return out;
    }

    // Partial Autocorrelation (Yule-Walker, naive)
    [[nodiscard]] Real partial_autocorrelation(const VecReal& x, int lag) {
        if (lag <= 0 || lag >= static_cast<int>(x.size())) return NaN();

        std::vector<Real> r(lag + 1, 0);
        const Real n = static_cast<Real>(x.size());
        Real mean = Functions::mean(x);
        for (size_t k = 0; k <= static_cast<size_t>(lag); ++k) {
            for (size_t i = 0; i < x.size() - k; ++i)
                r[k] += (x[i] - mean) * (x[i + k] - mean);
            r[k] /= n - static_cast<Real>(k);
        }

        if (lag == 1) return r[1] / r[0];

        std::vector<Real> phi(lag + 1, 0);
        std::vector<Real> phi_prev(lag + 1, 0);
        phi[1] = r[1] / r[0];

        for (int k = 2; k <= lag; ++k) {
            Real num = r[k];
            for (int j = 1; j < k; ++j)
                num -= phi_prev[j] * r[k - j];

            Real den = r[0];
            for (int j = 1; j < k; ++j)
                den -= phi_prev[j] * r[j];

            phi[k] = num / den;
            phi_prev = phi;
        }
        return phi[lag];
    }

    // Hurst Exponent (R/S method simplified)
    [[nodiscard]] Real hurst_exponent(const VecReal& x) {
        if (x.size() < 20) return NaN();
        VecReal y(x.size());
        std::partial_sum(x.begin(), x.end(), y.begin());

        Real mean = y.back() / static_cast<Real>(y.size());
        for (auto& v : y) v -= mean;

        Real R = *std::ranges::max_element(y) - *std::ranges::min_element(y);
        Real S = std::sqrt(std::inner_product(y.begin(), y.end(), y.begin(), Real{0}) / static_cast<Real>(y.size()));
        return std::log(R / S) / std::log(x.size());
    }

    // Detrend via linear regression
    [[nodiscard]] VecReal detrend(const VecReal& x) {
        const std::size_t n = x.size();
        if (n < 2) return x;

        Real sumX = 0;
        Real sumY = 0;
        Real sumXX = 0;
        Real sumXY = 0;

        for (std::size_t i = 0; i < n; ++i) {
            sumX += static_cast<Real>(i);
            sumY += x[i];
            sumXX += static_cast<Real>(i) * static_cast<Real>(i);
            sumXY += static_cast<Real>(i) * x[i];
        }

        Real denom = static_cast<Real>(n) * sumXX - sumX * sumX;
        if (denom == 0) return x;

        Real slope = (static_cast<Real>(n) * sumXY - sumX * sumY) / denom;
        Real intercept = (sumY - slope * sumX) / static_cast<Real>(n);

        VecReal out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = x[i] - (slope * static_cast<Real>(i) + intercept);

        return out;
    }

    // Differencing
    [[nodiscard]] VecReal difference(const VecReal& x, int order) {
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
    [[nodiscard]] Real lyapunov_exponent(const VecReal& x) {
        if (x.size() < 10) return NaN();
        Real sum = 0;
        for (std::size_t i = 1; i < x.size(); ++i) {
            Real denom = x[i - 1];
            if (denom == 0) return NaN();
            sum += std::log(abs(x[i] / denom));
        }
        return sum / (static_cast<Real>(x.size()) - 1);
    }

    // Takens Embedding
    [[nodiscard]] VecReal takens_map(const VecReal& signal, int dim, int tau) {
        if (dim <= 0 || tau <= 0 || signal.empty()) return {};
        size_t N = signal.size();
        if (N < (dim - 1) * tau + 1) return {};
        size_t n_embedded = N - (dim - 1) * tau;

        VecReal embedded;
        embedded.reserve(n_embedded * dim);

        for (size_t i = 0; i < n_embedded; ++i) {
            for (int j = 0; j < dim; ++j)
                embedded.push_back(signal[i + j * tau]);
        }
        return embedded;
    }

    // ==========================================================
    // ================= Sampling & Resampling ==================
    // ==========================================================

    [[nodiscard]] Real bootstrap_mean(const VecReal& x, int n) {
        if (x.empty() || n <= 0) return NaN();

        std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<std::size_t> d(0, x.size() - 1);

        Real acc = 0;
        for (int i = 0; i < n; ++i) {
            Real sum = 0;
            for (size_t j = 0; j < x.size(); ++j)
                sum += x[d(gen)];
            acc += sum / static_cast<Real>(x.size());
        }
        return acc / n;
    }

    [[nodiscard]] RealPair bootstrap_ci(const VecReal& x, Real alpha, int n = 1000) {
        if (x.empty() || alpha <= 0 || alpha >= 1) return {NaN(), NaN()};

        std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<std::size_t> d(0, x.size() - 1);
        VecReal samples;
        samples.reserve(n);

        for (int i = 0; i < n; ++i) {
            Real sum = 0;
            for (std::size_t j = 0; j < x.size(); ++j)
                sum += x[d(gen)];
            samples.push_back(sum / static_cast<Real>(x.size()));
        }

        using Diff = VecReal::difference_type;
        std::ranges::nth_element(samples.begin(), samples.begin() + (1 - static_cast<Diff>(alpha)) / 2 * n, samples.end());
        std::ranges::nth_element(samples.begin(), samples.begin() + (1 + static_cast<Diff>(alpha)) / 2 * n, samples.end());

        std::ranges::sort(samples);
        auto lo = static_cast<std::size_t>((1 - alpha)/2 * n);
        auto hi = static_cast<std::size_t>((1 + alpha)/2 * n);
        return {samples[lo], samples[hi]};

        return {samples[lo], samples[hi]};
    }

    [[nodiscard]] VecReal jackknife(const VecReal& x) {
        const std::size_t n = x.size();
        if (n == 0) return {};

        Real total = 0;
        for (Real v : x) total += v;

        VecReal out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = (total - x[i]) / static_cast<Real>(n - 1);

        return out;
    }

    [[nodiscard]] Real permutation_test(const VecReal& x, const VecReal& y, int trials = 1000) {
        if (x.size() != y.size() || x.empty()) return NaN();
        VecReal z;
        z.reserve(x.size() + y.size());
        z.insert(z.end(), x.begin(), x.end());
        z.insert(z.end(), y.begin(), y.end());

        Real obs = abs(mean(x) - mean(y));

        std::mt19937 gen{std::random_device{}()};
        int count = 0;

        using Diff = VecReal::difference_type;
        for (int i = 0; i < trials; ++i) {
            std::ranges::shuffle(z.begin(), z.end(), gen);
            Real m1 = std::accumulate(z.begin(), z.begin() + static_cast<Diff>(x.size()), Real{0}) / static_cast<Real>(x.size());
            Real m2 = std::accumulate(z.begin() + static_cast<Diff>(x.size()), z.end(), Real{0}) / static_cast<Real>(y.size());
            if (abs(m1 - m2) >= obs) ++count;
        }
        return static_cast<Real>(count) / trials;
    }

    // ==========================================================
    // ================= Regression & Estimation ================
    // ==========================================================

    [[nodiscard]] LinearRegressionResult linear_regression(const VecReal& x, const VecReal& y) {
        LinearRegressionResult r;

        if (x.size() != y.size() || x.empty()) {
            r.slope = NaN();
            r.intercept = NaN();
            r.r2 = NaN();
            return r;
        }

        const std::size_t n = x.size();
        Real mx = mean(x);
        Real my = mean(y);

        Real num = 0, den = 0;
        for (std::size_t i = 0; i < n; ++i) {
            Real dx = x[i] - mx;
            num += dx * (y[i] - my);
            den += dx * dx;
        }

        if (den == 0) {
            r.slope = 0.0;
            r.intercept = my;
            r.r2 = NaN();
            return r;
        }

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

    [[nodiscard]] VecReal polynomial_regression(const VecReal& x, const VecReal& y, int degree) {
        if (x.size() != y.size() || x.empty() || degree < 0) return {};
        const std::size_t n = x.size();
        const int m = degree + 1;

        // --- 1. Построение матрицы Вандермонда ---
        std::vector<VecReal> X(n, VecReal(m, 1.0));
        for (std::size_t i = 0; i < n; ++i)
            for (int j = 1; j < m; ++j)
                X[i][j] = X[i][j-1] * x[i];

        // --- 2. QR-разложение (классический Грам–Шмидт) ---
        std::vector<VecReal> Q = X; // копия для ортогонализации
        std::vector<VecReal> R(m, VecReal(m, 0.0));

        for (int j = 0; j < m; ++j) {
            // R[j][j] = ||Q_j||
            Real norm = 0;
            for (std::size_t i = 0; i < n; ++i)
                norm += Q[i][j] * Q[i][j];
            norm = std::sqrt(norm);
            if (norm < 1e-12) { // защита от сингулярной матрицы
                R[j][j] = 0;
                for (int k = j; k < m; ++k)
                    R[j][k] = 0;
                continue;
            }
            R[j][j] = norm;
            for (std::size_t i = 0; i < n; ++i)
                Q[i][j] /= norm;

            for (int k = j + 1; k < m; ++k) {
                Real rjk = 0;
                for (std::size_t i = 0; i < n; ++i)
                    rjk += Q[i][j] * Q[i][k];
                R[j][k] = rjk;
                for (std::size_t i = 0; i < n; ++i)
                    Q[i][k] -= rjk * Q[i][j];
            }
        }

        // --- 3. Вычисляем Qᵀ * y ---
        VecReal Qt_y(m, 0.0);
        for (int j = 0; j < m; ++j)
            for (std::size_t i = 0; i < n; ++i)
                Qt_y[j] += Q[i][j] * y[i];

        // --- 4. Решаем R * coef = Qᵀy (обратный ход) ---
        VecReal coef(m, 0.0);
        for (int i = m - 1; i >= 0; --i) {
            if (abs(R[i][i]) < 1e-12) { coef[i] = 0; continue; } // сингулярный случай
            Real sum = Qt_y[i];
            for (int j = i + 1; j < m; ++j)
                sum -= R[i][j] * coef[j];
            coef[i] = sum / R[i][i];
        }

        return coef;
    }

    [[nodiscard]] Real least_squares(const VecReal& residuals) {
        return std::inner_product(residuals.begin(), residuals.end(), residuals.begin(), Real{0});
    }

    // ==========================================================
    // ================= Outliers ===============================
    // ==========================================================

    [[nodiscard]] VecReal z_score(const VecReal& x) {
        if (x.empty()) return {};
        const std::size_t n = x.size();

        Real mean = Functions::mean(x);
        Real stddev = stddev_unbiased(x);
        if (stddev == 0) return VecReal(n, Real{0});

        VecReal out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = (x[i] - mean) / stddev;

        return out;
    }

    [[nodiscard]] VecReal modified_z_score(const VecReal& x) {
        if (x.empty()) return {};
        const std::size_t n = x.size();
        Real median = Functions::median(x);
        Real mad = median_absolute_deviation(x);
        if (mad == 0) return VecReal(n, Real{0});

        VecReal out(n);
        constexpr Real scale = 0.6745;
        for (std::size_t i = 0; i < n; ++i)
            out[i] = scale * (x[i] - median) / mad;

        return out;
    }

    [[nodiscard]] bool is_outlier(Real x, Real mean, Real stddev, Real threshold) {
        if (threshold <= 0) return false;
        return stddev > 0 && Functions::abs(x - mean) > threshold * stddev;
    }

    // Грубс-тест
    [[nodiscard]] bool grubbs_test(const VecReal& x, Real alpha = 0.05) {
        const std::size_t n = x.size();
        if (n < 3) return false;

        const Real mean   = Functions::mean(x);
        const Real stddev = stddev_unbiased(x);

        if (!(stddev > Real{0})) return false;

        Real max_dev = 0;
        for (Real v : x)
            max_dev = std::max(max_dev, Functions::abs(v - mean));

        const Real G = max_dev / stddev;

        const Real df = static_cast<Real>(n - 2);
        dist::StudentT t_dist{ df };

        const Real p = Real{1} - alpha / (Real{2} * static_cast<Real>(n));
        const Real t = dist::quantile(t_dist, p);

        const Real Gcrit =
            (static_cast<Real>(n - 1) / std::sqrt(static_cast<Real>(n))) *
            std::sqrt((t * t) / (df + t * t));

        return G > Gcrit;
    }

    // Критерий Шевенета
    [[nodiscard]] bool chauvenet_criterion(const VecReal& x) {
        const std::size_t n = x.size();
        if (n < 3) return false;

        Real median = Functions::median(x);
        Real mad = median_absolute_deviation(x);
        if (mad == 0) mad = Constants::EPS_12;

        constexpr Real scale = 0.6745;
        return std::ranges::any_of(x, [&](Real v) {
            Real mz = scale * (v - median) / mad;
            return Functions::abs(mz) > 3.5;
        });
    }

} // namespace functions
