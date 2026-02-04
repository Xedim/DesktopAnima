//Functions.h

#pragma once
#include <vector>
#include "../common/Types.h"
#include "../common/Constants.h"
#include "../common/Utils.h"
#include <boost/math/distributions/gamma.hpp>
#include "cmath"

namespace Functions {

    // ===========================================
    // ================= Sign ====================
    // ===========================================

    Real sign(Real x);
    Real abs(Real x);
    Real heaviside(Real x);

    // =============================================
    // ================= Algebraic =================
    // =============================================

    Real factorial(int n);
    Real binomial(int n, int k);
    Real combination(int n, int k);
    Real permutation(int n, int k);
    Real mod(Real x, Real y);
    Real polynomial(Real x, const VecReal& coefficients);
    Real rational(Real x, const VecReal& p, const VecReal& q);

    // ================================================
    // ================= Power / Root =================
    // ================================================

    Real pow(Real x, Real alpha);
    Real x_pow_y(Real x, Real y);
    Real algebraic_root(Real x, const VecReal& coefficients);
    Real sqrt(Real x);
    Real sqrt1pm1(Real x);
    Real cbrt(Real x);

    // =============================================================
    // ================= Exponential /Logarithmic ==================
    // =============================================================

    Real exp(Real x);
    Real exp2(Real x);
    Real expm1_safe(Real x);
    Real log2(Real x);
    Real log(Real x);
    Real log10(Real x);
    Real log_a(Real x, Real a);
    Real log1p(Real x);

    // =================================================
    // ================= Trigonometric =================
    // =================================================

    Real sin(Real x);
    Real cos(Real x);
    Real sec(Real x);
    Real csc(Real x);
    Real sinc(Real x);
    Real tan(Real x);
    Real cot(Real x);
    Real asin(Real x);
    Real acos(Real x);
    Real atan(Real x);
    Real atan2(Real y, Real x);
    Real hypot(Real x, Real y);

    // ==============================================
    // ================= Hyperbolic =================
    // ==============================================

    Real sinh(Real x);
    Real cosh(Real x);
    Real sech(Real x);
    Real csch(Real x);
    Real tanh(Real x);
    Real coth(Real x);
    Real asinh(Real x);
    Real acosh(Real x);
    Real atanh(Real x);

    // ===========================================
    // ================= Special =================
    // ===========================================

    Real erf(Real x);
    Real erfc(Real x);
    Real gamma(Real x);
    Real lgamma(Real x);
    Real beta(Real x, Real y);

    Real cyl_bessel_j(Real nu, Real x);
    Real cyl_neumann(Real nu, Real x);
    Real cyl_bessel_i(Real nu, Real x);
    Real cyl_bessel_k(Real nu, Real x);

    Real lambert_w(Real x);

    Real legendre(int l, Real x);
    Real assoc_legendre(int l, int m, Real x);

    Real riemann_zeta(Real s);
    Real zeta(Real s);

    // ===============================================
    // ================= Generalized =================
    // ===============================================

    Real dirac_delta(Real x, Real eps = 1e-3);
    Real geometric_sum(Real a, int N);

    // ====================================================
    // ==================== Numerical =====================
    // ====================================================

    Real round(Real x);
    Real floor(Real x);
    Real ceil(Real x);
    Real trunc(Real x);
    Real clamp(Real x, Real minVal, Real maxVal);
    Real lerp(Real a, Real b, Real t);
    Real fma(Real x, Real y, Real z);

    // ==============================================
    // ================= Fractals ===================
    // ==============================================

    Real weierstrass(Real x,
                     Real a = Constants::WEIERSTRASS_AMP_COEF,
                     Real b = Constants::WEIERSTRASS_FREQ_COEF,
                     int  N = Constants::WEIERSTRASS_ITER,
                     StabPolicy policy = StabPolicy::Reject);

    Real cantor(Real x,
                int max_iter = Constants::CANTOR_ITER,
                StabPolicy policy = StabPolicy::Reject);

    Real logistic(Real x,
                  Real r,
                  int n,
                  StabPolicy policy = StabPolicy::Reject);

    Real tent(Real x,
              int n,
              StabPolicy policy = StabPolicy::Reject);

    Complex julia(const Complex& z,
                  const Complex& c,
                  int n = Constants::JULIA_ITER,
                  StabPolicy policy = StabPolicy::Reject);

    bool escapes(Complex z0, Complex c, int max_iter, Real threshold = Constants::ESC_THRESHOLD);

    // =========================================
    // ================ Iterate ================
    // =========================================

    template<typename T, typename MapFunc>
    [[nodiscard]] T iterate(T x,
                            MapFunc&& f,
                            int n)
    {
        for (int i = 0; i < n; ++i) {
            x = f(x);

            if constexpr (std::is_same_v<T, Real>) {
                if (!std::isfinite(x))
                    return NaN();
            } else if constexpr (std::is_same_v<T, Complex>) {
                if (!std::isfinite(x.real()) || !std::isfinite(x.imag()))
                    return {NaN(), NaN()};
            }
        }
        return x;
    }

    [[nodiscard]] inline Real iterate(Real x,
                                      Real r,
                                      int n = Constants::MAP_ITER)
    {
        auto f = [r](Real x) {
            return r * x * (Real{1} - x);
        };

        return iterate(x, f, n);
    }

    // ==========================================================
    // ================= Descriptive Statistics =================
    // ==========================================================

    Real sum(const VecReal& x);
    Real mean(const VecReal& x);
    Real median(VecReal x);
    Real mode(const VecReal& x);

    Real min(const VecReal& x);
    Real max(const VecReal& x);
    Real range(const VecReal& x);

    Real variance(const VecReal& x);
    Real variance_unbiased(const VecReal& x);
    Real stddev(const VecReal& x);
    Real stddev_unbiased(const VecReal& x);
    Real mean_absolute_deviation(const VecReal& x);

    // ==========================================================
    // ================= Shape Statistics =======================
    // ==========================================================

    Real skewness(const VecReal& x);
    Real kurtosis(const VecReal& x);
    Real moment(const VecReal& x, int k);
    Real raw_moment(const VecReal& x, int k);

    // ==========================================================
    // ================= Order & Quantiles ======================
    // ==========================================================

    Real quantile(VecReal x, Real q);
    Real percentile(VecReal x, Real p);
    Quartiles quartiles(const VecReal& x);
    Real iqr(const VecReal& x);
    Real trimmed_mean(VecReal x, Real alpha);

    // ==========================================================
    // ================= Robust Statistics ======================
    // ==========================================================

    Real median_absolute_deviation(VecReal x);
    Real winsorized_mean(VecReal x, Real alpha);
    Real huber_mean(const VecReal& x, Real delta);
    Real biweight_mean(const VecReal& x);
    Real snr(const VecReal& signal);

    // ==========================================================
    // ================= Correlation & Dependence ===============
    // ==========================================================

    Real covariance(const VecReal& x, const VecReal& y);
    Real correlation_pearson(const VecReal& x, const VecReal& y);
    Real correlation_spearman(const VecReal& x, const VecReal& y);
    Real correlation_kendall(const VecReal& x, const VecReal& y);
    Real autocorrelation(const VecReal& x, int lag);
    Real cross_correlation(const VecReal& x, const VecReal& y, int lag);

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

        Real pdf(const Normal& d, Real x);
        Real cdf(const Normal& d, Real x);
        Real quantile(const Normal& d, Real p);
        Real log_likelihood(const Normal& d, const VecReal& data);

        struct LogNormal {
            Real mu;
            Real sigma;

            // cached
            Real inv_sigma;
            Real log_norm;

            explicit LogNormal(Real mu_, Real sigma_)
                : mu(mu_), sigma(sigma_) {

                if (sigma_ > 0) {
                    inv_sigma = Real{1} / sigma_;
                    log_norm = std::log(sigma_ * Constants::SQRT_2PI);
                } else {
                    inv_sigma = log_norm = NaN();
                }
            }
        };

        Real pdf(const LogNormal& d, Real x);
        Real cdf(const LogNormal& d, Real x);
        Real quantile(const LogNormal& d, Real p);
        Real log_likelihood(const LogNormal& d, const VecReal& data);

        struct Exponential {
            Real lambda;

            explicit Exponential(Real lambda_)
                : lambda(lambda_) {}
        };

        Real pdf(const Exponential& d, Real x);
        Real cdf(const Exponential& d, Real x);
        Real quantile(const Exponential& d, Real p);
        Real log_likelihood(const Exponential& d, const VecReal& data);

        struct Gamma {
            Real k;       // shape
            Real theta;   // scale

            // cached
            Real inv_theta;
            Real log_theta;
            Real log_gamma_k;

            explicit Gamma(Real k_, Real theta_)
                : k(k_), theta(theta_) {

                if (k_ > 0 && theta_ > 0) {
                    inv_theta = Real{1} / theta_;
                    log_theta = std::log(theta_);
                    log_gamma_k = boost::math::lgamma(k_);
                } else {
                    inv_theta = log_theta = log_gamma_k = NaN();
                }
            }
        };

        Real pdf(const Gamma& d, Real x);
        Real cdf(const Gamma& d, Real x);
        Real quantile(const Gamma& d, Real p);
        Real log_likelihood(const Gamma& d, const VecReal& data);

        struct Beta {
            Real alpha;
            Real beta;

            // cached
            Real log_beta_fn;   // log B(alpha, beta)

            explicit Beta(Real alpha_, Real beta_)
                : alpha(alpha_), beta(beta_) {

                if (alpha_ > 0 && beta_ > 0) {
                    log_beta_fn =
                        boost::math::lgamma(alpha_)
                      + boost::math::lgamma(beta_)
                      - boost::math::lgamma(alpha_ + beta_);
                } else {
                    log_beta_fn = NaN();
                }
            }
        };

        Real pdf(const Beta& d, Real x);
        Real cdf(const Beta& d, Real x);
        Real quantile(const Beta& d, Real p);
        Real log_likelihood(const Beta& d, const VecReal& data);

        struct Weibull {
            Real k;
            Real lambda;

            Real inv_lambda;
            Real log_lambda;

            explicit Weibull(Real k_, Real lambda_)
                : k(k_), lambda(lambda_) {

                inv_lambda = Real{1} / lambda_;
                log_lambda = std::log(lambda_);
            }
        };

        Real pdf(const Weibull& d, Real x);
        Real cdf(const Weibull& d, Real x);
        Real quantile(const Weibull& d, Real p);
        Real log_likelihood(const Weibull& d, const VecReal& data);

        struct Cauchy {
            Real x0;
            Real gamma;

            Real inv_gamma;
            Real log_norm;

            explicit Cauchy(Real x0_, Real gamma_)
                : x0(x0_), gamma(gamma_) {

                inv_gamma = Real{1} / gamma_;
                log_norm = std::log(Constants::PI * gamma_);
            }
        };

        Real pdf(const Cauchy& d, Real x);
        Real cdf(const Cauchy& d, Real x);
        Real quantile(const Cauchy& d, Real p);
        Real log_likelihood(const Cauchy& d, const VecReal& data);

        struct StudentT {
            Real nu;

            Real log_norm;

            explicit StudentT(Real nu_) : nu(nu_) {
                log_norm =
                    boost::math::lgamma((nu + 1) / 2) -
                    boost::math::lgamma(nu / 2) -
                    Real{0.5} * std::log(nu * Constants::PI);
            }
        };

        Real pdf(const StudentT& d, Real x);
        Real cdf(const StudentT& d, Real x);
        Real quantile(const StudentT& d, Real p);
        Real log_likelihood(const StudentT& d, const VecReal& data);

    } // namespace dist

    // ==========================================================
    // ================= Statistical Tests ======================
    // ==========================================================

    Real z_test(const VecReal& x, Real mu, Real sigma);
    Real t_test(const VecReal& x, Real mu);
    Real welch_t_test(const VecReal& x, const VecReal& y);
    Real mann_whitney_u(const VecReal& x, const VecReal& y);
    Real wilcoxon_signed_rank(const VecReal& x, const VecReal& y);
    Real ks_test(const VecReal& x, const VecReal& y);
    Real chi_square_test(const VecReal& observed, const VecReal& expected);
    Real anderson_darling(const VecReal& x);

    // ==========================================================
    // ================= Entropy & Information ==================
    // ==========================================================

    Real entropy(const VecReal& p);
    Real cross_entropy(const VecReal& p, const VecReal& q);
    Real kl_divergence(const VecReal& p, const VecReal& q);
    Real js_divergence(const VecReal& p, const VecReal& q);
    Real joint_entropy(const VecReal& pxy);
    Real mutual_information(const VecReal& x, const VecReal& y, const VecReal& pxy);
    Real conditional_entropy(const VecReal& x, const VecReal& y);

    // ==========================================================
    // ================= Characteristic Functions ==============
    // ==========================================================

    Complex normal_characteristic(Real t, Real mu, Real sigma);
    Complex samples_characteristic(const VecReal& samples, Real t);

    // ==========================================================
    // ================= Time Series Statistics =================
    // ==========================================================

    VecReal rolling_mean(const VecReal& x, std::size_t window);
    VecReal rolling_variance(const VecReal& x, std::size_t window);
    VecReal ema(const VecReal& x, Real alpha);
    Real autocovariance(const VecReal& x, int lag);
    Real partial_autocorrelation(const VecReal& x, int lag);
    Real hurst_exponent(const VecReal& x);
    VecReal detrend(const VecReal& x);
    VecReal difference(const VecReal& x, int order);
    Real lyapunov_exponent(const VecReal& x);
    VecReal takens_map(const VecReal& signal, int dim, int tau);

    // ==========================================================
    // ================= Sampling & Resampling ==================
    // ==========================================================

    Real bootstrap_mean(const VecReal& x, int n);
    RealPair bootstrap_ci(const VecReal& x, Real alpha, int n);
    VecReal jackknife(const VecReal& x);
    Real permutation_test(const VecReal& x, const VecReal& y, int trials);

    // ==========================================================
    // ================= Regression & Estimation ================
    // ==========================================================

    LinearRegressionResult linear_regression(const VecReal& x, const VecReal& y);
    VecReal polynomial_regression(const VecReal& x, const VecReal& y, int degree);
    Real least_squares(const VecReal& residuals);

    // ==========================================================
    // ================= Outliers & Anomaly =====================
    // ==========================================================

    VecReal z_score(const VecReal& x);
    VecReal modified_z_score(const VecReal& x);
    bool grubbs_test(const VecReal& x, Real alpha);
    bool chauvenet_criterion(const VecReal& x);
    bool is_outlier(Real x, Real mean, Real stddev, Real threshold = 3.0);

}
