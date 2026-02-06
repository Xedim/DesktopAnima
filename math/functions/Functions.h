//Functions.h

#pragma once
#include <vector>
#include "../types/Types.h"
#include "../helpers/Constants.h"
#include "../helpers/Utils.h"
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
    // ================= Exponential / Logarithmic =================
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

        Real pdf(const DistNormal& d, Real x);
        Real cdf(const DistNormal& d, Real x);
        Real quantile(const DistNormal& d, Real p);
        Real log_likelihood(const DistNormal& d, const VecReal& data);

        Real pdf(const DistLogNormal& d, Real x);
        Real cdf(const DistLogNormal& d, Real x);
        Real quantile(const DistLogNormal& d, Real p);
        Real log_likelihood(const DistLogNormal& d, const VecReal& data);

        Real pdf(const DistExp& d, Real x);
        Real cdf(const DistExp& d, Real x);
        Real quantile(const DistExp& d, Real p);
        Real log_likelihood(const DistExp& d, const VecReal& data);

        Real pdf(const DistGamma& d, Real x);
        Real cdf(const DistGamma& d, Real x);
        Real quantile(const DistGamma& d, Real p);
        Real log_likelihood(const DistGamma& d, const VecReal& data);

        Real pdf(const DistBeta& d, Real x);
        Real cdf(const DistBeta& d, Real x);
        Real quantile(const DistBeta& d, Real p);
        Real log_likelihood(const DistBeta& d, const VecReal& data);

        Real pdf(const DistWeibull& d, Real x);
        Real cdf(const DistWeibull& d, Real x);
        Real quantile(const DistWeibull& d, Real p);
        Real log_likelihood(const DistWeibull& d, const VecReal& data);

        Real pdf(const DistCauchy& d, Real x);
        Real cdf(const DistCauchy& d, Real x);
        Real quantile(const DistCauchy& d, Real p);
        Real log_likelihood(const DistCauchy& d, const VecReal& data);

        Real pdf(const DistStudentT& d, Real x);
        Real cdf(const DistStudentT& d, Real x);
        Real quantile(const DistStudentT& d, Real p);
        Real log_likelihood(const DistStudentT& d, const VecReal& data);

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
