#include <vector>
#include <cmath>
#include <functional>
#include <random>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/cauchy.hpp>
#include <boost/math/special_functions/zeta.hpp>
#include <complex>
#include <limits>

//--------------polynomial--------------
double polynomial(double x, const std::vector<double> &coefficients) {
    double result = 0.0;
    for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
        result = result * x + *it;
    }
    return result;
}

//--------------rational--------------
double rational(double x,
                const std::vector<double> &p,
                const std::vector<double> &q) {
    double num = polynomial(x, p);
    double den = polynomial(x, q);
    if (den == 0) { return 0; } else { return num / den; }
}

//--------------irrational--------------
//std::sqrt(x);
//std::cbrt(x);
double algebraic_root(double x, const std::vector<double> &coefficients) {
    return std::sqrt(polynomial(x, coefficients));
}

//--------------power-exponential--------------
//std::pow(x, alpha);
//std::abs(x);
//std::exp(x);

//--------------logarithmic--------------
//std::log(x);
//std::log10(x);

double log_a(double x, double a) {
    return std::log(x) / std::log(a);
}

//--------------trigonometric--------------
//std::sin(x);
//std::cos(x);
//std::tan(x)
//1 / std::tan(x)

//--------------reversed-trigonometric--------------
//std::asin(x)
//std::acos(x)
//std::atan(x)
//std::atan2(y, x);

//--------------hyperbolic--------------
//std::sinh(x);
//std::cosh(x);
//std::tanh(x);
//1.0 / std::tanh(x);

//--------------reversed-hyperbolic--------------
//std::asinh(x);
//std::acosh(x);
//std::atanh(x);

//--------------hybrid--------------
double x_pow_y(double x, double y) {
    return std::exp(y * std::log(x));
}

//--------------special--------------
//gamma
//std::tgamma(x);
//std::lgamma(x);
//std::beta(x, y);
//beta = std::tgamma(x) * std::tgamma(y) / std::tgamma(x + y);

//bessel
//std::cyl_bessel_j(nu, x);
//std::cyl_neumann(nu, x);
//std::cyl_bessel_i(nu, x);
//std::cyl_bessel_k(nu, x);

//lambert
//boost::math::lambert_w(x)

//legendre
//std::legendre(l, x);
//std::assoc_legendre(l, m, x);

//error-functions
//std::erf(x);
//std::erfc(x);

//riemann
//std::riemann_zeta(s);

//boost::math::zeta (s);

double weierstrass(double x,
                   double a = 0.5,
                   double b = 3.0,
                   int N = 50)
{
    double sum = 0.0;
    for (int n = 0; n < N; ++n) {
        sum += std::pow(a, n) * std::cos(std::pow(b, n) * M_PI * x);
    }
    return sum;
}

double cantor(double x, int max_iter = 32)
{
    double y = 0.0;
    double factor = 1.0;

    for (int i = 0; i < max_iter; ++i) {
        if (x < 1.0 / 3.0) {
            x *= 3.0;
        } else if (x > 2.0 / 3.0) {
            y += factor;
            x = 3.0 * x - 2.0;
        } else {
            return y + factor * 0.5;
        }
        factor *= 0.5;
    }
    return y;
}

std::vector<double> takens_map(const std::vector<double>& signal,
                               int dim,
                               int tau)
{
    std::vector<double> embedded;
    for (size_t i = 0; i + (dim - 1) * tau < signal.size(); ++i) {
        for (int j = 0; j < dim; ++j) {
            embedded.push_back(signal[i + j * tau]);
        }
    }
    return embedded;
}

double heaviside(double x)
{
    return x < 0.0 ? 0.0 : 1.0;
}

double smooth_heaviside(double x, double eps = 1e-3)
{
    return 0.5 * (1.0 + std::tanh(x / eps));
}

//-------------differential-------------
double derivative(double (*f)(double), double x) {
    const double h = 1e-6;
    return (f(x + h) - f(x - h)) / (2 * h);
}

//--------------generalised----------------
double dirac_delta(double x, double eps = 1e-3) {
    return std::exp(-x * x / eps) / std::sqrt(M_PI * eps);
}

//------------------norm----------------

using Function = std::function<double(double)>;

double L2_norm(Function f, double a, double b, int n = 10000) {
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; ++i) {
        double x = a + i * h;
        sum += f(x) * f(x);
    }
    return std::sqrt(sum * h);
}

//------------statistical-----------
//distributions
std::normal_distribution<double> normal(0.0, 1.0);
std::cauchy_distribution<double> cauchy(0.0, 1.0);
std::lognormal_distribution<double> lognormal(0.0, 1.0);

//density

//boost::math::normal nd(0.0, 1.0);
//double pdf = boost::math::pdf(nd, x);
//double cdf = boost::math::cdf(nd, x);

//characteristic

std::complex<double> normal_characteristic(double t,
                                            double mu,
                                            double sigma)
{
    double real = -0.5 * sigma * sigma * t * t;
    double imag = mu * t;
    return std::exp(std::complex<double>(real, imag));
}

std::complex<double> characteristic_from_samples(
    const std::vector<double>& samples,
    double t)
{
    std::complex<double> sum(0.0, 0.0);
    for (double x : samples)
        sum += std::exp(std::complex<double>(0.0, t * x));
    return sum / static_cast<double>(samples.size());
}

//-----------------complex-----------------
std::complex<double> clog(const std::complex<double>& z)
{
    return std::log(z);
}

std::complex<double> julia(std::complex<double> z,
                           std::complex<double> c)
{
    return z * z + c;
}

bool escapes(std::complex<double> z0,
             std::complex<double> c,
             int max_iter = 100)
{
    std::complex<double> z = z0;
    for (int i = 0; i < max_iter; ++i) {
        z = z * z + c;
        if (std::abs(z) > 2.0)
            return true;
    }
    return false;
}

//------------fractal--------------
double logistic(double x, double r)
{
    return r * x * (1.0 - x);
}

double iterate(double x0, double r, int n)
{
    double x = x0;
    for (int i = 0; i < n; ++i)
        x = logistic(x, r);
    return x;
}

double tent(double x)
{
    return x < 0.5 ? 2.0 * x : 2.0 * (1.0 - x);
}

//-----------properties------------

using Func = std::function<double(double)>;

bool isEvenFunction(const Func& f, double x, double eps = 1e-9)
{
    return std::abs(f(-x) - f(x)) < eps;
}

bool isOddFunction(const Func& f, double x, double eps = 1e-9)
{
    return std::abs(f(-x) + f(x)) < eps;
}

bool isPeriodic(const Func& f, double x, double T, double eps = 1e-9)
{
    return std::abs(f(x + T) - f(x)) < eps;
}

bool isIncreasing(const Func& f, double x1, double x2, double eps = 1e-9)
{
    return f(x2) >= f(x1) - eps;
}

bool isDecreasing(const Func& f, double x1, double x2, double eps = 1e-9)
{
    return f(x2) <= f(x1) + eps;
}

bool isBounded(const Func& f, double x)
{
    double y = f(x);
    return std::isfinite(y);
}

bool isConvex(const Func& f, double x, double h = 1e-5)
{
    double fpp = (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
    return fpp >= 0;
}

bool isConcave(const Func& f, double x, double h = 1e-5)
{
    double fpp = (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
    return fpp <= 0;
}

bool isNonNegative(const Func& f, double x)
{
    return f(x) >= 0;
}

bool isContinuous(const Func& f, double x, double h = 1e-6)
{
    return std::abs(f(x + h) - f(x)) < 1e-3;
}

//-----------fields--------------
//std::hypot


//------------libs--------------
//Готовые библиотеки
//Eigen
//Boost.Math
//GSL (GNU Scientific Library)
//deal.II / FEniCS