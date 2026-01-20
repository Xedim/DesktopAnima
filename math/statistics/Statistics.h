#pragma once
#include <vector>
#include <random>
#include <complex>
#include <cmath>

namespace Statistics {

    // Distributions
    extern std::normal_distribution<double> normal;
    extern std::cauchy_distribution<double> cauchy;
    extern std::lognormal_distribution<double> lognormal;

    // PDF / CDF wrappers (можно для всех дистрибуций)
    double pdf_normal(double x, double mu = 0.0, double sigma = 1.0);
    double cdf_normal(double x, double mu = 0.0, double sigma = 1.0);

    double pdf_cauchy(double x, double x0 = 0.0, double gamma = 1.0);
    double cdf_cauchy(double x, double x0 = 0.0, double gamma = 1.0);

    double pdf_lognormal(double x, double m = 0.0, double s = 1.0);
    double cdf_lognormal(double x, double m = 0.0, double s = 1.0);

    // Characteristic functions
    std::complex<double> normal_characteristic(double t,
                                               double mu,
                                               double sigma);

    std::complex<double> characteristic_from_samples(const std::vector<double>& samples,
                                                     double t);

}
