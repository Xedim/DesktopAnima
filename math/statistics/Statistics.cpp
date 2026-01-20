#include "Statistics.h"

namespace Statistics {

    // ---------------- Distributions ----------------
    std::normal_distribution<double> normal(0.0, 1.0);
    std::cauchy_distribution<double> cauchy(0.0, 1.0);
    std::lognormal_distribution<double> lognormal(0.0, 1.0);

    // ---------------- PDF / CDF Wrappers ----------------
    double pdf_normal(double x, double mu, double sigma) {
        double coeff = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
        double exp_term = std::exp(-0.5 * std::pow((x - mu) / sigma, 2));
        return coeff * exp_term;
    }

    double cdf_normal(double x, double mu, double sigma) {
        return 0.5 * (1.0 + std::erf((x - mu) / (sigma * std::sqrt(2.0))));
    }

    double pdf_cauchy(double x, double x0, double gamma) {
        return (1.0 / M_PI) * (gamma / (std::pow(x - x0, 2) + gamma * gamma));
    }

    double cdf_cauchy(double x, double x0, double gamma) {
        return 0.5 + (1.0 / M_PI) * std::atan((x - x0) / gamma);
    }

    double pdf_lognormal(double x, double m, double s) {
        if (x <= 0.0) return 0.0;
        return (1.0 / (x * s * std::sqrt(2.0 * M_PI))) *
               std::exp(-std::pow(std::log(x) - m, 2) / (2.0 * s * s));
    }

    double cdf_lognormal(double x, double m, double s) {
        if (x <= 0.0) return 0.0;
        return 0.5 + 0.5 * std::erf((std::log(x) - m) / (s * std::sqrt(2.0)));
    }

    // ---------------- Characteristic functions ----------------
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

}
