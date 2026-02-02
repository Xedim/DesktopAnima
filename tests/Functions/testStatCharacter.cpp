#include <gtest/gtest.h>
#include "../math/pattern/Functions.h"
#include <vector>
#include <complex>
#include <cmath>

using Complex = std::complex<double>;

TEST(CharacteristicFunctions, NormalAtZero) {
    double mu = 1.0, sigma = 2.0;
    Complex phi0 = Functions::normal_characteristic(0.0, mu, sigma);
    EXPECT_NEAR(std::abs(phi0), 1.0, 1e-12);
}

TEST(CharacteristicFunctions, NormalMagnitude) {
    double mu = 0.0, sigma = 1.0;
    for(double t : {-10, -1, 0, 1, 10}) {
        Complex phi = Functions::normal_characteristic(t, mu, sigma);
        EXPECT_LE(std::abs(phi), 1.0);
    }
}

TEST(CharacteristicFunctions, SamplesAtZero) {
    std::vector<double> samples = {0.0, 1.0, 2.0, 3.0};
    Complex phi0 = Functions::samples_characteristic(samples, 0.0);
    EXPECT_NEAR(std::abs(phi0), 1.0, 1e-12);
}

TEST(CharacteristicFunctions, SamplesMagnitude) {
    std::vector<double> samples = {0.0, 1.0, 2.0, 3.0};
    for(double t : {-10, -1, 0, 1, 10}) {
        Complex phi = Functions::samples_characteristic(samples, t);
        EXPECT_LE(std::abs(phi), 1.0);
    }
}

TEST(CharacteristicFunctions, SymmetryNormal) {
    double mu = 1.0, sigma = 2.0;
    for(double t : {-5, -1, 0, 1, 5}) {
        Complex phi_pos = Functions::normal_characteristic(t, mu, sigma);
        Complex phi_neg = Functions::normal_characteristic(-t, mu, sigma);
        EXPECT_NEAR(phi_pos.real(), phi_neg.real(), 1e-12);
        EXPECT_NEAR(phi_pos.imag(), -phi_neg.imag(), 1e-12);
    }
}

TEST(CharacteristicFunctions, SymmetrySamples) {
    std::vector<double> samples = {1.0, 2.0, 3.0};
    for(double t : {-5, -1, 0, 1, 5}) {
        Complex phi_pos = Functions::samples_characteristic(samples, t);
        Complex phi_neg = Functions::samples_characteristic(samples, -t);
        EXPECT_NEAR(phi_pos.real(), phi_neg.real(), 1e-12);
        EXPECT_NEAR(phi_pos.imag(), -phi_neg.imag(), 1e-12);
    }
}
