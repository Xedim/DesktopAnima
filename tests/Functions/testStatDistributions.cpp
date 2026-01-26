#include <gtest/gtest.h>
#include "../math/pattern/Functions.h"
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <vector>
#include <cmath>

struct DistCase {
    std::string name;
    std::function<Real(Real)> pdf;
    std::function<Real(Real)> cdf;
    std::function<Real(Real)> quantile;
    std::function<Real(const std::vector<Real>&)> log_likelihood;
    std::vector<Real> test_points;
    std::vector<Real> test_probs; // для quantile
    std::vector<Real> test_data;  // для log_likelihood
};

// Универсальный тест
class DistTests : public ::testing::TestWithParam<DistCase> {};

TEST_P(DistTests, PdfNonNegative) {
    const auto& d = GetParam();
    for (Real x : d.test_points) {
        EXPECT_GE(d.pdf(x), 0.0);
    }
}

TEST_P(DistTests, CdfMonotone) {
    const auto& d = GetParam();
    Real prev = d.cdf(d.test_points.front());
    for (Real x : d.test_points) {
        Real c = d.cdf(x);
        EXPECT_GE(c, prev);
        prev = c;
    }
}

TEST_P(DistTests, QuantileCdfConsistency) {
    const auto& d = GetParam();
    for (Real p : d.test_probs) {
        if (p > 0 && p < 1) {
            Real x = d.quantile(p);
            Real c = d.cdf(x);
            EXPECT_NEAR(c, p, 1e-6);
        } else {
            EXPECT_TRUE(std::isnan(d.quantile(p)));
        }
    }
}

TEST_P(DistTests, LogLikelihoodEmptyData) {
    const auto& d = GetParam();
    std::vector<Real> empty;
    EXPECT_TRUE(std::isnan(d.log_likelihood(empty)));
}

std::vector<DistCase> make_normal_cases() {
    Functions::dist::Normal d(0.0, 1.0);
    return {{
        "Normal(0,1)",
        [&](Real x){ return Functions::dist::pdf(d, x); },
        [&](Real x){ return Functions::dist::cdf(d, x); },
        [&](Real p){ return Functions::dist::quantile(d, p); },
        [&](const std::vector<Real>& data){ return Functions::dist::log_likelihood(d, data); },
        {-3, -2, -1, 0, 1, 2, 3},    // test_points
        {0.01, 0.1, 0.5, 0.9, 0.99}, // test_probs
        {0.0, 0.5, -0.5, 1.0, -1.0}   // test_data
    }};
}

std::vector<DistCase> make_lognormal_cases() {
    Functions::dist::LogNormal d(0.0, 1.0);
    return {{
        "LogNormal(0,1)",
        [&](Real x){ return Functions::dist::pdf(d, x); },
        [&](Real x){ return Functions::dist::cdf(d, x); },
        [&](Real p){ return Functions::dist::quantile(d, p); },
        [&](const std::vector<Real>& data){ return Functions::dist::log_likelihood(d, data); },
        {0.01, 0.1, 0.5, 1.0, 2.0, 5.0},
        {0.01, 0.1, 0.5, 0.9, 0.99},
        {0.5, 1.0, 2.0}
    }};
}

std::vector<DistCase> make_exponential_cases() {
    Functions::dist::Exponential d(2.0);
    return {{
        "Exponential(lambda=2)",
        [&](Real x){ return Functions::dist::pdf(d, x); },
        [&](Real x){ return Functions::dist::cdf(d, x); },
        [&](Real p){ return Functions::dist::quantile(d, p); },
        [&](const std::vector<Real>& data){ return Functions::dist::log_likelihood(d, data); },
        {0.0, 0.2, 0.5, 1.0, 2.0, 4.0},
        {0.01, 0.1, 0.5, 0.9},
        {0.1, 0.3, 1.0}
    }};
}

std::vector<DistCase> make_gamma_cases() {
    Functions::dist::Gamma d(2.0, 1.5);
    return {{
        "Gamma(k=2,theta=1.5)",
        [&](Real x){ return Functions::dist::pdf(d, x); },
        [&](Real x){ return Functions::dist::cdf(d, x); },
        [&](Real p){ return Functions::dist::quantile(d, p); },
        [&](const std::vector<Real>& data){ return Functions::dist::log_likelihood(d, data); },
        {0.1, 0.5, 1.0, 2.0, 4.0},
        {0.05, 0.25, 0.5, 0.75, 0.95},
        {0.5, 1.0, 2.0}
    }};
}

std::vector<DistCase> make_beta_cases() {
    Functions::dist::Beta d(2.0, 5.0);
    return {{
        "Beta(2,5)",
        [&](Real x){ return Functions::dist::pdf(d, x); },
        [&](Real x){ return Functions::dist::cdf(d, x); },
        [&](Real p){ return Functions::dist::quantile(d, p); },
        [&](const std::vector<Real>& data){ return Functions::dist::log_likelihood(d, data); },
        {0.01, 0.1, 0.3, 0.5, 0.7, 0.9},
        {0.05, 0.25, 0.5, 0.75, 0.95},
        {0.2, 0.4, 0.6}
    }};
}

std::vector<DistCase> make_weibull_cases() {
    Functions::dist::Weibull d(2.0, 1.0);
    return {{
        "Weibull(k=2,lambda=1)",
        [&](Real x){ return Functions::dist::pdf(d, x); },
        [&](Real x){ return Functions::dist::cdf(d, x); },
        [&](Real p){ return Functions::dist::quantile(d, p); },
        [&](const std::vector<Real>& data){ return Functions::dist::log_likelihood(d, data); },
        {0.01, 0.2, 0.5, 1.0, 2.0},
        {0.01, 0.1, 0.5, 0.9},
        {0.5, 1.0, 1.5}
    }};
}

std::vector<DistCase> make_cauchy_cases() {
    Functions::dist::Cauchy d(0.0, 1.0);
    return {{
        "Cauchy(0,1)",
        [&](Real x){ return Functions::dist::pdf(d, x); },
        [&](Real x){ return Functions::dist::cdf(d, x); },
        [&](Real p){ return Functions::dist::quantile(d, p); },
        [&](const std::vector<Real>& data){ return Functions::dist::log_likelihood(d, data); },
        {-10, -5, -1, 0, 1, 5, 10},
        {0.01, 0.1, 0.5, 0.9, 0.99},
        {-1.0, 0.0, 1.0}
    }};
}

std::vector<DistCase> make_student_cases() {
    Functions::dist::StudentT d(5.0);
    return {{
        "StudentT(nu=5)",
        [&](Real x){ return Functions::dist::pdf(d, x); },
        [&](Real x){ return Functions::dist::cdf(d, x); },
        [&](Real p){ return Functions::dist::quantile(d, p); },
        [&](const std::vector<Real>& data){ return Functions::dist::log_likelihood(d, data); },
        {-5, -2, -1, 0, 1, 2, 5},
        {0.01, 0.1, 0.5, 0.9, 0.99},
        {-1.0, 0.0, 1.0}
    }};
}

std::vector<DistCase> make_all_cases() {
    std::vector<DistCase> cases;

    auto append = [&](std::vector<DistCase>&& v) {
        cases.insert(cases.end(),
                     std::make_move_iterator(v.begin()),
                     std::make_move_iterator(v.end()));
    };

    append(make_normal_cases());
    append(make_lognormal_cases());
    append(make_exponential_cases());
    append(make_gamma_cases());
    append(make_beta_cases());
    append(make_weibull_cases());
    append(make_cauchy_cases());
    append(make_student_cases());

    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Distributions,
    DistTests,
    ::testing::ValuesIn(make_all_cases())
);

