//Types.h

#pragma once
#include <cstdint>
#include <complex>
#include <vector>
#include <functional>
#include <limits>
#include <algorithm>

// Алиасы для читаемости
using Real = double;
using Complex = std::complex<double>;
using VecReal = std::vector<Real>;
using VecComplex = std::vector<Complex>;
using Function1D = std::function<Real(Real)>;
using Function2D = std::function<Real(Real, Real)>;

// Можно добавить типы для статистики
using PDF = std::function<Real(Real)>;
using CDF = std::function<Real(Real)>;

// Единый NaN для всего проекта
inline Real NaN() noexcept {
    return std::numeric_limits<Real>::quiet_NaN();
}

enum class StabPolicy {
    Raw,        // без clamp
    Clamp,      // проекция
    Reject      // NaN при выходе
};

template<typename T>
[[nodiscard]] constexpr T clamp_range(T x, T lo, T hi) noexcept {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

[[nodiscard]] inline Real applyPolicy(Real x, Real lo, Real hi, StabPolicy policy) noexcept {
    switch (policy) {
        case StabPolicy::Raw:   return x;
        case StabPolicy::Clamp: return clamp_range(x, lo, hi);
        case StabPolicy::Reject: return (x < lo || x > hi) ? NaN() : x;
    }
    return NaN();
}

namespace Utils {

    // Сумма геометрической прогрессии: 1 + a + a^2 + ... + a^(N-1)
    [[nodiscard]] constexpr double geometric_sum(double a, int N) noexcept {
        if (a == 1.0) return static_cast<double>(N);
        double sum = 0.0;
        double term = 1.0;
        for (int i = 0; i < N; ++i) {
            sum += term;
            term *= a;
        }
        return sum;
    }

    [[nodiscard]] inline Real safeEval(const Function1D& f, Real x,
                                       Real lo = -std::numeric_limits<Real>::max(),
                                       Real hi = std::numeric_limits<Real>::max(),
                                       StabPolicy policy = StabPolicy::Reject) noexcept
    {
        if (!std::isfinite(x)) return NaN();

        Real y = f(x);
        if (!std::isfinite(y)) return NaN();

        return applyPolicy(y, lo, hi, policy);
    }

}
