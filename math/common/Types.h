//Types.h

#pragma once
#include <complex>
#include <vector>
#include <functional>
#include <limits>

// Алиасы для читаемости
using Real = double;
using RealPair = std::pair<Real, Real>;
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

//Policies
enum class StabPolicy {
    Raw,        // без clamp
    Clamp,      // проекция
    Reject      // NaN при выходе
};