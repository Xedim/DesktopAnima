//Utils.h

#pragma once
#include <complex>
#include <limits>
#include "Types.h"


namespace Utils {

    // Policy and Stability
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

    // ---------------------- Real ----------------------

    template<typename T>
    [[nodiscard]] bool isFiniteNum(const T& x) noexcept {
        static_assert(std::is_arithmetic_v<T>, "isFiniteAt: T must be arithmetic");
        return std::isfinite(x);
    }

    [[nodiscard]] inline Real checkStability(Real x,
                                             Real lo = -std::numeric_limits<Real>::max(),
                                             Real hi = std::numeric_limits<Real>::max(),
                                             StabPolicy policy = StabPolicy::Reject) noexcept
    {
        if (!isFiniteNum(x)) return std::numeric_limits<Real>::quiet_NaN();
        return applyPolicy(x, lo, hi, policy);
    }

    [[nodiscard]] inline Real evalStable(const Function1D& f, Real x,
                                         Real lo = -std::numeric_limits<Real>::max(),
                                         Real hi =  std::numeric_limits<Real>::max(),
                                         StabPolicy policy = StabPolicy::Reject) noexcept
    {
        if (!isFiniteNum(x)) return NaN();
        Real y = f(x);
        if (!isFiniteNum(y)) return NaN();
        return checkStability(y, lo, hi, policy);
    }

    [[nodiscard]] inline Real evalStable(const Function2D& f, Real x, Real y,
                                         Real lo = -std::numeric_limits<Real>::max(),
                                         Real hi =  std::numeric_limits<Real>::max(),
                                         StabPolicy policy = StabPolicy::Reject) noexcept
    {
        if (!isFiniteNum(x) || !isFiniteNum(y)) return NaN();
        Real val = f(x, y);
        if (!isFiniteNum(val)) return NaN();
        return checkStability(val, lo, hi, policy);
    }

    // ---------------------- Complex -------------------

    template<>
    [[nodiscard]] inline bool isFiniteNum(const Complex& x) noexcept {
        return std::isfinite(x.real()) && std::isfinite(x.imag());
    }

    [[nodiscard]] inline Complex checkStability(const Complex& z,
                                                Real lo = -std::numeric_limits<Real>::max(),
                                                Real hi = std::numeric_limits<Real>::max(),
                                                StabPolicy policy = StabPolicy::Reject) noexcept
    {
        Real re = checkStability(z.real(), lo, hi, policy);
        Real im = checkStability(z.imag(), lo, hi, policy);
        if (!isFiniteNum(re) || !isFiniteNum(im))
            return {std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::quiet_NaN()};
        return {re, im};
    }

    template<typename F>
    [[nodiscard]] Complex evalStable(const F& f, const Complex& z,
                                        Real lo = -std::numeric_limits<Real>::max(),
                                        Real hi =  std::numeric_limits<Real>::max(),
                                        StabPolicy policy = StabPolicy::Reject) noexcept
    {
        if (!isFiniteNum(z)) return {NaN(), NaN()};

        Complex y = f(z);
        if (!isFiniteNum(y)) return {NaN(), NaN()};

        return checkStability(y, lo, hi, policy);
    }

}
