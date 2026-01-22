// PatternExecutor.h
#pragma once
#include "PatternIdentity.h"
#include "PatternRegistry.h"
#include "Pattern1D.h"
#include "Functions.h"
#include "../common/Types.h"
#include <array>

namespace pattern1D {

    struct PatternExecutor {

        static constexpr size_t N = static_cast<size_t>(PatternID::_Count);

        // массив указателей на функции
        std::array<Function1D, N> functions{};

        // ---------------- 1D функции ----------------
        std::array<std::function<Real(Real)>, N> functions1D{};

        // ---------------- функции с VecReal ----------------
        std::array<std::function<Real(const VecReal&)>, N> functionsVec{};

        // ---------------- функции с int,int ----------------
        std::array<std::function<Real(int,int)>, N> functionsIntInt{};

        // конструктор инициализирует функции
        PatternExecutor() {

            // ---------------- Algebraic ----------------
            functions1D[static_cast<size_t>(PatternID::Sqrt)] = Functions::sqrt;
            functions1D[static_cast<size_t>(PatternID::Cbrt)] = Functions::cbrt;

            functions1D[static_cast<size_t>(PatternID::Sqrt)] = Functions::sign;
            functions1D[static_cast<size_t>(PatternID::Cbrt)] = Functions::abs;
            functions1D[static_cast<size_t>(PatternID::Exp)]  = Functions::exp;
            functions1D[static_cast<size_t>(PatternID::Exp2)]  = Functions::exp2;
            functions1D[static_cast<size_t>(PatternID::Expm1Safe)]  = Functions::expm1_safe;

            functions1D[static_cast<size_t>(PatternID::Log)]  = Functions::log;
            functions1D[static_cast<size_t>(PatternID::Log2)]  = Functions::log2;
            functions1D[static_cast<size_t>(PatternID::Log10)]  = Functions::log10;
            functions1D[static_cast<size_t>(PatternID::Log1p)]  = Functions::log1p;

            // ---------------- Trigonometric ----------------
            functions1D[static_cast<size_t>(PatternID::Sin)]            = Functions::sin;
            functions1D[static_cast<size_t>(PatternID::Cos)]            = Functions::cos;
            functions1D[static_cast<size_t>(PatternID::Sec)]            = Functions::sec;
            functions1D[static_cast<size_t>(PatternID::Csc)]            = Functions::csc;
            functions1D[static_cast<size_t>(PatternID::Sinc)]           = Functions::sinc;
            functions1D[static_cast<size_t>(PatternID::Tan)]            = Functions::tan;
            functions1D[static_cast<size_t>(PatternID::Cot)]            = Functions::cot;
            functions1D[static_cast<size_t>(PatternID::Asin)]           = Functions::asin;
            functions1D[static_cast<size_t>(PatternID::Acos)]           = Functions::acos;
            functions1D[static_cast<size_t>(PatternID::Atan)]           = Functions::atan;

            // ---------------- Hyperbolic ----------------
            functions[static_cast<size_t>(PatternID::Sinh)]           = Functions::sinh;
            functions[static_cast<size_t>(PatternID::Cosh)]           = Functions::cosh;
            functions[static_cast<size_t>(PatternID::Sech)]           = Functions::sech;
            functions[static_cast<size_t>(PatternID::Csch)]           = Functions::csch;
            functions[static_cast<size_t>(PatternID::Tanh)]           = Functions::tanh;
            functions[static_cast<size_t>(PatternID::Coth)]           = Functions::coth;
            functions[static_cast<size_t>(PatternID::Asinh)]          = Functions::asinh;
            functions[static_cast<size_t>(PatternID::Acosh)]          = Functions::acosh;
            functions[static_cast<size_t>(PatternID::Atanh)]          = Functions::atanh;

            // ---------------- Hybrid / Numerical ----------------
            functions[static_cast<size_t>(PatternID::Sqrt1pM1)]       = Functions::sqrt1pm1;
            functions[static_cast<size_t>(PatternID::Heaviside)]      = Functions::heaviside;

            // ---------------- Special Functions ----------------
            functions[static_cast<size_t>(PatternID::Erf)]            = Functions::erf;
            functions[static_cast<size_t>(PatternID::Erfc)]           = Functions::erfc;
            functions[static_cast<size_t>(PatternID::Gamma)]          = Functions::gamma;
            functions[static_cast<size_t>(PatternID::LGamma)]         = Functions::lgamma;
            functions[static_cast<size_t>(PatternID::LambertW)]       = Functions::lambert_w;
            functions[static_cast<size_t>(PatternID::RiemannZeta)]    = Functions::riemann_zeta;
            functions[static_cast<size_t>(PatternID::Zeta)]           = Functions::zeta;

        }

        [[nodiscard]] Real operator()(PatternID id, Real x) const {
            size_t idx = static_cast<size_t>(id);
            if (idx >= N || !functions[idx]) return NaN();
            return functions[idx](x);
        }

        [[nodiscard]] constexpr const PatternDescriptor& descriptor(PatternID id) const {
            return pattern1D::descriptor(id);
        }

        [[nodiscard]] constexpr Interval domain(PatternID id) const {
            return pattern1D::domain(id);
        }

        [[nodiscard]] constexpr Interval range(PatternID id) const {
            return pattern1D::range(id);
        }
    };

} // namespace pattern1D
