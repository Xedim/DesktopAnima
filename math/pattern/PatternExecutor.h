// PatternExecutor.h
#pragma once
#include "PatternIdentity.h"
#include "PatternRegistry.h"
#include "Functions.h"
#include "../common/Types.h"
#include <array>
#include <variant>
#include <cassert>

namespace pattern {

    struct Executor {

        Executor() {
            #define X(id, sig, fn, name, domain, range, kind) \
            register_##sig(PatternID::id, Functions::fn);
            #include "PatternList.h"
            #undef X
        }

        std::array<UnaryFns, size_t(PatternID::_Count)> unary{};
        void register_unary(PatternID id, UnaryFns f) {
            unary[size_t(id)] = f;
        }

        Real call_unary(PatternID id, ArgVariant x) const {
            const auto& fvar = unary[size_t(id)];
            return std::visit([x](auto&& f) -> Real {
                using T = std::decay_t<decltype(f)>;
                if constexpr (std::is_same_v<T, RealRealFn>) {
                    return f(std::get<Real>(x));
                } else if constexpr (std::is_same_v<T, RealIntFn>) {
                    return f(std::get<int>(x));
                } else if constexpr (std::is_same_v<T, RealVecFn>) {
                    return f(std::get<VecReal>(x));
                } else {
                    assert(false && "Pattern ID is not a unary function");
                    return 0;
                }
            }, fvar);
        }

        std::array<BinaryFns, size_t(PatternID::_Count)> binary{};
        void register_binary(PatternID id, BinaryFns f) {
            binary[size_t(id)] = f;
        }

        Real call_binary(PatternID id, ArgVariant a, ArgVariant b) const {
            const auto& f = binary[size_t(id)];
            return std::visit([&](auto&& fn) -> Real {
                using T = std::decay_t<decltype(fn)>;
                if constexpr (std::is_same_v<T, RealRealFn>) {
                    return fn(std::get<Real>(a), std::get<Real>(b));
                } else if constexpr (std::is_same_v<T, RealIntFn>) {
                    return fn(std::get<Real>(a), std::get<int>(b));
                } else if constexpr (std::is_same_v<T, RealVecFn>) {
                    return fn(std::get<Real>(a), std::get<VecReal>(b));
                } else if constexpr (std::is_same_v<T, IntRealFn>) {
                    return fn(std::get<int>(a), std::get<Real>(b));
                } else if constexpr (std::is_same_v<T, IntIntFn>) {
                    return fn(std::get<int>(a), std::get<int>(b));
                } else if constexpr (std::is_same_v<T, IntVecFn>) {
                    return fn(std::get<int>(a), std::get<VecReal>(b));
                } else if constexpr (std::is_same_v<T, VecRealFn>) {
                    return fn(std::get<VecReal>(a), std::get<Real>(b));
                } else if constexpr (std::is_same_v<T, VecIntFn>) {
                    return fn(std::get<VecReal>(a), std::get<int>(b));
                } else if constexpr (std::is_same_v<T, VecVecFn>) {
                    return fn(std::get<VecReal>(a), std::get<VecReal>(b));
                } else {
                    assert(false && "Unknown binary function type");
                    return 0;
                }
            }, f);
        }

        std::array<TernaryFns, size_t(PatternID::_Count)> ternary{};
        void register_ternary(PatternID id, TernaryFns f) {
            ternary[size_t(id)] = f;
        }

        Real call_ternary(PatternID id, ArgVariant a, ArgVariant b, ArgVariant c) const {
            const auto& f = ternary[size_t(id)];
            return std::visit([&](auto&& fn) -> Real {
                using T = std::decay_t<decltype(fn)>;

                if constexpr (std::is_same_v<T, RealRealRealFn>)
                    return fn(std::get<Real>(a), std::get<Real>(b), std::get<Real>(c));
                else if constexpr (std::is_same_v<T, RealRealIntFn>)
                    return fn(std::get<Real>(a), std::get<Real>(b), std::get<int>(c));
                else if constexpr (std::is_same_v<T, RealRealVecFn>)
                    return fn(std::get<Real>(a), std::get<Real>(b), std::get<VecReal>(c));
                else if constexpr (std::is_same_v<T, RealIntRealFn>)
                    return fn(std::get<Real>(a), std::get<int>(b), std::get<Real>(c));
                else if constexpr (std::is_same_v<T, RealIntIntFn>)
                    return fn(std::get<Real>(a), std::get<int>(b), std::get<int>(c));
                else if constexpr (std::is_same_v<T, RealIntVecFn>)
                    return fn(std::get<Real>(a), std::get<int>(b), std::get<VecReal>(c));
                else if constexpr (std::is_same_v<T, RealVecRealFn>)
                    return fn(std::get<Real>(a), std::get<VecReal>(b), std::get<Real>(c));
                else if constexpr (std::is_same_v<T, RealVecIntFn>)
                    return fn(std::get<Real>(a), std::get<VecReal>(b), std::get<int>(c));
                else if constexpr (std::is_same_v<T, RealVecVecFn>)
                    return fn(std::get<Real>(a), std::get<VecReal>(b), std::get<VecReal>(c));
                else if constexpr (std::is_same_v<T, IntRealRealFn>)
                    return fn(std::get<int>(a), std::get<Real>(b), std::get<Real>(c));
                else if constexpr (std::is_same_v<T, IntRealIntFn>)
                    return fn(std::get<int>(a), std::get<Real>(b), std::get<int>(c));
                else if constexpr (std::is_same_v<T, IntRealVecFn>)
                    return fn(std::get<int>(a), std::get<Real>(b), std::get<VecReal>(c));
                else if constexpr (std::is_same_v<T, IntIntRealFn>)
                    return fn(std::get<int>(a), std::get<int>(b), std::get<Real>(c));
                else if constexpr (std::is_same_v<T, IntIntIntFn>)
                    return fn(std::get<int>(a), std::get<int>(b), std::get<int>(c));
                else if constexpr (std::is_same_v<T, IntIntVecFn>)
                    return fn(std::get<int>(a), std::get<int>(b), std::get<VecReal>(c));
                else if constexpr (std::is_same_v<T, IntVecRealFn>)
                    return fn(std::get<int>(a), std::get<VecReal>(b), std::get<Real>(c));
                else if constexpr (std::is_same_v<T, IntVecIntFn>)
                    return fn(std::get<int>(a), std::get<VecReal>(b), std::get<int>(c));
                else if constexpr (std::is_same_v<T, IntVecVecFn>)
                    return fn(std::get<int>(a), std::get<VecReal>(b), std::get<VecReal>(c));
                else if constexpr (std::is_same_v<T, VecRealRealFn>)
                    return fn(std::get<VecReal>(a), std::get<Real>(b), std::get<Real>(c));
                else if constexpr (std::is_same_v<T, VecRealIntFn>)
                    return fn(std::get<VecReal>(a), std::get<Real>(b), std::get<int>(c));
                else if constexpr (std::is_same_v<T, VecRealVecFn>)
                    return fn(std::get<VecReal>(a), std::get<Real>(b), std::get<VecReal>(c));
                else if constexpr (std::is_same_v<T, VecIntRealFn>)
                    return fn(std::get<VecReal>(a), std::get<int>(b), std::get<Real>(c));
                else if constexpr (std::is_same_v<T, VecIntIntFn>)
                    return fn(std::get<VecReal>(a), std::get<int>(b), std::get<int>(c));
                else if constexpr (std::is_same_v<T, VecIntVecFn>)
                    return fn(std::get<VecReal>(a), std::get<int>(b), std::get<VecReal>(c));
                else if constexpr (std::is_same_v<T, VecVecRealFn>)
                    return fn(std::get<VecReal>(a), std::get<VecReal>(b), std::get<Real>(c));
                else if constexpr (std::is_same_v<T, VecVecIntFn>)
                    return fn(std::get<VecReal>(a), std::get<VecReal>(b), std::get<int>(c));
                else if constexpr (std::is_same_v<T, VecVecVecFn>)
                    return fn(std::get<VecReal>(a), std::get<VecReal>(b), std::get<VecReal>(c));
                else {
                    assert(false && "Unknown ternary function type");
                    return 0;
                }
            }, f);
        }

        std::array<ComplexFns, size_t(PatternID::_Count)> complex{};
        void register_complex(PatternID id, ComplexFns f) {
            complex[size_t(id)] = f;
        }

        ArgVariant call_complex_to_bool(PatternID id, ArgVariant a, ArgVariant b, ArgVariant c = {}, ArgVariant d = {}) const {
            const auto& f = complex[size_t(id)];
            return std::visit([&](auto&& fn) -> ArgVariant {
                using T = std::decay_t<decltype(fn)>;
                if constexpr (std::is_same_v<T, ComplexComplexFn>)
                    return fn(std::get<Complex>(a), std::get<Complex>(b));
                else if constexpr (std::is_same_v<T, ComplexComplexRealRealFn>)
                    return fn(std::get<Complex>(a), std::get<Complex>(b), std::get<Real>(c), std::get<Real>(d));
                else if constexpr (std::is_same_v<T, Complex_3RealFn>)
                    return fn(std::get<Real>(a), std::get<Real>(b), std::get<Real>(c));
                else if constexpr (std::is_same_v<T, Complex_VecRealFn>)
                    return fn(std::get<VecReal>(a), std::get<Real>(b));
                else {
                    assert(false && "Function does not return bool");
                    return false;
                }
            }, f);
        }

        std::array<PolicyFns, size_t(PatternID::_Count)> political{};
        void register_political(PatternID id, PolicyFns f) {
            political[size_t(id)] = f;
        }

        Real call_policy(PatternID id, ArgVariant a, ArgVariant b = {}, ArgVariant c = {},
                 ArgVariant d = {}, ArgVariant e = {}) const {
            const auto& f = political[size_t(id)];
            return std::visit([&](auto&& fn) -> Real {
                using T = std::decay_t<decltype(fn)>;

                if constexpr (std::is_same_v<T, RealRealRealIntPolicyFn>) {
                    return fn(
                        std::get<Real>(a),
                        std::get<Real>(b),
                        std::get<Real>(c),
                        std::get<int>(d),
                        std::get<StabPolicy>(e)
                    );
                }
                else if constexpr (std::is_same_v<T, RealIntPolicyFn>) {
                    return fn(
                        std::get<Real>(a),
                        std::get<int>(b),
                        std::get<StabPolicy>(c)
                    );
                }
                else if constexpr (std::is_same_v<T, RealRealPolicyFn>) {
                    return fn(
                        std::get<Real>(a),
                        std::get<Real>(b),
                        std::get<StabPolicy>(c)
                    );
                }
                else if constexpr (std::is_same_v<T, RealPolicyFn>) {
                    return fn(
                        std::get<Real>(a),
                        std::get<StabPolicy>(b)
                    );
                }
                else if constexpr (std::is_same_v<T, RealRealIntPolicyFn>) {
                    return fn(
                        std::get<Real>(a),
                        std::get<Real>(b),
                        std::get<int>(c),
                        std::get<StabPolicy>(d)
                    );
                }
                else {
                    assert(false && "Unknown policy function type");
                    return 0;
                }
            }, f);
        }

        std::array<BoolFns, size_t(PatternID::_Count)> bools{};
        void register_bool(PatternID id, BoolFns f) {
            bools[size_t(id)] = f;
        }

        // Вызов функций, возвращающих bool
        bool call_bool(PatternID id, ArgVariant a, ArgVariant b = {}, ArgVariant c = {}, ArgVariant d = {}) const {
            const auto& f = bools[size_t(id)];
            return std::visit([&](auto&& fn) -> bool {
                using T = std::decay_t<decltype(fn)>;

                if constexpr (std::is_same_v<T, bool_VecRealFn>) {
                    return fn(std::get<VecReal>(a), std::get<Real>(b));
                }
                else if constexpr (std::is_same_v<T, bool_VecFn>) {
                    return fn(std::get<VecReal>(a));
                }
                else if constexpr (std::is_same_v<T, Bool4RealFn>) {
                    return fn(std::get<Real>(a), std::get<Real>(b), std::get<Real>(c), std::get<Real>(d));
                }
                else {
                    assert(false && "Unknown bool function type");
                    return false;
                }
            }, f);
        }

        std::array<SpecialFns, size_t(PatternID::_Count)> special{};
        void register_special(PatternID id, SpecialFns f) {
            special[size_t(id)] = f;
        }

        ArgVariant call_special(PatternID id, ArgVariant a, ArgVariant b = {}, ArgVariant c = {}) const {
            const auto& f = special[size_t(id)];
            return std::visit([&](auto&& fn) -> ArgVariant {
                using T = std::decay_t<decltype(fn)>;

                if constexpr (std::is_same_v<T, Pair_VecRealFn>) {
                    return fn(std::get<VecReal>(a), std::get<Real>(b));
                }
                else if constexpr (std::is_same_v<T, LR_VecVecFn>) {
                    return fn(std::get<VecReal>(a), std::get<VecReal>(b));
                }
                else if constexpr (std::is_same_v<T, Quartiles_VecFn>) {
                    return fn(std::get<VecReal>(a));
                }
                else {
                    assert(false && "Unknown special function type");
                    return ArgVariant{};
                }
            }, f);
        }

        std::array<DistFns, size_t(PatternID::_Count)> dist_fns{};

        template<typename Dist>
        Real call_dist(PatternID id, const Dist& dist, Real x) const {
            const auto& fn = std::get<DistRealFnT<Dist>>(dist_fns[size_t(id)]);
            return fn(dist, x);
        }

        template<typename Dist>
        Real call_dist(PatternID id, const Dist& dist, const VecReal& vec) const {
            const auto& fn = std::get<DistVecRealFnT<Dist>>(dist_fns[size_t(id)]);
            return fn(dist, vec);
        }

    };

} // namespace pattern
