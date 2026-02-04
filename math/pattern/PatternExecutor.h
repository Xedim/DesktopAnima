// PatternExecutor.h
#pragma once
#include "../common/Types.h"
#include "PatternIdentity.h"
#include <array>
#include <variant>
#include <tuple>
#include <type_traits>
#include <stdexcept>

template<typename R, typename... Args>
auto make_fn(R(*f)(Args...)) {
    return WrapFn<R, std::tuple<Args...>>{std::function<R(Args...)>(f)};
}

namespace pattern {

    struct ExecutorUnified {
        std::array<AnyFnVariant, static_cast<size_t>(PatternID::Count)> registry{};

        // -------------------------
        // Регистрация функции
        // -------------------------
        template<typename FnVariant>
        void register_fn(PatternID id, FnVariant&& f) {
            registry[static_cast<size_t>(id)] = std::forward<FnVariant>(f);
        }

        // -------------------------
        // Универсальный вызов с проверкой типов
        // -------------------------
        template<typename Result = ResultVariant, typename... Args>
        Result call(PatternID id, Args&&... args) const {
            const auto& any_f = registry.at(static_cast<size_t>(id));
            return visit_recursive<Result>(any_f, std::forward<Args>(args)...);
        }

    private:

        // -------------------------
        // Рекурсивный visitor для вложенных variant
        // -------------------------
        template<typename Result, typename Variant, typename... Args>
        static Result visit_recursive(const Variant& var, Args&&... args) {
            return std::visit([&](auto&& wrapped_inner) -> Result {
                using T = std::decay_t<decltype(wrapped_inner)>;
                if constexpr (is_wrapfn_v<T>) {
                    return call_wrapfn<Result>(wrapped_inner, std::forward<Args>(args)...);
                } else {
                    return visit_recursive<Result>(wrapped_inner, std::forward<Args>(args)...);
                }
            }, var);
        }

        // -------------------------
        // Вызов WrapFn
        // -------------------------
        template<typename Result, typename WrapFnType, typename... Args>
        static Result call_wrapfn(const WrapFnType& wrapped, Args&&... args) {
            using Tuple = typename WrapFnType::args_tuple;
            using Ret   = typename WrapFnType::result_type;

            // Проверка количества аргументов
            if (sizeof...(args) != std::tuple_size_v<Tuple>) {
                throw std::invalid_argument("ExecutorUnified: количество аргументов не совпадает");
            }

            // Построение кортежа
            auto tuple_args = build_tuple_safe<Tuple>(std::forward<Args>(args)...);

            if constexpr (std::is_same_v<Ret, void>) {
                std::apply(wrapped.fn, tuple_args);
                return std::monostate{};
            } else {
                return std::apply(wrapped.fn, tuple_args);
            }
        }

        // -------------------------
        // Построение кортежа
        // -------------------------
        template<typename Tuple, std::size_t... I, typename... Args>
        static Tuple build_tuple_impl(std::index_sequence<I...>, Args&&... args) {
            return Tuple{ safe_get<std::tuple_element_t<I, Tuple>>(std::forward<Args>(args))... };
        }

        template<typename Tuple, typename... Args>
        static Tuple build_tuple_safe(Args&&... args) {
            return build_tuple_impl<Tuple>(
                std::make_index_sequence<std::tuple_size_v<Tuple>>{},
                std::forward<Args>(args)...
            );
        }

        // -------------------------
        // Безопасное извлечение из ArgVariant
        // -------------------------
        template<typename T, typename Arg>
        static T safe_get(Arg&& arg) {
            if constexpr (std::is_same_v<T, std::monostate>) {
                return {};
            } else if constexpr (std::is_reference_v<T>) {
                using Base = std::remove_reference_t<T>;
                if (auto p = std::get_if<Base>(&arg)) return *p;
                else throw std::bad_variant_access();
            } else {
                if (auto p = std::get_if<T>(&arg)) return *p;
                else throw std::bad_variant_access();
            }
        }

    // -------------------------
    // Проверка, является ли тип WrapFn
    // -------------------------
    template<typename T>
    struct is_wrapfn : std::false_type {};

    template<typename R, typename Tuple>
    struct is_wrapfn<WrapFn<R, Tuple>> : std::true_type {};

    template<typename T>
    inline static constexpr bool is_wrapfn_v = is_wrapfn<T>::value;
};

} // namespace pattern

