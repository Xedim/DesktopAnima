// PatternExecutor.h
#pragma once
#include "math/common/Types.h"
#include "PatternIdentity.h"
#include <array>
#include <variant>
#include <tuple>
#include <type_traits>
#include <stdexcept>

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

            return std::visit([&]<typename Wrapped>(Wrapped&& wrapped) -> Result {
                using FnType = std::decay_t<decltype(wrapped)>;
                using Tuple   = typename FnType::args_tuple;
                using Ret     = typename FnType::result_type;

                // Проверяем количество аргументов
                constexpr size_t N = std::tuple_size_v<Tuple>;
                if constexpr (sizeof...(args) != N) {
                    throw std::invalid_argument("ExecutorUnified: аргументы не совпадают по размеру");
                }

                // Преобразуем ArgVariant в кортеж
                auto tuple_args = build_tuple_safe<Tuple>(std::forward<Args>(args)...);

                // Вызов функции
                if constexpr (std::is_same_v<Ret, void>) {
                    std::apply(wrapped.fn, tuple_args);
                    return std::monostate{};
                } else {
                    return std::apply(wrapped.fn, tuple_args);
                }
            }, any_f);
        }

    private:
        // -------------------------
        // Построение кортежа с безопасным извлечением
        // -------------------------
        template<typename Tuple, typename... Args, std::size_t... I>
        static Tuple build_tuple_impl(std::index_sequence<I...>, Args&&... args) {
            return Tuple{ safe_get<std::tuple_element_t<I, Tuple>>(std::forward<Args>(args))... };
        }

        template<typename Tuple, typename... Args>
        static Tuple build_tuple_safe(Args&&... args) {
            return build_tuple_impl<Tuple>(
                std::make_index_sequence<sizeof...(Args)>{},
                std::forward<Args>(args)...
            );
        }

        // -------------------------
        // Безопасное извлечение значения из ArgVariant
        // -------------------------
        template<typename T>
        static T safe_get(const ArgVariant& arg) {
            if constexpr (std::is_same_v<T, std::monostate>) {
                return {};
            } else {
                if (auto p = std::get_if<T>(&arg)) {
                    return *p;
                } else {
                    throw std::bad_variant_access();
                }
            }
        }
    };

} // namespace pattern

