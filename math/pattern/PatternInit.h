#pragma once
#include "PatternExecutor.h"
#include "Functions.h"
#include <tuple>
#include <stdexcept>
#include <type_traits>

namespace pattern {

// -------------------------
// Вспомогательная функция safe_get_arg
// -------------------------
template<typename T>
T safe_get_arg(const ArgVariant& arg) {
    using Base = std::remove_cv_t<std::remove_reference_t<T>>;

    return std::visit([](auto&& v) -> T {
        using V = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<Base, std::monostate>) {
            return {};
        } else if constexpr (std::is_same_v<Base, V>) {
            if constexpr (std::is_reference_v<T>) return v;
            else return v;
        } else if constexpr (std::is_integral_v<V> && std::is_integral_v<Base>) {
            return static_cast<Base>(v);
        } else {
            throw std::invalid_argument("Argument type mismatch in safe_get_arg");
        }
    }, arg);
}

// -------------------------
// Преобразование std::vector<ArgVariant> в кортеж
// -------------------------
template<typename... Args, std::size_t... I>
std::tuple<Args...> build_tuple_from_args(const std::vector<ArgVariant>& args, std::index_sequence<I...>) {
    return std::tuple<Args...>{ safe_get_arg<Args>(args[I])... };
}

template<typename... Args>
std::tuple<Args...> build_tuple_from_args(const std::vector<ArgVariant>& args) {
    if (args.size() != sizeof...(Args))
        throw std::invalid_argument("Incorrect number of arguments");
    return build_tuple_from_args<Args...>(args, std::index_sequence_for<Args...>{});
}

// -------------------------
// make_fn: адаптация обычной функции к EngineFn
// -------------------------
template<typename R, typename... Args>
EngineFn make_fn(R(*f)(Args...)) {
    return [f](const std::vector<ArgVariant>& args) -> ResultVariant {
        auto tuple_args = build_tuple_from_args<Args...>(args);

        if constexpr (std::is_same_v<R, void>) {
            std::apply(f, tuple_args);
            return std::monostate{}; // void -> monostate
        } else {
            return std::apply(f, tuple_args);
        }
    };
}

// -------------------------
// bind_all: регистрация всех функций из PatternList
// -------------------------
inline void bind_all(ExecutorUnified& exec) {
#define X(id, fn, sig, name, domain, range, kind) \
    exec.register_fn(PatternID::id, make_fn(static_cast<sig*>(&fn)));

#include "PatternList.h"
#undef X
}

} // namespace pattern
