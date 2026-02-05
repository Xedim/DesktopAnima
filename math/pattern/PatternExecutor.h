// PatternExecutor.h
#pragma once
#include "../common/Types.h"
#include "PatternIdentity.h"
#include <array>
#include <variant>
#include <stdexcept>

namespace pattern {

    // Унифицированный тип функции для движка
    using EngineFn = std::function<ResultVariant(const std::vector<ArgVariant>&)>;

    struct ExecutorUnified {
        std::array<EngineFn, static_cast<size_t>(PatternID::Count)> registry{};

        void register_fn(PatternID id, EngineFn fn) {
            registry[static_cast<size_t>(id)] = std::move(fn);
        }

        [[nodiscard]] ResultVariant call(PatternID id, const std::vector<ArgVariant>& args) const {
            const auto& fn = registry.at(static_cast<size_t>(id));
            if (!fn) throw std::runtime_error("Function not registered");
            return fn(args);
        }
    };

} // namespace pattern