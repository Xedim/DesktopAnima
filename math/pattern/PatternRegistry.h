//PatternRegistry.h
#pragma once
#include "PatternIdentity.h"
#include "../common/Types.h"
#include <array>
#include <limits>
#include <string_view>

struct Interval {
    Real min;
    Real max;
};

struct PatternDescriptor {
    std::string_view name;
    Interval domain;
    Interval range;
};

namespace pattern1D {

    constexpr double INF = std::numeric_limits<double>::infinity();

    // Здесь регистрируем только те функции, для которых известны domain/range
    constexpr std::array<PatternDescriptor,
        static_cast<size_t>(PatternID::_Count)> registry = {{
            { "sin",            { -INF, INF }, { -1.0, 1.0 } },
            { "cos",            { -INF, INF }, { -1.0, 1.0 } },
            { "log",            { 0.0, INF },  { -INF, INF } },
            { "exp",            { -INF, INF }, { 0.0, INF } },
            { "dirac_delta",    { -INF, INF }, { 0.0, INF } },
            // Остальные функции по необходимости можно оставить пустыми или с INF
            { "", { -INF, INF }, { -INF, INF } } // заглушка
        }};

    [[nodiscard]] constexpr const PatternDescriptor& descriptor(PatternID id) {
        return registry[static_cast<size_t>(id)];
    }

    [[nodiscard]] constexpr Interval domain(PatternID id) {
        return descriptor(id).domain;
    }

    [[nodiscard]] constexpr Interval range(PatternID id) {
        return descriptor(id).range;
    }

} // namespace pattern1D
