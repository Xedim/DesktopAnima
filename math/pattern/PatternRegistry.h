#pragma once
#include "PatternIdentity.h"
#include "../common/Types.h"
#include <array>
#include <limits>

namespace pattern1D {

    constexpr double INF = std::numeric_limits<double>::infinity();

    constexpr std::array<PatternDescriptor,
        static_cast<size_t>(PatternIdentity::_Count)> registry = {{
            {
                "sin",
                { -INF, INF },
                { -1.0, 1.0 },
                PatternKind::Trigonometric
            },
            {
                "log",
                { 0.0, INF },
                { -INF, INF },
                PatternKind::Logarithmic
            },
            {
                "dirac_delta",
                { -INF, INF },
                { 0.0, INF },
                PatternKind::Generalized
            }
        }};

}
