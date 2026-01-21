//Pattern1D.cpp

#include "Pattern1D.h"
#include "PatternRegistry.h"
#include <cmath>
#include <vector>

namespace pattern1D {

    enum class PatternID {
        Sin,
        Log,
        Dirac,
        _Count
    };

    [[nodiscard]] constexpr const PatternDescriptor&
    descriptor(PatternID id) {
        return registry[static_cast<size_t>(id)];
    }

    [[nodiscard]] constexpr Interval
    domain(PatternID id) {
        return descriptor(id).domain;
    }

    [[nodiscard]] constexpr Interval
    range(PatternID id) {
        return descriptor(id).range;
    }

    [[nodiscard]] constexpr PatternKind
    kind(PatternID id) {
        return descriptor(id).kind;
    }

} // namespace pattern1D
