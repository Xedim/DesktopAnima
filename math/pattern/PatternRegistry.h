#pragma once
#include "PatternID.h"
#include "PatternTypes.h"
#include <array>

struct PatternDescriptor {
    std::string_view name;
    Interval domain;
    Interval range;
    PatternKind kind;
    PatternSignature signature;
};

namespace pattern {

    constexpr std::array<PatternDescriptor, size_t(PatternID::_Count)> registry = {{
        #define X(id, sig, fn, name, domain, range, kind) \
        PatternDescriptor{ name, domain, range, kind, PatternSignature::sig },
        #include "PatternList.h"
        #undef X
    }};

    constexpr const PatternDescriptor& descriptor(PatternID id) {
        return registry[size_t(id)];
    }

} // namespace pattern