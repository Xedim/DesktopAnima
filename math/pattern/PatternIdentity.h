//PatternIdentity.h
#pragma once

enum class PatternID {
    #define X(id, fn, sig, name, domain, range, kind) id,
    #include "PatternList.h"
    #undef X
        Count
};
