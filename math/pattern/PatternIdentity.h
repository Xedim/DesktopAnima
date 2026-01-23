//PatternIdentity.h
#pragma once

enum class PatternID {
    #define X(id, fn, name, domain, range, kind) id,
    #include "PatternList.h"
    #undef X
        _Count
};
