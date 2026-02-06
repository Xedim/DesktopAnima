//PatternIdentity.h
#pragma once

enum class PatternID {
    #define X(id, fn, name, sig, kind) id,
    #include "PatternList.h"
    #undef X
        Count
};
