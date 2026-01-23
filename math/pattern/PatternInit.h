//PatternInit.h
#pragma once
#include "PatternExecutor.h"
#include "Functions.h"

namespace pattern {

    inline void bind_all(ExecutorUnified& exec) {
        #define X(id, fn, ...) \
        exec.register_fn(PatternID::id, make_##sig(&Functions::fn));

        #include "PatternList.h"
        #undef X
    }

}
