//MathEngine.h
#pragma once
#include "../common/Types.h"
#include "PatternIdentity.h"
#include "PatternExecutor.h"
#include "PatternInit.h"

class MathEngine {
public:
    MathEngine() {
        // Инициализация: создаём executor и привязываем функции
        pattern::bind_all(exec_);
    }

    // Вариадический compute для скаляров и векторов
    template<typename... Args>
        ResultVariant compute(size_t id, Args&&... args) {
        auto pid = static_cast<PatternID>(id);

        return exec_.call<ResultVariant>(pid, toArgVariant(std::forward<Args>(args))...);
    }


private:
    pattern::ExecutorUnified exec_;

    // --- helper: конвертация в ArgVariant ---
    static ArgVariant toArgVariant(std::size_t v)                 { return v; }
    static ArgVariant toArgVariant(Real v)                        { return v; }
    static ArgVariant toArgVariant(int v)                         { return v; }
    static ArgVariant toArgVariant(const VecReal& v)             { return v; }
    static ArgVariant toArgVariant(std::vector<Real> v)          { return v; } // initializer_list будет преобразован через vector
    static ArgVariant toArgVariant(const RealPair& p)            { return p; }
    static ArgVariant toArgVariant(const Complex& c)             { return c; }
    static ArgVariant toArgVariant(const StabPolicy& p)          { return p; }
    static ArgVariant toArgVariant(const LinearRegressionResult& r) { return r; }
    static ArgVariant toArgVariant(const Quartiles& q)           { return q; }

    static ArgVariant toArgVariant(std::initializer_list<Real> l) {
        if (l.size() == 2) {
            // Автоматически считаем RealPair
            auto it = l.begin();
            return RealPair{*it, *(it + 1)};
        }
        return VecReal(l);
    }
};