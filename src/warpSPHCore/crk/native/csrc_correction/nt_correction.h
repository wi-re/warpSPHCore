#pragma once
#include "config.h"
#include <cstdint>

namespace sph_native::crk {
template <class T> struct Fields {
    const T *A = nullptr, *B = nullptr, *gradA = nullptr, *gradB = nullptr;
};

template <class T, int D, bool Gradient> struct Coefficients;
template <class T, int D> struct Coefficients<T, D, false> {
    T A = T(1), B[D] = {};
};
template <class T, int D> struct Coefficients<T, D, true> : Coefficients<T, D, false> {
    T gradA[D] = {}, gradB[D * D] = {};
};

// Load query coefficients once, independently of CSR/cell traversal and of
// how a source record is represented. A null A disables the correction.
template <class T, int D, bool Gradient>
SPH_HD Coefficients<T, D, Gradient> prepare(Fields<T> fields, int64_t i) {
    Coefficients<T, D, Gradient> result{};
    if (fields.A) {
        result.A = fields.A[i];
        for (int d = 0; d < D; ++d) result.B[d] = fields.B[i * D + d];
        if constexpr (Gradient) {
            for (int d = 0; d < D; ++d) result.gradA[d] = fields.gradA[i * D + d];
            for (int c = 0; c < D * D; ++c) result.gradB[c] = fields.gradB[i * D * D + c];
        }
    }
    return result;
}

template <class T, int D, bool Gradient>
SPH_HD T polynomial(const Coefficients<T, D, Gradient>& q, const T (&dx)[D]) {
    T result = T(0);
    for (int d = 0; d < D; ++d) result += q.B[d] * dx[d];
    return T(1) + result;
}

template <class T, int D>
SPH_HD T value(const Coefficients<T, D, false>& q, const T (&dx)[D], T w) {
    return q.A * polynomial(q, dx) * w;
}

template <class T, int D>
SPH_HD void gradient(const Coefficients<T, D, true>& q, const T (&dx)[D],
                     T w, T radial_gradient, T (&result)[D]) {
    const T p = polynomial(q, dx), Aw = q.A * w;
    for (int d = 0; d < D; ++d) {
        T product = T(0);
        // gradB[derivative, component], deliberately not its transpose.
        for (int c = 0; c < D; ++c) product += q.gradB[d * D + c] * dx[c];
        result[d] = Aw * q.B[d] + (q.A * p) * (radial_gradient * dx[d]) +
                    (p * w) * q.gradA[d] + Aw * product;
    }
}
} // namespace sph_native::crk
