#pragma once
#include "kernels/kernel.h"

namespace sph_native {
// Interpolation and density consume only the radial value. Keep derivative
// evaluation available separately for analytic adjoints and gradient operators.
template <class T, int D, bool Wendland2 = false> SPH_HD T kernel_value(T r, T h, int kernel) {
    if (h <= T(0) || r > h)
        return T(0);
    const T q = r / h;
    T value;
    if constexpr (Wendland2) {
        const T t = T(1) - q;
        T power = t * t * t;
        if constexpr (D != 1)
            power *= t;
        value = power * (T(1) + T(D == 1 ? 3 : 4) * q);
    } else {
        value = radial_shape_raw<T, D>(q, kernel).v;
    }
    return normalization<T, D>(Wendland2 ? 0 : kernel) * value / power<T, D>(h, D);
}

template <class T, int D, bool WendlandGather = false>
SPH_HD T pair_kernel_value(const RowArgs<T, D> &a, int64_t i, int64_t j, T r) {
    if constexpr (WendlandGather) {
        return kernel_value<T, D, true>(r, a.qh[i], 0);
    } else {
        if (a.support == 14 || a.support == 15)
            return (kernel_value<T, D>(r, a.qh[i], a.kernel) +
                    kernel_value<T, D>(r, a.rh[j], a.kernel)) /
                   T(2);
        return kernel_value<T, D>(r, a.radius(i, j), a.kernel);
    }
}
} // namespace sph_native
