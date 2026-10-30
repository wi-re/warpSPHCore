#pragma once
#include "dataTypes/state.h"
#include "kernels/kernelFunctions/wendland2.h"

namespace sph_native {
template <typename T, int D>
SPH_HD void pair_kernel(const RowArgs<T, D>& a, int64_t i, int64_t j,
                        T r, T& value, T& gradient) {
    if (a.support == 14 || a.support == 15) {
        T value_j, gradient_j;
        wendland<T, D>(r, a.qh[i], value, gradient);
        wendland<T, D>(r, a.rh[j], value_j, gradient_j);
        value = (value + value_j) / T(2);
        gradient = (gradient + gradient_j) / T(2);
    } else {
        wendland<T, D>(r, a.radius(i, j), value, gradient);
    }
}
} // namespace sph_native
