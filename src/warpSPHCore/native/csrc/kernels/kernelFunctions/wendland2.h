#pragma once
#include "math/pow.h"

namespace sph_native {
// Wendland C2 value and coefficient multiplying displacement in its gradient.
template <typename T, int D> SPH_HD void wendland(T r, T h, T& value, T& gradient) {
    value = gradient = T(0);
    if (h <= T(0) || r > h) return;
    const T q = r / h, z = T(1) - q;
    const T pi = T(3.141592653589793238462643383279502884);
    const T normal = D == 1 ? T(5) / T(4) : (D == 2 ? T(7) / pi : T(21) / (T(2) * pi));
    if (D == 1) {
        value = normal * power<T, D>(z, 3) * (T(1) + T(3) * q) / power<T, D>(h, D);
        gradient = T(-12) * normal * z * z / power<T, D>(h, D + 2);
    } else {
        value = normal * power<T, D>(z, 4) * (T(1) + T(4) * q) / power<T, D>(h, D);
        gradient = T(-20) * normal * power<T, D>(z, 3) / power<T, D>(h, D + 2);
    }
}

} // namespace sph_native
