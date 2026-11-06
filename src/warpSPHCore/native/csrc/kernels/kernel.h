#pragma once
#include "dataTypes/state.h"
#include "kernels/radial.h"
#include "kernels/gradient_shape.h"
#include "math/pow.h"
namespace sph_native {
SPH_HD constexpr bool force_operation(int operation) {
    return (operation>=2 && operation<=5) || operation==7;
}
template <class T, int D>
SPH_HD void kernel_pair_value(T r, T h, int kernel, T &value, T &gradient) {
    value = gradient = T(0);
    if (h <= T(0) || r > h)
        return;
    if (kernel == 0) {
        Wendland2Primal<T, D>(h).evaluate(r, value, gradient);
        return;
    }
    auto f = radial_shape<T, D>(r / h, kernel);
    T C = normalization<T, D>(kernel), hd = power<T, D>(h, D);
    value = C * f.v / hd;
    gradient = r > T(0) ? C * f.d1 / (hd * h * r) : C * f.d2 / (hd * h * h);
}
template <typename T, int D>
SPH_HD void pair_kernel(const RowArgs<T, D> &a, int64_t i, int64_t j, T r, T &value, T &gradient) {
    if (a.support == 14 || a.support == 15) {
        T vj, gj;
        kernel_pair_value<T, D>(r, a.qh[i], a.kernel, value, gradient);
        kernel_pair_value<T, D>(r, a.rh[j], a.kernel, vj, gj);
        value = (value + vj) / T(2);
        gradient = (gradient + gj) / T(2);
    } else
        kernel_pair_value<T, D>(r, a.radius(i, j), a.kernel, value, gradient);
    if(force_operation(a.operation)) {
        auto radial=[&](T h) {
            if(h<=T(0)||r>h)return T(0);
            const T eps=sizeof(T)==4?T(1e-7):T(1e-15);
            const auto f=radial_shape<T,D>(r/h,a.kernel);
            return normalization<T,D>(a.kernel)*warp_compat::gradient_shape_derivative<T,D>(r/h,a.kernel,f.d1)/
                (power<T,D>(h,D+1)*(r>eps?r:eps));
        };
        gradient=(a.support==14||a.support==15)?(radial(a.qh[i])+radial(a.rh[j]))/T(2):radial(a.radius(i,j));
    }
}
} // namespace sph_native
