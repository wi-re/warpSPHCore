#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> wendland4_shape(RadialJet<T> q) {
    if (q.v > T(1)) return {};
    // Exact derivatives in factored form. Differentiating the two O(1)
    // factors separately subtracts nearly equal terms in f', and costs a full
    // jet product for every power. These are the same spline polynomials.
    const T x=q.v,z=T(1)-x,z2=z*z,z3=z2*z,z4=z3*z,z5=z4*z,z6=z5*z;
    T value,first,second,third;
    if constexpr (D==1) {
        value=z5*(T(1)+T(5)*x+T(8)*(x*x));
        first=T(-14)*x*(T(4)*x+T(1))*z4;
        second=T(14)*(T(24)*(x*x)-T(3)*x-T(1))*z3;
        third=T(-840)*x*(T(2)*x-T(1))*z2;
    } else {
        value=z6*(T(1)+T(6)*x+(T(35)/T(3))*(x*x));
        first=(T(-56)/T(3))*x*(T(5)*x+T(1))*z5;
        second=(T(56)/T(3))*(T(35)*(x*x)-T(4)*x-T(1))*z4;
        third=T(-560)*x*(T(7)*x-T(3))*z3;
    }
    return {value,first*q.d1,second*q.d1*q.d1+first*q.d2,
            third*q.d1*q.d1*q.d1+T(3)*second*q.d1*q.d2+first*q.d3};
}
} // namespace sph_native
