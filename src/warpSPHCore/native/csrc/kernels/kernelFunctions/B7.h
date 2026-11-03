#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> B7_shape(RadialJet<T> q) {
    // Form the derivative polynomial before multiplying by its factorial.
    // Differentiating each power by repeated product rules separately adds
    // avoidable FP32 cancellation error near the central spline knots.
    const T knots[4]={T(1),T(5)/T(7),T(3)/T(7),T(1)/T(7)};
    const T coefficients[4]={T(1),T(-7),T(21),T(-35)};
    T value=T(0),first=T(0),second=T(0),third=T(0);
    for(int k=0;k<4;++k) {
        const T x=q.v<knots[k]?q.v-knots[k]:T(0);
        T cube=x*x*x,fourth=cube*x,fifth=fourth*x,sixth=fifth*x;
        value+=coefficients[k]*sixth;first+=coefficients[k]*fifth;
        second+=coefficients[k]*fourth;third+=coefficients[k]*cube;
    }
    first*=T(6);second*=T(30);third*=T(120);
    return {value,first*q.d1,second*q.d1*q.d1+first*q.d2,
            third*q.d1*q.d1*q.d1+T(3)*second*q.d1*q.d2+first*q.d3};
}
} // namespace sph_native
