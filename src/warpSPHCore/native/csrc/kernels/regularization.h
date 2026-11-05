#pragma once
#include "config.h"
#include "dataTypes/state.h"
#include "math/distance.h"
namespace sph_native::warp_compat {
// Warp's vectorNormalize primal uses x/max(|x|, epsilon), with an absolute
// distance floor. Frontend legacy methods request it explicitly; the native
// analytic kernel/AD primitive remains unchanged.
template<class T> SPH_HD T gradient_scale(T radius) {
    const T epsilon=sizeof(T)==4?T(1e-7):T(1e-15);
    return radius>=epsilon?T(1):radius/epsilon;
}
template<class T,int D> SPH_HD T minimum_image_displacement(const T* q,const T* r,const T* lo,const T* hi,
        const bool* periodic,T (&dx)[D]) {
    // Frontend pairs share the core's cancellation-safe periodic geometry.
    // The namespace also retains the explicit legacy gradient-floor policy.
    return sph_native::minimum_image_displacement<T,D>(q,r,lo,hi,periodic,dx);
}
template<class T,int D> SPH_HD T displacement(const RowArgs<T,D>& a,int64_t i,int64_t j,T (&dx)[D]) {
    return minimum_image_displacement<T,D>(a.qx+i*D,a.rx+j*D,a.lo,a.hi,a.periodic,dx);
}
}
