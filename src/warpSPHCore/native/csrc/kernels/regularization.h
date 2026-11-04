#pragma once
#include "config.h"
#include "dataTypes/state.h"
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
    T squared=T(0);
    for(int c=0;c<D;++c) {
        T x=q[c],y=r[c];
        if(periodic[c]) {
            const T length=hi[c]-lo[c],half=length/T(2);
            x-=length*scalar_floor((x-lo[c])/length);y-=length*scalar_floor((y-lo[c])/length);
            // Preserve the reference's add/wrap/subtract order, including
            // cancellation at sub-ulp periodic displacements and half-box ties.
            const T shifted=(x-y)+half;
            dx[c]=(shifted-scalar_floor(shifted/length)*length)-half;
        }else dx[c]=x-y;
        squared+=dx[c]*dx[c];
    }
    return scalar_sqrt(squared);
}
template<class T,int D> SPH_HD T displacement(const RowArgs<T,D>& a,int64_t i,int64_t j,T (&dx)[D]) {
    return minimum_image_displacement<T,D>(a.qx+i*D,a.rx+j*D,a.lo,a.hi,a.periodic,dx);
}
}
