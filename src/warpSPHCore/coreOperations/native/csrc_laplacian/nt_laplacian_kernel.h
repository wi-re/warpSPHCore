#pragma once
#include "kernels/warp_derivatives.h"
namespace sph_native {
// Laplacian estimators contract the canonical vector force gradient, including
// its component-wise singular arithmetic at a compact-support boundary.
template<class T,int D> SPH_HD void laplacian_kernel_gradient(const RowArgs<T,D>& a,
        int64_t i,int64_t j,const T (&dx)[D],T r,T (&gradient)[D]) {
    auto evaluate=[&](T h,T (&g)[D]) {
        for(int d=0;d<D;++d)g[d]=T(0);
        T w=0,wx[D]={},wh=0,gx[D*D]={},gh[D]={};
        warp_compat::force_derivatives<T,D>(dx,r,h,a.kernel,w,g,wx,wh,gx,gh,false);
    };
    if(a.support==14||a.support==15) {
        T left[D],right[D];evaluate(a.qh[i],left);evaluate(a.rh[j],right);
        for(int d=0;d<D;++d)gradient[d]=(left[d]+right[d])/T(2);
    } else evaluate(a.radius(i,j),gradient);
}
}
