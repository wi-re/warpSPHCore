#pragma once
#include "kernels/warp_derivatives.h"
namespace sph_native {
template<class T,int D> SPH_HD void covariance_kernel_gradient(const T (&dx)[D],T r,
        T hi,T hj,int kernel,int support,T (&gradient)[D]) {
    auto evaluate=[&](T h,T (&g)[D]) {
        for(int d=0;d<D;++d)g[d]=T(0);
        T w=0,wx[D]={},wh=0,gx[D*D]={},gh[D]={};
        warp_compat::force_derivatives<T,D>(dx,r,h,kernel,w,g,wx,wh,gx,gh,false);
    };
    if(support==14||support==15) {
        T left[D],right[D];evaluate(hi,left);evaluate(hj,right);
        for(int d=0;d<D;++d)gradient[d]=(left[d]+right[d])/T(2);
    } else evaluate(pairwise_support(hi,hj,support),gradient);
}
}
