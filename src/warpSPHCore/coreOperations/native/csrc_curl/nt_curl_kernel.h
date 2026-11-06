#pragma once
#include "kernels/kernel.h"
namespace sph_native {
// Curl consumes vector gradient components. Evaluating a radial coefficient
// and multiplying by dx later changes their rounding and singular arithmetic.
// Keep the component order of the reference force-gradient primitive.
template<class T,int D> SPH_HD void curl_kernel_gradient(const RowArgs<T,D>& a,
        int64_t i,int64_t j,const T (&dx)[D],T r,T (&gradient)[D]) {
    auto evaluate=[&](T h,T (&g)[D]) {
        for(int d=0;d<D;++d)g[d]=T(0);
        if(h<=T(0)||r>h)return;
        const T eps=sizeof(T)==4?T(1e-7):T(1e-15),den=r>eps?r:eps;
        const T d1=warp_compat::gradient_shape_derivative<T,D>(r/h,a.kernel,radial_shape<T,D>(r/h,a.kernel).d1);
        for(int d=0;d<D;++d)g[d]=dx[d]/den*normalization<T,D>(a.kernel)*d1/power<T,D>(h,D+1);
    };
    if(a.support==14||a.support==15) {
        T left[D],right[D];evaluate(a.qh[i],left);evaluate(a.rh[j],right);
        for(int d=0;d<D;++d)gradient[d]=(left[d]+right[d])/T(2);
    } else evaluate(a.radius(i,j),gradient);
}
} // namespace sph_native
