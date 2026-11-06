#pragma once
#include "kernels/warp_derivatives.h"
namespace sph_native::warp_compat {
template<class T> SPH_HD T support_sum(T a,T b,T c,T d) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr(sizeof(T)==4)return __fmaf_rn(a,b,__fmul_rn(c,d));
    else return __fma_rn(a,b,__dmul_rn(c,d));
#else
    return warp_compat::reference_weighted_sum(a,b,c,d);
#endif
}
template<class T> SPH_HD void power_pullback(T x,int n,T bar,T& bx) {
    T p[6];p[0]=x;for(int k=1;k<n;++k)p[k]=p[k-1]*x;
    for(int k=n-1;k>0;--k){bx+=warp_compat::rounded_scalar_term(p[k-1]*bar);bar=x*bar;}
    bx+=bar;
}
template<class T,int D> SPH_HD T support_value(T r,T h,int kernel) {
    if(h<=T(0)||r>h)return T(0);
    const T q=r/h;const auto f=radial_shape_raw<T,D>(q,kernel);
    const T dk=warp_compat::gradient_shape_derivative<T,D>(q,kernel,f.d1);
    return (-normalization<T,D>(kernel)/power<T,D>(h,D+2))*support_sum(T(D)*h,f.v,r,dk);
}
template<class T,int D> SPH_HD void support_pullback(const T (&dx)[D],T r,T h,int kernel,T bar,T (&bx)[D],T& bh) {
    if(h<=T(0)||r>h)return;
    const T q=r/h,C=normalization<T,D>(kernel),hp=power<T,D>(h,D+2);
    const auto f=radial_shape_raw<T,D>(q,kernel);
    const T dk=warp_compat::gradient_shape_derivative<T,D>(q,kernel,f.d1);
    const T kq=warp_compat::value_shape_derivative<T,D>(q,kernel,f.d1);
    T dkq=f.d2;
    if(T(1)-q==T(1)) {
        if(kernel==21)dkq=T(0);
        else if(kernel==30)dkq-=T(6);
        else if(kernel==31)dkq-=T(12);
        else if(kernel==32)dkq-=T(20);
    }
    const T norm=-C/hp,u=support_sum(T(D)*h,f.v,r,dk);
    const T bu=norm*bar,bnorm=u*bar,bk=(T(D)*h)*bu,bdk=r*bu;
    T br=dk*bu,bq=0;
    bh+=warp_compat::rounded_scalar_term(T(D)*(f.v*bu));
    power_pullback(h,D+2,-bnorm*norm/hp,bh);
    // Keep the singular rational kernel's taped scalar contributions separate.
    // Combining its analytic second derivative loses cancellation residuals.
    if(kernel==40) {
        bq+=warp_compat::rounded_scalar_term(T(2)*bdk);
        power_pullback(q,2,bdk*(T(.5)/(q*q))/(q*q),bq);
        power_pullback(q,2,T(-1.5)*bdk,bq);
        bq-=warp_compat::rounded_scalar_term(bk*(T(.5)/q)/q);
        power_pullback(q,2,bk,bq);
        power_pullback(q,3,T(-.5)*bk,bq);
    } else {
        bq+=warp_compat::rounded_scalar_term(bdk*dkq);
        bq+=warp_compat::rounded_scalar_term(bk*kq);
    }
    br+=bq/h;bh-=warp_compat::rounded_scalar_term(bq*q/h);
    const T eps=sizeof(T)==4?T(1e-7):T(1e-15),den=r>eps?r:eps;
    for(int c=0;c<D;++c)bx[c]+=dx[c]/den*br;
}
// Taped value pullback, separate from the analytic gradient's custom adjoint.
template<class T,int D> SPH_HD void value_shape_pullback(T q,int kernel,T bar,T& bq) {
    if(kernel==40){
        bq-=rounded_scalar_term(bar*(T(.5)/q)/q);
        power_pullback(q,2,bar,bq);power_pullback(q,3,T(-.5)*bar,bq);
    }else{
        const auto f=radial_shape_raw<T,D>(q,kernel);
        const T d1=kernel==42 && q==T(1)?T(NAN):value_shape_derivative<T,D>(q,kernel,f.d1);
        bq+=rounded_scalar_term(bar*d1);
    }
}
template<class T,int D> SPH_HD void value_pullback(const T (&dx)[D],T r,T h,int kernel,T bar,T (&bx)[D],T& bh){
    if(h<=T(0)||r>h)return;
    const T q=r/h,hd=power<T,D>(h,D),C=normalization<T,D>(kernel);
    const auto f=radial_shape_raw<T,D>(q,kernel);const T w=f.v*C/hd;
    power_pullback(h,D,-bar*w/hd,bh);
    T bq=0;value_shape_pullback<T,D>(q,kernel,C*(bar/hd),bq);
    bh-=rounded_scalar_term(bq*q/h);
    const T eps=sizeof(T)==4?T(1e-7):T(1e-15),den=r>eps?r:eps;
    for(int c=0;c<D;++c)bx[c]+=dx[c]/den*(bq/h);
}
}
