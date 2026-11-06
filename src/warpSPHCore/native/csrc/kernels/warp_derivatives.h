#pragma once
#include "kernels/kernel.h"

namespace sph_native::warp_compat {
// Kernel values are differentiated by Warp through their clamped polynomial
// factors; kernel gradients instead have an explicit analytic custom adjoint.
// At a nonzero q smaller than half an ulp, 1-q rounds to the upper clamp
// boundary. The clamp's right operand owns that tie, so its input derivative
// stops. Keep this distinct from the smooth analytic radial derivative.
template<class T,int D>SPH_HD T value_shape_derivative(T q,int kernel,T analytic) {
    if(T(1)-q!=T(1))return analytic;
    switch(kernel) {
    case 0:return T(D==1?3:4);
    case 1:return D==1?T(5)+T(16)*q:T(6)+(T(70)/T(3))*q;
    case 2:return D==1?T(7)+T(38)*q+T(63)*q*q:T(8)+T(50)*q+T(96)*q*q;
    case 21:return T(0);
    case 30:{const T z=T(.5)-q;return T(12)*z*z;}
    case 31:{const T a=T(.6)-q,b=T(.2)-q;return T(20)*a*a*a-T(40)*b*b*b;}
    case 32:{const T a=T(2)/T(3)-q,b=T(1)/T(3)-q;return T(30)*a*a*a*a-T(75)*b*b*b*b;}
    default:return analytic;
    }
}

// The reference evaluates these spline first derivatives as differences of
// powers. Their cancellation is observable in the custom Hessian near zero;
// the core's smooth-origin expansion is a different, analytic convention.
template<class T>SPH_HD T clamped_power(T x,int n) {
    x=x<T(0)?T(0):x>T(1)?T(1):x;T result=T(1);
    for(int i=0;i<n;++i)result*=x;return result;
}
template<class T>SPH_HD T reference_weighted_sum(T ca,T a,T cb,T b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return std::fma(cb,b,ca*a);
#else
    // Warp's CPU kernel compiler rounds these scalar polynomial terms before
    // adding. Contracting them on a CPU with FMA changes its near-zero Hessian.
    volatile T left=ca*a,right=cb*b;return left+right;
#endif
}
template<class T,int D>SPH_HD T gradient_shape_derivative(T q,int kernel,T analytic) {
    if(q>=(sizeof(T)==4?T(1e-5):T(1e-9)))return analytic;
    switch(kernel) {
    case 30:return reference_weighted_sum(T(-3),clamped_power(T(1)-q,2),T(12),clamped_power(T(.5)-q,2));
    case 31:{const T first=reference_weighted_sum(T(-4),clamped_power(T(1)-q,3),T(20),clamped_power(T(.6)-q,3));
        return reference_weighted_sum(T(1),first,T(-40),clamped_power(T(.2)-q,3));}
    case 32:{const T first=reference_weighted_sum(T(-5),clamped_power(T(1)-q,4),T(30),clamped_power(T(2)/T(3)-q,4));
        return reference_weighted_sum(T(1),first,T(-75),clamped_power(T(1)/T(3)-q,4));}
    case 33:{const T first=reference_weighted_sum(T(56),clamped_power(T(.25)-q,6),T(-28),clamped_power(T(.5)-q,6));
        const T second=reference_weighted_sum(T(1),first,T(8),clamped_power(T(.75)-q,6));
        return (second-clamped_power(T(1)-q,6))*T(7);}
    default:return analytic;
    }
}
// Use the reference's custom gradient adjoint, including its near-origin
// Hessian regularization, while sharing core radial values and derivatives.
template<class T,int D>
SPH_HD void kernel_derivatives(const T (&dx)[D],T r,T h,int kernel,
    T& w,T (&g)[D],T (&wx)[D],T& wh,T (&gx)[D*D],T (&gh)[D]) {
    if(h<=T(0)||r>h)return;
    const auto f=radial_shape<T,D>(r/h,kernel);
    const T C=normalization<T,D>(kernel),hd=power<T,D>(h,D);
    const T kd1=gradient_shape_derivative<T,D>(r/h,kernel,f.d1);
    const T k1=C*kd1/(hd*h),k2=C*f.d2/(hd*h*h);
    const T eps=sizeof(T)==4?T(1e-7):T(1e-15),den=r>eps?r:eps;
    const T value_d1=value_shape_derivative<T,D>(r/h,kernel,f.d1);
    w=C*f.v/hd;wh=-C*(T(D)*f.v+(r/h)*value_d1)/(hd*h);
    for(int a=0;a<D;++a) {
        g[a]=dx[a]/den*C*f.d1/(hd*h);wx[a]=dx[a]/den*C*value_d1/(hd*h);
        gh[a]=-dx[a]/den*C*((r/h)*f.d2+T(D+1)*kd1)/(hd*h*h);
        for(int b=0;b<D;++b) {
            T A=dx[a]*dx[b]/(r*r+eps*eps*h*h);
            if(r/h<eps&&a==b)A=T(1);
            const T B=-dx[a]*dx[b]/(r*r*r+eps*eps*eps*h*h*h)
                       +(a==b?T(1)/(r+eps*eps*h):T(0));
            gx[a*D+b]=A*k2+B*k1;
        }
    }
}
}
