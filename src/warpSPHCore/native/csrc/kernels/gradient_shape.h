#pragma once
#include "kernels/radial.h"
namespace sph_native::warp_compat {
// Warp CPU vector adjoints round each scalar contribution before reducing.
// Device compilers retain their ordinary fused accumulation policy.
template<class T>SPH_HD T rounded_scalar_term(T value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return value;
#else
    volatile T rounded=value;return rounded;
#endif
}
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
// Warp contracts the first positive product in B8's subtraction. Reversing
// the FMA operands erases its observable FP64 near-origin Hessian residual.
template<class T>SPH_HD T reference_difference(T ca,T a,T cb,T b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return std::fma(ca,a,-cb*b);
#else
    volatile T left=ca*a,right=cb*b;return left-right;
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
    case 33:{const T first=reference_difference(T(56),clamped_power(T(.25)-q,6),T(28),clamped_power(T(.5)-q,6));
        const T second=reference_weighted_sum(T(1),first,T(8),clamped_power(T(.75)-q,6));
        return (second-clamped_power(T(1)-q,6))*T(7);}
    default:return analytic;
    }
}
}
