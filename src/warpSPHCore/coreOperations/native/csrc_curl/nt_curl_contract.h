#pragma once
#include "kernels/gradient_shape.h"
namespace sph_native {
// Warp reverses the z/y/x assignments in the 3D cross. Preserve that product
// order: a rounded cancellation residual is amplified by cusp Hessians.
template<class T> SPH_HD T curl_sum(T ca,T a,T cb,T b) {
    // The reference adjoint stores the vector component contributions before
    // reducing them, including on devices. Do not fuse these two products.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr(sizeof(T)==4)return __fadd_rn(__fmul_rn(ca,a),__fmul_rn(cb,b));
    else return __dadd_rn(__dmul_rn(ca,a),__dmul_rn(cb,b));
#else
    volatile T left=ca*a,right=cb*b;return left+right;
#endif
}
template<class T> SPH_HD void curl_pair_bars(const T* u,const T* f,const T* g,T* bf,T* bg) {
    bf[0]=curl_sum(-u[2],g[1],u[1],g[2]);
    bf[1]=curl_sum(u[2],g[0],-u[0],g[2]);
    bf[2]=curl_sum(-u[1],g[0],u[0],g[1]);
    bg[0]=curl_sum(u[2],f[1],-u[1],f[2]);
    bg[1]=curl_sum(-u[2],f[0],u[0],f[2]);
    bg[2]=curl_sum(u[1],f[0],-u[0],f[1]);
}
} // namespace sph_native
