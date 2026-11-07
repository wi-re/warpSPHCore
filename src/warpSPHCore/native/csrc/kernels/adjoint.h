#pragma once
#include "kernels/evaluation.h"
namespace sph_native {
template <class T, int D>
SPH_HD void kernel_vjp(const T (&dx)[D], T h, int kernel, T bw, const T (&bg)[D], T (&bx)[D],
                       T &bh) {
    if (h <= T(0))
        return;
    auto e = evaluate_kernel<T, D>(dx, h, kernel);
    for (int d = 0; d < D; ++d) {
        bx[d] += bw * e.gradient[d];
        bh += bg[d] * e.gradient_dh[d];
        for (int c = 0; c < D; ++c)
            bx[d] += e.hessian[d * D + c] * bg[c];
    }
    bh += bw * e.dh;
}
template <class T, int D> SPH_HD void laplacian_vjp(T r, T h, int kernel, T bar, T &br, T &bh) {
    const T eps = sizeof(T) == 4 ? T(1e-7) : T(1e-15);
    if (h <= T(0) || r / h < eps || r / h > T(1))
        return;
    T q = r / h, A = r + eps * h, R = r * r, C = normalization<T, D>(kernel),
      hd = power<T, D>(h, D);
    auto f = radial_shape<T, D>(q, kernel);
    f.d1 = warp_compat::gradient_shape_derivative<T,D>(q,kernel,f.d1);
    T k1 = C * f.d1 / (hd * h), k2 = C * f.d2 / (hd * h * h), k3 = C * f.d3 / (hd * h * h * h);
    T s = R / (A * A), t = -R / (A * A * A) + T(D) / A;
    T ds = T(2) * r / (A * A) - T(2) * R / (A * A * A);
    T dt = -T(2) * r / (A * A * A) + T(3) * R / (A * A * A * A) - T(D) / (A * A);
    br += bar * (ds * k2 + s * k3 + dt * k1 + t * k2);
    bh += bar * (-T(2) * eps * R / (A * A * A) * k2 + s * (-T(D + 2) * k2 / h - q * k3) +
                 (T(3) * eps * R / (A * A * A * A) - T(D) * eps / (A * A)) * k1 +
                 t * (-T(D + 1) * k1 / h - q * k2));
}
// Naive Laplacian's taped reverse is distinct from its explicit analytic JVP.
// Preserve adj_div's seed * quotient / denominator order, including weighted
// seeds near the singular overflow threshold. Keep quotient/power pullbacks
// before their reductions: refactoring these
// products suppresses observable overflow in the singular viscosity kernel.
template<class T,int D> SPH_HD void laplacian_pullback(const T (&dx)[D],T r,T h,
        int kernel,T bar,T (&bx)[D],T& bh) {
    const T eps=sizeof(T)==4?T(1e-7):T(1e-15),q=r/h;
    if(h<=T(0)||q<eps||q>T(1))return;
    const auto f=radial_shape<T,D>(q,kernel);
    const T d1=warp_compat::gradient_shape_derivative<T,D>(q,kernel,f.d1),C=normalization<T,D>(kernel);
    const T h1=power<T,D>(h,D+1),h2=power<T,D>(h,D+2),k1=d1*C/h1,k2=f.d2*C/h2;
    T R=T(0);for(int d=0;d<D;++d)R+=dx[d]*dx[d];
    const T A=r+eps*h,A2=A*A,A3=A2*A,s=R/A2,t=-R/A3+T(D)/A;
    const T bs=k2*bar,bt=k1*bar,bk1=t*bar,bk2=s*bar;
    const T bR=bs/A2-bt/A3;
    T bA=-bt*(T(D)/A)/A;
    const T bA3=bt*(R/A3)/A3;
    bA+=A2*bA3+A*(A*bA3)+A*(A*bA3);
    const T bA2=-bs*(R/A2)/A2;
    bA+=A*bA2+A*bA2;
    const T b1=(bk1/h1)*C,b2=(bk2/h2)*C;
    T bq=b1*f.d2+b2*f.d3;
    if(kernel==40) {
        const T q2=q*q,q3=q2*q;
        const T bq2=b1*(T(.5)/q2)/q2,bq3=-b2*(T(1)/q3)/q3;
        bq=(-T(3)*q+T(2))*b1-T(3)*b2;
        bq+=q*bq2+q*bq2;
        bq+=q2*bq3+q*(q*bq3)+q*(q*bq3);
    }
    bh+=-bq*(r/h)/h+eps*bA;
    bh+=(-bk1*k1/h1)*T(D+1)*power<T,D>(h,D);
    bh+=(-bk2*k2/h2)*T(D+2)*power<T,D>(h,D+1);
    const T den=r>eps?r:eps,br=bA+bq/h;
    for(int d=0;d<D;++d)bx[d]+=T(2)*dx[d]*bR+(dx[d]/den)*br;
}
} // namespace sph_native
