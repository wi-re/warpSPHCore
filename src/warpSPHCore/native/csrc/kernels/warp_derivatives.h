#pragma once
#include "kernels/kernel.h"
#include "kernels/gradient_shape.h"

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
// Value-only products never need a force-kernel Hessian. Keep the taped
// value's clamp derivatives separate from the explicit analytic JVP convention.
template<class T,int D>
SPH_HD void value_derivatives(const T (&dx)[D],T r,T h,int kernel,
    T& w,T (&wx)[D],T& wh,bool reverse=true) {
    if(h<=T(0)||r>h)return;
    const T q=r/h,C=normalization<T,D>(kernel),hd=power<T,D>(h,D);
    const auto f=radial_shape_raw<T,D>(q,kernel);
    // At the adhesion cutoff the taped fractional power distributes infinite
    // bars through opposing polynomial terms: its radial bar is NaN.
    const T d1=reverse && kernel==42 && q==T(1) ? T(NAN)
                      : reverse ? value_shape_derivative<T,D>(q,kernel,f.d1)
                      : gradient_shape_derivative<T,D>(q,kernel,radial_shape<T,D>(q,kernel).d1);
    const T eps=sizeof(T)==4?T(1e-7):T(1e-15),den=r>eps?r:eps;
    w=C*f.v/hd;
    wh=-C*(T(D)*f.v+q*d1)/(hd*h);
    for(int d=0;d<D;++d)wx[d]=dx[d]/den*C*d1/(hd*h);
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
// Force operators use Warp's analytic custom gradient adjoint. The value
// polynomial has a distinct taped reverse, needed only by CRK corrections.
template<class T,int D>
SPH_HD void force_derivatives(const T (&dx)[D],T r,T h,int kernel,
    T& w,T (&g)[D],T (&wx)[D],T& wh,T (&gx)[D*D],T (&gh)[D],bool reverse=true) {
    kernel_derivatives<T,D>(dx,r,h,kernel,w,g,wx,wh,gx,gh);
    value_derivatives<T,D>(dx,r,h,kernel,w,wx,wh,reverse);
    if(h<=T(0)||r>h)return;
    const T eps=sizeof(T)==4?T(1e-7):T(1e-15),den=r>eps?r:eps;
    const T d1=gradient_shape_derivative<T,D>(r/h,kernel,radial_shape<T,D>(r/h,kernel).d1);
    for(int d=0;d<D;++d)g[d]=dx[d]/den*normalization<T,D>(kernel)*d1/power<T,D>(h,D+1);
}
template<class T,int D>
SPH_HD void force_pullback(const T (&dx)[D],T r,T h,int kernel,T bw,const T (&bg)[D],
    T (&bx)[D],T& bh,bool value_used=false) {
    T w=0,g[D]={},wx[D]={},wh=0,gx[D*D]={},gh[D]={};
    force_derivatives<T,D>(dx,r,h,kernel,w,g,wx,wh,gx,gh);
    for(int d=0;d<D;++d) {
        if(value_used)bx[d]+=bw*wx[d];
        bh+=bg[d]*gh[d];
        for(int c=0;c<D;++c)bx[d]+=gx[d*D+c]*bg[c];
    }
    if(value_used)bh+=bw*wh;
}

}
