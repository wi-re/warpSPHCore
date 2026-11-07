#pragma once
#include "coreOperations/native/csrc_interpolate/nt_interpolate_jvp.h"
namespace sph_native::polyfit {
// Raw polynomial helpers permit supplied nonpositive supports. Preserve that
// reference extrapolation locally; physical kernel owners retain their policy.
template<class T>SPH_HD RadialJet<T> literal_power(T knot,RadialJet<T> q,int n){
    if(knot-q.v>T(1))return RadialJet<T>(T(1));
    return positive_power(knot,q,n);
}
template<class T,int D>SPH_HD RadialJet<T> literal_shape(T q,int kernel){
    if(!(q<T(0)))return radial_shape_raw<T,D>(q,kernel);
    const RadialJet<T> x(q,T(1)),one(T(1));
    switch(kernel){
    case 0:return literal_power(T(1),x,D==1?3:4)*(one+x*T(D==1?3:4));
    case 1:return literal_power(T(1),x,D==1?5:6)*(one+x*T(D==1?5:6)+jet_power(x,2)*(D==1?T(8):T(35)/T(3)));
    case 2:return literal_power(T(1),x,D==1?7:8)*(one+x*T(D==1?7:8)+jet_power(x,2)*T(D==1?19:25)+jet_power(x,3)*T(D==1?21:32));
    case 21:return literal_power(T(1),x,3);
    case 30:return literal_power(T(1),x,3)-literal_power(T(.5),x,3)*T(4);
    case 31:return literal_power(T(1),x,4)-literal_power(T(.6),x,4)*T(5)+literal_power(T(.2),x,4)*T(10);
    case 32:return literal_power(T(1),x,5)-literal_power(T(2)/T(3),x,5)*T(6)+literal_power(T(1)/T(3),x,5)*T(15);
    default:return radial_shape_raw<T,D>(q,kernel);
    }
}
template<class T,int D>SPH_HD RadialJet<T> literal_slope(T q,int kernel){
    const RadialJet<T> x(q,T(1));
    switch(kernel){
    case 0:return x*literal_power(T(1),x,D==1?2:3)*T(D==1?-12:-20);
    case 1:return x*(x*T(D==1?4:5)+RadialJet<T>(T(1)))*literal_power(T(1),x,D==1?4:5)*(D==1?T(-14):T(-56)/T(3));
    case 2:return x*(jet_power(x,2)*T(D==1?35:16)+x*T(D==1?18:7)+RadialJet<T>(T(D==1?3:1)))*literal_power(T(1),x,D==1?6:7)*T(D==1?-6:-22);
    case 21:return literal_power(T(1),x,2)*T(-3);
    case 30:return literal_power(T(1),x,2)*T(-3)+literal_power(T(.5),x,2)*T(12);
    case 31:return literal_power(T(1),x,3)*T(-4)+literal_power(T(.6),x,3)*T(20)-literal_power(T(.2),x,3)*T(40);
    case 32:return literal_power(T(1),x,4)*T(-5)+literal_power(T(2)/T(3),x,4)*T(30)-literal_power(T(1)/T(3),x,4)*T(75);
    default:{const auto f=radial_shape_raw<T,D>(q,kernel);return {f.d1,f.d2,f.d3};}
    }
}
template<class T,int D>SPH_HD T literal_second(T q,int kernel){
    switch(kernel){
    case 0:return D==1?T(-12)*(T(3)*q*q-T(4)*q+T(1)):T(20)*(T(4)*q-T(1))*warp_compat::clamped_power(T(1)-q,2);
    case 1:return D==1?T(14)*(T(24)*q*q-T(3)*q-T(1))*warp_compat::clamped_power(T(1)-q,3):T(56)/T(3)*(T(35)*q*q-T(4)*q-T(1))*warp_compat::clamped_power(T(1)-q,4);
    case 2:return D==1?T(18)*(T(105)*q*q*q+T(13)*q*q-T(5)*q-T(1))*warp_compat::clamped_power(T(1)-q,5):T(22)*(T(160)*q*q*q+T(15)*q*q-T(6)*q-T(1))*warp_compat::clamped_power(T(1)-q,6);
    case 21:return T(6)*(T(1)-q);
    case 30:return T(6)*(T(1)-q)-T(24)*(T(.5)-q);
    case 31:return T(12)*warp_compat::clamped_power(T(1)-q,2)-T(60)*warp_compat::clamped_power(T(.6)-q,2)+T(120)*warp_compat::clamped_power(T(.2)-q,2);
    case 32:return T(20)*warp_compat::clamped_power(T(1)-q,3)-T(120)*warp_compat::clamped_power(T(2)/T(3)-q,3)+T(300)*warp_compat::clamped_power(T(1)/T(3)-q,3);
    default:return radial_shape_raw<T,D>(q,kernel).d2;
    }
}
template<class T,int D>struct LiteralKernel{
    const T (&x)[D];T r,h;int name;
    SPH_HD T value()const{
        if(h>T(0))return kernel_value<T,D>(r,h,name);
        const T q=r/h;if(q>T(1))return T(0);
        return literal_shape<T,D>(q,name).v*normalization<T,D>(name)/power<T,D>(h,D);
    }
    SPH_HD T tangent(const T(&dt)[D],T dh)const{
        if(h>T(0))return warp_compat::ValueProductKernel<T,D>(x,r,h,name).tangent(dt,dh);
        const T q=r/h;if(q>T(1))return T(0);
        const auto f=literal_shape<T,D>(q,name),s=literal_slope<T,D>(q,name);
        const T C=normalization<T,D>(name),eps=sizeof(T)==4?T(1e-7):T(1e-15),den=r>eps?r:eps;
        T dot=0;for(int c=0;c<D;++c)dot+=x[c]/den*(C*s.v/power<T,D>(h,D+1))*dt[c];
        return dot+(-C/power<T,D>(h,D+2))*(T(D)*h*f.v+r*s.v)*dh;
    }
    SPH_HD void value_reverse(T bw,T(&bx)[D],T& bh)const{
        if(h>T(0)){warp_compat::value_pullback<T,D>(x,r,h,name,bw,bx,bh);return;}
        const T q=r/h;if(q>T(1))return;
        const auto f=literal_shape<T,D>(q,name);const T C=normalization<T,D>(name),hd=power<T,D>(h,D);
        const T bq=C*(bw/hd)*f.d1,eps=sizeof(T)==4?T(1e-7):T(1e-15),den=r>eps?r:eps;
        warp_compat::power_pullback(h,D,-bw*value()/hd,bh);bh-=bq*q/h;
        for(int c=0;c<D;++c)bx[c]+=x[c]/den*(bq/h);
    }
    SPH_HD void backward(const T(&dt)[D],T dh,T bw,T bd,T(&bx)[D],T& bh,T(&bt)[D],T& bth)const{
        if(h>T(0)){warp_compat::ValueProductKernel<T,D>(x,r,h,name).backward(name,dt,dh,bw,bd,bx,bh,bt,bth);return;}
        const T q=r/h;if(q>T(1))return;
        const auto f=literal_shape<T,D>(q,name),s=literal_slope<T,D>(q,name);const T second=literal_second<T,D>(q,name);
        const T C=normalization<T,D>(name),eps=sizeof(T)==4?T(1e-7):T(1e-15),den=r>eps?r:eps;
        const T norm=-C/power<T,D>(h,D+2),u=T(D)*h*f.v+r*s.v,bu=norm*(dh*bd);
        T br=s.v*bu;const T bq=(T(D)*h*f.d1+r*s.d1)*bu;
        br+=bq/h;bh+=T(D)*f.v*bu-bq*q/h;
        warp_compat::power_pullback(h,D+2,-(u*(dh*bd))*norm/power<T,D>(h,D+2),bh);
        for(int a=0;a<D;++a){
            const T g=x[a]/den*(C*s.v/power<T,D>(h,D+1));bt[a]+=g*bd;
            bh+=(-x[a]/den*C/power<T,D>(h,D+2)*(q*second+T(D+1)*s.v))*(dt[a]*bd);
            for(int b=0;b<D;++b){T A=x[a]*x[b]/(r*r+eps*eps*h*h);if(q<eps&&a==b)A=T(1);
                const T B=-x[a]*x[b]/(r*r*r+eps*eps*eps*h*h*h)+(a==b?T(1)/(r+eps*eps*h):T(0));
                bx[b]+=(A*(C*second/power<T,D>(h,D+2))+B*(C*s.v/power<T,D>(h,D+1)))*(dt[a]*bd);}
            bx[a]+=x[a]/den*br;
        }
        bth+=norm*u*bd;value_reverse(bw,bx,bh);
    }
};
} // namespace sph_native::polyfit
