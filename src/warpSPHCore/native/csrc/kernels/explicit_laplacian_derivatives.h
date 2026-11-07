#pragma once
#include "kernels/explicit_force_derivatives.h"
#include "kernels/adjoint.h"
namespace sph_native::warp_compat {
template<class S>SPH_HD S third_positive(S x,int n){return x.v<=0?S(0):explicit_ad::power(x,n);}
template<class S>SPH_HD S third_negative(S x,int n){return x.v>=0?S(0):explicit_ad::power(x,n);}
template<int D,class S>SPH_HD S third_shape(S q,int name){
    const S q2=q*q;
    switch(name){
    case 0:if constexpr(D==1)return S(48)-S(72)*q;else return S(120)*(S(2)*q2-S(3)*q+S(1));
    case 1:if constexpr(D==1)return S(-840)*q*(S(2)*q-S(1))*third_positive(S(1)-q,2);
           else return S(-560)*q*(S(7)*q-S(3))*third_positive(S(1)-q,3);
    case 2:if constexpr(D==1)return S(-1008)*q*(S(15)*q2-S(4)*q-S(1))*third_positive(S(1)-q,4);
           else return S(-1584)*q*(S(20)*q2-S(5)*q-S(1))*third_positive(S(1)-q,5);
    case 20:return S(-6)*(S(20)*q2*q-S(12)*q);
    case 21:return S(-6);
    case 30:return q.v>=.5?S(-6):S(18);
    case 31:return S(-24)*(S(1)-q)+S(120)*third_positive(S(.6)-q,1)-S(240)*third_positive(S(.2)-q,1);
    case 32:return S(-60)*third_positive(S(1)-q,2)+S(360)*third_positive(S(2./3.)-q,2)-S(900)*third_positive(S(1./3.)-q,2);
    case 34:if(q.v<1./7.)return q*(S(24)*S(60./7.)+q2*S(120)*S(-20));
            return S(120)*(third_negative(q-S(1),3)-S(7)*third_negative(q-S(5./7.),3)+S(21)*third_negative(q-S(3./7.),3)-S(35)*third_negative(q-S(1./7.),3));
    case 33:return S(210)*(S(56)*third_negative(q-S(.25),4)-S(28)*third_negative(q-S(.5),4)+S(8)*third_negative(q-S(.75),4)-third_negative(q-S(1),4));
    case 35:if(q.v<=.214108111463)return S(0);
            return S(24)*third_negative(q-S(1),1)+S(24.*16./5.)*third_negative(q-S(.75),1)-S(24.*94./5.)*third_negative(q-S(.5),1);
    case 36:return (S(196608)*q-S(16777216)*q2*q)*pair_ad::exp(S(-128)*q2);
    case 40:return explicit_ad::divide(S(-3),explicit_ad::power(q,4))-S(3);
    case 41:{S z=S(-120)*q2*q+S(180)*q2-S(72)*q+S(6);return q.v<.5?S(2)*z:z;}
    case 42:if(q.v<=.5)return S(0);else{
        const S c=S(2)*q2-S(3)*q+S(1),b=S(-4)*q2+S(6)*q-S(2);
        return explicit_ad::divide(S(3)*(S(3)-S(4)*q)*(S(16)*q2-S(24)*q+S(15)),S(32)*c*c*pair_ad::pow(b,S(.75)));}
    default:return S(0);
    }
}
template<class T,int D,bool Reverse,int C>SPH_HD void laplacian_partials(
    const T(&x)[D],T r,T h,int name,pair_ad::Number<T,Reverse,C>(&xx)[D],pair_ad::Number<T,Reverse,C> hh,
    pair_ad::Number<T,Reverse,C>(&gradient)[D],pair_ad::Number<T,Reverse,C>& support){
    using S=pair_ad::Number<T,Reverse,C>;
    const T eps=sizeof(T)==4?T(1e-7):T(1e-15),q0=r/h,den=r>eps?r:eps;
    if(h<=0||q0<eps||q0>1)return;
    T partial[D];for(int c=0;c<D;++c)partial[c]=x[c]/den;
    const S rr=pair_ad::primitive(r,xx,partial),q=explicit_ad::divide(rr,hh);
    const auto raw=radial_shape_raw<T,D>(q0,name);
    const T d1=gradient_shape_derivative<T,D>(q0,name,radial_shape<T,D>(q0,name).d1);
    S first=S::unary(d1,q,raw.d2),second=S::unary(raw.d2,q,raw.d3),third=third_shape<D>(q,name);
    if(name==40){first=S(-1.5)*q*q-explicit_ad::divide(S(.5),q*q)+S(2)*q;
        second=explicit_ad::divide(S(1),explicit_ad::power(q,3))-S(3)*q+S(2);}
    const S k1=explicit_ad::divide(first*S(normalization<T,D>(name)),explicit_ad::power(hh,D+1));
    const S k2=explicit_ad::divide(second*S(normalization<T,D>(name)),explicit_ad::power(hh,D+2));
    const S k3=explicit_ad::divide(third*S(normalization<T,D>(name)),explicit_ad::power(hh,D+3));
    const S re=rr+S(eps)*hh,re2=explicit_ad::power(re,2),re3=explicit_ad::power(re,3),re4=explicit_ad::power(re,4);
    S R(0);for(int c=0;c<D;++c)R+=xx[c]*xx[c];
    const S s=explicit_ad::divide(R,re2),t=-explicit_ad::divide(R,re3)+explicit_ad::divide(S(D),re);
    for(int a=0;a<D;++a){
        T normal_partial[D];const T shifted=r+eps,normal_den=shifted*shifted*shifted+eps*eps*eps;
        for(int c=0;c<D;++c)normal_partial[c]=((a==c?shifted*shifted+eps*eps*eps:T(0))-x[a]*x[c])/normal_den;
        const S direction=pair_ad::primitive(x[a]/den,xx,normal_partial);
        const S ds=explicit_ad::divide(S(2)*rr*S(eps)*hh,re3)*direction;
        const S dt=direction*(explicit_ad::divide(S(3)*R-S(2)*rr*re,re4)-explicit_ad::divide(S(D),re2));
        gradient[a]=k2*ds+k1*dt+(s*k3+t*k2)*direction;
    }
    const S ds=-explicit_ad::divide(S(2)*S(eps)*R,re3);
    const S dt=explicit_ad::divide(S(3)*S(eps)*R,re4)-explicit_ad::divide(S(D)*S(eps),re2);
    const S M=s*k3+t*k2;
    support=k2*ds+k1*dt-q*M-explicit_ad::divide(s*S(D+2)*k2+t*S(D+1)*k1,hh);
}
template<class T,int D>SPH_HD void laplacian_product_value(const T(&x)[D],T r,T h,int name,const T(&dt)[D],T dh,T& value,T& product){
    value=evaluate_kernel<T,D>(x,h,name).laplacian;
    using S=pair_ad::Number<T,false,768>;S xx[D],g[D],hh(h),support(0);
    for(int c=0;c<D;++c)xx[c]=S(x[c]);
    laplacian_partials<T,D,false,768>(x,r,h,name,xx,hh,g,support);
    for(int c=0;c<D;++c)product+=g[c].v*dt[c];product+=support.v*dh;
}
template<class T,int D>SPH_HD void laplacian_clipped_replay(const T(&x)[D],T r,T h,int name,T(&bx)[D],T& bh,explicit_ad::Trace<T,768>& trace){
    const T eps=sizeof(T)==4?T(1e-7):T(1e-15);
    if(h<=0||(r/h>=eps&&r/h<=1))return;
    using S=pair_ad::Number<T,true,768>;trace.count=trace.inputs=0;for(int k=0;k<768;++k)trace.division[k]=0;S xx[D];
    T partial[D];for(int c=0;c<D;++c){xx[c]=S::leaf(x[c],0,&trace,0,c,true);partial[c]=x[c]/(r>eps?r:eps);}
    const S hh=S::leaf(h,0,&trace,0,D,true),rr=pair_ad::primitive(r,xx,partial),q=explicit_ad::divide(rr,hh);
    const auto raw=(name==42&&r/h>1)?adhesionKernel_shape<T,D>(RadialJet<T>(r/h,T(1))):radial_shape_raw<T,D>(r/h,name);
    const T first_value=r/h>1?raw.d1:gradient_shape_derivative<T,D>(r/h,name,radial_shape<T,D>(r/h,name).d1);
    S first=S::unary(first_value,q,raw.d2),second=S::unary(raw.d2,q,raw.d3);
    if(name==40){first=S(-1.5)*q*q-explicit_ad::divide(S(.5),q*q)+S(2)*q;second=explicit_ad::divide(S(1),explicit_ad::power(q,3))-S(3)*q+S(2);}
    const S k1=explicit_ad::divide(first*S(normalization<T,D>(name)),explicit_ad::power(hh,D+1));
    const S k2=explicit_ad::divide(second*S(normalization<T,D>(name)),explicit_ad::power(hh,D+2));
    const S re=rr+S(eps)*hh;S R(0);for(int c=0;c<D;++c)R+=xx[c]*xx[c];
    const S original_s=explicit_ad::divide(R,explicit_ad::power(re,2));
    const S t=-explicit_ad::divide(R,explicit_ad::power(re,3))+explicit_ad::divide(S(D),re);
    // L is overwritten outside support and s is also overwritten below epsilon.
    // Replay their operations with
    // no output seed so IEEE 0*Inf still follows the reference reverse.
    const S replay=(r/h<eps?S(1):original_s)*k2+t*k1;(void)original_s;(void)replay;
    trace.reverse();for(int k=0;k<trace.inputs;++k){const auto leaf=trace.leaves[k];const T bar=trace.nodes[leaf.node].bar;if(leaf.at==D)bh+=bar;else bx[leaf.at]+=bar;}
}
template<class T,int D>SPH_HD void laplacian_product_pullback(const T(&x)[D],T r,T h,int name,const T(&dt)[D],T dh,T bv,T bp,T(&bx)[D],T& bh,T(&bt)[D],T& bth,explicit_ad::Trace<T,768>& trace){
    laplacian_pullback<T,D>(x,r,h,name,bv,bx,bh);
    laplacian_clipped_replay<T,D>(x,r,h,name,bx,bh,trace);
    using S=pair_ad::Number<T,true,768>;trace.count=trace.inputs=0;for(int k=0;k<768;++k)trace.division[k]=0;S xx[D],g[D],support(0);
    for(int c=0;c<D;++c)xx[c]=S::leaf(x[c],0,&trace,0,c,true);
    const S hh=S::leaf(h,0,&trace,0,D,true);
    laplacian_partials<T,D,true,768>(x,r,h,name,xx,hh,g,support);
    for(int c=0;c<D;++c){g[c].seed(dt[c]*bp);bt[c]+=g[c].v*bp;}support.seed(dh*bp);bth+=support.v*bp;
    trace.reverse();for(int k=0;k<trace.inputs;++k){const auto leaf=trace.leaves[k];const T bar=trace.nodes[leaf.node].bar;if(leaf.at==D)bh+=bar;else bx[leaf.at]+=bar;}
}
}
