#pragma once
#include "kernels/explicit_value_derivatives.h"
#include "math/explicit_ad.h"
namespace sph_native::warp_compat {
// Reverse of the explicitly exposed H*dx + grad_h(G)*dh product. H is itself
// differentiated here; the ordinary gradient's custom Hessian stays separate.
template<class T,int D>SPH_HD void force_product_pullback(
        const T (&x)[D],T r,T h,int name,const T (&dt)[D],T dh,
        const T (&bg)[D],const T (&bdg)[D],T (&bx)[D],T& bh,T (&bt)[D],T& bth,explicit_ad::Trace<T,512>& trace){
    if(h<=T(0)||r>h)return;
    constexpr int C=512;using S=pair_ad::Number<T,true,C>;
    trace.count=trace.inputs=0;for(int k=0;k<C;++k)trace.division[k]=0;S inputs[D+1],xx[D];
    for(int c=0;c<D;++c)inputs[c]=xx[c]=S::leaf(x[c],0,&trace,0,c,true);
    inputs[D]=S::leaf(h,0,&trace,0,D,true);const S hh=inputs[D];
    const T eps=sizeof(T)==4?T(1e-7):T(1e-15),den=r>eps?r:eps;
    T partial[D];for(int c=0;c<D;++c)partial[c]=x[c]/den;
    const S rr=pair_ad::primitive(r,xx,partial),q=explicit_ad::divide(rr,hh);
    const auto raw=radial_shape_raw<T,D>(r/h,name);
    const T d1=gradient_shape_derivative<T,D>(r/h,name,radial_shape<T,D>(r/h,name).d1);
    T d1q=raw.d2;
    if(T(1)-r/h==T(1)){if(name==21)d1q=0;else if(name==30)d1q-=T(6);else if(name==31)d1q-=T(12);else if(name==32)d1q-=T(20);}
    S first=S::unary(d1,q,d1q),second=S::unary(raw.d2,q,raw.d3);
    if(name==40){first=S(-1.5)*explicit_ad::power(q,2)-explicit_ad::divide(S(.5),explicit_ad::power(q,2))+S(2)*q;
        second=explicit_ad::divide(S(1),explicit_ad::power(q,3))-S(3)*q+S(2);}
    const S k1=explicit_ad::divide(first*S(normalization<T,D>(name)),explicit_ad::power(hh,D+1));
    const S k2=explicit_ad::divide(second*S(normalization<T,D>(name)),explicit_ad::power(hh,D+2));
    const S s=explicit_ad::power(rr,2)+S(eps*eps)*explicit_ad::power(hh,2);
    const S t=explicit_ad::power(rr,3)+S(eps*eps*eps)*explicit_ad::power(hh,3);
    const S u=rr+S(eps*eps)*hh;
    const S norm=explicit_ad::divide(S(-normalization<T,D>(name)),explicit_ad::power(hh,D+2));
    const bool conditioned_support=D==1 && name==40 && r>T(0);
    const bool conditioned_hessian=D==1 && sizeof(T)==4 && use_conditioned_hessian(r,h,eps);
    const S support_shape=conditioned_support ? S(6)*q*(S(1)-q) : q*second+S(D+1)*first;
    const S support=norm*support_shape;
    T w=0,g[D]={},wx[D]={},wh=0,H[D*D]={},gh[D]={};
    force_derivatives<T,D>(x,r,h,name,w,g,wx,wh,H,gh);
    for(int a=0;a<D;++a){
        T normal_partial[D];const T shifted=r+eps,normal_den=shifted*shifted*shifted+eps*eps*eps;
        for(int c=0;c<D;++c)normal_partial[c]=((a==c?shifted*shifted+eps*eps*eps:T(0))-x[a]*x[c])/normal_den;
        const S direction=pair_ad::primitive(x[a]/den,xx,normal_partial);
        (direction*support).seed(dh*bdg[a]);
        bth+=gh[a]*bdg[a];bh+=gh[a]*bg[a];
        for(int b=0;b<D;++b){
            S A=explicit_ad::matrix_divide(xx[a]*xx[b],s);
            // Warp's reverse replays the pre-overwrite matrix and retains
            // its input dependency through the in-place diagonal assignment.
            S B;
            if(conditioned_hessian) {
                const ConditionedHessianFactor factor{double(r),double(h),double(eps)};
                const double rd=factor.r,hd=factor.h,e=factor.eps;
                const double e2=e*e,e2h=e2*hd,e3=e2*e;
                const double radial_cube=rd*rd*rd+e3*hd*hd*hd;
                const double value=factor.value(),denominator=factor.denominator();
                const double nr=-2*e2h*rd,nh=e2*(3*e*hd*hd-rd*rd);
                const double dr=3*rd*rd*(rd+e2h)+radial_cube;
                const double dh=3*e3*hd*hd*(rd+e2h)+e2*radial_cube;
                double radius_partial=(nr-value*dr)/denominator;
                double support_partial=(nh-value*dh)/denominator;
                double correction=0,position_partial=0;
                if(r<eps) {
                    // Below the clamp the adjoint of r differs from that of
                    // |x|. Preserve the dependence on both inputs.
                    const double xd=x[a];
                    correction=(rd*rd-xd*xd)/radial_cube;
                    radius_partial+=(2*rd-correction*3*rd*rd)/radial_cube;
                    support_partial-=correction*3*e3*hd*hd/radial_cube;
                    position_partial=-2*xd/radial_cube;
                }
                const T coefficient=T(value+correction);
                B=S::binary(coefficient,rr,hh,T(radius_partial),T(support_partial));
                if(r<eps)B=S::binary(coefficient,B,xx[a],T(1),T(position_partial));
            } else {
                B=explicit_ad::matrix_divide(-xx[a]*xx[b],t);
                if(a==b)B+=explicit_ad::matrix_divide(S(1),u);
            }
            (A*k2+B*k1).seed(dt[b]*bdg[a]);
            bx[b]+=H[a*D+b]*bg[a];bt[b]+=H[a*D+b]*bdg[a];
        }
    }
    trace.reverse();
    for(int k=0;k<trace.inputs;++k){const auto leaf=trace.leaves[k];const T bar=trace.nodes[leaf.node].bar;
        if(leaf.at==D)bh+=bar;else bx[leaf.at]+=bar;}
}
}

namespace sph_native::warp_compat {
template<class T,int D>SPH_HD void force_product_pullback(
        const T (&x)[D],T r,T h,int name,const T (&dt)[D],T dh,
        const T (&bg)[D],const T (&bdg)[D],T (&bx)[D],T& bh,T (&bt)[D],T& bth){
    explicit_ad::Trace<T,512> trace;
    force_product_pullback(x,r,h,name,dt,dh,bg,bdg,bx,bh,bt,bth,trace);
}
}
