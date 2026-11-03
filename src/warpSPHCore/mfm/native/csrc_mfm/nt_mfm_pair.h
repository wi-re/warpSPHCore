#pragma once
#include "math/small.h"
#include "kernels/kernel.h"
namespace sph_native::mfm {
template<class T> SPH_HD T minimum(T a,T b) {return a<b ? a:b;}
template<class T> SPH_HD T maximum(T a,T b) {return a>b ? a:b;}
template<class T> SPH_HD T absolute(T a) {return a<T(0) ? -a:a;}
template<class T> SPH_HD bool finite(T value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}
template<class T> SPH_HD T scalar_power(T value,T exponent) {
    if constexpr(sizeof(T)==4) return ::powf(value,exponent);
    else return ::pow(value,exponent);
}
template<class T,int D> SPH_HD T weight(const SmallVector<T,D>& d,T h,int kernel) {
    T w,g;kernel_pair_value<T,D>(length(d),h,kernel,w,g);return w;
}
template<class T,int D> SPH_HD SmallVector<T,D> guarded_face(const SmallVector<T,D>& d,T Vi,T Vj,
        const SmallMatrix<T,D>& Ei,const SmallMatrix<T,D>& Ej,T hi,T hj,bool badI,bool badJ,int kernel,bool centred,bool areaCap) {
    const T r=length(d),Wi=weight(d,hi,kernel),Wj=weight(d,hj,kernel),tiny=T(1e-30);
    T wti=Vi,wtj=Vj;
    if (centred && absolute(Vi-Vj)/minimum(Vi,Vj)/T(D)>T(1.25)) {
        const T denominator=Vi*Wi+Vj*Wj;
        if (denominator>T(0)) wti=wtj=Vi*Vj*(Wi+Wj)/denominator;
    }
    auto face=(wti*Wi)*(Ei*d)+(wtj*Wj)*(Ej*d);
    if (badI || badJ || dot(face,d)<T(0) || !finite(length(face))) {
        T wi,gi,wj,gj;kernel_pair_value<T,D>(r,hi,kernel,wi,gi);kernel_pair_value<T,D>(r,hj,kernel,wj,gj);
        // radial is (dW/dr)/r. Retain the reference's finite origin limit.
        face=(-(wti*Vi*gi*r+wtj*Vj*gj*r)/maximum(r,tiny))*d;
    }
    if (areaCap) {
        T ai=2,aj=2;
        if constexpr(D==2) {ai=T(2)*T(3.141592653589793)*scalar_sqrt(Vi);aj=T(2)*T(3.141592653589793)*scalar_sqrt(Vj);}
        if constexpr(D==3) {ai=T(4)*T(3.141592653589793)*scalar_power(Vi,T(2./3));aj=T(4)*T(3.141592653589793)*scalar_power(Vj,T(2./3));}
        const T cap=minimum(ai,aj),magnitude=length(face);
        if (magnitude>cap) face=face*(cap/magnitude);
    }
    return face;
}
template<class T> SPH_HD T alpha(T value,T high,T low,T midHigh,T midLow,T beta) {
    const T up=midHigh-value,down=value-midLow;
    const T upper=up>T(1e-30) ? (high-value)/up:T(1e30);
    const T lower=down>T(1e-30) ? (value-low)/down:T(1e30);
    return minimum(T(1),beta*minimum(upper,lower));
}
template<class T> SPH_HD T sign(T v) {return v>T(0) ? T(1):v<T(0) ? T(-1):T(0);}
template<class T> SPH_HD T pair_limit(T i,T j,T face,T fraction,T psi1,T psi2) {
    if (i==j) return i;
    const T delta=absolute(i-j),d1=psi1*delta,d2=psi2*delta,average=i+fraction*(j-i);
    const T low=minimum(i,j),high=maximum(i,j);
    T floor=low-d1,ceiling=high+d1;
    if (sign(floor)!=sign(low)) floor=low/(T(1)+d1/maximum(absolute(low),T(1e-30)));
    if (sign(ceiling)!=sign(high)) ceiling=high/(T(1)+d1/maximum(absolute(high),T(1e-30)));
    return i<j ? maximum(floor,minimum(average+d2,face)):minimum(ceiling,maximum(average-d2,face));
}
template<class T> struct Star {T left,right,contact,pressure;};
template<class T> SPH_HD Star<T> star_from_speeds(T left,T right,T rhoL,T uL,T pL,T rhoR,T uR,T pR) {
    const T contact=(pR-pL+rhoL*uL*(left-uL)-rhoR*uR*(right-uR))/(rhoL*(left-uL)-rhoR*(right-uR));
    return {left,right,contact,pL+rhoL*(left-uL)*(contact-uL)};
}
template<class T> SPH_HD bool valid_star(const Star<T>& star) {return star.pressure>T(0) && finite(star.pressure) && finite(star.contact);}
template<class T,int D> SPH_HD Star<T> hllc_star(T rhoL,T uL,const SmallVector<T,D>& vtL,T pL,T rhoR,T uR,const SmallVector<T,D>& vtR,T pR,T gamma) {
    const T aL=scalar_sqrt(gamma*pL/rhoL),aR=scalar_sqrt(gamma*pR/rhoR);
    auto star=star_from_speeds(minimum(uL,uR)-maximum(aL,aR),maximum(uL,uR)+maximum(aL,aR),rhoL,uL,pL,rhoR,uR,pR);
    if (!valid_star(star)) {
        const T ratio=scalar_sqrt(rhoR/rhoL);
        const T EL=pL/(gamma-T(1))+T(.5)*rhoL*(uL*uL+dot(vtL,vtL));
        const T ER=pR/(gamma-T(1))+T(.5)*rhoR*(uR*uR+dot(vtR,vtR));
        const T ut=(uL+ratio*uR)/(T(1)+ratio);
        const auto vt=(vtL+ratio*vtR)/(T(1)+ratio);
        const T H=((EL+pL)/rhoL+ratio*(ER+pR)/rhoR)/(T(1)+ratio);
        const T sound=scalar_sqrt(maximum((gamma-T(1))*(H-T(.5)*(ut*ut+dot(vt,vt))),T(0)));
        star=star_from_speeds(minimum(uL-aL,ut-sound),maximum(uR+aR,ut+sound),rhoL,uL,pL,rhoR,uR,pR);
        if (!valid_star(star)) {
            const T pressure=maximum(T(.5)*(pL+pR)-T(.5)*(uR-uL)*T(.5)*(rhoL+rhoR)*T(.5)*(aL+aR),T(0));
            const T qL=pressure>pL ? scalar_sqrt(T(1)+(gamma+T(1))/(T(2)*gamma)*(pressure/pL-T(1))):T(1);
            const T qR=pressure>pR ? scalar_sqrt(T(1)+(gamma+T(1))/(T(2)*gamma)*(pressure/pR-T(1))):T(1);
            star=star_from_speeds(uL-aL*qL,uR+aR*qR,rhoL,uL,pL,rhoR,uR,pR);
            if (!valid_star(star)) {
                const T speed=maximum(absolute(uL)+aL,absolute(uR)+aR);
                star=star_from_speeds(-speed,speed,rhoL,uL,pL,rhoR,uR,pR);
            }
        }
    }
    return star;
}
template<class T,int D> struct SideFlux {T mass,normal;SmallVector<T,D> tangent;T energy;};
template<class T,int D> SPH_HD SideFlux<T,D> hllc_side(T rho,T velocity,const SmallVector<T,D>& vt,T pressure,T speed,T contact,T gamma,bool star) {
    const T energy=pressure/(gamma-T(1))+T(.5)*rho*(velocity*velocity+dot(vt,vt));
    SideFlux<T,D> flux{rho*velocity,rho*velocity*velocity+pressure,rho*velocity*vt,(energy+pressure)*velocity};
    if (star) {
        const T coefficient=rho*(speed-velocity)/(speed-contact);
        flux.mass+=speed*(coefficient-rho);
        flux.normal+=speed*(coefficient*contact-rho*velocity);
        flux.tangent+=speed*(coefficient*vt-rho*vt);
        flux.energy+=speed*(coefficient*(energy/rho+(contact-velocity)*(contact+pressure/(rho*(speed-velocity))))-energy);
    }
    return flux;
}
template<class T,int D> struct FaceFlux {T mass;SmallVector<T,D> momentum;T energy,contact,pressure;};
template<class T,int D> SPH_HD FaceFlux<T,D> face_flux(T rhoL,T uL,const SmallVector<T,D>& vtL,T pL,
        T rhoR,T uR,const SmallVector<T,D>& vtR,T pR,T gamma,int mode,const SmallVector<T,D>& normal,const SmallVector<T,D>& frame) {
    const auto star=hllc_star(rhoL,uL,vtL,pL,rhoR,uR,vtR,pR,gamma);
    if (mode==0) return {T(0),star.pressure*normal,star.pressure*(star.contact+dot(frame,normal)),star.contact,star.pressure};
    SideFlux<T,D> flux{};
    if (star.left>=T(0)) flux=hllc_side(rhoL,uL,vtL,pL,star.left,star.contact,gamma,false);
    else if (star.contact>=T(0)) flux=hllc_side(rhoL,uL,vtL,pL,star.left,star.contact,gamma,true);
    else if (star.right>T(0)) flux=hllc_side(rhoR,uR,vtR,pR,star.right,star.contact,gamma,true);
    else flux=hllc_side(rhoR,uR,vtR,pR,star.right,star.contact,gamma,false);
    const auto momentum=flux.normal*normal+flux.tangent;
    return {flux.mass,momentum+flux.mass*frame,flux.energy+dot(frame,momentum)+T(.5)*dot(frame,frame)*flux.mass,star.contact,star.pressure};
}
template<class T,int D> struct Primitive {T density;SmallVector<T,D> velocity;T pressure;};
template<class T,int D> SPH_HD Primitive<T,D> time_derivative(T rho,const SmallVector<T,D>& velocity,T pressure,
        const SmallVector<T,D>& gRho,const SmallMatrix<T,D>& gVel,const SmallVector<T,D>& gP,T gamma) {
    T divergence=0;for(int d=0;d<D;++d)divergence+=gVel(d,d);
    return {-(dot(velocity,gRho)+rho*divergence),-(gVel*velocity+gP/rho),-(dot(velocity,gP)+gamma*pressure*divergence)};
}
template<class T,int D> SPH_HD SmallMatrix<T,D> limited_gradient(SmallMatrix<T,D> gradient,const SmallVector<T,D>& limiter) {
    for(int c=0;c<D;++c)for(int k=0;k<D;++k)gradient(c,k)*=limiter[c];return gradient;
}
template<class T,int D> SPH_HD Primitive<T,D> reconstruct(T rho,const SmallVector<T,D>& velocity,T pressure,T otherRho,
        const SmallVector<T,D>& otherVelocity,T otherPressure,const SmallVector<T,D>& gRho,const SmallMatrix<T,D>& gVel,
        const SmallVector<T,D>& gP,T aRho,const SmallVector<T,D>& aVel,T aP,const SmallVector<T,D>& displacement,
        T fraction,const SmallVector<T,D>& frame,T dt,T gamma,int order,T psi1,T psi2) {
    Primitive<T,D> face{rho,velocity,pressure};
    if(order>=2) {
        face.density=pair_limit(rho,otherRho,rho+aRho*dot(gRho,displacement),fraction,psi1,psi2);
        face.pressure=pair_limit(pressure,otherPressure,pressure+aP*dot(gP,displacement),fraction,psi1,psi2);
        const auto linear=gVel*displacement;
        for(int c=0;c<D;++c)face.velocity[c]=pair_limit(velocity[c],otherVelocity[c],velocity[c]+aVel[c]*linear[c],fraction,psi1,psi2);
    }
    face.velocity=face.velocity-frame;
    if(order>=2 && dt>T(0)) {
        const auto derivative=time_derivative(rho,velocity,pressure,aRho*gRho,limited_gradient(gVel,aVel),aP*gP,gamma);
        const T half=T(.5)*dt,rhoPredicted=face.density+half*derivative.density,pPredicted=face.pressure+half*derivative.pressure;
        if(rhoPredicted>T(0) && pPredicted>T(0)) {face.density=rhoPredicted;face.pressure=pPredicted;face.velocity+=half*derivative.velocity;}
    }
    return face;
}
} // namespace sph_native::mfm
