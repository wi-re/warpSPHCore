#pragma once
#include "math/pair_ad.h"
namespace sph_native::pair_ad {
template<class T,bool R,int N>SPH_HD Number<T,R,N>& operator/=(Number<T,R,N>& a,Number<T,R,N> b){a=a/b;return a;}
template<class T,bool R,int N>SPH_HD Number<T,R,N> scalar_sqrt(Number<T,R,N> a){return sqrt(a);}
template<class T,bool R,int N>SPH_HD Number<T,R,N> scalar_power(Number<T,R,N> a,Number<T,R,N> b){return pow(a,b);}
template<class T,bool R,int N>SPH_HD bool finite(Number<T,R,N> a){return std::isfinite(a.v);}
}
#include "kernels/regularization.h"
namespace sph_native::warp_compat {
template<class T,bool R,int N>SPH_HD pair_ad::Number<T,R,N> gradient_scale(pair_ad::Number<T,R,N> r){
 using S=pair_ad::Number<T,R,N>;const S eps(sizeof(T)==4?T(1e-7):T(1e-15));return r>=eps?S(1):r/eps;
}
}
#include "nt_mfm_pair.h"
#include "kernels/warp_derivatives.h"
namespace sph_native::mfm {
template<class T,bool R,int N,int D>struct KernelPair<pair_ad::Number<T,R,N>,D> {
 using S=pair_ad::Number<T,R,N>;
 SPH_HD static void evaluate(S r,S h,int kernel,S& w,S& g){
  T weight=0,radial=0;kernel_pair_value<T,D>(r.v,h.v,kernel,weight,radial);
  T wp[2]={},gp[2]={};
  if(h.v>T(0)&&r.v<=h.v){
   const T q=r.v/h.v,C=normalization<T,D>(kernel),hd=power<T,D>(h.v,D);const auto f=radial_shape<T,D>(q,kernel);
   const T first=warp_compat::value_shape_derivative<T,D>(q,kernel,f.d1);
   wp[0]=C*first/(hd*h.v);wp[1]=-C*(T(D)*f.v+q*first)/(hd*h.v);
   if(r.v>T(0)){
    gp[0]=C*(f.d2/(h.v*r.v)-f.d1/(r.v*r.v))/(hd*h.v);
    gp[1]=-C*(T(D+1)*f.d1+q*f.d2)/(hd*h.v*h.v*r.v);
   }else{wp[0]=T(0);gp[1]=-T(D+2)*radial/h.v;}
  }
  const S inputs[2]={r,h};w=pair_ad::primitive(weight,inputs,wp);g=pair_ad::primitive(radial,inputs,gp);
 }
};
}
