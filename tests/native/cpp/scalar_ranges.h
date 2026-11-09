#pragma once
#include "math/symmetric.h"
#include "math/pair_ad.h"
#include "math/directional.h"
#include <cmath>
#include <limits>

// Analytic a/b products at rounded inputs. Local Jacobians are representable.
template<class T> SPH_HD int quotient_cases(){
 using namespace sph_native;
 using F=pair_ad::Number<T,false,32>;
 using R=pair_ad::Number<T,true,32>;
 const T scales[]={T(1e-25),T(1e-22),T(1),T(1e20),T(1e38)};
 const T signs[]={T(-1),T(1)};
 for(T magnitude:scales)for(T sign:signs){
  const T b=sign*magnitude;
  const F x=F::leaf(b,T(0),nullptr,0,0,true),y=F::leaf(b,b,nullptr,1,0,true);
  const F z=x/y;
  const auto dz=Directional<T>(b,T(0))/Directional<T>(b,b);
  const T tolerance=T(8)*std::numeric_limits<T>::epsilon();
  if(z.v!=T(1)||std::abs(z.d+T(1))>tolerance||dz.value!=T(1)||std::abs(dz.tangent+T(1))>tolerance)return 1;
  pair_ad::Trace<T,32> trace;
  const R rx=R::leaf(b,T(0),&trace,0,0,true),ry=R::leaf(b,T(0),&trace,1,0,true);
  const R rz=rx/ry;rz.seed(b);trace.reverse();
  if(trace.count!=3||trace.inputs!=2)return 2;
  if(std::abs(trace.nodes[rx.node].bar-T(1))>tolerance||std::abs(trace.nodes[ry.node].bar+T(1))>tolerance)return 3;
  // The total direction is zero even when the unscaled numerator overflows.
  const T tangent=T(1e30);
  const auto cancel=F::leaf(b,tangent,nullptr,0,0,true)/F::leaf(b,tangent,nullptr,1,0,true);
  const auto cancel_direction=Directional<T>(b,tangent)/Directional<T>(b,tangent);
  if constexpr(sizeof(T)==sizeof(float))
   if((!std::isfinite(b*b)||std::abs(b*b)<std::numeric_limits<T>::min()) &&
      (cancel.d!=T(0)||cancel_direction.tangent!=T(0)))return 4;
 }
 return 0;
}


template<class T,int D> SPH_HD int spectrum_cases(){
 const T scale=T(1e38),tolerance=T(32)*std::numeric_limits<T>::epsilon();
 T matrix[D*D]={},inverse[D*D]={};
 matrix[0]=matrix[D+1]=scale;matrix[1]=matrix[D]=T(.25)*scale;
 if constexpr(D==3)matrix[8]=-scale;
 const auto spectrum=sph_native::symmetric_spectrum<T,D>(matrix);
 for(int i=0;i<D;++i)for(int j=0;j<D;++j){
  T reconstructed=0,orthogonal=0;
  for(int k=0;k<D;++k){reconstructed+=spectrum.vectors[i*3+k]*(spectrum.values[k]/scale)*spectrum.vectors[j*3+k];orthogonal+=spectrum.vectors[k*3+i]*spectrum.vectors[k*3+j];}
  if(!std::isfinite(reconstructed)||std::abs(reconstructed-matrix[i*D+j]/scale)>tolerance)return 5;
  if(std::abs(orthogonal-T(i==j))>tolerance)return 6;
 }
 sph_native::symmetric_inverse<T,D>(matrix,T(1e-6),inverse);
 for(int i=0;i<D;++i)for(int j=0;j<D;++j){
  T product=0;for(int k=0;k<D;++k)product+=(matrix[i*D+k]/scale)*(inverse[k*D+j]*scale);
  if(!std::isfinite(product)||std::abs(product-T(i==j))>tolerance)return 7;
 }
 return 0;
}
template<class T>SPH_HD int scalar_range_cases(){const int q=quotient_cases<T>();if(q)return q;const int a=spectrum_cases<T,2>();if(a)return a;return spectrum_cases<T,3>();}
