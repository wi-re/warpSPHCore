#pragma once
#include "nt_mfm_ad_math.h"
#include "nt_mfm_rows.h"
#include "math/pair_ad.h"
#include "execution/atomic.h"
#include "kernels/warp_derivatives.h"
#include <type_traits>
namespace sph_native::mfm {
constexpr int ProductSlots=OUTPUT+FieldCount;
struct ProductBuffers {Buffers primal;Tensors bars,directions,gradients;int64_t mask;bool deterministic;};
template<class T,int D,int Task,bool Reverse>struct ProductRow {
 RowArgs<T,D> a;const T* f[FieldCount]={};const int32_t* bad=nullptr;const int32_t* materials=nullptr;Options options;
 const T* bars[3]={};const T* directions[ProductSlots]={};T* gradients[ProductSlots]={};T* outputs[3]={};
 int64_t mask,rows;bool deterministic;
 SPH_HD void add(int k,int64_t at,T v)const{if(deterministic)gradients[k][at]+=v;else add_atomic(gradients[k]+at,v);}
 SPH_HD void pair(int64_t row,int64_t i,int64_t j,bool self)const{
  constexpr int Nodes=768;using Trace=pair_ad::Trace<T,Nodes>;using S=pair_ad::Number<T,Reverse,Nodes>;
  using Storage=std::conditional_t<Reverse,Trace,char>;Storage storage;Trace* trace=nullptr;if constexpr(Reverse)trace=&storage;
  auto leaf=[&](T v,int k,int64_t at){return S::leaf(v,directions[k]?directions[k][at]:T(0),trace,k,at,Reverse?bool(mask&(int64_t(1)<<k)):bool(directions[k]));};
  auto field=[&](int k,int64_t at){return leaf(f[k][at],OUTPUT+k,at);};
  T raw[D]={};if(!self){warp_compat::minimum_image_displacement<T,D>(a.rx+i*D,a.rx+j*D,a.lo,a.hi,a.periodic,raw);for(int c=0;c<D;++c)raw[c]=-raw[c];}
  T squared=0;for(int c=0;c<D;++c)squared+=raw[c]*raw[c];const T radius=scalar_sqrt(squared);
  if(!self && (i==j || radius>maximum(a.rh[i],a.rh[j])))return;
  if constexpr(Task==Gradients)if(self||bad[i])return;
  S dx[D];for(int c=0;c<D;++c){dx[c]=self?S(0):leaf(a.rx[j*D+c],RX,j*D+c)-leaf(a.rx[i*D+c],RX,i*D+c);
   if(!self&&a.periodic[c]){const T winding=std::round((a.rx[j*D+c]-a.rx[i*D+c]-raw[c])/(a.hi[c]-a.lo[c]));if(winding!=T(0))dx[c]+=S(winding)*(leaf(a.lo[c],LO,c)-leaf(a.hi[c],HI,c));}dx[c].v=raw[c];}
  const S h=leaf(a.rh[i],RH,i);T w=0,g=0;kernel_pair_value<T,D>(radius,h.v,a.kernel,w,g);
  if constexpr(Task==Gradients)if(w<=T(0))return;
  T partial[D+1]={};
  if(h.v>T(0)&&radius<=h.v){const T q=radius/h.v,C=normalization<T,D>(a.kernel),hd=power<T,D>(h.v,D);const auto radial=radial_shape_raw<T,D>(q,a.kernel);
   const T first=warp_compat::value_shape_derivative<T,D>(q,a.kernel,radial.d1);
   for(int c=0;c<D;++c)partial[c]=radius>T(0)?(raw[c]/radius)*(C*first/(hd*h.v)):T(0);
   partial[D]=-C*(T(D)*radial.v+q*first)/(hd*h.v);
  }
  S inputs[D+1];for(int c=0;c<D;++c)inputs[c]=dx[c];inputs[D]=h;const S weight=pair_ad::primitive(w,inputs,partial);
  // NVCC can value-initialize later subarrays of a defaulted AD Number
  // constructor with node=0 rather than its inactive node=-1. That seeds
  // an unrelated position leaf. Construct every accumulator explicitly.
  S value[3][D*D];for(int output=0;output<3;++output)for(int c=0;c<D*D;++c)value[output][c]=S(T(0));
  if constexpr(Task==Moments){value[0][0]=weight;for(int c=0;c<D;++c)for(int d=0;d<D;++d)value[1][c*D+d]=dx[c]*dx[d]*weight;}
  else if constexpr(Task==Gradients){
   const S dr=field(Rho,j)-field(Rho,i),dp=field(Pressure,j)-field(Pressure,i);
   S inverse[D*D];for(int c=0;c<D*D;++c)inverse[c]=field(Inverse,i*D*D+c);
   for(int c=0;c<D;++c)for(int d=0;d<D;++d){value[0][c]+=inverse[c*D+d]*(weight*dr*dx[d]);value[2][c]+=inverse[c*D+d]*(weight*dp*dx[d]);}
   for(int c=0;c<D;++c){const S dv=field(Velocity,j*D+c)-field(Velocity,i*D+c);for(int d=0;d<D;++d)for(int k=0;k<D;++k)value[1][c*D+d]+=(dv*dx[k]*weight)*inverse[k*D+d];}
  }
  constexpr int A=Task==Moments?1:D,B=D*D,C=Task==Moments?0:D;
  if constexpr(Reverse){
   for(int c=0;c<A;++c)value[0][c].seed(bars[0]?bars[0][row*A+c]:T(0));
   for(int c=0;c<B;++c)value[1][c].seed(bars[1]?bars[1][row*B+c]:T(0));
   for(int c=0;c<C;++c)value[2][c].seed(bars[2]?bars[2][row*C+c]:T(0));
   trace->reverse();for(int k=0;k<trace->inputs;++k){auto l=trace->leaves[k];add(l.slot,l.at,trace->nodes[l.node].bar);}
  }else{
   // Separate output buffers carry the three differently-shaped row products.
   for(int c=0;c<A;++c)outputs[0][row*A+c]+=value[0][c].d;
   for(int c=0;c<B;++c)outputs[1][row*B+c]+=value[1][c].d;
   for(int c=0;c<C;++c)outputs[2][row*C+c]+=value[2][c].d;
  }
 }
 SPH_HD void flux_pair(int64_t row,int64_t i,int64_t j)const{
  if(i==j)return;const int64_t aidx=i<j?i:j,bidx=i<j?j:i;
  T raw[D];warp_compat::minimum_image_displacement<T,D>(a.rx+aidx*D,a.rx+bidx*D,a.lo,a.hi,a.periodic,raw);
  T r2=0;for(int c=0;c<D;++c){raw[c]=-raw[c];r2+=raw[c]*raw[c];}
  if(scalar_sqrt(r2)>maximum(a.rh[aidx],a.rh[bidx]))return;
  constexpr int Nodes=3072;using Trace=pair_ad::Trace<T,Nodes>;using S=pair_ad::Number<T,Reverse,Nodes>;
  using Storage=std::conditional_t<Reverse,Trace,char>;Storage storage;Trace* trace=nullptr;if constexpr(Reverse)trace=&storage;
  auto leaf=[&](T v,int k,int64_t at){return S::leaf(v,directions[k]?directions[k][at]:T(0),trace,k,at,Reverse?bool(mask&(int64_t(1)<<k)):bool(directions[k]));};
  Row<S,D,Flux> local{};local.options=options;local.args.kernel=a.kernel;
  int32_t localBad[2]={bad[aidx],bad[bidx]},localMaterials[2]={materials?materials[aidx]:0,materials?materials[bidx]:0};
  local.bad=localBad;if(materials)local.materials=localMaterials;
  S supports[2]={leaf(a.rh[aidx],RH,aidx),leaf(a.rh[bidx],RH,bidx)};local.args.rh=supports;
  S fields[FieldCount][2*D*D];
  for(int k=0;k<FieldCount;++k)if(f[k]){
   const int stride=k==Inverse||k==GradientVelocity?D*D:k==Velocity||k==GradientRho||k==GradientPressure||k==AlphaVelocity||k==Potential||k==Vector?D:1;
   local.fields[k]=fields[k];
   for(int q=0;q<2;++q)for(int c=0;c<stride;++c){const int64_t at=k==DeviceTimestep?0:(q==0?aidx:bidx)*stride+c;fields[k][q*stride+c]=leaf(f[k][at],OUTPUT+k,at);}
  }
  SmallVector<S,D> dx{};
  for(int c=0;c<D;++c){dx[c]=leaf(a.rx[bidx*D+c],RX,bidx*D+c)-leaf(a.rx[aidx*D+c],RX,aidx*D+c);
   if(a.periodic[c]){const T winding=std::round((a.rx[bidx*D+c]-a.rx[aidx*D+c]-raw[c])/(a.hi[c]-a.lo[c]));if(winding!=T(0))dx[c]+=S(winding)*(leaf(a.lo[c],LO,c)-leaf(a.hi[c],HI,c));}dx[c].v=raw[c];}
  const auto original=local.face(0,1,dx);const S k=scalar_power(length(original),S(options.power));
  auto difference=local.vector(Potential,1)-local.vector(Potential,0);const S magnitude=length(difference);
  if(options.closureCap>0&&magnitude>S(options.closureCap))difference=difference*(S(options.closureCap)/magnitude);
  auto corrected=original+k*difference;
  if(options.faceGuard){
   const S before=dot(original,dx),after=dot(corrected,dx);
   if(options.faceGuard==1){if(after<=S(0))corrected=original;}
   else if(options.faceGuard>=3){const auto change=corrected-original;const S norm=length(change),cap=(options.faceGuard==4?S(.25):S(.5))*length(original);if(norm>cap)corrected=original+change*(cap/norm);}
   else if(after<S(.25)*before){const S change=after-before,scale=change<S(0)?maximum(S(0),(S(.25)*before-before)/change):S(0);corrected=original+scale*(corrected-original);}
  }
  const S area=length(corrected);if(area<=S(0))return;
  const auto flux=local.flux(0,1,dx,corrected,area);const S sign=i==aidx?S(-1):S(1);
  S values[D+2];values[0]=sign*area*flux.mass;for(int c=0;c<D;++c)values[1+c]=sign*area*flux.momentum[c];values[D+1]=sign*area*flux.energy;
  if constexpr(Reverse){values[0].seed(bars[0]?bars[0][row]:T(0));for(int c=0;c<D;++c)values[1+c].seed(bars[1]?bars[1][row*D+c]:T(0));values[D+1].seed(bars[2]?bars[2][row]:T(0));trace->reverse();
   for(int k=0;k<trace->inputs;++k){auto l=trace->leaves[k];add(l.slot,l.at,trace->nodes[l.node].bar);}
  }else{outputs[0][row]+=values[0].d;for(int c=0;c<D;++c)outputs[1][row*D+c]+=values[1+c].d;outputs[2][row]+=values[D+1].d;}
 }
 // Extrema membership is frozen at the literal forward selection. Only the
 // four selected samples enter a bounded trace; neighborhood size is unbounded.
 SPH_HD void limiter(int64_t row,int64_t i,int component)const{
  const int fieldId=component==0?Rho:component==D+1?Pressure:Velocity;
  const int gradId=component==0?GradientRho:component==D+1?GradientPressure:GradientVelocity;
  const int c=fieldId==Velocity?component-1:0,stride=fieldId==Velocity?D:1;
  auto value=[&](int64_t j){return f[fieldId][j*stride+c];};
  const T own=value(i);T high=own,low=own,mhi=own,mlo=own;int64_t hj=i,lj=i,mhj=-1,mlj=-1;
  for(int64_t e=0;e<a.counts[row];++e){const auto j=a.neighbors[a.offsets[row]+e];
   T dx[D];warp_compat::minimum_image_displacement<T,D>(a.rx+i*D,a.rx+j*D,a.lo,a.hi,a.periodic,dx);
   T r2=0;for(int d=0;d<D;++d){dx[d]=-dx[d];r2+=dx[d]*dx[d];}
   if(i==j||scalar_sqrt(r2)>maximum(a.rh[i],a.rh[j]))continue;
   const T fraction=a.rh[i]/(a.rh[i]+a.rh[j]);T mid=own;
   // Match literal dot accumulation, including its rounded fraction*dx.
   T delta=0;for(int d=0;d<D;++d)delta+=f[gradId][(i*stride+c)*D+d]*(fraction*dx[d]);mid+=delta;
   const T v=value(j);if(!(high>v)){high=v;hj=j;}if(!(low<v)){low=v;lj=j;}
   if(!(mhi>mid)){mhi=mid;mhj=j;}if(!(mlo<mid)){mlo=mid;mlj=j;}
  }
  constexpr int Nodes=256;using Trace=pair_ad::Trace<T,Nodes>;using S=pair_ad::Number<T,Reverse,Nodes>;
  using Storage=std::conditional_t<Reverse,Trace,char>;Storage storage;Trace* trace=nullptr;if constexpr(Reverse)trace=&storage;
  auto leaf=[&](T v,int k,int64_t at){return S::leaf(v,directions[k]?directions[k][at]:T(0),trace,k,at,Reverse?bool(mask&(int64_t(1)<<k)):bool(directions[k]));};
  auto sample=[&](int64_t j){return leaf(value(j),OUTPUT+fieldId,j*stride+c);};
  const S phi=sample(i);
  auto middle=[&](int64_t j){
   if(j<0)return phi;
   T raw[D];warp_compat::minimum_image_displacement<T,D>(a.rx+i*D,a.rx+j*D,a.lo,a.hi,a.periodic,raw);
   const S hi=leaf(a.rh[i],RH,i),hj=leaf(a.rh[j],RH,j);const S fraction=hi/(hi+hj);S delta(0);
   for(int d=0;d<D;++d){S dx=leaf(a.rx[j*D+d],RX,j*D+d)-leaf(a.rx[i*D+d],RX,i*D+d);
    if(a.periodic[d]){const T winding=std::round((a.rx[j*D+d]-a.rx[i*D+d]+raw[d])/(a.hi[d]-a.lo[d]));if(winding!=T(0))dx+=S(winding)*(leaf(a.lo[d],LO,d)-leaf(a.hi[d],HI,d));}
    dx.v=-raw[d];delta+=leaf(f[gradId][(i*stride+c)*D+d],OUTPUT+gradId,(i*stride+c)*D+d)*(fraction*dx);
   }return phi+delta;
  };
  const S result=alpha(phi,sample(hj),sample(lj),middle(mhj),middle(mlj),leaf(f[Beta][i],OUTPUT+Beta,i));
  const int output=fieldId==Rho?0:fieldId==Pressure?2:1;const int64_t outAt=row*stride+c;
  if constexpr(Reverse){result.seed(bars[output]?bars[output][outAt]:T(0));trace->reverse();
   for(int k=0;k<trace->inputs;++k){auto l=trace->leaves[k];add(l.slot,l.at,trace->nodes[l.node].bar);}
  }else outputs[output][outAt]=result.d;
 }
 SPH_HD void row(int64_t row)const{
  const auto i=row+options.rowStart;
  if constexpr(Task==Limiter){for(int c=0;c<D+2;++c)limiter(row,i,c);}
  else if constexpr(Task==Flux){for(int64_t e=0;e<a.counts[row];++e)flux_pair(row,i,a.neighbors[a.offsets[row]+e]);}
  else{
   if constexpr(Task==Moments)pair(row,i,i,true);
   for(int64_t e=0;e<a.counts[row];++e)pair(row,i,a.neighbors[a.offsets[row]+e],false);
  }
 }
 SPH_HD void operator()(int64_t at)const{if(Reverse&&deterministic)for(int64_t r=0;r<rows;++r)row(r);else row(at);}
};
struct ProductRows {
 const ProductBuffers& b;int task;bool reverse;
 template<class T,int D,int Task,bool Reverse,class Launch>void typed(int64_t n,RowArgs<T,D>a,const Launch& launch)const{
  ProductRow<T,D,Task,Reverse> row{};row.a=a;row.rows=n;row.mask=b.mask;row.options=b.primal.options;row.deterministic=b.deterministic;
  for(int k=0;k<FieldCount;++k)if(k==Bad)row.bad=ptr<int32_t>(b.primal.values[k]);else if(k==Material)row.materials=ptr<int32_t>(b.primal.values[k]);else if(k!=Cell)row.f[k]=ptr<T>(b.primal.values[k]);
  for(int k=0;k<ProductSlots;++k){row.directions[k]=ptr<T>(b.directions[k]);row.gradients[k]=b.gradients[k].defined()?b.gradients[k].mutable_data_ptr<T>():nullptr;}
  for(int k=0;k<3;++k){row.bars[k]=ptr<T>(b.bars[k]);if(!Reverse&&b.primal.output[k].defined())row.outputs[k]=b.primal.output[k].mutable_data_ptr<T>();}
  launch(Reverse&&b.deterministic?1:n,row);
 }
 template<class T,int D,class Launch>void operator()(int64_t n,RowArgs<T,D>a,const Launch& launch)const{
  if(task==Moments){if(reverse)typed<T,D,Moments,true>(n,a,launch);else typed<T,D,Moments,false>(n,a,launch);}
  else if(task==Flux){if(reverse)typed<T,D,Flux,true>(n,a,launch);else typed<T,D,Flux,false>(n,a,launch);}
  else if(task==Limiter){if(reverse)typed<T,D,Limiter,true>(n,a,launch);else typed<T,D,Limiter,false>(n,a,launch);}
  else{if(reverse)typed<T,D,Gradients,true>(n,a,launch);else typed<T,D,Gradients,false>(n,a,launch);}
 }
};
void products_gpu(const Tensors&,const ProductBuffers&,int,int,bool);
}
