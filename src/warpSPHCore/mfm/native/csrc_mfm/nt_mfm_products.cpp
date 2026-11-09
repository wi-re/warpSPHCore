#include "nt_mfm_prepare.h"
#include "nt_mfm_products.h"
#include "execution/cpu.h"
namespace sph_native::mfm {
OptionalTensors products(const OptionalTensors& geometry,const OptionalTensors& fields,const std::vector<double>& scalars,const std::vector<int64_t>& integers,int64_t task,int64_t kernel,const OptionalTensors& bars,const OptionalTensors& directions,int64_t mask,bool reverse,bool deterministic){
 ProductBuffers b{};auto t=prepare_mfm(geometry,fields,scalars,integers,task,b.primal);const auto& x=t[QX];const auto n=x.size(0),d=x.size(1);
 STD_TORCH_CHECK((task==Moments||task==Gradients||task==Limiter||task==Flux)&&bars.size()==3&&directions.size()==ProductSlots&&mask>=0&&mask<(int64_t(1)<<ProductSlots),"Unsupported meshless product layout/pass");
 b.mask=mask;b.deterministic=deterministic;
 for(int k=0;k<3;++k){b.bars.push_back(bars[k]?*bars[k]:Tensor{});if(bars[k]){
  STD_TORCH_CHECK(task!=Moments||k!=2,"Meshless neighbor counts are nondifferentiable");
  tensor(*bars[k],x,x.scalar_type(),"meshless cotangent");const std::vector<int64_t> shape=(task==Limiter||task==Flux)?(k==1?std::vector<int64_t>{n,d}:std::vector<int64_t>{n}):k==1?std::vector<int64_t>{n,d,d}:task==Moments?std::vector<int64_t>{n}:std::vector<int64_t>{n,d};
  STD_TORCH_CHECK(bars[k]->sizes().vec()==shape,"Meshless cotangent shape mismatch");}}
 Tensors flat(t.begin(),t.begin()+OUTPUT);flat.insert(flat.end(),b.primal.values.begin(),b.primal.values.end());OptionalTensors result;
 for(int k=0;k<ProductSlots;++k){b.directions.push_back(directions[k]?*directions[k]:Tensor{});b.gradients.push_back(Tensor{});
  if(directions[k]||(mask&(int64_t(1)<<k)))STD_TORCH_CHECK(flat[k].defined()&&flat[k].scalar_type()==x.scalar_type(),"Meshless AD requires a floating slot");
  if(directions[k]){tensor(*directions[k],x,x.scalar_type(),"meshless direction");STD_TORCH_CHECK(directions[k]->sizes()==flat[k].sizes(),"Meshless direction shape mismatch");}
  if(mask&(int64_t(1)<<k))b.gradients[k]=stable::new_zeros(flat[k],flat[k].sizes());result.push_back(b.gradients[k].defined()?std::optional<Tensor>(b.gradients[k]):std::nullopt);
 }
 if(!reverse && (task==Limiter||task==Flux))b.primal.output={stable::new_zeros(x,{n}),stable::new_zeros(x,{n,d}),stable::new_zeros(x,{n})};
 else if(!reverse){b.primal.output={stable::new_zeros(x,task==Moments?std::vector<int64_t>{n}:std::vector<int64_t>{n,d}),stable::new_zeros(x,{n,d,d}),task==Moments?Tensor{}:stable::new_zeros(x,{n,d})};}
 if(x.is_cuda()){
#ifdef WITH_CUDA
 products_gpu(t,b,int(task),int(kernel),reverse);
#else
 STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
 }else execute(ProductRows{b,int(task),reverse},t,11,0,1,CpuLauncher{},int(kernel));
 if(!reverse)return {b.primal.output[0],b.primal.output[1],task==Moments?std::nullopt:std::optional<Tensor>(b.primal.output[2])};return result;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m){m.def("meshless_products(Tensor?[] geometry, Tensor?[] fields, float[] scalars, int[] integers, int task, int kernel, Tensor?[] bars, Tensor?[] directions, int mask, bool reverse, bool deterministic) -> Tensor?[]");m.impl("meshless_products",TORCH_BOX(&products));}
}
