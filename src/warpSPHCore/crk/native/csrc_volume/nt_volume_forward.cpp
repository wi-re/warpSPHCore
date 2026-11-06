#include "crk/native/csrc_preparation/nt_preparation_torch.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "execution/rowNeighborhood.h"
#include "nt_volume_operator.h"

namespace sph_native {
void crk_volume_forward_gpu(const Tensors&, int, int, int);
Tensor crk_volume_forward(const OptionalTensors& inputs, int64_t support, int64_t direction, int64_t kernel) {
    auto t = crk::prepare(inputs, support, direction, kernel);
    t[OUTPUT] = stable::new_empty(t[QX], {t[QX].size(0)});
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        crk_volume_forward_gpu(t, support, direction, kernel);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else {
        using Rows = neighborhood::SerialSphRows<neighborhood::ApparentVolume, neighborhood::Weight::None>;
        execute(Rows{}, t, support, direction, 6, CpuLauncher{}, kernel);
    }
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("crk_volume_forward(Tensor?[] inputs, int support, int direction, int kernel) -> Tensor");
    m.impl("crk_volume_forward", TORCH_BOX(&crk_volume_forward));
}
} // namespace sph_native

#include "nt_volume_jvp.h"
namespace sph_native {
void crk_volume_jvp_vjp_gpu(const Tensors&,const crk_volume_jvp::Buffers&,int,int);
Tensors crk_volume_product(const OptionalTensors& inputs,const OptionalTensors& directions,
        const Tensor& volumes,const std::optional<Tensor>& cotangent,const Tensors& transpose,int64_t direction,int64_t kernel) {
    auto t=crk::prepare(inputs,11,direction,kernel);
    const auto& x=t[QX];const auto n=x.size(0),m=t[RX].size(0),d=x.size(1);
    vector(volumes,x,x.scalar_type(),n,"apparent volumes");if(cotangent)vector(*cotangent,x,x.scalar_type(),n,"volume JVP cotangent");
    STD_TORCH_CHECK(directions.size()==4,"volume JVP direction layout");
    for(int slot=0;slot<4;++slot)if(directions[slot]){
        tensor(*directions[slot],x,x.scalar_type(),"volume JVP direction");
        STD_TORCH_CHECK(directions[slot]->sizes()==t[slot].sizes(),"volume JVP direction shape");
    }
    if(cotangent){
    STD_TORCH_CHECK(transpose.size()==3,"Volume JVP pullback requires CSC");
    vector(transpose[0],x,ScalarType::Long,m,"CSC offsets");vector(transpose[1],x,ScalarType::Int,m,"CSC counts");
    vector(transpose[2],x,ScalarType::Long,t[NEIGHBORS].numel(),"CSC IDs");
    }
    Tensors gradients;
    if(cotangent)gradients={stable::new_zeros(x,{n,d}),stable::new_zeros(x,{m,d}),stable::new_zeros(x,{n}),stable::new_zeros(x,{m}),
        stable::new_zeros(x,{n,d}),stable::new_zeros(x,{m,d}),stable::new_zeros(x,{n}),stable::new_zeros(x,{m}),stable::new_zeros(x,{n})};
    if(!cotangent){gradients={stable::new_empty(x,{n})};t[OUTPUT]=gradients[0];}
    const crk_volume_jvp::Buffers b{directions,transpose,gradients,cotangent?*cotangent:Tensor{},volumes};
    if(x.is_cuda()){
#ifdef WITH_CUDA
        crk_volume_jvp_vjp_gpu(t,b,direction,kernel);
#else
        STD_TORCH_CHECK(false,"Native extension was built for CPU only");
#endif
    }else execute(crk_volume_jvp::Rows{b},t,11,direction,8,CpuLauncher{},kernel);
    return gradients;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m){
    m.def("crk_volume_product(Tensor?[] inputs, Tensor?[] directions, Tensor volumes, Tensor? cotangent, Tensor[] transpose, int direction, int kernel) -> Tensor[]");
    m.impl("crk_volume_product",TORCH_BOX(&crk_volume_product));
}
}
