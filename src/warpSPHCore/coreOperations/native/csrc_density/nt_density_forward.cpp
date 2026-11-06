#include <torch/csrc/stable/library.h>
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "nt_density_forward.h"
namespace sph_native {
void density_forward_gpu(const Tensors &, int, int, int);
Tensor density_forward(const Tensors &inputs, int64_t support, int64_t direction, int64_t kernel) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    Tensors t = prepare_operator(inputs, support, direction);
    t[OUTPUT] = stable::new_empty(t[QX], {t[QX].size(0)});
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        density_forward_gpu(t, support, direction, kernel);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only; select "
                               "WARPSPHCORE_NATIVE_BUILD=gpu and rebuild");
#endif
    } else
        execute(DensityRows{}, t, support, direction, 6, CpuLauncher{}, kernel);
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("density_forward(Tensor[] inputs, int support, int direction, int kernel=0) -> Tensor");
    m.impl("density_forward", TORCH_BOX(&density_forward));
}
} // namespace sph_native

#include "nt_density_jvp.h"
namespace sph_native {
void density_product_gpu(const Tensors&,const density_jvp::Buffers&,int,int,int);
Tensors density_product(const OptionalTensors& inputs,const OptionalTensors& directions,const std::optional<Tensor>& cotangent,
        const Tensors& transpose,int64_t support,int64_t direction,int64_t kernel){
    STD_TORCH_CHECK(valid_kernel(kernel),"Unknown native kernel");
    Tensors borrowed;for(const auto& value:inputs)borrowed.push_back(value?*value:Tensor{});
    auto t=prepare_operator(borrowed,support,direction);const auto& x=t[QX];const auto n=x.size(0),m=t[RX].size(0),d=x.size(1);
    STD_TORCH_CHECK(directions.size()==5,"density JVP direction layout");
    for(int k=0;k<5;++k)if(directions[k]){tensor(*directions[k],x,x.scalar_type(),"density JVP direction");STD_TORCH_CHECK(directions[k]->sizes()==t[k].sizes(),"density JVP direction shape");}
    Tensors output;
    if(cotangent){
        vector(*cotangent,x,x.scalar_type(),n,"density JVP cotangent");STD_TORCH_CHECK(transpose.size()==3,"Density JVP reverse requires CSC");
        vector(transpose[0],x,ScalarType::Long,m,"CSC offsets");vector(transpose[1],x,ScalarType::Int,m,"CSC counts");vector(transpose[2],x,ScalarType::Long,t[NEIGHBORS].numel(),"CSC IDs");
        output={stable::new_zeros(x,{n,d}),stable::new_zeros(x,{m,d}),stable::new_zeros(x,{n}),stable::new_zeros(x,{m}),stable::new_zeros(x,{m}),
            stable::new_zeros(x,{n,d}),stable::new_zeros(x,{m,d}),stable::new_zeros(x,{n}),stable::new_zeros(x,{m}),stable::new_zeros(x,{m})};
    }else{output={stable::new_empty(x,{n})};t[OUTPUT]=output[0];}
    const density_jvp::Buffers buffers{directions,transpose,output,cotangent?*cotangent:Tensor{}};
    if(x.is_cuda()){
#ifdef WITH_CUDA
        density_product_gpu(t,buffers,support,direction,kernel);
#else
        STD_TORCH_CHECK(false,"Native extension was built for CPU only");
#endif
    }else execute(density_jvp::Rows{buffers},t,support,direction,6,CpuLauncher{},kernel);
    return output;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m){
    m.def("density_product(Tensor?[] inputs, Tensor?[] directions, Tensor? cotangent, Tensor[] transpose, int support, int direction, int kernel) -> Tensor[]");
    m.impl("density_product",TORCH_BOX(&density_product));
}
}
