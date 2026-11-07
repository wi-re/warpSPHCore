#include "nt_gradient_forward.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.h"
namespace sph_native {
void gradient_corrected_forward_gpu(const Tensors &, const OptionalTensors &, int, int, int, int);
void gradient_forward_gpu(const Tensors &, int, int, int, int, int, bool);
Tensor gradient_forward(const Tensors &inputs, int64_t support, int64_t direction, int64_t kernel,
                        int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_field_operator(inputs, support, direction, 2, scheme, lap, dot);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        gradient_forward_gpu(t, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(GradientRows{}, t, support, direction, 2, CpuLauncher{}, kernel, scheme, lap, dot);
    return t[OUTPUT];
}
Tensor gradient_corrected_forward(const OptionalTensors& inputs,
        const OptionalTensors& corrections, int64_t support, int64_t direction,
        int64_t kernel, int64_t scheme) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_corrected_operator(inputs, corrections, support, direction, 2, scheme);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        gradient_corrected_forward_gpu(t, corrections, support, direction, kernel, scheme);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else {
        using Rows = neighborhood::CorrectedRows<neighborhood::CorrectedGradient,
                                                 neighborhood::SerialCorrectionExecutor>;
        execute(Rows{corrections}, t, support, direction, 2, CpuLauncher{}, kernel, scheme);
    }
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("gradient_corrected_forward(Tensor?[] inputs, Tensor?[] corrections, int support, "
          "int direction, int kernel=0, int scheme=3) -> Tensor");
    m.impl("gradient_corrected_forward", TORCH_BOX(&gradient_corrected_forward));
    m.def("gradient_forward(Tensor[] inputs, int support, int direction, int kernel=0, int "
          "scheme=3, int lap=2, bool dot=False) -> Tensor");
    m.impl("gradient_forward", TORCH_BOX(&gradient_forward));
}
} // namespace sph_native

#include "nt_gradient_jvp.h"
namespace sph_native{
void gradient_product_gpu(const Tensors&,const gradient_jvp::Buffers&,int,int,int,int);
Tensors gradient_product(const OptionalTensors& flat,const OptionalTensors& directions,const std::optional<Tensor>& cotangent,
        const Tensors& transpose,int64_t support,int64_t direction,int64_t kernel,int64_t scheme){
    STD_TORCH_CHECK(valid_kernel(kernel),"Unknown native kernel");STD_TORCH_CHECK(flat.size()==26&&directions.size()==26,"Gradient JVP input layout");
    OptionalTensors inputs(flat.begin(),flat.begin()+17),corrections(flat.begin()+17,flat.end());
    auto t=prepare_corrected_operator(inputs,corrections,support,direction,2,scheme,2,false,false,cotangent?&*cotangent:nullptr);
    const auto& x=t[QX];const auto m=t[RX].size(0),d=x.size(1);
    STD_TORCH_CHECK(t[RVALUE].dim()==2&&t[RVALUE].size(1)>0&&t[RVALUE].size(1)<=d*d,"Gradient JVP field block");
    for(int k=0;k<26;++k)if(directions[k]){
        STD_TORCH_CHECK(flat[k],"Gradient JVP tangent requires a primal");tensor(*directions[k],x,x.scalar_type(),"Gradient JVP tangent");
        STD_TORCH_CHECK(directions[k]->sizes()==flat[k]->sizes(),"Gradient JVP tangent shape");
    }
    Tensors output;
    if(cotangent){
        STD_TORCH_CHECK(transpose.size()==3,"Gradient JVP reverse requires CSC");
        vector(transpose[0],x,ScalarType::Long,m,"CSC offsets");vector(transpose[1],x,ScalarType::Int,m,"CSC counts");vector(transpose[2],x,ScalarType::Long,t[NEIGHBORS].numel(),"CSC IDs");
        for(int k=0;k<52;++k){const auto& value=k<26?flat[k]:directions[k-26];
            std::vector<int64_t> shape;
            if(value&&(value->scalar_type()==ScalarType::Float||value->scalar_type()==ScalarType::Double)){for(int a=0;a<value->dim();++a)shape.push_back(value->size(a));}
            else shape.push_back(0);
            output.push_back(stable::new_zeros(x,shape));
        }
    }else output={t[OUTPUT]};
    const gradient_jvp::Buffers b{flat,directions,transpose,output,cotangent?*cotangent:Tensor{}};
    if(x.is_cuda()){
#ifdef WITH_CUDA
        gradient_product_gpu(t,b,support,direction,kernel,scheme);
#else
        STD_TORCH_CHECK(false,"Native extension was built for CPU only");
#endif
    }else execute(gradient_jvp::Rows{b},t,support,direction,2,CpuLauncher{},kernel,scheme);
    return output;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m){
    m.def("gradient_product(Tensor?[] flat, Tensor?[] directions, Tensor? cotangent, Tensor[] transpose, int support, int direction, int kernel, int scheme) -> Tensor[]");
    m.impl("gradient_product",TORCH_BOX(&gradient_product));
}
}
