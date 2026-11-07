#include "nt_interpolate_forward.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.h"
namespace sph_native {
void interpolate_corrected_forward_gpu(const Tensors &, const OptionalTensors &, int, int, int, int);
void interpolate_forward_gpu(const Tensors &, int, int, int, int, int, bool);
Tensor interpolate_forward(const Tensors &inputs, int64_t support, int64_t direction,
                           int64_t kernel, int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_field_operator(inputs, support, direction, 1, scheme, lap, dot);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        interpolate_forward_gpu(t, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(InterpolateRows{}, t, support, direction, 1, CpuLauncher{}, kernel, scheme, lap,
                dot);
    return t[OUTPUT];
}
Tensor interpolate_corrected_forward(const OptionalTensors& inputs,
        const OptionalTensors& corrections, int64_t support, int64_t direction,
        int64_t kernel, int64_t scheme) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_corrected_operator(inputs, corrections, support, direction, 1, scheme);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        interpolate_corrected_forward_gpu(t, corrections, support, direction, kernel, scheme);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else {
        using Rows = neighborhood::CorrectedRows<neighborhood::CorrectedInterpolate,
                                                 neighborhood::SerialCorrectionExecutor>;
        execute(Rows{corrections}, t, support, direction, 1, CpuLauncher{}, kernel, scheme);
    }
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("interpolate_corrected_forward(Tensor?[] inputs, Tensor?[] corrections, int support, "
          "int direction, int kernel=0, int scheme=3) -> Tensor");
    m.impl("interpolate_corrected_forward", TORCH_BOX(&interpolate_corrected_forward));
    m.def("interpolate_forward(Tensor[] inputs, int support, int direction, int kernel=0, int "
          "scheme=3, int lap=2, bool dot=False) -> Tensor");
    m.impl("interpolate_forward", TORCH_BOX(&interpolate_forward));
}
} // namespace sph_native

#include "nt_interpolate_jvp.h"
namespace sph_native{
void interpolate_product_gpu(const Tensors&,const interpolate_jvp::Buffers&,int,int,int);
Tensors interpolate_product(const OptionalTensors& flat,const OptionalTensors& directions,const std::optional<Tensor>& cotangent,
        const Tensors& transpose,int64_t support,int64_t direction,int64_t kernel){
    STD_TORCH_CHECK(valid_kernel(kernel),"Unknown native kernel");STD_TORCH_CHECK(flat.size()==26&&directions.size()==26,"Interpolation JVP input layout");
    OptionalTensors inputs(flat.begin(),flat.begin()+17),corrections(flat.begin()+17,flat.end());
    auto t=prepare_corrected_operator(inputs,corrections,support,direction,1,1,2,false,false,cotangent?&*cotangent:nullptr);
    const auto& x=t[QX];const auto n=x.size(0),m=t[RX].size(0),d=x.size(1),f=t[RVALUE].size(1);
    STD_TORCH_CHECK(t[RVALUE].dim()==2&&f>0&&f<=d*d,"Interpolation JVP field block");
    for(int k=0;k<26;++k)if(directions[k]){
        STD_TORCH_CHECK(flat[k],"Interpolation JVP tangent requires a primal");tensor(*directions[k],x,x.scalar_type(),"Interpolation JVP tangent");
        STD_TORCH_CHECK(directions[k]->sizes()==flat[k]->sizes(),"Interpolation JVP tangent shape");
    }
    Tensors output;
    if(cotangent){
        STD_TORCH_CHECK(transpose.size()==3,"Interpolation JVP reverse requires CSC");
        vector(transpose[0],x,ScalarType::Long,m,"CSC offsets");vector(transpose[1],x,ScalarType::Int,m,"CSC counts");vector(transpose[2],x,ScalarType::Long,t[NEIGHBORS].numel(),"CSC IDs");
        for(int repeat=0;repeat<2;++repeat){
            output.push_back(stable::new_zeros(x,{n,d}));output.push_back(stable::new_zeros(x,{m,d}));
            output.push_back(stable::new_zeros(x,{n}));output.push_back(stable::new_zeros(x,{m}));
            output.push_back(stable::new_zeros(x,{m}));output.push_back(stable::new_zeros(x,{m}));
            output.push_back(stable::new_zeros(x,{m,f}));output.push_back(stable::new_zeros(x,{m}));
            output.push_back(stable::new_zeros(x,{n}));output.push_back(stable::new_zeros(x,{n,d}));
        }
    }else output={t[OUTPUT]};
    const interpolate_jvp::Buffers b{flat,directions,transpose,output,cotangent?*cotangent:Tensor{}};
    if(x.is_cuda()){
#ifdef WITH_CUDA
        interpolate_product_gpu(t,b,support,direction,kernel);
#else
        STD_TORCH_CHECK(false,"Native extension was built for CPU only");
#endif
    }else execute(interpolate_jvp::Rows{b},t,support,direction,1,CpuLauncher{},kernel);
    return output;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m){
    m.def("interpolate_product(Tensor?[] flat, Tensor?[] directions, Tensor? cotangent, Tensor[] transpose, int support, int direction, int kernel) -> Tensor[]");
    m.impl("interpolate_product",TORCH_BOX(&interpolate_product));
}
}
