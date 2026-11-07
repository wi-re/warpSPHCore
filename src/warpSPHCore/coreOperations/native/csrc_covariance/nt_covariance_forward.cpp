#include "nt_covariance_forward.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "nt_covariance_operator.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.h"
#include "nt_covariance_torch.h"
namespace sph_native {
void covariance_corrected_forward_gpu(const Tensors& t, const OptionalTensors& corrections, int support, int direction, int kernel, int scheme, int lap, bool dot, const Tensor& counts);
void covariance_forward_gpu(const Tensors &, int, int, int, int, int, bool);
Tensor covariance_forward(const Tensors &inputs, int64_t support, int64_t direction, int64_t kernel,
                          int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_field_operator(inputs, support, direction, 7, scheme, lap, dot);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        covariance_forward_gpu(t, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(CovarianceRows{}, t, support, direction, 7, CpuLauncher{}, kernel, scheme, lap,
                dot);
    return t[OUTPUT];
}
Tensors covariance_corrected_forward(const OptionalTensors& inputs, const OptionalTensors& corrections,
        int64_t support, int64_t direction, int64_t kernel, int64_t scheme, int64_t lap, bool dot, bool return_counts) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_corrected_operator(inputs, corrections, support, direction, 7, scheme, lap, dot);
    const Tensor counts = return_counts ? stable::new_empty(t[COUNTS], {t[QX].size(0)}) : Tensor{};
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        covariance_corrected_forward_gpu(t, corrections, support, direction, kernel, scheme, lap, dot, counts);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else {
        using Rows = neighborhood::CorrectedRows<neighborhood::CorrectedCovariance,
            neighborhood::SerialCorrectionExecutor, neighborhood::CovarianceCorrectionOptions>;
        execute(Rows{corrections, {counts}}, t, support, direction, 7, CpuLauncher{}, kernel, scheme, lap, dot);
    }
    return return_counts ? Tensors{t[OUTPUT], counts} : Tensors{t[OUTPUT]};
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("covariance_corrected_forward(Tensor?[] inputs, Tensor?[] corrections, int support, int direction, "
          "int kernel=0, int scheme=3, int lap=2, bool dot=False, bool return_counts=False) -> Tensor[]");
    m.impl("covariance_corrected_forward", TORCH_BOX(&covariance_corrected_forward));

    m.def("covariance_forward(Tensor[] inputs, int support, int direction, int kernel=0, int "
          "scheme=3, int lap=2, bool dot=False) -> Tensor");
    m.impl("covariance_forward", TORCH_BOX(&covariance_forward));
}
} // namespace sph_native

#include "nt_covariance_jvp.h"
namespace sph_native{
void covariance_product_gpu(const Tensors&,const covariance_jvp::Buffers&,int,int,int,int);
Tensors covariance_product(const OptionalTensors& flat,const OptionalTensors& directions,const std::optional<Tensor>& cotangent,
        const Tensors& transpose,int64_t support,int64_t direction,int64_t kernel,int64_t scheme){
    STD_TORCH_CHECK(valid_kernel(kernel),"Unknown native kernel");STD_TORCH_CHECK(flat.size()==26&&directions.size()==26,"Covariance JVP input layout");
    OptionalTensors inputs(flat.begin(),flat.begin()+17),corrections(flat.begin()+17,flat.end());
    auto t=prepare_corrected_operator(inputs,corrections,support,direction,7,scheme,2,false,false,cotangent?&*cotangent:nullptr);
    const auto& x=t[QX];const auto m=t[RX].size(0),d=x.size(1);
    for(int k=0;k<26;++k)if(directions[k]){
        STD_TORCH_CHECK(flat[k],"Covariance JVP tangent requires a primal");tensor(*directions[k],x,x.scalar_type(),"Covariance JVP tangent");
        STD_TORCH_CHECK(directions[k]->sizes()==flat[k]->sizes(),"Covariance JVP tangent shape");
    }
    Tensors output;
    if(cotangent){
        STD_TORCH_CHECK(transpose.size()==3,"Covariance JVP reverse requires CSC");
        vector(transpose[0],x,ScalarType::Long,m,"CSC offsets");vector(transpose[1],x,ScalarType::Int,m,"CSC counts");vector(transpose[2],x,ScalarType::Long,t[NEIGHBORS].numel(),"CSC IDs");
        for(int k=0;k<52;++k){const auto& value=k<26?flat[k]:directions[k-26];
            std::vector<int64_t> shape;
            if(value&&(value->scalar_type()==ScalarType::Float||value->scalar_type()==ScalarType::Double)){for(int a=0;a<value->dim();++a)shape.push_back(value->size(a));}
            else shape.push_back(0);
            output.push_back(stable::new_zeros(x,shape));
        }
    }else output={t[OUTPUT]};
    const covariance_jvp::Buffers b{flat,directions,transpose,output,cotangent?*cotangent:Tensor{}};
    if(x.is_cuda()){
#ifdef WITH_CUDA
        covariance_product_gpu(t,b,support,direction,kernel,scheme);
#else
        STD_TORCH_CHECK(false,"Native extension was built for CPU only");
#endif
    }else execute(covariance_jvp::Rows{b},t,support,direction,7,CpuLauncher{},kernel,scheme);
    return output;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m){
    m.def("covariance_product(Tensor?[] flat, Tensor?[] directions, Tensor? cotangent, Tensor[] transpose, int support, int direction, int kernel, int scheme) -> Tensor[]");
    m.impl("covariance_product",TORCH_BOX(&covariance_product));
}
}
