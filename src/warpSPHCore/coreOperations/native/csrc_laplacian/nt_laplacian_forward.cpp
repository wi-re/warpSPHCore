#include "nt_laplacian_forward.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "nt_laplacian_operator.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.h"
namespace sph_native {
void laplacian_corrected_forward_gpu(const Tensors& t, const OptionalTensors& corrections, int support, int direction, int kernel, int scheme, int lap, bool dot);
void laplacian_forward_gpu(const Tensors &, int, int, int, int, int, bool);
Tensor laplacian_forward(const Tensors &inputs, int64_t support, int64_t direction, int64_t kernel,
                         int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_field_operator(inputs, support, direction, 5, scheme, lap, dot);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        laplacian_forward_gpu(t, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(LaplacianRows{}, t, support, direction, 5, CpuLauncher{}, kernel, scheme, lap, dot);
    return t[OUTPUT];
}
Tensor laplacian_corrected_forward(const OptionalTensors& inputs, const OptionalTensors& corrections,
        int64_t support, int64_t direction, int64_t kernel, int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_corrected_operator(inputs, corrections, support, direction, 5, scheme, lap, dot);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        laplacian_corrected_forward_gpu(t, corrections, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else {
        using Rows = neighborhood::CorrectedRows<neighborhood::CorrectedLaplacian,
            neighborhood::SerialCorrectionExecutor>;
        execute(Rows{corrections}, t, support, direction, 5, CpuLauncher{}, kernel, scheme, lap, dot);
    }
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("laplacian_corrected_forward(Tensor?[] inputs, Tensor?[] corrections, int support, int direction, "
          "int kernel=0, int scheme=3, int lap=2, bool dot=False) -> Tensor");
    m.impl("laplacian_corrected_forward", TORCH_BOX(&laplacian_corrected_forward));

    m.def("laplacian_forward(Tensor[] inputs, int support, int direction, int kernel=0, int "
          "scheme=3, int lap=2, bool dot=False) -> Tensor");
    m.impl("laplacian_forward", TORCH_BOX(&laplacian_forward));
}
} // namespace sph_native

#include "nt_laplacian_jvp.h"
namespace sph_native{
void laplacian_product_gpu(const Tensors&,const laplacian_jvp::Buffers&,int,int,int,int,int);
Tensors laplacian_product(const OptionalTensors& flat,const OptionalTensors& directions,const std::optional<Tensor>& cotangent,
        const Tensors& transpose,int64_t support,int64_t direction,int64_t kernel,int64_t scheme,int64_t lap,bool bounds_ad,const OptionalTensors& gates){
    STD_TORCH_CHECK(valid_kernel(kernel),"Unknown native kernel");STD_TORCH_CHECK(flat.size()==26&&directions.size()==26,"Laplacian JVP input layout");
    OptionalTensors inputs(flat.begin(),flat.begin()+17),corrections(flat.begin()+17,flat.end());
    auto t=prepare_corrected_operator(inputs,corrections,support,direction,5,scheme,lap,false,false,cotangent?&*cotangent:nullptr);
    const auto& x=t[QX];const auto m=t[RX].size(0),d=x.size(1);
    STD_TORCH_CHECK(field_components(t)>0&&field_components(t)<=d*d,"Laplacian JVP field block");
    for(int k=0;k<26;++k)if(directions[k]){
        STD_TORCH_CHECK(flat[k],"Laplacian JVP tangent requires a primal");tensor(*directions[k],x,x.scalar_type(),"Laplacian JVP tangent");
        STD_TORCH_CHECK(directions[k]->sizes()==flat[k]->sizes(),"Laplacian JVP tangent shape");
    }
    STD_TORCH_CHECK(gates.empty()||gates.size()==2,"Laplacian JVP gate layout");
    if(!gates.empty())for(int k=0;k<2;++k){STD_TORCH_CHECK(gates[k]&&flat[13+k],"Laplacian JVP gate fields");
        tensor(*gates[k],x,x.scalar_type(),"Laplacian JVP gate");STD_TORCH_CHECK(gates[k]->sizes()==flat[13+k]->sizes(),"Laplacian JVP gate shape");}
    Tensors output;
    if(cotangent){
        STD_TORCH_CHECK(transpose.size()==3,"Laplacian JVP reverse requires CSC");
        vector(transpose[0],x,ScalarType::Long,m,"CSC offsets");vector(transpose[1],x,ScalarType::Int,m,"CSC counts");vector(transpose[2],x,ScalarType::Long,t[NEIGHBORS].numel(),"CSC IDs");
        for(int k=0;k<52;++k){const auto& value=k<26?flat[k]:directions[k-26];
            std::vector<int64_t> shape;
            if(value&&(value->scalar_type()==ScalarType::Float||value->scalar_type()==ScalarType::Double)){if(bounds_ad&&(k%26==7||k%26==8))shape={x.size(0),d};else for(int a=0;a<value->dim();++a)shape.push_back(value->size(a));}
            else shape.push_back(0);
            output.push_back(stable::new_zeros(x,shape));
        }
    }else output={t[OUTPUT]};
    const laplacian_jvp::Buffers b{flat,directions,transpose,output,cotangent?*cotangent:Tensor{},false,bounds_ad,gates};
    if(x.is_cuda()){
#ifdef WITH_CUDA
        laplacian_product_gpu(t,b,support,direction,kernel,scheme,lap);
#else
        STD_TORCH_CHECK(false,"Native extension was built for CPU only");
#endif
    }else execute(laplacian_jvp::Rows{b},t,support,direction,5,CpuLauncher{},kernel,scheme,lap);
    return output;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m){
    m.def("laplacian_product(Tensor?[] flat, Tensor?[] directions, Tensor? cotangent, Tensor[] transpose, int support, int direction, int kernel, int scheme, int lap, bool bounds_ad, Tensor?[] gates) -> Tensor[]");
    m.impl("laplacian_product",TORCH_BOX(&laplacian_product));
}
}
