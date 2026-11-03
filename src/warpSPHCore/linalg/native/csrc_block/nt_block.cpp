#include "nt_block_rows.h"
#include "torch/validation.h"
#include "execution/cpu.h"
namespace sph_native::block {
void apply(const Tensor& inverse,const Tensor& values,const Tensor& output,bool transpose) {
    STD_TORCH_CHECK(values.scalar_type()==ScalarType::Float || values.scalar_type()==ScalarType::Double,"Block application requires float32/float64");
    STD_TORCH_CHECK(values.device().is_cpu() || values.is_cuda(),"Block application requires CPU/CUDA/HIP");
    tensor(inverse,values,values.scalar_type(),"block inverse");
    STD_TORCH_CHECK(output.device()==values.device() && output.scalar_type()==values.scalar_type(),"Block vectors must share dtype/device");
    STD_TORCH_CHECK(inverse.dim()==3 && inverse.size(1)==inverse.size(2) && inverse.size(1)>=1 && inverse.size(1)<=3,"Expected (n,d,d) blocks, d<=3");
    STD_TORCH_CHECK(values.dim()==1 && output.dim()==1 && values.numel()==inverse.size(0)*inverse.size(1) && output.numel()==values.numel(),"Block application shape mismatch");
    if (values.is_cuda()) {
#ifdef WITH_CUDA
        apply_gpu(inverse,values,output,transpose);
#else
        STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
    } else execute(inverse,values,output,transpose,CpuLauncher{});
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("block_apply(Tensor inverse, Tensor values, Tensor(a!) output, bool transpose) -> ()");
    m.impl("block_apply",TORCH_BOX(&apply));
}
} // namespace sph_native::block
