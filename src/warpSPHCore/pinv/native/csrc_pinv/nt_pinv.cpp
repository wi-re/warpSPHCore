#include "nt_pinv_rows.h"
#include "torch/validation.h"
#include "execution/cpu.h"
namespace sph_native::dense {
std::tuple<Tensor,std::optional<Tensor>> inverse(const Tensor& matrix,const std::optional<Tensor>& values,
                                const std::optional<Tensor>& counts,int64_t mode,double rtol) {
    STD_TORCH_CHECK(matrix.device().is_cpu() || matrix.is_cuda(),"Dense kernels require CPU/CUDA/HIP");
    STD_TORCH_CHECK(matrix.scalar_type()==ScalarType::Float || matrix.scalar_type()==ScalarType::Double,"Dense kernels require float32/float64");
    tensor(matrix,matrix,matrix.scalar_type(),"matrices");
    STD_TORCH_CHECK(matrix.dim()==3 && matrix.size(1)>=1 && matrix.size(1)<=3 && matrix.size(1)==matrix.size(2),"Expected (n,d,d), d=1,2,3");
    STD_TORCH_CHECK(mode>=0 && mode<=2 && rtol>=0 && std::isfinite(rtol),"Invalid dense policy");
    const auto n=matrix.size(0),d=matrix.size(1);
    if (mode==1) {
        STD_TORCH_CHECK(d<=2,"Renormalization policy is specialised for 1D/2D");
        if (d==2) { STD_TORCH_CHECK(counts.has_value(),"2D renormalization needs counts"); vector(*counts,matrix,ScalarType::Int,n,"neighbor counts"); }
    }
    if (mode==2) {
        STD_TORCH_CHECK(values.has_value(),"Fused inverse application needs values");
        tensor(*values,matrix,matrix.scalar_type(),"values");
        STD_TORCH_CHECK(values->dim()==3 && values->size(0)==n && values->size(2)==d,"Values must have shape (n,p,d)");
    }
    auto output=mode==2 ? stable::empty_like(*values):stable::empty_like(matrix);
    auto eigenvalues=mode==1 ? stable::new_empty(matrix,{n,d}):Tensor{};
    Buffers buffers{matrix,values ? *values:Tensor{},counts ? *counts:Tensor{},output,eigenvalues};
    if (matrix.is_cuda()) {
#ifdef WITH_CUDA
        inverse_gpu(buffers,int(mode),rtol);
#else
        STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
    } else execute(buffers,int(mode),rtol,CpuLauncher{});
    return {output,mode==1 ? std::optional<Tensor>(eigenvalues):std::nullopt};
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("small_inverse(Tensor matrix, Tensor? values, Tensor? counts, int mode, float rtol) -> (Tensor, Tensor?)");
    m.impl("small_inverse",TORCH_BOX(&inverse));
}
} // namespace sph_native::dense
