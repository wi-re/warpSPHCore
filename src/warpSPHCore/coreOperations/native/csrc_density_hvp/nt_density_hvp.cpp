#include "nt_density_hvp.h"
#include "torch/validation.h"
#include "execution/cpu.h"
namespace sph_native {
void density_hvp_gpu(const Tensors&,const density_hvp::Buffers&,int,int,int);
Tensors density_position_hvp(const OptionalTensors& inputs,const Tensors& transpose,
        const std::optional<Tensor>& cotangent,int64_t support,int64_t direction,int64_t kernel,int64_t row_start) {
    STD_TORCH_CHECK(valid_kernel(kernel),"Unknown native kernel");
    Tensors borrowed; for (const auto& value:inputs) borrowed.push_back(value ? *value:Tensor{});
    auto t=prepare_operator(borrowed,support,direction);
    const auto& x=t[QX]; const auto n=x.size(0),nr=t[RX].size(0),d=x.size(1);
    STD_TORCH_CHECK(row_start>=0,"row start must be nonnegative");
    auto field=[&](const Tensor& value,int64_t rows) {
        tensor(value,x,x.scalar_type(),"HVP vector");
        STD_TORCH_CHECK(value.dim()==2 && value.size(0)==rows && value.size(1)==d,"HVP vector shape");
    };
    field(t[QVALUE],n);
    if (t[RVALUE].defined()) field(t[RVALUE],nr);
    Tensors result;
    if (cotangent) {
        field(*cotangent,n);
        STD_TORCH_CHECK(transpose.size()==3,"HVP pullback requires CSC");
        vector(transpose[0],x,ScalarType::Long,nr,"CSC offsets");
        vector(transpose[1],x,ScalarType::Int,nr,"CSC counts");
        vector(transpose[2],x,ScalarType::Long,t[NEIGHBORS].numel(),"CSC IDs");
        result={stable::new_empty(x,{n,d}),stable::new_empty(x,{nr,d}),stable::new_empty(x,{nr})};
    } else result={stable::new_empty(x,{n,d})};
    t[OUTPUT]=result[0];
    const density_hvp::Buffers buffers{cotangent ? *cotangent:Tensor{},transpose,cotangent ? result:Tensors{},row_start};
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        density_hvp_gpu(t,buffers,support,direction,kernel);
#else
        STD_TORCH_CHECK(false,"Native extension was built for CPU only");
#endif
    } else execute(density_hvp::Rows{buffers},t,support,direction,6,CpuLauncher{},kernel);
    return result;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("density_position_hvp(Tensor?[] inputs, Tensor[] transpose, Tensor? cotangent, int support, int direction, int kernel, int row_start) -> Tensor[]");
    m.impl("density_position_hvp",TORCH_BOX(&density_position_hvp));
}
} // namespace sph_native
