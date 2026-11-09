#include "nt_mfm_prepare.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include <limits>
namespace sph_native::mfm {
Tensors evaluate(const OptionalTensors& inputs,const OptionalTensors& values,const std::vector<double>& scalars,
                 const std::vector<int64_t>& integers,int64_t task,int64_t kernel) {
    Buffers b{};auto t=prepare_mfm(inputs,values,scalars,integers,task,b);
    const auto& x=t[QX];const int64_t n=x.size(0),d=x.size(1);
    auto scalar=[&](){return stable::new_empty(x,{n});};
    auto vector=[&](){return stable::new_empty(x,{n,d});};
    auto matrix=[&](){return stable::new_empty(x,{n,d,d});};
    switch(task) {
    case Moments:b.output={scalar(),matrix(),stable::new_empty(x,{n},ScalarType::Int)};break;
    case Gradients:b.output={vector(),matrix(),vector()};break;
    case Limiter:b.output={scalar(),vector(),scalar()};break;
    case Residual:case Matvec:b.output={vector(),scalar()};break;
    case Weights:b.output={stable::new_zeros(x,{t[NEIGHBORS].numel()})};break;
    case Coarse:b.output={stable::new_empty(x,{n,d==1 ? 3:d==2 ? 9:27})};break;
    case Flux:b.output={scalar(),vector(),scalar()};break;
    case Timestep:b.output={scalar()};break;
    }
    if(x.is_cuda()) {
#ifdef WITH_CUDA
        run_gpu(t,b,int(task),int(kernel));
#else
        STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
    } else run(t,b,int(task),int(kernel),CpuLauncher{});
    return b.output;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("meshless(Tensor?[] inputs, Tensor?[] values, float[] scalars, int[] integers, int task, int kernel) -> Tensor[]");
    m.impl("meshless",TORCH_BOX(&evaluate));
}
} // namespace sph_native::mfm
