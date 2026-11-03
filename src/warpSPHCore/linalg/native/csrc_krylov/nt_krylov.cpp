#include "nt_krylov_rows.h"
#include "torch/validation.h"
#include "execution/cpu.h"
namespace sph_native::krylov {
void update(const Tensors& arrays,int64_t task,int64_t size,double threshold,bool hasThreshold,int64_t dimension,bool history) {
    constexpr int count[] = {7, 9, 8, 5, 3, 5, 6, 7, 8, 4, 3, 10, 10, 10, 6, 5, 3};
    constexpr const char* kinds[] = {"ffifbbb", "ffffffffi", "ffffiifb", "ffffi", "fff", "ffffi", "ffffib", "ffffffi", "fffffffi", "ffii", "fff", "ffffffffii", "ffffffffii", "ffffffffii", "ffiifb", "ffffi", "fff"};
    constexpr int minima[][ 10] = {
        {1, 1, 5, 1, 1, 1, 1},
        {-1, -1, -1, -1, 1, 1, -1, -1, 2},
        {1, 1, 1, 1, 2, 5, 1, 1},
        {-1, -1, -1, -1, 5},
        {-1, -1, -1},
        {-1, -1, 1, 1, 5},
        {1, 1, 1, 1, 9, 1},
        {1, 1, 1, 1, 1, 1, 9},
        {-1, -1, -1, -1, 1, -1, -1, 3},
        {1, 1, 3, 9},
        {1, 1, 1},
        {-1, -1, -1, -1, -1, -1, 1, 1, 3, 9},
        {1, 1, 1, 1, 1, 1, 1, 1, 3, 9},
        {-1, -1, -1, -1, -1, -1, 1, 1, 3, 9},
        {1, 1, 3, 9, 1, 1},
        {-1, -1, -1, -1, 9},
        {-1, -1, -1},
    };
    STD_TORCH_CHECK(task>=0 && task<int(sizeof(count)/sizeof(count[0])),"Unknown Krylov update");
    STD_TORCH_CHECK(arrays.size()==size_t(count[task]) && size>=0 && dimension>0,"Invalid Krylov command layout");
    const auto& example=arrays[0];
    STD_TORCH_CHECK(example.device().is_cpu() || example.is_cuda(),"Krylov updates require CPU/CUDA/HIP");
    STD_TORCH_CHECK(example.scalar_type()==ScalarType::Float || example.scalar_type()==ScalarType::Double,"Krylov updates require float32/float64");
    for (size_t slot=0;slot<arrays.size();++slot) {
        const auto dtype=kinds[task][slot]=='i' ? ScalarType::Int:kinds[task][slot]=='b' ? ScalarType::Bool:example.scalar_type();
        if ((task==4 || task==16) && slot==1) {
            // The external diagonal may be strided; workspace buffers are dense.
            STD_TORCH_CHECK(arrays[slot].device()==example.device() && arrays[slot].scalar_type()==dtype,"Krylov diagonal dtype/device mismatch");
        } else tensor(arrays[slot],example,dtype,"Krylov buffer");
        STD_TORCH_CHECK(arrays[slot].dim()==1 && arrays[slot].numel()>=(minima[task][slot]<0 ? size:minima[task][slot]),"Krylov buffer shape mismatch");
    }
    if (example.is_cuda()) {
#ifdef WITH_CUDA
        update_gpu(arrays,int(task),size,threshold,hasThreshold,int(dimension),history);
#else
        STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
    } else execute(arrays,int(task),size,threshold,hasThreshold,int(dimension),history,CpuLauncher{});
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("krylov_update(Tensor(a!)[] arrays, int task, int size, float threshold, bool has_threshold, int dimension, bool history) -> ()");
    m.impl("krylov_update",TORCH_BOX(&update));
}
} // namespace sph_native::krylov
