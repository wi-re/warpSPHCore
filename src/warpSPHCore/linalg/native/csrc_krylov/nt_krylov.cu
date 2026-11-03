#include "execution/gpu.cuh"
#include "nt_krylov_rows.h"
namespace sph_native::krylov {
void update_gpu(const Tensors& arrays,int task,int64_t size,double threshold,bool hasThreshold,int dimension,bool history) {
    const stable::accelerator::DeviceGuard guard(arrays[0].get_device_index());
    execute(arrays,task,size,threshold,hasThreshold,dimension,history,GpuLauncher{current_stream(arrays[0])});
}
} // namespace sph_native::krylov
