#include "execution/gpu.cuh"
#include "nt_block_rows.h"
namespace sph_native::block {
void apply_gpu(const Tensor& inverse,const Tensor& values,const Tensor& output,bool transpose) {
    const stable::accelerator::DeviceGuard guard(values.get_device_index());
    execute(inverse,values,output,transpose,GpuLauncher{current_stream(values)});
}
} // namespace sph_native::block
