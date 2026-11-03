#include "execution/gpu.cuh"
#include "nt_kernel_forward.h"
namespace sph_native {
void kernel_forward_gpu(const Tensors &t, int kernel) {
    const stable::accelerator::DeviceGuard guard(t[0].get_device_index());
    kernel_execute(t, kernel, GpuLauncher{current_stream(t[0])});
}
} // namespace sph_native
