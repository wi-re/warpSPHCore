#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "nt_interpolate_forward.cuh"
namespace sph_native {
void interpolate_forward_gpu(const Tensors &t, int support, int direction, int kernel, int scheme,
                             int lap, bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    execute(InterpolateGpuRows{current_stream(t[QX])}, t, support, direction, 1, GpuLauncher{},
            kernel, scheme, lap, dot);
}
} // namespace sph_native
