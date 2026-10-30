#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "nt_interpolate_forward.h"
namespace sph_native {
void interpolate_forward_gpu(const Tensors &t, int support, int direction, int kernel, int scheme,
                             int lap, bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    execute(InterpolateRows{}, t, support, direction, 1, GpuLauncher{current_stream(t[QX])}, kernel,
            scheme, lap, dot);
}
} // namespace sph_native
