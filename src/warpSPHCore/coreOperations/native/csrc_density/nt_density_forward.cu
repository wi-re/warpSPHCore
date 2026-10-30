#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "nt_density_forward.h"
namespace sph_native {
void density_forward_gpu(const Tensors &t, int support, int direction, int kernel) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    execute(DensityRows{}, t, support, direction, 6, GpuLauncher{current_stream(t[QX])}, kernel);
}
} // namespace sph_native
