#include "execution/gpu.cuh"
#include "nt_radius_dispatch.h"
namespace sph_native {
void radius_gpu(RadiusTask task, const Tensors &t, int support, compact::View index) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    execute(RadiusRows{task, index}, t, support, 0, 6, GpuLauncher{current_stream(t[QX])});
}
} // namespace sph_native
