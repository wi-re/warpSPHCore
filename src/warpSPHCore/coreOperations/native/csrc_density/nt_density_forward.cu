#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "execution/valueTraversal.cuh"
namespace sph_native {
void density_forward_gpu(const Tensors &t, int support, int direction, int kernel) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    execute(ValueGpuRows<true>{}, t, support, direction, 6,
            CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
}
} // namespace sph_native
