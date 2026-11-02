#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "execution/rowNeighborhood.cuh"
#include "nt_volume_operator.h"

namespace sph_native {
void crk_volume_forward_gpu(const Tensors& t, int support, int direction, int kernel) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    if (t[QX].scalar_type() == ScalarType::Float)
        execute_precision<float>(neighborhood::CooperativeSphRows<neighborhood::ApparentVolume,
            float, neighborhood::Weight::None>{}, t, support, direction, 6,
            CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
    else
        execute_precision<double>(neighborhood::CooperativeSphRows<neighborhood::ApparentVolume,
            double, neighborhood::Weight::None>{}, t, support, direction, 6,
            CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
}
} // namespace sph_native
