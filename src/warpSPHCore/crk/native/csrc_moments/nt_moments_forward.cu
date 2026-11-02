#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "crk/native/csrc_preparation/nt_preparation_rows.cuh"

namespace sph_native {
void crk_moments_forward_gpu(const Tensors& t, const crk::PreparationTensors& buffers,
        int support, int direction, int kernel) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    execute(crk::PreparationRows<true, crk::CooperativePreparationExecutor>{buffers}, t,
            support, direction, 6, CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
}
} // namespace sph_native
