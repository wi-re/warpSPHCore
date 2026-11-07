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

#include "nt_moments_jvp.h"
namespace sph_native {
void crk_moments_product_gpu(const Tensors& t,const crk_moments_jvp::Buffers& buffers,
        int support,int direction,int kernel) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    execute(crk_moments_jvp::Rows{buffers},t,support,direction,9,GpuLauncher{current_stream(t[QX])},kernel);
}
} // namespace sph_native
