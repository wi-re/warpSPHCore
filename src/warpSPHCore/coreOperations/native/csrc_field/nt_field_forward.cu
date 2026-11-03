#include "execution/gpu.cuh"
#include "execution/rowNeighborhood.cuh"
#include "nt_field_dispatch.h"
namespace sph_native {
void field_product_gpu(const Tensors& t, const OptionalTensors& corrections,
        const neighborhood::FieldBuffers& buffers, int operation, int support, int direction,
        int kernel, int scheme, int lap, bool dot, bool consistent, bool pullback) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const auto launch = CooperativeGpuLauncher<8>{current_stream(t[QX])};
    if (pullback)
        execute_field(t, corrections, neighborhood::FieldExecutor<true, true>{buffers},
            operation, support, direction, kernel, scheme, lap, dot, consistent, launch);
    else
        execute_field(t, corrections, neighborhood::FieldExecutor<true, false>{buffers},
            operation, support, direction, kernel, scheme, lap, dot, consistent, launch);
}
} // namespace sph_native
