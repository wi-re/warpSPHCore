#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "nt_interpolate_adjoint.h"
namespace sph_native {
void interpolate_adjoint_gpu(const Tensors &t, const Tensor &cot, const Tensors &gradients,
                             int64_t mask, int support, int direction, int kernel, int scheme,
                             int lap, bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    execute(AdjointRows<InterpolateAdjointPolicy>{gradients, cot, mask}, t, support, direction, 1,
            GpuLauncher{current_stream(t[QX])}, kernel, scheme, lap, dot);
}
} // namespace sph_native
