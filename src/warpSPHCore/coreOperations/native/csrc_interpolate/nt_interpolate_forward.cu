#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "execution/valueTraversal.cuh"
#include "execution/packedValues.cuh"
namespace sph_native {
void interpolate_forward_gpu(const Tensors &t, int support, int direction, int kernel, int scheme,
                             int lap, bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const int pitch = packed_reference_pitch(t, support, direction, kernel, field_components(t));
    if (pitch) {
        const Tensor packed = stable::new_empty(t[RX], {t[RX].size(0), pitch});
        execute_precision<float>(PackedValuesRows<false>{packed.mutable_data_ptr<float>()}, t,
                                 support, direction, 1,
                                 CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
        return;
    }
    execute(ValueGpuRows<false>{}, t, support, direction, 1,
            CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
}
} // namespace sph_native
