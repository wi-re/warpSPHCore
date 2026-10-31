#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "execution/valueTraversal.cuh"
#include "execution/packedValues.cuh"
namespace sph_native {
void density_forward_gpu(const Tensors &t, int support, int direction, int kernel) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const int pitch = packed_reference_pitch(t, support, direction, kernel, 1);
    if (pitch) {
        const Tensor packed = stable::new_empty(t[RX], {t[RX].size(0), pitch});
        execute_precision<float>(PackedValuesRows<true>{packed.mutable_data_ptr<float>()}, t,
                                 support, direction, 6,
                                 CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
        return;
    }
    execute(ValueGpuRows<true>{}, t, support, direction, 6,
            CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
}
} // namespace sph_native
