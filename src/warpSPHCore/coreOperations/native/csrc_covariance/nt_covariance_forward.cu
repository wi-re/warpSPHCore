#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "nt_covariance_forward.h"
#include "execution/fieldTraversal.cuh"
#include "execution/packedFieldTraversal.cuh"
namespace sph_native {
void covariance_forward_gpu(const Tensors &t, int support, int direction, int kernel, int scheme,
                            int lap, bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const int pitch = (scheme == 3) ? packed_reference_pitch(t, support, direction, kernel, 1) : 0;
    if (pitch) {
        const Tensor packed = stable::new_empty(t[RX], {t[RX].size(0), pitch});
        execute_precision<float>(
            PackedFieldRows<CovariancePair, 7>{packed.mutable_data_ptr<float>()}, t, support,
            direction, 7, CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap,
            dot);
        return;
    }
    execute(CooperativeFieldRows<CovariancePair, 7>{}, t, support, direction, 7,
            CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
}
} // namespace sph_native
