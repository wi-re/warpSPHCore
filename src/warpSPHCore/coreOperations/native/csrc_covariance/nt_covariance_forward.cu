#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "nt_covariance_forward.h"
#include "execution/fieldTraversal.cuh"
#include "execution/packedFieldTraversal.cuh"
#include "nt_covariance_operator.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.cuh"
#include "nt_covariance_torch.h"
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
void covariance_corrected_forward_gpu(const Tensors& t, const OptionalTensors& corrections, int support, int direction, int kernel, int scheme, int lap, bool dot, const Tensor& counts) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    using Rows = neighborhood::CorrectedRows<neighborhood::CorrectedCovariance,
        neighborhood::CooperativeCorrectionExecutor, neighborhood::CovarianceCorrectionOptions>;
    execute(Rows{corrections, {counts}}, t, support, direction, 7,
        CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
}
} // namespace sph_native
