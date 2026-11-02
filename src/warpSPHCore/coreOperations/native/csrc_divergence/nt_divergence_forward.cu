#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "nt_divergence_forward.h"
#include "execution/fieldTraversal.cuh"
#include "execution/packedFieldTraversal.cuh"
#include "nt_divergence_operator.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.cuh"
namespace sph_native {
void divergence_forward_gpu(const Tensors &t, int support, int direction, int kernel, int scheme,
                            int lap, bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const int pitch = (scheme == 3) ? packed_reference_pitch(t, support, direction, kernel,
                                                             field_components(t) + 1)
                                    : 0;
    if (pitch) {
        const Tensor packed = stable::new_empty(t[RX], {t[RX].size(0), pitch});
        execute_precision<float>(
            PackedFieldRows<DivergencePair, 3>{packed.mutable_data_ptr<float>()}, t, support,
            direction, 3, CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap,
            dot);
        return;
    }
    execute(CooperativeFieldRows<DivergencePair, 3>{}, t, support, direction, 3,
            CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
}
void divergence_corrected_forward_gpu(const Tensors& t, const OptionalTensors& corrections, int support, int direction, int kernel, int scheme, int lap, bool dot, bool consistent) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    using Rows = neighborhood::CorrectedRows<neighborhood::CorrectedDivergence,
        neighborhood::CooperativeCorrectionExecutor, neighborhood::DivergenceCorrectionOptions>;
    execute(Rows{corrections, {consistent}}, t, support, direction, 3,
        CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
}
} // namespace sph_native
