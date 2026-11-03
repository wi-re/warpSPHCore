#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "nt_laplacian_forward.h"
#include "execution/fieldTraversal.cuh"
#include "execution/packedFieldTraversal.cuh"
#include "nt_laplacian_operator.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.cuh"
namespace sph_native {
void laplacian_forward_gpu(const Tensors &t, int support, int direction, int kernel, int scheme,
                           int lap, bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const int pitch =
        (scheme == 3 && lap == 2)
            ? packed_reference_pitch(t, support, direction, kernel, field_components(t) + 1)
            : 0;
    if (pitch) {
        const Tensor packed = stable::new_empty(t[RX], {t[RX].size(0), pitch});
        execute_precision<float>(
            PackedFieldRows<LaplacianPair, 5>{packed.mutable_data_ptr<float>()}, t, support,
            direction, 5, CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap,
            dot);
        return;
    }
    execute(CooperativeFieldRows<LaplacianPair, 5>{}, t, support, direction, 5,
            CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
}
void laplacian_corrected_forward_gpu(const Tensors& t, const OptionalTensors& corrections, int support, int direction, int kernel, int scheme, int lap, bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    using Rows = neighborhood::CorrectedRows<neighborhood::CorrectedLaplacian,
        neighborhood::CooperativeCorrectionExecutor>;
    execute(Rows{corrections}, t, support, direction, 5,
        CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
}
} // namespace sph_native
