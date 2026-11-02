#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "execution/valueTraversal.cuh"
#include "execution/packedValues.cuh"
#include "execution/rowNeighborhood.cuh"
#include "nt_density_operator.h"
namespace sph_native {
namespace {
template <class T, bool Packed = false>
using DensityNeighborhoodRows = neighborhood::CooperativeSphRows<
    neighborhood::DensityPolicy, T, neighborhood::Weight::Mass, Packed>;
} // namespace

void density_forward_gpu(const Tensors &t, int support, int direction, int kernel) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const int pitch = packed_reference_pitch(t, support, direction, kernel, 1);
    if (pitch) {
        const Tensor packed = stable::new_empty(t[RX], {t[RX].size(0), pitch});
        const auto rows = DensityNeighborhoodRows<float, true>{packed.mutable_data_ptr<float>()};
        // Use only storage metadata; sparse/low-dimensional work keeps its
        // established schedule, while dense 3D rows use a full warp per query.
        const bool dense_3d = t[QX].size(1) == 3 && t[NEIGHBORS].numel() / 128 >= t[QX].size(0);
        if (t[QX].size(1) == 2)
            // Retain the existing packed 2D row, which wins the paired
            // low/mid-degree FP32 measurements with this record layout.
            execute_precision<float>(PackedValuesRows<true>{packed.mutable_data_ptr<float>()},
                                     t, support, direction, 6,
                                     CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
        else if (dense_3d)
            execute_precision<float>(rows, t, support, direction, 6,
                                     CooperativeGpuLauncher<32>{current_stream(t[QX])}, kernel);
        else
            execute_precision<float>(rows, t, support, direction, 6,
                                     CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
        return;
    }
    if (kernel == 0 && support == 11) {
        if (t[QX].scalar_type() == ScalarType::Float)
            execute_precision<float>(DensityNeighborhoodRows<float>{}, t, support, direction, 6,
                CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
        else
            execute_precision<double>(DensityNeighborhoodRows<double>{}, t, support, direction, 6,
                CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
    } else {
        execute(ValueGpuRows<true>{}, t, support, direction, 6,
                CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel);
    }
}
} // namespace sph_native
