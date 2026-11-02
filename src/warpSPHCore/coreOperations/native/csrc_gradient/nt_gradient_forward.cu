#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "nt_gradient_forward.h"
#include "execution/fieldTraversal.cuh"
#include "execution/packedFieldTraversal.cuh"
#include "execution/rowNeighborhood.cuh"
namespace sph_native {
namespace {
template <class T, bool Packed = false>
using GradientNeighborhoodRows = neighborhood::CooperativeSphRows<
    neighborhood::Gradient, T, neighborhood::Weight::Volume, Packed>;
} // namespace

void gradient_forward_gpu(const Tensors &t, int support, int direction, int kernel, int scheme,
                          int lap, bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const int pitch = (scheme == 3) ? packed_reference_pitch(t, support, direction, kernel,
                                                             field_components(t) + 1)
                                    : 0;
    if (pitch) {
        const Tensor packed = stable::new_empty(t[RX], {t[RX].size(0), pitch});
        const auto rows = GradientNeighborhoodRows<float, true>{packed.mutable_data_ptr<float>()};
        const bool dense_3d = t[QX].size(1) == 3 && t[NEIGHBORS].numel() / 128 >= t[QX].size(0);
        if (dense_3d)
            execute_precision<float>(rows, t, support, direction, 2,
                                     CooperativeGpuLauncher<16>{current_stream(t[QX])}, kernel, scheme, lap, dot);
        else
            execute_precision<float>(rows, t, support, direction, 2,
                                     CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
        return;
    }
    if (kernel == 0 && support == 11 && scheme == 3) {
        if (t[QX].scalar_type() == ScalarType::Float)
            execute_precision<float>(GradientNeighborhoodRows<float>{}, t, support, direction, 2,
                CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
        else
            execute_precision<double>(GradientNeighborhoodRows<double>{}, t, support, direction, 2,
                CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
    } else {
        execute(CooperativeFieldRows<GradientPair, 2>{}, t, support, direction, 2,
                CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
    }
}
} // namespace sph_native
