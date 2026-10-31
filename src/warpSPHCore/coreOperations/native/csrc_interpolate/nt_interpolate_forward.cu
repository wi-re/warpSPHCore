#include "execution/dispatch.h"
#include "execution/gpu.cuh"
#include "execution/packedValues.cuh"
#include "execution/valueTraversal.cuh"
namespace sph_native {
namespace {
template <class T, int Width, int Batch>
void packed_interpolate(const Tensors &t, int pitch, int support, int direction, int kernel) {
    const Tensor packed = stable::new_empty(t[RX], {t[RX].size(0), pitch});
    execute_precision<T>(PackedValuesRows<false, T, Batch>{packed.mutable_data_ptr<T>()}, t,
                         support, direction, 1,
                         CooperativeGpuLauncher<Width>{current_stream(t[QX])}, kernel);
}
} // namespace
void interpolate_forward_gpu(const Tensors &t, int support, int direction, int kernel, int scheme,
                             int lap, bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const int components = field_components(t), dimension = int(t[QX].size(1));
    // FP64 packing is limited to the scalar/vector cases qualified below.
    const bool scalar_or_vector = components <= dimension;
    const int pitch = scalar_or_vector
                          ? packed_reference_pitch<true>(t, support, direction, kernel, components)
                          : packed_reference_pitch(t, support, direction, kernel, components);
    if (pitch) {
        // Storage sizes provide a cheap degree estimate; inspect no CSR values.
        const bool dense_3d = dimension == 3 && t[NEIGHBORS].numel() / 128 >= t[QX].size(0);
        if (t[QX].scalar_type() == ScalarType::Double) {
            if (dimension == 1)
                packed_interpolate<double, 4, 2>(t, pitch, support, direction, kernel);
            else if (dense_3d)
                packed_interpolate<double, 8, 1>(t, pitch, support, direction, kernel);
            else
                packed_interpolate<double, 4, 1>(t, pitch, support, direction, kernel);
        } else if (!scalar_or_vector || (dimension == 3 && !dense_3d)) {
            packed_interpolate<float, 8, 1>(t, pitch, support, direction, kernel);
        } else if (dense_3d) {
            packed_interpolate<float, 16, 2>(t, pitch, support, direction, kernel);
        } else {
            packed_interpolate<float, 4, 2>(t, pitch, support, direction, kernel);
        }
        return;
    }
    execute(ValueGpuRows<false>{}, t, support, direction, 1,
            CooperativeGpuLauncher<8>{current_stream(t[QX])}, kernel, scheme, lap, dot);
}
} // namespace sph_native
