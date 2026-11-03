#include "crk/native/csrc_preparation/nt_preparation_rows.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"

namespace sph_native {
void crk_moments_forward_gpu(const Tensors&, const crk::PreparationTensors&, int, int, int);
Tensors crk_moments_forward(const OptionalTensors& inputs, const Tensor& volumes,
                           int64_t support, int64_t direction, int64_t kernel) {
    auto t = crk::prepare(inputs, support, direction, kernel);
    crk::volume(volumes, t);
    const auto n = t[QX].size(0), d = t[QX].size(1);
    Tensors moments;
    for (const auto& shape : std::vector<std::vector<int64_t>>{
            {n}, {n,d}, {n,d,d}, {n,d}, {n,d,d}, {n,d,d,d}})
        moments.push_back(stable::new_empty(t[QX], shape));
    const auto counts = stable::new_empty(t[COUNTS], {n});
    const crk::PreparationTensors buffers{volumes, {}, {}, moments, counts};
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        crk_moments_forward_gpu(t, buffers, support, direction, kernel);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(crk::PreparationRows<true, crk::SerialPreparationExecutor>{buffers}, t,
                support, direction, 6, CpuLauncher{}, kernel);
    moments.push_back(counts);
    return moments;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("crk_moments_forward(Tensor?[] inputs, Tensor volumes, int support, int direction, int kernel) -> Tensor[]");
    m.impl("crk_moments_forward", TORCH_BOX(&crk_moments_forward));
}
} // namespace sph_native
