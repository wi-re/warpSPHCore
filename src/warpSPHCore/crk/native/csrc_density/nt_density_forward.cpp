#include "crk/native/csrc_preparation/nt_preparation_rows.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"

namespace sph_native {
void crk_density_forward_gpu(const Tensors&, const crk::PreparationTensors&, int, int, int);
Tensor crk_density_forward(const OptionalTensors& inputs, const Tensor& volumes,
        const Tensor& A, const Tensor& B, int64_t support, int64_t direction, int64_t kernel) {
    auto t = crk::prepare(inputs, support, direction, kernel);
    crk::volume(volumes, t);
    const auto& x = t[QX];
    vector(t[MASS], x, x.scalar_type(), t[RX].size(0), "reference masses");
    vector(A, x, x.scalar_type(), x.size(0), "CRK A");
    tensor(B, x, x.scalar_type(), "CRK B");
    STD_TORCH_CHECK(B.dim() == 2 && B.size(0) == x.size(0) && B.size(1) == x.size(1),
                    "CRK B has an incompatible shape");
    t[OUTPUT] = stable::new_empty(x, {x.size(0)});
    const crk::PreparationTensors buffers{volumes, A, B, {}, {}};
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        crk_density_forward_gpu(t, buffers, support, direction, kernel);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(crk::PreparationRows<false, crk::SerialPreparationExecutor>{buffers}, t,
                support, direction, 6, CpuLauncher{}, kernel);
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("crk_density_forward(Tensor?[] inputs, Tensor volumes, Tensor A, Tensor B, int support, int direction, int kernel) -> Tensor");
    m.impl("crk_density_forward", TORCH_BOX(&crk_density_forward));
}
} // namespace sph_native
