#include "nt_density_adjoint.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
namespace sph_native {
void density_adjoint_gpu(const Tensors &, const Tensor &, const Tensors &, int64_t, int, int, int,
                         int, int, bool);
Tensors density_adjoint(const Tensors &inputs, const Tensor &cot, int64_t mask, int64_t support,
                        int64_t direction, int64_t kernel, int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_operator(inputs, support, direction, false);
    t[OUTPUT] = stable::new_empty(t[QX], {t[QX].size(0)});
    tensor(cot, t[QX], t[QX].scalar_type(), "cotangent");
    STD_TORCH_CHECK(cot.dim() == t[OUTPUT].dim() && cot.numel() == t[OUTPUT].numel(),
                    "cotangent shape must match output");
    for (int d = 0; d < cot.dim(); ++d)
        STD_TORCH_CHECK(cot.size(d) == t[OUTPUT].size(d), "cotangent shape must match output");
    auto gradients = gradient_buffers(t, mask);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        density_adjoint_gpu(t, cot, gradients, mask, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(AdjointRows<DensityAdjointPolicy>{gradients, cot, mask}, t, support, direction, 6,
                CpuLauncher{}, kernel, scheme, lap, dot);
    return gradients;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("density_adjoint(Tensor[] inputs, Tensor cot, int mask, int support, int direction, int "
          "kernel, int scheme, int lap, bool dot) -> Tensor[]");
    m.impl("density_adjoint", TORCH_BOX(&density_adjoint));
}
} // namespace sph_native
