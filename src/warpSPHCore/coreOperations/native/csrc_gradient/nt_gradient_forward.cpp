#include <torch/csrc/stable/library.h>
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "nt_gradient_forward.h"
namespace sph_native {
void gradient_forward_gpu(const Tensors&, int, int);
Tensor gradient_forward(const Tensors& inputs, int64_t support, int64_t direction) {
    Tensors t = prepare_operator(inputs, support, direction, true);
    t[OUTPUT] = stable::new_empty(t[QX], {t[QX].size(0), t[QX].size(1)});
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        gradient_forward_gpu(t, support, direction);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only; select WARPSPHCORE_NATIVE_BUILD=gpu and rebuild");
#endif
    } else execute(GradientRows{}, t, support, direction, 2, CpuLauncher{});
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("gradient_forward(Tensor[] inputs, int support, int direction) -> Tensor");
    m.impl("gradient_forward", TORCH_BOX(&gradient_forward));
}
} // namespace sph_native
