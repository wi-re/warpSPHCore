#include <torch/csrc/stable/library.h>
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "nt_density_forward.h"
namespace sph_native {
void density_forward_gpu(const Tensors &, int, int, int);
Tensor density_forward(const Tensors &inputs, int64_t support, int64_t direction, int64_t kernel) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    Tensors t = prepare_operator(inputs, support, direction, false);
    t[OUTPUT] = stable::new_empty(t[QX], {t[QX].size(0)});
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        density_forward_gpu(t, support, direction, kernel);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only; select "
                               "WARPSPHCORE_NATIVE_BUILD=gpu and rebuild");
#endif
    } else
        execute(DensityRows{}, t, support, direction, 6, CpuLauncher{}, kernel);
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("density_forward(Tensor[] inputs, int support, int direction, int kernel=0) -> Tensor");
    m.impl("density_forward", TORCH_BOX(&density_forward));
}
} // namespace sph_native
