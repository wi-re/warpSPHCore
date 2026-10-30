#include "nt_interpolate_forward.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
namespace sph_native {
void interpolate_forward_gpu(const Tensors &, int, int, int, int, int, bool);
Tensor interpolate_forward(const Tensors &inputs, int64_t support, int64_t direction,
                           int64_t kernel, int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_field_operator(inputs, support, direction, 1, scheme, lap, dot);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        interpolate_forward_gpu(t, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(InterpolateRows{}, t, support, direction, 1, CpuLauncher{}, kernel, scheme, lap,
                dot);
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("interpolate_forward(Tensor[] inputs, int support, int direction, int kernel=0, int "
          "scheme=3, int lap=2, bool dot=False) -> Tensor");
    m.impl("interpolate_forward", TORCH_BOX(&interpolate_forward));
}
} // namespace sph_native
