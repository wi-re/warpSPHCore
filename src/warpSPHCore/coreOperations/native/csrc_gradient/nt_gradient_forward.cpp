#include "nt_gradient_forward.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.h"
namespace sph_native {
void gradient_corrected_forward_gpu(const Tensors &, const OptionalTensors &, int, int, int, int);
void gradient_forward_gpu(const Tensors &, int, int, int, int, int, bool);
Tensor gradient_forward(const Tensors &inputs, int64_t support, int64_t direction, int64_t kernel,
                        int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_field_operator(inputs, support, direction, 2, scheme, lap, dot);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        gradient_forward_gpu(t, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(GradientRows{}, t, support, direction, 2, CpuLauncher{}, kernel, scheme, lap, dot);
    return t[OUTPUT];
}
Tensor gradient_corrected_forward(const OptionalTensors& inputs,
        const OptionalTensors& corrections, int64_t support, int64_t direction,
        int64_t kernel, int64_t scheme) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_corrected_operator(inputs, corrections, support, direction, 2, scheme);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        gradient_corrected_forward_gpu(t, corrections, support, direction, kernel, scheme);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else {
        using Rows = neighborhood::CorrectedRows<neighborhood::CorrectedGradient,
                                                 neighborhood::SerialCorrectionExecutor>;
        execute(Rows{corrections}, t, support, direction, 2, CpuLauncher{}, kernel, scheme);
    }
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("gradient_corrected_forward(Tensor?[] inputs, Tensor?[] corrections, int support, "
          "int direction, int kernel=0, int scheme=3) -> Tensor");
    m.impl("gradient_corrected_forward", TORCH_BOX(&gradient_corrected_forward));
    m.def("gradient_forward(Tensor[] inputs, int support, int direction, int kernel=0, int "
          "scheme=3, int lap=2, bool dot=False) -> Tensor");
    m.impl("gradient_forward", TORCH_BOX(&gradient_forward));
}
} // namespace sph_native
