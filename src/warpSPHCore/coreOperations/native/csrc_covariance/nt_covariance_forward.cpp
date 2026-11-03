#include "nt_covariance_forward.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "nt_covariance_operator.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.h"
#include "nt_covariance_torch.h"
namespace sph_native {
void covariance_corrected_forward_gpu(const Tensors& t, const OptionalTensors& corrections, int support, int direction, int kernel, int scheme, int lap, bool dot, const Tensor& counts);
void covariance_forward_gpu(const Tensors &, int, int, int, int, int, bool);
Tensor covariance_forward(const Tensors &inputs, int64_t support, int64_t direction, int64_t kernel,
                          int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_field_operator(inputs, support, direction, 7, scheme, lap, dot);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        covariance_forward_gpu(t, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(CovarianceRows{}, t, support, direction, 7, CpuLauncher{}, kernel, scheme, lap,
                dot);
    return t[OUTPUT];
}
Tensors covariance_corrected_forward(const OptionalTensors& inputs, const OptionalTensors& corrections,
        int64_t support, int64_t direction, int64_t kernel, int64_t scheme, int64_t lap, bool dot, bool return_counts) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_corrected_operator(inputs, corrections, support, direction, 7, scheme, lap, dot);
    const Tensor counts = return_counts ? stable::new_empty(t[COUNTS], {t[QX].size(0)}) : Tensor{};
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        covariance_corrected_forward_gpu(t, corrections, support, direction, kernel, scheme, lap, dot, counts);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else {
        using Rows = neighborhood::CorrectedRows<neighborhood::CorrectedCovariance,
            neighborhood::SerialCorrectionExecutor, neighborhood::CovarianceCorrectionOptions>;
        execute(Rows{corrections, {counts}}, t, support, direction, 7, CpuLauncher{}, kernel, scheme, lap, dot);
    }
    return return_counts ? Tensors{t[OUTPUT], counts} : Tensors{t[OUTPUT]};
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("covariance_corrected_forward(Tensor?[] inputs, Tensor?[] corrections, int support, int direction, "
          "int kernel=0, int scheme=3, int lap=2, bool dot=False, bool return_counts=False) -> Tensor[]");
    m.impl("covariance_corrected_forward", TORCH_BOX(&covariance_corrected_forward));

    m.def("covariance_forward(Tensor[] inputs, int support, int direction, int kernel=0, int "
          "scheme=3, int lap=2, bool dot=False) -> Tensor");
    m.impl("covariance_forward", TORCH_BOX(&covariance_forward));
}
} // namespace sph_native
