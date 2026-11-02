#include "nt_curl_forward.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "nt_curl_operator.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.h"
namespace sph_native {
void curl_corrected_forward_gpu(const Tensors& t, const OptionalTensors& corrections, int support, int direction, int kernel, int scheme, int lap, bool dot);
void curl_forward_gpu(const Tensors &, int, int, int, int, int, bool);
Tensor curl_forward(const Tensors &inputs, int64_t support, int64_t direction, int64_t kernel,
                    int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_field_operator(inputs, support, direction, 4, scheme, lap, dot);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        curl_forward_gpu(t, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(CurlRows{}, t, support, direction, 4, CpuLauncher{}, kernel, scheme, lap, dot);
    return t[OUTPUT];
}
Tensor curl_corrected_forward(const OptionalTensors& inputs, const OptionalTensors& corrections,
        int64_t support, int64_t direction, int64_t kernel, int64_t scheme, int64_t lap, bool dot) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    auto t = prepare_corrected_operator(inputs, corrections, support, direction, 4, scheme, lap, dot);
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        curl_corrected_forward_gpu(t, corrections, support, direction, kernel, scheme, lap, dot);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else {
        using Rows = neighborhood::CorrectedRows<neighborhood::CorrectedCurl,
            neighborhood::SerialCorrectionExecutor>;
        execute(Rows{corrections}, t, support, direction, 4, CpuLauncher{}, kernel, scheme, lap, dot);
    }
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("curl_corrected_forward(Tensor?[] inputs, Tensor?[] corrections, int support, int direction, "
          "int kernel=0, int scheme=3, int lap=2, bool dot=False) -> Tensor");
    m.impl("curl_corrected_forward", TORCH_BOX(&curl_corrected_forward));

    m.def("curl_forward(Tensor[] inputs, int support, int direction, int kernel=0, int scheme=3, "
          "int lap=2, bool dot=False) -> Tensor");
    m.impl("curl_forward", TORCH_BOX(&curl_forward));
}
} // namespace sph_native
