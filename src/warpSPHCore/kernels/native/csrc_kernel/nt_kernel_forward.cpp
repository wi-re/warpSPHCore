#include "nt_kernel_forward.h"
#include "torch/validation.h"
#include "execution/cpu.h"
namespace sph_native {
void kernel_forward_gpu(const Tensors &, int);
Tensors kernel_forward(const Tensor &x, const Tensor &h, int64_t kernel) {
    STD_TORCH_CHECK(x.scalar_type() == ScalarType::Float || x.scalar_type() == ScalarType::Double,
                    "Expected float32 or float64");
    STD_TORCH_CHECK(x.dim() == 2 && x.size(1) >= 1 && x.size(1) <= 3,
                    "displacement must have shape [N,D], D=1,2,3");
    STD_TORCH_CHECK(x.is_cpu() || x.is_cuda(), "Expected CPU or CUDA/HIP tensors");
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    tensor(x, x, x.scalar_type(), "displacement");
    vector(h, x, x.scalar_type(), x.size(0), "supports");
    const auto n = x.size(0), d = x.size(1);
    Tensors t{x,
              h,
              stable::new_empty(x, {n}),
              stable::new_empty(x, {n, d}),
              stable::new_empty(x, {n, d, d}),
              stable::new_empty(x, {n}),
              stable::new_empty(x, {n, d}),
              stable::new_empty(x, {n}),
              stable::new_empty(x, {n, 7})};
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        kernel_forward_gpu(t, kernel);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        kernel_execute(t, kernel, CpuLauncher{});
    return Tensors(t.begin() + 2, t.end());
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("kernel_forward(Tensor x, Tensor h, int kernel) -> Tensor[]");
    m.impl("kernel_forward", TORCH_BOX(&kernel_forward));
}
} // namespace sph_native
