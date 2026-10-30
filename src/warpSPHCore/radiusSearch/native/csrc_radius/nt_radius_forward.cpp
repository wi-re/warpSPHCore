#include <torch/csrc/stable/library.h>
#include "torch/validation.h"
#include "torch/ops.h"
#include "execution/cpu.h"
#include "nt_radius_dispatch.h"
namespace sph_native {
void launch_radius(RadiusTask task, const Tensors& t, int support) {
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        radius_gpu(task, t, support);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only; select WARPSPHCORE_NATIVE_BUILD=gpu and rebuild");
#endif
    } else execute(RadiusRows{task}, t, support, 0, 6, CpuLauncher{});
}
std::tuple<Tensor, Tensor, Tensor> radius(
    const Tensor& qx, const Tensor& rx, const Tensor& qh, const Tensor& rh,
    const Tensor& lo, const Tensor& hi, const Tensor& periodic, int64_t support) {
    Tensors t(SLOTS);
    t[QX] = qx; t[RX] = rx; t[QH] = qh; t[RH] = rh;
    t[LO] = lo; t[HI] = hi; t[PERIODIC] = periodic;
    geometry(t, support);
    t[COUNTS] = stable::new_empty(qx, {qx.size(0)}, ScalarType::Int);
    launch_radius(RadiusTask::Count, t, support);
    const auto counts = wide(t[COUNTS]);
    t[OFFSETS] = exclusive_scan(counts);
    const int64_t total = sum_int(counts); // exact allocation boundary
    t[NEIGHBORS] = stable::new_empty(qx, {total}, ScalarType::Long);
    launch_radius(RadiusTask::Collect, t, support);
    return {t[NEIGHBORS], t[COUNTS], t[OFFSETS]};
}

STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("radius(Tensor qx, Tensor rx, Tensor qh, Tensor rh, Tensor lo, Tensor hi, Tensor periodic, int support) -> (Tensor, Tensor, Tensor)");
    m.impl("radius", TORCH_BOX(&radius));
}
} // namespace sph_native
