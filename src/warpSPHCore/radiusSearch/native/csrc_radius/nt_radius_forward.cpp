#include <torch/csrc/stable/library.h>
#include "torch/validation.h"
#include "execution/scan.h"
#include "execution/cpu.h"
#include "nt_radius_dispatch.h"
#include "radiusSearch/native/csrc_compact/nt_compact_backend.h"
namespace sph_native {
void launch_radius(RadiusTask task, const Tensors &t, int support, compact::View index) {
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        radius_gpu(task, t, support, index);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only; select "
                               "WARPSPHCORE_NATIVE_BUILD=gpu and rebuild");
#endif
    } else
        execute(RadiusRows{task, index}, t, support, 0, 6, CpuLauncher{});
}
std::tuple<Tensor, Tensor, Tensor> radius(const Tensor &qx, const Tensor &rx, const Tensor &qh,
                                          const Tensor &rh, const Tensor &lo, const Tensor &hi,
                                          const Tensor &periodic, int64_t support) {
    Tensors t(SLOTS);
    t[QX] = qx;
    t[RX] = rx;
    t[QH] = qh;
    t[RH] = rh;
    t[LO] = lo;
    t[HI] = hi;
    t[PERIODIC] = periodic;
    geometry(t, support);
    // The framework-free builder borrows Torch's existing tensor memory. Its
    // resulting index feeds the same traversal rows as the standalone API.
    auto buffer = [](const Tensor &value) {
        return ConstBuffer{value.const_data_ptr(), value.numel()};
    };
    Problem problem{};
    problem.precision =
        qx.scalar_type() == ScalarType::Float ? Precision::Float32 : Precision::Float64;
    problem.dimension = int(qx.size(1));
    problem.query = {qx.size(0), buffer(qx), buffer(qh)};
    problem.reference = {rx.size(0), buffer(rx), buffer(rh)};
    problem.domain = {buffer(lo), buffer(hi), buffer(periodic)};
    problem.properties.support = Support(support);
    Context context{};
    if (qx.is_cuda()) {
#ifdef WITH_CUDA
#if defined(USE_ROCM)
        context.device = Device::HIP;
#else
        context.device = Device::CUDA;
#endif
        context.device_index = qx.get_device_index();
        STABLE_TORCH_ERROR_CODE_CHECK(
            aoti_torch_get_current_cuda_stream(context.device_index, &context.stream));
#endif
    }
    auto index = build_compact_hash(context, problem);
    const auto grid = compact::view(index);
    t[COUNTS] = stable::new_empty(qx, {qx.size(0)}, ScalarType::Int);
    launch_radius(RadiusTask::Count, t, support, grid);
    t[OFFSETS] = stable::new_empty(qx, {qx.size(0)}, ScalarType::Long);
    int64_t total = 0;
    if (!qx.is_cuda())
        total = exclusive_scan_cpu(t[COUNTS].const_data_ptr<int32_t>(),
                                   t[OFFSETS].mutable_data_ptr<int64_t>(), qx.size(0));
#ifdef WITH_CUDA
    else
        total = runtime::gpu_scan(context, qx.size(0), buffer(t[COUNTS]),
                                  {t[OFFSETS].mutable_data_ptr(), t[OFFSETS].numel()});
#endif
    t[NEIGHBORS] = stable::new_empty(qx, {total}, ScalarType::Long);
    launch_radius(RadiusTask::Collect, t, support, grid);
    return {t[NEIGHBORS], t[COUNTS], t[OFFSETS]};
}

STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("radius(Tensor qx, Tensor rx, Tensor qh, Tensor rh, Tensor lo, Tensor hi, Tensor "
          "periodic, int support) -> (Tensor, Tensor, Tensor)");
    m.impl("radius", TORCH_BOX(&radius));
}
} // namespace sph_native
