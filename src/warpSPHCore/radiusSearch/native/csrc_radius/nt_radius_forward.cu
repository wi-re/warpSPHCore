#include "execution/gpu.cuh"
#include "nt_radius_dispatch.h"
#include "execution/rowSort.cuh"
namespace sph_native {
void radius_gpu(RadiusTask task, const Tensors &t, int support, compact::View index) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const auto stream = current_stream(t[QX]);
    execute(RadiusRows<false>{task, index}, t, support, 0, 6, GpuLauncher{stream});
    if (task == RadiusTask::Collect)
        sort_csr_rows(t[QX].size(0), index_view(t[OFFSETS]), t[COUNTS].const_data_ptr<int32_t>(),
                      t[NEIGHBORS].mutable_data_ptr<int64_t>(), stream);
}
} // namespace sph_native
