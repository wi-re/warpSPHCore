#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "execution/transpose.h"
#include "execution/rowSort.cuh"
namespace sph_native {
template <class Task> struct DeviceTranspose {
    Task task;
    __device__ void operator()(int64_t i) const { task(i); }
};
void transpose_count_gpu(const Tensor& offsets, const Tensor& counts, const Tensor& ids, const Tensor& result) {
    const stable::accelerator::DeviceGuard guard(counts.get_device_index());
    GpuLauncher{current_stream(counts)}(counts.numel(), DeviceTranspose<TransposeCount>{{
        index_view(offsets), index_view(ids), counts.const_data_ptr<int32_t>(), result.mutable_data_ptr<int32_t>()}});
}
void transpose_fill_gpu(const Tensor& offsets, const Tensor& counts, const Tensor& ids,
        const Tensor& target_offsets, const Tensor& target_counts, const Tensor& cursors, const Tensor& result) {
    const stable::accelerator::DeviceGuard guard(counts.get_device_index());
    const auto stream = current_stream(counts);
    GpuLauncher{stream}(counts.numel(), DeviceTranspose<TransposeFill>{{
        index_view(offsets), index_view(ids), counts.const_data_ptr<int32_t>(),
        target_offsets.const_data_ptr<int64_t>(), cursors.mutable_data_ptr<int32_t>(), result.mutable_data_ptr<int64_t>()}});
    sort_csr_rows(target_counts.numel(), index_view(target_offsets), target_counts.const_data_ptr<int32_t>(),
                  result.mutable_data_ptr<int64_t>(), stream);
}
} // namespace sph_native
