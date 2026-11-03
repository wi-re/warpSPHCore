#include "execution/gpu.cuh"
#include "nt_radius_dispatch.h"
#include "execution/rowSort.cuh"
#include "nt_radius_forward.cuh"
namespace sph_native {
namespace {
struct CooperativeRadiusRows {
    RadiusTask task;
    compact::View index;
    bool reordered;
    template <class T, int D, class Launch>
    void operator()(int64_t n, RowArgs<T, D> a, const Launch &launch) const {
        if (reordered) {
            if (a.support == 11)
                launch(n, compact::CooperativeSearchRow<T, D, Launch::width, true, true>{
                              a, index, task == RadiusTask::Collect});
            else
                launch(n, compact::CooperativeSearchRow<T, D, Launch::width, false, true>{
                              a, index, task == RadiusTask::Collect});
        } else {
            if (a.support == 11)
                launch(n, compact::CooperativeSearchRow<T, D, Launch::width, true, false>{
                              a, index, task == RadiusTask::Collect});
            else
                launch(n, compact::CooperativeSearchRow<T, D, Launch::width, false, false>{
                              a, index, task == RadiusTask::Collect});
        }
    }
};
} // namespace
void radius_gpu(RadiusTask task, const Tensors &t, int support, compact::View index) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const auto stream = current_stream(t[QX]);
    // Self searches can visit queries in the index's existing cell order. Output
    // rows still use original query IDs; distinct query/reference sets retain
    // input order and require no additional sorting or permutation buffers.
    const bool reordered =
        t[QX].size(0) == t[RX].size(0) && t[QX].const_data_ptr() == t[RX].const_data_ptr();
    execute(CooperativeRadiusRows{task, index, reordered}, t, support, 0, 6,
            CooperativeGpuLauncher<4>{stream});
    if (task == RadiusTask::Collect && t[NEIGHBORS].scalar_type()==ScalarType::Long)
        sort_csr_rows(t[QX].size(0), index_view(t[OFFSETS]), t[COUNTS].const_data_ptr<int32_t>(),
                      t[NEIGHBORS].mutable_data_ptr<int64_t>(), stream);
    else if (task == RadiusTask::Collect)
        sort_csr_rows(t[QX].size(0),index_view(t[OFFSETS]),t[COUNTS].const_data_ptr<int32_t>(),
                      t[NEIGHBORS].mutable_data_ptr<int32_t>(),stream);
}
} // namespace sph_native
