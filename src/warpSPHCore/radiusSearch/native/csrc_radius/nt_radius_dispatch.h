#pragma once
#include "execution/dispatch.h"
#include "nt_radius_forward.h"
#include "radiusSearch/native/csrc_compact/nt_compact.h"
namespace sph_native {
enum class RadiusTask { Count, Collect };
struct RadiusRows {
    RadiusTask task;
    compact::View index;
    template <typename T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        launch(n, compact::SearchRow<T, D>{a, index, task == RadiusTask::Collect});
    }
};
void radius_gpu(RadiusTask, const Tensors &, int, compact::View);
} // namespace sph_native
