#pragma once
#include "execution/dispatch.h"
#include "nt_radius_forward.h"
namespace sph_native {
enum class RadiusTask { Count, Collect };
struct RadiusRows {
    RadiusTask task;
    template <typename T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        if (task == RadiusTask::Count)
            launch(n, CountRow<T, D>{a});
        else
            launch(n, CollectRow<T, D>{a});
    }
};
void radius_gpu(RadiusTask, const Tensors &, int);
} // namespace sph_native
