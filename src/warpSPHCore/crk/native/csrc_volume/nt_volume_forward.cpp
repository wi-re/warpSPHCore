#include "crk/native/csrc_preparation/nt_preparation_torch.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"
#include "execution/rowNeighborhood.h"
#include "nt_volume_operator.h"

namespace sph_native {
void crk_volume_forward_gpu(const Tensors&, int, int, int);
Tensor crk_volume_forward(const OptionalTensors& inputs, int64_t support, int64_t direction, int64_t kernel) {
    auto t = crk::prepare(inputs, support, direction, kernel);
    t[OUTPUT] = stable::new_empty(t[QX], {t[QX].size(0)});
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        crk_volume_forward_gpu(t, support, direction, kernel);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else {
        using Rows = neighborhood::SerialSphRows<neighborhood::ApparentVolume, neighborhood::Weight::None>;
        execute(Rows{}, t, support, direction, 6, CpuLauncher{}, kernel);
    }
    return t[OUTPUT];
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("crk_volume_forward(Tensor?[] inputs, int support, int direction, int kernel) -> Tensor");
    m.impl("crk_volume_forward", TORCH_BOX(&crk_volume_forward));
}
} // namespace sph_native
