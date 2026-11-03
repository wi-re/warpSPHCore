#include "nt_density_hvp.h"
#include "execution/gpu.cuh"
namespace sph_native {
void density_hvp_gpu(const Tensors& t,const density_hvp::Buffers& buffers,int support,int direction,int kernel) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    execute(density_hvp::Rows{buffers},t,support,direction,6,GpuLauncher{current_stream(t[QX])},kernel);
}
} // namespace sph_native
