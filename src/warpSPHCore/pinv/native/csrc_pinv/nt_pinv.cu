#include "execution/gpu.cuh"
#include "nt_pinv_rows.h"
namespace sph_native::dense {
void inverse_gpu(const Buffers& buffers,int mode,double rtol) {
    const stable::accelerator::DeviceGuard guard(buffers.matrix.get_device_index());
    execute(buffers,mode,rtol,GpuLauncher{current_stream(buffers.matrix)});
}
} // namespace sph_native::dense
