#include "execution/gpu.cuh"
#include "execution/dispatch.h"
#include "nt_curl_forward.h"
namespace sph_native {
void curl_forward_gpu(const Tensors &t, int support, int direction, int kernel, int scheme, int lap,
                      bool dot) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    execute(CurlRows{}, t, support, direction, 4, GpuLauncher{current_stream(t[QX])}, kernel,
            scheme, lap, dot);
}
} // namespace sph_native
