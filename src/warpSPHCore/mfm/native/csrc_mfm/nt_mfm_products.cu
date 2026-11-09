#include "nt_mfm_products.h"
#include "execution/gpu.cuh"
namespace sph_native::mfm {
void products_gpu(const Tensors& t,const ProductBuffers& b,int task,int kernel,bool reverse){const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());execute(ProductRows{b,task,reverse},t,11,0,1,GpuLauncher{current_stream(t[QX])},kernel);}
}
