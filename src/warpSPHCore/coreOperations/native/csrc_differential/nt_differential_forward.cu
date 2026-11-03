#include "nt_differential_rows.h"
#include "execution/gpu.cuh"
namespace sph_native {
void corrected_differential_gpu(const Tensors& t,const OptionalTensors& flat,const differential::Buffers& buffers,
        int operation,int support,int direction,int kernel,int scheme,int lap,bool dot,int mode) {
    const stable::accelerator::DeviceGuard guard(t[QX].get_device_index());
    const auto launch=GpuLauncher{current_stream(t[QX])};
    if (mode==0) execute(differential::Rows<0>{flat,buffers},t,support,direction,operation,launch,kernel,scheme,lap,dot);
    else if (mode==1) execute(differential::Rows<1>{flat,buffers},t,support,direction,operation,launch,kernel,scheme,lap,dot);
    else execute(differential::Rows<2>{flat,buffers},t,support,direction,operation,launch,kernel,scheme,lap,dot);
}
} // namespace sph_native
