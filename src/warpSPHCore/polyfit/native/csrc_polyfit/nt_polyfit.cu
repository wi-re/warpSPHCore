#include "nt_polyfit_rows.h"
#include "execution/gpu.cuh"
namespace sph_native::polyfit {
void run_gpu(const Tensors& t,const Buffers& b,int support,int direction,int kernel,int role){
    execute(Dispatch<GpuLauncher>{b,role},t,support,direction,1,GpuLauncher{current_stream(t[QX])},kernel);
}
}
