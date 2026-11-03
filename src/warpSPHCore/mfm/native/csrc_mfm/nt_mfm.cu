#include "nt_mfm_rows.h"
#include "execution/gpu.cuh"
namespace sph_native::mfm {
void run_gpu(const Tensors& t,const Buffers& b,int task,int kernel) {
    run(t,b,task,kernel,GpuLauncher{current_stream(t[QX])});
}
}
