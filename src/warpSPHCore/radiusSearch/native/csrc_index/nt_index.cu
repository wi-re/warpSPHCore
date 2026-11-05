#include "nt_index.h"
#include "execution/gpu.cuh"
namespace sph_native::search_index {
void copy_gpu(const Tensor& target,const void* source,size_t bytes) {
    const stable::accelerator::DeviceGuard guard(target.get_device_index());
    const auto status=cudaMemcpyAsync(target.mutable_data_ptr(),source,bytes,cudaMemcpyDeviceToDevice,current_stream(target));
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
void fill_gpu(const Tensor& offsets,const Tensor& counts,const Tensor& output,int64_t start) {
    const stable::accelerator::DeviceGuard guard(output.get_device_index());fill_rows(offsets,counts,output,start,GpuLauncher{current_stream(output)});
}
}

namespace sph_native::search_index {
void fixed_keys_gpu(const Tensor& cells,const Tensor& counts,const Tensor& output,int64_t buckets) {
    const stable::accelerator::DeviceGuard guard(cells.get_device_index());
    fixed_keys(cells,counts,output,buckets,GpuLauncher{current_stream(cells)});
}
}
