#include "nt_closure_rows.h"
#include "execution/gpu.cuh"
namespace sph_native::mfm {
void graph_gpu(const GraphBuffers& b) {
    const stable::accelerator::DeviceGuard guard(b.values.get_device_index());
    graph(b,GpuLauncher{current_stream(b.values)});}
void segment_gpu(const SegmentBuffers& b) {
    const stable::accelerator::DeviceGuard guard(b.values.get_device_index());
    segment(b,GpuLauncher{current_stream(b.values)});}
}
