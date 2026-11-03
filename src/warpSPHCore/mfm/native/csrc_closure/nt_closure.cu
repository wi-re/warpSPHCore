#include "nt_closure_rows.h"
#include "execution/gpu.cuh"
namespace sph_native::mfm {
void graph_gpu(const GraphBuffers& b) {graph(b,GpuLauncher{current_stream(b.values)});}
void segment_gpu(const SegmentBuffers& b) {segment(b,GpuLauncher{current_stream(b.values)});}
}
