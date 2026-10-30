#include <torch/csrc/stable/library.h>
namespace sph_native {
bool has_gpu() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
}
STABLE_TORCH_LIBRARY(warpsph_native, m) { m.def("has_gpu() -> bool"); m.impl("has_gpu", TORCH_BOX(&has_gpu)); }
} // namespace sph_native
