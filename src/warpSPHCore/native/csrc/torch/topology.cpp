#include "torch/validation.h"
#include "execution/dispatch.h"
#include "execution/cpu.h"
#include "execution/transpose.h"
namespace sph_native {
void transpose_count_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&);
void transpose_fill_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                        const Tensor&, const Tensor&, const Tensor&);
namespace {
void index(const Tensor& value, const Tensor& example, const char* name) {
    STD_TORCH_CHECK(value.scalar_type() == ScalarType::Int || value.scalar_type() == ScalarType::Long,
                    name, " must be int32 or int64");
    tensor(value, example, value.scalar_type(), name);
    STD_TORCH_CHECK(value.dim() == 1, name, " must be one dimensional");
}
void metadata(const Tensor& offsets, const Tensor& counts, const Tensor& ids, const Tensor& target) {
    index(offsets, counts, "CSR offsets"); index(ids, counts, "CSR IDs");
    vector(counts, counts, ScalarType::Int, offsets.numel(), "CSR counts");
    vector(target, counts, ScalarType::Int, target.numel(), "transpose counts");
    STD_TORCH_CHECK(offsets.numel() <= INT32_MAX, "transpose per-row counts must fit int32");
}
}
// Counting and integer fill are construction, not validation. No CSR contents
// are inspected on the host for GPU inputs, and capacity avoids an edge-count
// readback. Sorting makes floating-point owned reductions repeatable.
void csr_transpose_count(const Tensor& offsets, const Tensor& counts, const Tensor& ids, const Tensor& result) {
    metadata(offsets, counts, ids, result);
    if (counts.is_cuda()) {
#ifdef WITH_CUDA
        transpose_count_gpu(offsets, counts, ids, result);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else CpuLauncher{}(counts.numel(), TransposeCount{index_view(offsets), index_view(ids),
                                counts.const_data_ptr<int32_t>(), result.mutable_data_ptr<int32_t>()});
}
void csr_transpose_fill(const Tensor& offsets, const Tensor& counts, const Tensor& ids,
        const Tensor& target_offsets, const Tensor& target_counts, const Tensor& cursors, const Tensor& result) {
    metadata(offsets, counts, ids, target_counts);
    vector(target_offsets, counts, ScalarType::Long, target_counts.numel(), "transpose offsets");
    vector(cursors, counts, ScalarType::Int, target_counts.numel(), "transpose cursors");
    vector(result, counts, ScalarType::Long, ids.numel(), "transpose IDs");
    if (counts.is_cuda()) {
#ifdef WITH_CUDA
        transpose_fill_gpu(offsets, counts, ids, target_offsets, target_counts, cursors, result);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else {
        CpuLauncher{}(counts.numel(), TransposeFill{index_view(offsets), index_view(ids),
            counts.const_data_ptr<int32_t>(), target_offsets.const_data_ptr<int64_t>(),
            cursors.mutable_data_ptr<int32_t>(), result.mutable_data_ptr<int64_t>()});
        CpuLauncher{}(target_counts.numel(), SortTranspose{target_offsets.const_data_ptr<int64_t>(),
            target_counts.const_data_ptr<int32_t>(), result.mutable_data_ptr<int64_t>()});
    }
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("csr_transpose_count(Tensor offsets, Tensor counts, Tensor ids, Tensor(a!) result) -> ()");
    m.impl("csr_transpose_count", TORCH_BOX(&csr_transpose_count));
    m.def("csr_transpose_fill(Tensor offsets, Tensor counts, Tensor ids, Tensor target_offsets, "
          "Tensor target_counts, Tensor(a!) cursors, Tensor(b!) result) -> ()");
    m.impl("csr_transpose_fill", TORCH_BOX(&csr_transpose_fill));
}
} // namespace sph_native
