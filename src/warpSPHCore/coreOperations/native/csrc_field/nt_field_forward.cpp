#include "nt_field_dispatch.h"
#include "execution/cpu.h"
namespace sph_native {
void field_product_gpu(const Tensors&, const OptionalTensors&, const neighborhood::FieldBuffers&,
                      int, int, int, int, int, int, bool, bool, bool);
namespace {
Tensors prepare(const OptionalTensors& inputs, const OptionalTensors& corrections,
        int operation, int support, int direction, int kernel, int scheme, int lap,
        bool dot, bool consistent, const Tensor* output = nullptr) {
    STD_TORCH_CHECK(operation >= 1 && operation <= 5, "Field derivatives support interpolation, gradient, divergence, curl and Laplacian");
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    return prepare_corrected_operator(inputs, corrections, support, direction, operation, scheme,
                                       lap, dot, consistent, output);
}
void field(const Tensor& value, const Tensor& example, const char* name) {
    tensor(value, example, example.scalar_type(), name);
    STD_TORCH_CHECK(value.dim() == example.dim(), name, " has an incompatible rank");
    for (int d = 0; d < value.dim(); ++d)
        STD_TORCH_CHECK(value.size(d) == example.size(d), name, " has an incompatible shape");
}
Tensor empty_field(const Tensor& example) {
    std::vector<int64_t> shape;
    for (int d = 0; d < example.dim(); ++d) shape.push_back(example.size(d));
    return stable::new_empty(example, shape);
}
}
Tensor field_product(const OptionalTensors& inputs, const OptionalTensors& corrections,
        const std::optional<Tensor>& query, const std::optional<Tensor>& reference,
        int64_t operation, int64_t support, int64_t direction, int64_t kernel,
        int64_t scheme, int64_t lap, bool dot, bool consistent) {
    auto t = prepare(inputs, corrections, operation, support, direction, kernel, scheme, lap, dot, consistent);
    if (query && t[QVALUE].defined()) field(*query, t[QVALUE], "query tangent");
    if (reference) field(*reference, t[RVALUE], "reference tangent");
    const neighborhood::FieldBuffers buffers{query && t[QVALUE].defined() ? *query : Tensor{}, reference ? *reference : Tensor{}, {}, {}};
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        field_product_gpu(t, corrections, buffers, operation, support, direction, kernel, scheme, lap, dot, consistent, false);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute_field(t, corrections, neighborhood::FieldExecutor<false, false>{buffers},
            operation, support, direction, kernel, scheme, lap, dot, consistent, CpuLauncher{});
    return t[OUTPUT];
}
OptionalTensors field_pullback(const OptionalTensors& inputs, const OptionalTensors& corrections,
        const Tensor& cotangent, const Tensors& transpose, bool query, bool reference,
        int64_t operation, int64_t support, int64_t direction, int64_t kernel,
        int64_t scheme, int64_t lap, bool dot, bool consistent) {
    auto t = prepare(inputs, corrections, operation, support, direction, kernel, scheme, lap, dot, consistent, &cotangent);
    query = query && t[QVALUE].defined();
    if (reference) {
        STD_TORCH_CHECK(transpose.size() == 3, "Reference VJP requires a CSR transpose");
        vector(transpose[0], t[QX], ScalarType::Long, t[RX].size(0), "CSC offsets");
        vector(transpose[1], t[QX], ScalarType::Int, t[RX].size(0), "CSC counts");
        vector(transpose[2], t[QX], ScalarType::Long, t[NEIGHBORS].numel(), "CSC IDs");
    }
    const neighborhood::FieldBuffers buffers{query ? empty_field(t[QVALUE]) : Tensor{},
        reference ? empty_field(t[RVALUE]) : Tensor{}, cotangent, transpose};
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        field_product_gpu(t, corrections, buffers, operation, support, direction, kernel, scheme, lap, dot, consistent, true);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute_field(t, corrections, neighborhood::FieldExecutor<false, true>{buffers},
            operation, support, direction, kernel, scheme, lap, dot, consistent, CpuLauncher{});
    return {query ? std::optional<Tensor>(buffers.query) : std::nullopt,
            reference ? std::optional<Tensor>(buffers.reference) : std::nullopt};
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("field_product(Tensor?[] inputs, Tensor?[] corrections, Tensor? query, Tensor? reference, "
          "int operation, int support, int direction, int kernel, int scheme, int lap, bool dot, bool consistent) -> Tensor");
    m.impl("field_product", TORCH_BOX(&field_product));
    m.def("field_pullback(Tensor?[] inputs, Tensor?[] corrections, Tensor cotangent, Tensor[] transpose, "
          "bool query, bool reference, int operation, int support, int direction, int kernel, int scheme, int lap, bool dot, bool consistent) -> Tensor?[]");
    m.impl("field_pullback", TORCH_BOX(&field_pullback));
}
} // namespace sph_native
