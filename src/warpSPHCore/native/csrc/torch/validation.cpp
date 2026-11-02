#include <limits>
#include "torch/validation.h"

namespace sph_native {
void tensor(const Tensor &x, const Tensor &example, ScalarType dtype, const char *name) {
    STD_TORCH_CHECK(x.defined(), name, " is required");
    STD_TORCH_CHECK(x.device() == example.device(), name, " must be on the positions device");
    STD_TORCH_CHECK(x.scalar_type() == dtype, name, " has an incompatible dtype");
    STD_TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    // The Python entry points enforce the supported AD policy. Stable Tensor
    // intentionally exposes no requires_grad query; do not reach into ATen.
}
void vector(const Tensor &x, const Tensor &example, ScalarType dtype, int64_t n, const char *name) {
    tensor(x, example, dtype, name);
    STD_TORCH_CHECK(x.dim() == 1 && x.size(0) == n, name, " has an incompatible shape");
}

void geometry(const Tensors &t, int support) {
    const auto &q = t[QX];
    const auto &r = t[RX];
    STD_TORCH_CHECK(q.device().is_cpu() || q.is_cuda(),
                    "Native backend supports CPU and CUDA/HIP tensors");
    STD_TORCH_CHECK(q.scalar_type() == ScalarType::Float || q.scalar_type() == ScalarType::Double,
                    "positions must be float32 or float64");
    STD_TORCH_CHECK(q.dim() == 2 && q.size(1) >= 1 && q.size(1) <= 3,
                    "positions must have shape [N, D], D=1,2,3");
    tensor(q, q, q.scalar_type(), "query positions");
    tensor(r, q, q.scalar_type(), "reference positions");
    STD_TORCH_CHECK(r.dim() == 2 && r.size(1) == q.size(1),
                    "reference positions have an incompatible shape");
    STD_TORCH_CHECK(r.size(0) <= std::numeric_limits<int32_t>::max(),
                    "per-row neighbor counts must fit int32");
    vector(t[QH], q, q.scalar_type(), q.size(0), "query supports");
    vector(t[RH], q, q.scalar_type(), r.size(0), "reference supports");
    vector(t[LO], q, q.scalar_type(), q.size(1), "domain min");
    vector(t[HI], q, q.scalar_type(), q.size(1), "domain max");
    vector(t[PERIODIC], q, ScalarType::Bool, q.size(1), "periodicity");
    STD_TORCH_CHECK(support >= 11 && support <= 15,
                    "Native pilot supports Gather, Scatter, MeanSymmetric, KernelMeanSymmetric and "
                    "SuperSymmetric");
}

Tensors prepare_pair_geometry(const Tensors &inputs, int support, int direction) {
    STD_TORCH_CHECK(inputs.size() == OUTPUT, "invalid native operator tensor layout");
    Tensors t = inputs;
    t.resize(SLOTS);
    geometry(t, support);
    const auto &q = t[QX];
    const auto n = q.size(0), nr = t[RX].size(0);
    STD_TORCH_CHECK(direction >= 0 && direction <= 14, "invalid operation direction");
    vector(t[QKIND], q, ScalarType::Int, n, "query kinds");
    vector(t[RKIND], q, ScalarType::Int, nr, "reference kinds");
    vector(t[COUNTS], q, ScalarType::Int, n, "neighbor counts");
    for (auto slot : {OFFSETS, NEIGHBORS}) {
        const auto dtype = t[slot].scalar_type();
        STD_TORCH_CHECK(dtype == ScalarType::Int || dtype == ScalarType::Long,
                        "CSR indices and offsets must be int32 or int64");
        tensor(t[slot], q, dtype, "CSR storage");
        STD_TORCH_CHECK(t[slot].dim() == 1, "CSR storage must be one dimensional");
    }
    STD_TORCH_CHECK(t[OFFSETS].size(0) == n, "CSR offsets must have one entry per query");
    return t;
}
Tensors prepare_operator(const Tensors &inputs, int support, int direction) {
    auto t = prepare_pair_geometry(inputs, support, direction);
    vector(t[MASS], t[QX], t[QX].scalar_type(), t[RX].size(0), "reference masses");
    return t;
}
Tensors prepare_field_operator(const Tensors &inputs, int support, int direction, int op,
                               int scheme, int lap, bool dot) {
    Tensors t = prepare_operator(inputs, support, direction);
    vector(t[DENSITY], t[QX], t[QX].scalar_type(), t[RX].size(0), "reference densities");
    return prepare_field_output(std::move(t), op, scheme, lap, dot);
}
Tensors prepare_field_output(Tensors t, int op, int scheme, int lap, bool dot) {
    STD_TORCH_CHECK(op >= 1 && op <= 7 && op != 6, "Unknown field operator");
    STD_TORCH_CHECK(scheme >= 1 && scheme <= 4, "Unknown gradient scheme");
    STD_TORCH_CHECK(lap >= 1 && lap <= 4, "Unknown Laplacian scheme");
    const auto &x = t[QX];
    const auto n = x.size(0), nr = t[RX].size(0), D = x.size(1);

    std::vector<int64_t> shape;
    if (op == 7)
        shape = {D, D};
    else {
        tensor(t[RVALUE], x, x.scalar_type(), "reference values");
        STD_TORCH_CHECK(t[RVALUE].dim() >= 1 && t[RVALUE].dim() <= 3 && t[RVALUE].size(0) == nr,
                        "reference values must be scalar, vector or matrix fields");
        for (int d = 1; d < t[RVALUE].dim(); ++d) {
            STD_TORCH_CHECK(t[RVALUE].size(d) == D, "field axes must match spatial dimension");
            shape.push_back(D);
        }

        if (op != 1) {
            tensor(t[QVALUE], x, x.scalar_type(), "query values");
            STD_TORCH_CHECK(t[QVALUE].dim() == t[RVALUE].dim() && t[QVALUE].size(0) == n,
                            "query values have an incompatible shape");
            for (int d = 1; d < t[QVALUE].dim(); ++d)
                STD_TORCH_CHECK(t[QVALUE].size(d) == D,
                                "query field axes must match spatial dimension");
        }
        if (op == 2)
            shape.push_back(D);
        if (op == 3) {
            STD_TORCH_CHECK(!shape.empty(), "Divergence requires a vector or matrix field");
            shape.erase(dot ? shape.begin() : shape.end() - 1);
        }
        if (op == 4) {
            STD_TORCH_CHECK(D == 1 || !shape.empty(), "Curl requires a vector or matrix field");
            if (D == 1)
                shape.clear();
            else if (D == 2) {
                shape.erase(shape.begin());
                if (shape.empty())
                    shape.push_back(1);
            }
        }
        if (op == 5 && lap == 3)
            STD_TORCH_CHECK(D == 1 || t[RVALUE].dim() > 1,
                            "Dot Laplacian requires a vector or matrix field");
    }
    if (scheme == 2 && op != 1 && op != 7) {
        vector(t[QDENSITY], x, x.scalar_type(), n, "query densities");
    }
    shape.insert(shape.begin(), n);
    t[OUTPUT] = stable::new_empty(x, shape);
    return t;
}
} // namespace sph_native
