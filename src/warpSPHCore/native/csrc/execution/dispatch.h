#pragma once
#include "torch/types.h"
#include "dataTypes/state.h"

namespace sph_native {
template <typename T> const T *ptr(const Tensor &t) {
    return t.defined() ? t.const_data_ptr<T>() : nullptr;
}
inline IndexView index_view(const Tensor &t) {
    return {t.defined() ? t.const_data_ptr() : nullptr,
            !t.defined() || t.scalar_type() == ScalarType::Long};
}
template <typename T, int D, class Task, class Launcher>
void execute_typed(const Task &task, const Tensors &t, int support, int direction, int operation,
                   const Launcher &launch, int kernel = 0, int scheme = 3, int laplacian = 2,
                   bool dot = false) {
    RowArgs<T, D> a{};
    a.qx = ptr<T>(t[QX]);
    a.rx = ptr<T>(t[RX]);
    a.qh = ptr<T>(t[QH]);
    a.rh = ptr<T>(t[RH]);
    a.mass = ptr<T>(t[MASS]);
    a.qkind = ptr<int32_t>(t[QKIND]);
    a.rkind = ptr<int32_t>(t[RKIND]);
    a.lo = ptr<T>(t[LO]);
    a.hi = ptr<T>(t[HI]);
    a.periodic = ptr<bool>(t[PERIODIC]);
    a.qvalue = ptr<T>(t[QVALUE]);
    a.rvalue = ptr<T>(t[RVALUE]);
    a.density = ptr<T>(t[DENSITY]);
    a.offsets = index_view(t[OFFSETS]);
    a.neighbors = index_view(t[NEIGHBORS]);
    a.counts = t[COUNTS].mutable_data_ptr<int32_t>();
    a.collected = t[NEIGHBORS].defined() && t[NEIGHBORS].scalar_type() == ScalarType::Long
                      ? t[NEIGHBORS].mutable_data_ptr<int64_t>()
                      : nullptr;
    a.output = t[OUTPUT].defined() ? t[OUTPUT].mutable_data_ptr<T>() : nullptr;
    a.qdensity = ptr<T>(t[QDENSITY]);
    a.kernel = kernel;
    a.scheme = scheme;
    a.laplacian_mode = laplacian;
    a.dot_mode = dot;
    if (t[RVALUE].defined() && t[RVALUE].dim() > 0) {
        a.fields = 1;
        for (int d = 1; d < t[RVALUE].dim(); ++d)
            a.fields *= int(t[RVALUE].size(d));
    }
    if (t[OUTPUT].defined()) {
        a.outputs = 1;
        for (int d = 1; d < t[OUTPUT].dim(); ++d)
            a.outputs *= int(t[OUTPUT].size(d));
    }
    a.references = t[RX].size(0);
    a.support = support;
    a.direction = direction;
    a.operation = operation;
    const auto n = t[QX].size(0);
    task(n, a, launch);
}
template <class Task, class Launcher>
void execute(const Task &task, const Tensors &t, int support, int direction, int operation,
             const Launcher &launch, int kernel = 0, int scheme = 3, int laplacian = 2,
             bool dot = false) {
    if (t[QX].scalar_type() == ScalarType::Float) {
        switch (t[QX].size(1)) {
        case 1:
            execute_typed<float, 1>(task, t, support, direction, operation, launch, kernel, scheme,
                                    laplacian, dot);
            break;
        case 2:
            execute_typed<float, 2>(task, t, support, direction, operation, launch, kernel, scheme,
                                    laplacian, dot);
            break;
        case 3:
            execute_typed<float, 3>(task, t, support, direction, operation, launch, kernel, scheme,
                                    laplacian, dot);
            break;
        }
    } else {
        switch (t[QX].size(1)) {
        case 1:
            execute_typed<double, 1>(task, t, support, direction, operation, launch, kernel, scheme,
                                     laplacian, dot);
            break;
        case 2:
            execute_typed<double, 2>(task, t, support, direction, operation, launch, kernel, scheme,
                                     laplacian, dot);
            break;
        case 3:
            execute_typed<double, 3>(task, t, support, direction, operation, launch, kernel, scheme,
                                     laplacian, dot);
            break;
        }
    }
}
} // namespace sph_native
