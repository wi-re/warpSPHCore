#pragma once
#include "execution/traversal.h"
#include "math/queryGeometry.h"
namespace sph_native {
// A packed forward row retains the original query and support metadata while
// providing a register-resident reference record and cached query field.
// Only the Difference/Gather path constructs this adapter.
template <class T, int D> struct PackedFieldArgs : RowArgs<T, D> {
    const T *reference;
    const T *query_field;
};
template <class T, int D> SPH_HD T reference_volume(const RowArgs<T, D> &a, int64_t j) {
    return a.mass[j] / a.density[j];
}
template <class T, int D> SPH_HD T reference_volume(const PackedFieldArgs<T, D> &a, int64_t) {
    return a.reference[D];
}
template <class T, int D>
SPH_HD T field_weight(const PackedFieldArgs<T, D> &a, int64_t, int64_t, int c, bool = false) {
    return (a.reference[D + 1 + c] - a.query_field[c]) * a.reference[D];
}
template <class T, int D>
SPH_HD T field_weight(const RowArgs<T, D> &a, int64_t i, int64_t j, int c, bool lap = false) {
    T fi = a.qvalue[i * a.fields + c], fj = a.rvalue[j * a.fields + c],
      volume = reference_volume(a, j);
    if (lap)
        return (fj - fi) *
               (a.scheme == 2 ? a.mass[j] * a.qdensity[i] / (a.density[j] * a.density[j]) : volume);
    if (a.scheme == 1)
        return fj * volume;
    if (a.scheme == 2)
        return a.mass[j] * a.qdensity[i] *
               (fi / (a.qdensity[i] * a.qdensity[i]) + fj / (a.density[j] * a.density[j]));
    return (a.scheme == 3 ? fj - fi : fj + fi) * volume;
}
template <class T, int D, class Pair> struct FieldOperatorRow {
    RowArgs<T, D> a;
    SPH_HD void operator()(int64_t i) const {
        for (int c = 0; c < a.outputs; ++c)
            a.output[i * a.outputs + c] = T(0);
        if (!query_allowed(a.qkind[i], a.direction))
            return;
        T sum[D * D * D] = {};
        int64_t start = a.offsets[i];
        const QueryGeometry<T, D> geometry(a.qx + i * D, a.lo, a.hi, a.periodic);
        const bool prepared = a.kernel == 0 && a.support == 11;
        const Wendland2Primal<T, D> kernel(a.qh[i]);
        for (int32_t e = 0; e < a.counts[i]; ++e) {
            int64_t j = a.neighbors[start + e];
            if (!reference_allowed(a.rkind[j], a.direction))
                continue;
            T dx[D], w, g, r = geometry.distance(a.rx + j * D, dx);
            if (prepared)
                kernel.evaluate(r, w, g);
            else
                pair_kernel(a, i, j, r, w, g);
            Pair::accumulate(a, i, j, dx, r, w, g, sum);
        }
        for (int c = 0; c < a.outputs; ++c)
            a.output[i * a.outputs + c] = sum[c];
    }
};
} // namespace sph_native
